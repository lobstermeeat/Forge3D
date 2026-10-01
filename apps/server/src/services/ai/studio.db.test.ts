/**
 * The views step on Postgres: the ai_generations columns as drizzle-kit creates them, and the
 * guard that lets only one of two overlapping polls start the preview. It runs where
 * DATABASE_URL reaches a server that lets it create a database (CI's Postgres service), in a
 * database of its own made from the schema and dropped afterwards, so it never touches the app's
 * tables. Without a reachable server it is skipped.
 */
import { randomUUID } from 'node:crypto';
import { afterAll, beforeAll, describe, expect, it, vi } from 'vitest';
import postgres from 'postgres';
import sharp from 'sharp';
import { drizzle } from 'drizzle-orm/postgres-js';
import * as schema from '../../db/schema';
import type { StorageProvider } from '../storage';
import { MOCK_VIEW_AZIMUTHS, MockWorkers } from './providers/mock';
import { AIStudio, drizzleGenerationStore, type GenerationStore } from './studio';
import type { StudioWorkers } from './types';

const url =
  process.env['DATABASE_URL'] ?? 'postgresql://forge3d:forge3d_dev@localhost:5432/forge3d';

async function reachable(): Promise<boolean> {
  const sql = postgres(url, { max: 1, connect_timeout: 3, onnotice: () => {} });
  try {
    await sql`select 1`;
    return true;
  } catch {
    return false;
  } finally {
    await sql.end({ timeout: 1 });
  }
}

const available = await reachable();
if (!available) console.info('[studio.db.test] No Postgres at DATABASE_URL, so skipped');

function memoryStorage(): StorageProvider & { files: Map<string, Buffer> } {
  const files = new Map<string, Buffer>();
  return {
    files,
    async write(key, data) {
      files.set(key, data);
      return `https://files.test/${key}`;
    },
    async read(key) {
      const data = files.get(key);
      if (!data) throw new Error(`no ${key}`);
      return data;
    },
    async delete(key) {
      files.delete(key);
    },
    getUrl: (key) => `https://files.test/${key}`,
  };
}

const photo = (side: number) =>
  sharp({ create: { width: side, height: side, channels: 3, background: '#888' } })
    .png()
    .toBuffer();

describe.skipIf(!available)('AIStudio on Postgres (ai_generations)', () => {
  const name = `forge3d_test_${randomUUID().replace(/-/g, '').slice(0, 16)}`;
  let admin: postgres.Sql;
  let sql: postgres.Sql;
  let store: GenerationStore;

  /** The generation's row as Postgres has it */
  const row = async (id: string) => {
    const [found] = await sql`
      select status, job_id, job_kind, views, views_error from ai_generations where id = ${id}`;
    return found!;
  };

  beforeAll(async () => {
    admin = postgres(url, { max: 1, onnotice: () => {} });
    await admin.unsafe(`create database "${name}"`);
    const testUrl = new URL(url);
    testUrl.pathname = `/${name}`;
    sql = postgres(testUrl.toString(), { max: 4, onnotice: () => {} });
    // The tables as `drizzle-kit push` makes them from the schema
    const { generateDrizzleJson, generateMigration } = await import('drizzle-kit/api');
    const statements = await generateMigration(
      generateDrizzleJson({}),
      generateDrizzleJson(schema),
    );
    for (const statement of statements) await sql.unsafe(statement);
    const db = drizzle(sql, { schema });
    await db.insert(schema.users).values({ id: 'u1', email: 'u1@example.test', name: 'Tester' });
    store = drizzleGenerationStore(db);
  }, 60_000);

  afterAll(async () => {
    await sql?.end({ timeout: 5 });
    await admin?.unsafe(`drop database if exists "${name}" with (force)`);
    await admin?.end({ timeout: 5 });
  });

  it('keeps the views with the generation, or why there are none', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    try {
      const storage = memoryStorage();
      const studio = new AIStudio({ workers: new MockWorkers(0), store, storage, multiview: true });
      let gen = await studio.startFromPrompt('u1', 'a stack of old books with a candle');
      gen = await studio.get('u1', gen.id);
      gen = await studio.pick('u1', gen.id, gen.recommended!);
      expect(await row(gen.id)).toMatchObject({
        status: 'viewing',
        job_kind: 'views',
        views: null,
        views_error: null,
      });

      gen = await studio.get('u1', gen.id);
      const stored = await row(gen.id);
      expect(stored).toMatchObject({
        status: 'previewing',
        job_kind: 'preview',
        views_error: null,
      });
      expect(stored['views']).toEqual(gen.views);
      expect((stored['views'] as { azimuth: number }[]).map((view) => view.azimuth)).toEqual(
        MOCK_VIEW_AZIMUTHS,
      );

      expect((await studio.get('u1', gen.id)).status).toBe('reviewing');
      await studio.keep('u1', gen.id);
      expect((await studio.get('u1', gen.id)).status).toBe('done');
      expect((await row(gen.id))['views']).toEqual(gen.views);

      // A photo whose views fail goes on to its preview, and the reason is kept
      const small = await studio.startFromPhoto('u1', await photo(100));
      await studio.get('u1', small.id);
      expect(await row(small.id)).toMatchObject({
        status: 'previewing',
        views: null,
        views_error: expect.stringContaining('128 px'),
      });
      expect((await studio.recent('u1')).map((g) => g.status)).toEqual(['previewing', 'done']);
    } finally {
      warn.mockRestore();
    }
  });

  it('lets one of two overlapping polls start the preview', async () => {
    let release: () => void = () => {};
    const downloaded = new Promise<void>((resolve) => (release = resolve));
    const view = await photo(64);
    let models = 0;
    const cancelled: string[] = [];
    const workers: StudioWorkers = {
      name: 'db-test',
      prompts: false,
      multiview: true,
      startReferences: () => Promise.reject(new Error('photos only')),
      references: () => Promise.reject(new Error('photos only')),
      startViews: async () => 'views-1',
      views: async () => ({
        status: 'done',
        output: {
          views: [{ file: { url: 'https://r2.test/view-0.png' }, azimuth: 0, elevation: 0 }],
          seconds: 2,
        },
      }),
      startModel: async () => `model-${++models}`,
      model: async () => ({ status: 'running' }),
      cancel: async (kind, jobId) => {
        cancelled.push(`${kind}:${jobId}`);
      },
    };
    const studio = new AIStudio({
      workers,
      store,
      storage: memoryStorage(),
      multiview: true,
      // Both polls wait here with the views done, then race to start the preview
      fetchImpl: async () => {
        await downloaded;
        return new Response(view);
      },
    });
    const gen = await studio.startFromPhoto('u1', await photo(256));
    const polls = [studio.get('u1', gen.id), studio.get('u1', gen.id)];
    await new Promise((resolve) => setTimeout(resolve, 50));
    release();
    for (const poll of await Promise.all(polls)) expect(poll.status).toBe('previewing');

    expect(models).toBe(2);
    const kept = (await row(gen.id))['job_id'] as string;
    expect(['model-1', 'model-2']).toContain(kept);
    expect(cancelled).toEqual([`model:${kept === 'model-1' ? 'model-2' : 'model-1'}`]);
  });
});
