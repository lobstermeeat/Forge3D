/**
 * The views step, the texture options and the judge's pick on Postgres: the ai_generations columns
 * as drizzle-kit creates them, and the guards that let only one of two overlapping polls start the
 * preview, or the textures job. It runs where DATABASE_URL reaches a server that lets it create a database
 * (CI's Postgres service), in a database of its own made from the schema and dropped afterwards,
 * so it never touches the app's tables. Without a reachable server it is skipped.
 */
import { randomUUID } from 'node:crypto';
import { afterAll, beforeAll, describe, expect, it, vi } from 'vitest';
import postgres from 'postgres';
import sharp from 'sharp';
import { drizzle } from 'drizzle-orm/postgres-js';
import type { Database } from '../../db';
import * as schema from '../../db/schema';
import type { StorageProvider } from '../storage';
import { MOCK_JUDGE_WHY, MOCK_VIEW_AZIMUTHS, MockWorkers } from './providers/mock';
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
  let db: Database;
  let store: GenerationStore;

  /** The generation's row as Postgres has it */
  const row = async (id: string) => {
    const [found] = await sql`
      select status, job_id, job_kind, views, views_error from ai_generations where id = ${id}`;
    return found!;
  };

  /** The generation's final and texture options as Postgres has them */
  const texturesRow = async (id: string) => {
    const [found] = await sql`
      select status, final_pipeline, textures_status, textures_job_id, textures, textures_error
      from ai_generations where id = ${id}`;
    return found!;
  };

  /** The judge's pick and the creator's choice as Postgres has them */
  const pickRow = async (id: string) => {
    const [found] = await sql`
      select textures_pick, textures_pick_why, textures_judge_error, textures_pick_applied,
        textures_chosen
      from ai_generations where id = ${id}`;
    return found!;
  };
  const NO_PICK = {
    textures_pick: null,
    textures_pick_why: null,
    textures_judge_error: null,
    textures_pick_applied: false,
    textures_chosen: null,
  };

  /** Another user, so each test's models in progress stay under the limit */
  const user = async (id: string) => {
    await db.insert(schema.users).values({ id, email: `${id}@example.test`, name: 'Tester' });
    return id;
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
    db = drizzle(sql, { schema });
    await user('u1');
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

  it('keeps the texture options with the generation, through a restart', async () => {
    const id = await user('u-textures');
    const storage = memoryStorage();
    // The GPU jobs outlive a server process, so a restarted studio gets the same workers
    const workers = new MockWorkers(0);
    const studio = new AIStudio({ workers, store, storage });
    let gen = await studio.startFromPrompt(id, 'a brass ship compass');
    gen = await studio.get(id, gen.id);
    gen = await studio.pick(id, gen.id, gen.recommended!);
    expect((await studio.get(id, gen.id)).status).toBe('reviewing');
    await studio.keep(id, gen.id);
    gen = await studio.get(id, gen.id);
    // The final is done at once, and its textures job is kept with it
    expect(gen).toMatchObject({ status: 'done', textures: { status: 'running', options: [] } });
    expect(await texturesRow(gen.id)).toMatchObject({
      status: 'done',
      final_pipeline: '1024_cascade',
      textures_status: 'running',
      textures_job_id: expect.stringMatching(/^mock-/),
      textures: null,
      textures_error: null,
    });

    const restarted = new AIStudio({ workers, store, storage });
    gen = await restarted.get(id, gen.id);
    expect(gen.textures).toMatchObject({ status: 'done', error: null });
    expect(gen.textures!.options).toHaveLength(3);
    const stored = await texturesRow(gen.id);
    expect(stored).toMatchObject({
      status: 'done',
      textures_status: 'done',
      textures_job_id: null,
      textures_error: null,
    });
    expect(stored['textures']).toEqual(gen.textures!.options);

    // Another picture clears them, with the final
    await restarted.pick(id, gen.id, 0);
    expect(await texturesRow(gen.id)).toMatchObject({
      status: 'previewing',
      final_pipeline: null,
      textures_status: null,
      textures_job_id: null,
      textures: null,
      textures_error: null,
    });
  });

  it("keeps the judge's pick and the creator's choice with the generation", async () => {
    const id = await user('u-judge');
    const storage = memoryStorage();
    const workers = new MockWorkers(0);
    const studio = new AIStudio({ workers, store, storage, textureJudge: true });
    let gen = await studio.startFromPrompt(id, 'a brass desk lamp');
    // The columns' defaults: no pick, not applied, nothing chosen
    expect(await pickRow(gen.id)).toEqual(NO_PICK);

    gen = await studio.get(id, gen.id);
    gen = await studio.pick(id, gen.id, gen.recommended!);
    expect((await studio.get(id, gen.id)).status).toBe('reviewing');
    await studio.keep(id, gen.id);
    expect((await studio.get(id, gen.id)).textures?.status).toBe('running');
    gen = await studio.get(id, gen.id);
    expect(gen.textures).toMatchObject({
      status: 'done',
      recommended: { number: 3, why: MOCK_JUDGE_WHY, applied: false },
      chosen: null,
      judgeError: null,
    });
    expect(await pickRow(gen.id)).toEqual({
      ...NO_PICK,
      textures_pick: 3,
      textures_pick_why: MOCK_JUDGE_WHY,
    });

    // The panel applied the pick, then the creator chose another
    await studio.chooseTexture(id, gen.id, 3, 'judge');
    await studio.chooseTexture(id, gen.id, 2, 'creator');
    expect(await pickRow(gen.id)).toEqual({
      ...NO_PICK,
      textures_pick: 3,
      textures_pick_why: MOCK_JUDGE_WHY,
      textures_pick_applied: true,
      textures_chosen: 2,
    });

    // A restarted server tells the panel the same, so the pick never goes in again
    const restarted = new AIStudio({ workers, store, storage, textureJudge: true });
    expect((await restarted.get(id, gen.id)).textures).toMatchObject({
      recommended: { number: 3, applied: true },
      chosen: 2,
    });

    // Another picture clears them, with the final
    await restarted.pick(id, gen.id, 0);
    expect(await pickRow(gen.id)).toEqual(NO_PICK);
  });

  it('lets one of two overlapping polls start the textures job', async () => {
    const id = await user('u-race');
    let release: () => void = () => {};
    const ready = new Promise<void>((resolve) => (release = resolve));
    let texturesJobs = 0;
    const cancelled: string[] = [];
    const workers: StudioWorkers = {
      name: 'db-test',
      prompts: false,
      multiview: false,
      startReferences: () => Promise.reject(new Error('photos only')),
      references: () => Promise.reject(new Error('photos only')),
      startViews: () => Promise.reject(new Error('no views')),
      views: () => Promise.reject(new Error('no views')),
      startModel: async ({ mode }) => `${mode}-1`,
      model: async (jobId) => ({
        status: 'done',
        output: {
          file: { data: Buffer.from(jobId) },
          seed: 11,
          triangles: 1,
          bytes: 1,
          seconds: 1,
          credits: [],
          pipeline: jobId.startsWith('final') ? '1024_cascade' : '512',
        },
      }),
      // Both polls wait here, then race to record their textures job
      startTextures: async () => {
        await ready;
        return `textures-${++texturesJobs}`;
      },
      textures: async () => ({ status: 'running' }),
      cancel: async (kind, jobId) => {
        cancelled.push(`${kind}:${jobId}`);
      },
    };
    const studio = new AIStudio({ workers, store, storage: memoryStorage() });
    const gen = await studio.startFromPhoto(id, await photo(256));
    expect((await studio.get(id, gen.id)).status).toBe('reviewing');
    await studio.keep(id, gen.id);

    const polls = [studio.get(id, gen.id), studio.get(id, gen.id)];
    await new Promise((resolve) => setTimeout(resolve, 50));
    release();
    for (const poll of await Promise.all(polls)) {
      expect(poll).toMatchObject({ status: 'done', textures: { status: 'running' } });
    }
    expect(texturesJobs).toBe(2);
    const kept = (await texturesRow(gen.id))['textures_job_id'] as string;
    expect(['textures-1', 'textures-2']).toContain(kept);
    expect(cancelled).toEqual([`model:${kept === 'textures-1' ? 'textures-2' : 'textures-1'}`]);
  });
});
