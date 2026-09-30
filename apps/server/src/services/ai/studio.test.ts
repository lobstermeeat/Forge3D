import { describe, expect, it } from 'vitest';
import sharp from 'sharp';
import { randomUUID } from 'node:crypto';
import type { StorageProvider } from '../storage';
import {
  AIStudio,
  LIMITS,
  StudioError,
  type GenerationRecord,
  type GenerationStore,
} from './studio';
import type { ModelOutput, ReferencesOutput, StudioWorkers, WorkerJobState } from './types';

/** Generations in memory, shaped like the ai_generations table's rows. */
function memoryStore(): GenerationStore & { rows: Map<string, GenerationRecord> } {
  const rows = new Map<string, GenerationRecord>();
  let clock = Date.parse('2026-09-30T00:00:00Z');
  return {
    rows,
    async create(values) {
      const row = {
        id: randomUUID(),
        sceneId: null,
        prompt: null,
        imageUrl: null,
        jobId: null,
        jobKind: null,
        referenceImages: null,
        seed: null,
        previewUrl: null,
        previewTriangles: null,
        finalUrl: null,
        finalTriangles: null,
        credits: null,
        resultAssetId: null,
        creditsUsed: 0,
        durationMs: null,
        errorMessage: null,
        source: 'prompt',
        status: 'drawing',
        createdAt: new Date((clock += 1000)),
        updatedAt: new Date(clock),
        ...values,
      } as GenerationRecord;
      rows.set(row.id, row);
      return { ...row };
    },
    async find(id, userId) {
      const row = rows.get(id);
      return row && row.userId === userId ? { ...row } : null;
    },
    async update(id, patch) {
      const row = { ...rows.get(id)!, ...patch } as GenerationRecord;
      rows.set(id, row);
      return { ...row };
    },
    async finishJob(id, jobId, patch) {
      if (rows.get(id)?.jobId !== jobId) return null;
      return this.update(id, patch);
    },
    async recent(userId, limit) {
      return [...rows.values()]
        .filter((r) => r.userId === userId)
        .sort((a, b) => b.createdAt.getTime() - a.createdAt.getTime())
        .slice(0, limit);
    },
  };
}

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

/** Workers whose jobs finish (or fail) when the test says so. */
function scriptedWorkers(prompts = true) {
  const references = new Map<string, WorkerJobState<ReferencesOutput> | Error>();
  const models = new Map<string, WorkerJobState<ModelOutput> | Error>();
  const started: { kind: string; input: Record<string, unknown> }[] = [];
  let next = 1;
  const workers: StudioWorkers = {
    name: 'test-workers',
    prompts,
    async startReferences(input) {
      const id = `ref-${next++}`;
      started.push({ kind: 'references', input });
      references.set(id, { status: 'running' });
      return id;
    },
    async references(jobId) {
      const state = references.get(jobId)!;
      if (state instanceof Error) throw state;
      return state;
    },
    async startModel(input) {
      const id = `model-${next++}`;
      started.push({ kind: input.mode, input });
      models.set(id, { status: 'running' });
      return id;
    },
    async model(jobId) {
      const state = models.get(jobId)!;
      if (state instanceof Error) throw state;
      return state;
    },
  };
  return { workers, references, models, started };
}

const png = (colour: string) =>
  sharp({ create: { width: 256, height: 256, channels: 4, background: colour } })
    .png()
    .toBuffer();

const modelOutput = (seed: number, mode: string): ModelOutput => ({
  file: { data: Buffer.from(`glb-${mode}-${seed}`) },
  seed,
  triangles: mode === 'final' ? 100_000 : 30_000,
  bytes: 10,
  seconds: mode === 'final' ? 40.5 : 12.25,
  credits: ['Built with DINOv3'],
});

function setup(prompts = true) {
  const store = memoryStore();
  const storage = memoryStorage();
  const jobs = scriptedWorkers(prompts);
  const studio = new AIStudio({ workers: jobs.workers, store, storage });
  return { studio, store, storage, ...jobs };
}

describe('AIStudio', () => {
  it('takes a prompt through pictures, preview and final', async () => {
    const { studio, storage, references, models, started } = setup();
    let gen = await studio.startFromPrompt('u1', '  a brass pocket watch ');
    expect(gen).toMatchObject({
      status: 'drawing',
      prompt: 'a brass pocket watch',
      source: 'prompt',
    });
    expect(started[0]).toEqual({
      kind: 'references',
      input: { prompt: 'a brass pocket watch', count: 4, requestId: gen.id },
    });

    // Still drawing
    expect((await studio.get('u1', gen.id)).status).toBe('drawing');

    references.set('ref-1', {
      status: 'done',
      output: {
        images: await Promise.all(
          ['#f00', '#0f0', '#00f', '#ff0'].map(async (c, i) => ({
            file: { data: await png(c) },
            seed: 7 + i,
          })),
        ),
      },
    });
    gen = await studio.get('u1', gen.id);
    expect(gen.status).toBe('picking');
    expect(gen.references).toEqual(
      [7, 8, 9, 10].map((seed) => ({
        url: `https://files.test/ai/${gen.id}/reference-${seed}.png`,
        seed,
      })),
    );

    // The picked picture's bytes go to TRELLIS.2; the preview gets a fresh seed
    gen = await studio.pick('u1', gen.id, 2);
    expect(gen).toMatchObject({
      status: 'previewing',
      image: `https://files.test/ai/${gen.id}/reference-9.png`,
    });
    const previewStart = started.at(-1)!;
    expect(previewStart.kind).toBe('preview');
    expect(previewStart.input['image']).toEqual(storage.files.get(`ai/${gen.id}/reference-9.png`));
    expect(previewStart.input['seed']).toBeUndefined();

    models.set('model-2', { status: 'done', output: modelOutput(4242, 'preview') });
    gen = await studio.get('u1', gen.id);
    expect(gen).toMatchObject({
      status: 'reviewing',
      preview: { url: `https://files.test/ai/${gen.id}/preview-4242.glb`, triangles: 30_000 },
      credits: ['Built with DINOv3'],
    });
    expect(storage.files.get(`ai/${gen.id}/preview-4242.glb`)?.toString()).toBe('glb-preview-4242');

    // The final reuses the picture and the preview's seed
    gen = await studio.keep('u1', gen.id);
    expect(gen.status).toBe('finishing');
    expect(started.at(-1)).toMatchObject({ kind: 'final', input: { seed: 4242 } });
    expect(started.at(-1)!.input['image']).toEqual(previewStart.input['image']);

    models.set('model-3', { status: 'done', output: modelOutput(4242, 'final') });
    gen = await studio.get('u1', gen.id);
    expect(gen).toMatchObject({
      status: 'done',
      final: { url: `https://files.test/ai/${gen.id}/final-4242.glb`, triangles: 100_000 },
    });
  });

  it('starts a photo at the preview, upright and scaled down', async () => {
    const { studio, storage, started } = setup(false);
    const photo = await sharp({
      create: { width: 3000, height: 1500, channels: 3, background: '#888' },
    })
      .jpeg()
      .toBuffer();
    const gen = await studio.startFromPhoto('u1', photo);
    expect(gen).toMatchObject({ status: 'previewing', source: 'photo' });
    expect(gen.image).toBe(`https://files.test/ai/${gen.id}/input.jpg`);
    const stored = storage.files.get(`ai/${gen.id}/input.jpg`)!;
    expect(await sharp(stored).metadata()).toMatchObject({
      width: 2048,
      height: 1024,
      format: 'jpeg',
    });
    expect(started[0]).toMatchObject({ kind: 'preview', input: { image: stored } });

    // Transparency is kept (TRELLIS.2 then skips background removal)
    const cutout = await studio.startFromPhoto('u1', await png('#ff000080'));
    expect(cutout.image).toMatch(/input\.png$/);
  });

  it('records failures, and retry runs the failed step again', async () => {
    const { studio, references, models, started } = setup();
    let gen = await studio.startFromPrompt('u1', 'a teapot');
    references.set('ref-1', { status: 'failed', message: 'CUDA out of memory' });
    gen = await studio.get('u1', gen.id);
    expect(gen).toMatchObject({ status: 'failed', error: 'CUDA out of memory' });

    gen = await studio.retry('u1', gen.id);
    expect(gen).toMatchObject({ status: 'drawing', error: null });
    expect(started.filter((s) => s.kind === 'references')).toHaveLength(2);

    references.set('ref-2', {
      status: 'done',
      output: { images: [{ file: { data: await png('#fff') }, seed: 1 }] },
    });
    await studio.get('u1', gen.id);
    await studio.pick('u1', gen.id, 0);
    models.set('model-3', { status: 'done', output: modelOutput(9, 'preview') });
    await studio.get('u1', gen.id);
    await studio.keep('u1', gen.id);
    models.set('model-4', { status: 'failed', message: 'container exited with code 137' });
    gen = await studio.get('u1', gen.id);
    expect(gen.status).toBe('failed');

    gen = await studio.retry('u1', gen.id);
    expect(gen.status).toBe('finishing');
    expect(started.at(-1)).toMatchObject({ kind: 'final', input: { seed: 9 } });
  });

  it('keeps polling through network trouble, but fails on a lost job', async () => {
    const { studio, references } = setup();
    const gen = await studio.startFromPrompt('u1', 'a lamp');
    references.set(
      'ref-1',
      new Error('AI worker status failed: 503 {"detail":"Modal can\'t be reached"}'),
    );
    expect((await studio.get('u1', gen.id)).status).toBe('drawing');
    references.set('ref-1', new Error('AI worker status failed: 404 {"detail":"unknown job"}'));
    expect((await studio.get('u1', gen.id)).status).toBe('failed');
  });

  it('downloads outputs that workers stored at a URL', async () => {
    const store = memoryStore();
    const storage = memoryStorage();
    const jobs = scriptedWorkers();
    const fetched: string[] = [];
    const studio = new AIStudio({
      workers: jobs.workers,
      store,
      storage,
      fetchImpl: async (url) => {
        fetched.push(url);
        return new Response(await png('#123'));
      },
    });
    const gen = await studio.startFromPrompt('u1', 'a crate');
    jobs.references.set('ref-1', {
      status: 'done',
      output: { images: [{ file: { url: 'https://r2.test/ai/x/reference-5.png' }, seed: 5 }] },
    });
    await studio.get('u1', gen.id);
    expect(fetched).toEqual(['https://r2.test/ai/x/reference-5.png']);
    expect(storage.files.has(`ai/${gen.id}/reference-5.png`)).toBe(true);
  });

  it("refuses what doesn't make sense", async () => {
    const { studio, store } = setup(false);
    const expectError = async (call: Promise<unknown>, code: StudioError['code'], text: RegExp) => {
      const err = await call.catch((e: unknown) => e);
      expect(err).toBeInstanceOf(StudioError);
      expect(err).toMatchObject({ code });
      expect((err as Error).message).toMatch(text);
    };
    await expectError(studio.startFromPrompt('u1', 'a lamp'), 'UNAVAILABLE', /photos only/);
    await expectError(
      studio.startFromPhoto('u1', Buffer.from('not a picture')),
      'BAD_REQUEST',
      /PNG, JPEG/,
    );

    const gen = await studio.startFromPhoto('u1', await png('#fff'));
    await expectError(studio.get('u2', gen.id), 'NOT_FOUND', /not found/);
    await expectError(studio.keep('u1', gen.id), 'BAD_REQUEST', /finished preview/);
    await expectError(studio.pick('u1', gen.id, 0), 'BAD_REQUEST', /no such picture/);
    await expectError(studio.retry('u1', gen.id), 'BAD_REQUEST', /failed step/);
    expect(store.rows.size).toBe(1);

    const none = new AIStudio({ workers: null, store, storage: memoryStorage() });
    expect(none.capabilities()).toEqual({
      available: false,
      prompts: false,
      photos: false,
      mock: false,
    });
    await expectError(none.startFromPhoto('u1', await png('#fff')), 'UNAVAILABLE', /not set up/);
  });

  it('bounds how many models a user can have running', async () => {
    const { studio } = setup();
    for (let i = 0; i < LIMITS.running; i++) await studio.startFromPrompt('u1', `thing ${i}`);
    await expect(studio.startFromPrompt('u1', 'one more')).rejects.toThrow(/in progress/);
    // Someone else isn't affected
    await expect(studio.startFromPrompt('u2', 'a chair')).resolves.toMatchObject({
      status: 'drawing',
    });
  });

  it("doesn't let a slow poll undo Keep", async () => {
    const store = memoryStore();
    const storage = memoryStorage();
    const jobs = scriptedWorkers(false);
    let release: () => void = () => {};
    let downloads = 0;
    const studio = new AIStudio({
      workers: jobs.workers,
      store,
      storage,
      // The first download (a slow poll) waits until the test lets it go
      fetchImpl: async () => {
        if (downloads++ === 0) await new Promise<void>((resolve) => (release = resolve));
        return new Response('glb-preview');
      },
    });
    const gen = await studio.startFromPhoto('u1', await png('#fff'));
    jobs.models.set('model-1', {
      status: 'done',
      output: { ...modelOutput(5, 'preview'), file: { url: 'https://r2.test/preview.glb' } },
    });
    const slow = studio.get('u1', gen.id);
    await new Promise((resolve) => setTimeout(resolve, 10));
    expect((await studio.get('u1', gen.id)).status).toBe('reviewing');
    expect((await studio.keep('u1', gen.id)).status).toBe('finishing');

    release();
    expect((await slow).status).toBe('finishing');
    expect((await studio.get('u1', gen.id)).status).toBe('finishing');
  });
});
