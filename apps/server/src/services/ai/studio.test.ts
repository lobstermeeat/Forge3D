import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import sharp from 'sharp';
import { randomUUID } from 'node:crypto';
import type { StorageProvider } from '../storage';
import { MOCK_VIEW_AZIMUTHS, MockWorkers } from './providers/mock';
import {
  AIStudio,
  LIMITS,
  StudioError,
  VIEWS_TIMEOUT_MS,
  WARM_INTERVAL_MS,
  type GenerationRecord,
  type GenerationStore,
  type GenerationView,
} from './studio';
import type {
  ModelOutput,
  ModelView,
  ReferencesOutput,
  StudioWorkers,
  ViewsOutput,
  WorkerJobState,
  WorkerKind,
} from './types';

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
        views: null,
        viewsError: null,
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
  const views = new Map<string, WorkerJobState<ViewsOutput> | Error>();
  const models = new Map<string, WorkerJobState<ModelOutput> | Error>();
  const started: { kind: string; input: Record<string, unknown> }[] = [];
  /** The GPUs asked to start early, in order */
  const warmed: WorkerKind[] = [];
  /** The jobs the studio stopped, as `kind:jobId` */
  const cancelled: string[] = [];
  /**
   * What the test makes happen: pictures, views or models that can't start, and what starting a
   * GPU early does
   */
  const control: {
    startError?: Error;
    viewsStartError?: Error;
    modelStartError?: Error;
    warm?: () => Promise<void>;
  } = {};
  let next = 1;
  const workers: StudioWorkers = {
    name: 'test-workers',
    prompts,
    multiview: true,
    async startReferences(input) {
      if (control.startError) throw control.startError;
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
    async startViews(input) {
      if (control.viewsStartError) throw control.viewsStartError;
      const id = `views-${next++}`;
      started.push({ kind: 'views', input });
      views.set(id, { status: 'running' });
      return id;
    },
    async views(jobId) {
      const state = views.get(jobId)!;
      if (state instanceof Error) throw state;
      return state;
    },
    async startModel(input) {
      if (control.modelStartError) throw control.modelStartError;
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
    async warm(kind) {
      warmed.push(kind);
      await control.warm?.();
    },
    async cancel(kind, jobId) {
      cancelled.push(`${kind}:${jobId}`);
    },
  };
  return { workers, references, views, models, started, warmed, cancelled, control };
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

/** 6 views like the multiview worker's, each a different grey so their order shows */
const viewsOutput = async (): Promise<ViewsOutput> => ({
  views: await Promise.all(
    MOCK_VIEW_AZIMUTHS.map(async (azimuth, i) => ({
      file: { data: await png(`#${i}${i}${i}`) },
      azimuth,
      elevation: 0,
    })),
  ),
  seconds: 21.5,
});

/** The views a job was started with, as the studio sent them */
const sentViews = (start: { input: Record<string, unknown> }) =>
  start.input['views'] as ModelView[] | undefined;

/** AIStudio with scripted workers; `multiview` is AI_MULTIVIEW=1 */
function setup(prompts = true, { multiview = false }: { multiview?: boolean } = {}) {
  const store = memoryStore();
  const storage = memoryStorage();
  const jobs = scriptedWorkers(prompts);
  const studio = new AIStudio({ workers: jobs.workers, store, storage, multiview });
  return { studio, store, storage, ...jobs };
}

/** A prompt's generation with one picture drawn (seed 3), for the user to pick */
async function picking(
  { studio, references }: ReturnType<typeof setup>,
  user = 'u1',
): Promise<GenerationView> {
  const gen = await studio.startFromPrompt(user, 'a vintage film camera');
  references.set([...references.keys()].at(-1)!, {
    status: 'done',
    output: { images: [{ file: { data: await png('#c84') }, seed: 3 }] },
  });
  return studio.get(user, gen.id);
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

    // AI_MULTIVIEW is off by default: no views step, and the models get the picture alone
    expect(started.map((s) => s.kind)).toEqual(['references', 'preview', 'final']);
    expect(started.some((s) => 'views' in s.input)).toBe(false);
    expect(gen).toMatchObject({ views: [], viewsError: null });
    expect(studio.capabilities().multiview).toBe(false);
  });

  it('starts a photo at the preview, upright and scaled down', async () => {
    const { studio, storage, started } = setup(false);
    const photo = await sharp({
      create: { width: 3000, height: 1500, channels: 3, background: '#888' },
    })
      .jpeg()
      .toBuffer();
    const gen = await studio.startFromPhoto('u1', photo);
    expect(gen).toMatchObject({
      status: 'previewing',
      source: 'photo',
      references: [],
      recommended: null,
    });
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
      multiview: false,
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

  it('keeps how the worker rated each picture and recommends the best start for 3D', async () => {
    const { studio, store, references } = setup();
    let gen = await studio.startFromPrompt('u1', 'a desk lamp');
    expect(gen.recommended).toBeNull();

    const ratings = [
      { score: 0.42, issues: ['cut off at the bottom', 'more than one object'] },
      { score: 0.91 },
      { score: 0.91, issues: [] },
      {},
    ];
    references.set('ref-1', {
      status: 'done',
      output: {
        images: await Promise.all(
          ratings.map(async (rating, i) => ({
            file: { data: await png('#abc') },
            seed: 20 + i,
            ...rating,
          })),
        ),
      },
    });
    gen = await studio.get('u1', gen.id);
    // The first of equal scores; a picture without one is never recommended
    expect(gen).toMatchObject({ status: 'picking', recommended: 1 });
    const url = (seed: number) => `https://files.test/ai/${gen.id}/reference-${seed}.png`;
    // Stored with each picture, leaving out what the worker didn't say
    expect(store.rows.get(gen.id)!.referenceImages).toStrictEqual([
      {
        url: url(20),
        seed: 20,
        score: 0.42,
        issues: ['cut off at the bottom', 'more than one object'],
      },
      { url: url(21), seed: 21, score: 0.91 },
      { url: url(22), seed: 22, score: 0.91 },
      { url: url(23), seed: 23 },
    ]);
    expect(gen.references).toStrictEqual(store.rows.get(gen.id)!.referenceImages);

    // The user still chooses, and the recommendation stays for trying another picture later
    gen = await studio.pick('u1', gen.id, 3);
    expect(gen).toMatchObject({ status: 'previewing', image: url(23), recommended: 1 });
  });

  it('recommends the highest score, however low', async () => {
    const { studio, references } = setup();
    const gen = await studio.startFromPrompt('u1', 'a forest clearing with three tents');
    references.set('ref-1', {
      status: 'done',
      output: {
        images: await Promise.all(
          [0.05, undefined, 0.12, 0].map(async (score, i) => ({
            file: { data: await png('#abc') },
            seed: 30 + i,
            ...(score === undefined ? {} : { score, issues: ['more than one object'] }),
          })),
        ),
      },
    });
    expect(await studio.get('u1', gen.id)).toMatchObject({ status: 'picking', recommended: 2 });
  });

  it('recommends nothing for pictures drawn before the worker rated them', async () => {
    const { studio, store } = setup();
    // A row from before scores: its pictures have only a URL and a seed
    const old = await store.create({
      userId: 'u1',
      provider: 'forge3d-trellis2',
      prompt: 'a chair',
      status: 'picking',
      referenceImages: [
        { url: 'https://files.test/ai/old/reference-1.png', seed: 1 },
        { url: 'https://files.test/ai/old/reference-2.png', seed: 2 },
      ],
    });
    expect(await studio.get('u1', old.id)).toMatchObject({
      status: 'picking',
      references: [{ seed: 1 }, { seed: 2 }],
      recommended: null,
    });
    expect((await studio.recent('u1'))[0]).toMatchObject({ id: old.id, recommended: null });
    expect((await studio.pick('u1', old.id, 1)).image).toBe(
      'https://files.test/ai/old/reference-2.png',
    );
  });

  it('starts each GPU early at most once every 2 minutes per user', () => {
    expect(WARM_INTERVAL_MS).toBe(2 * 60 * 1000);
    vi.useFakeTimers({ toFake: ['Date'] });
    try {
      const start = Date.parse('2026-09-30T12:00:00Z');
      vi.setSystemTime(start);
      const { studio, warmed } = setup();
      studio.warm('u1', 'references');
      studio.warm('u1', 'references');
      studio.warm('u1', 'model');
      studio.warm('u2', 'references');
      expect(warmed).toEqual(['references', 'model', 'references']);

      vi.setSystemTime(start + WARM_INTERVAL_MS - 1);
      studio.warm('u1', 'references');
      expect(warmed).toHaveLength(3);
      vi.setSystemTime(start + WARM_INTERVAL_MS);
      studio.warm('u1', 'references');
      studio.warm('u1', 'references');
      studio.warm('u2', 'model');
      expect(warmed).toEqual(['references', 'model', 'references', 'references', 'model']);
    } finally {
      vi.useRealTimers();
    }
  });

  it("never fails the caller when a GPU can't be started early", async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    try {
      const { studio, workers, warmed, control } = setup();
      control.warm = async () => {
        throw new Error('AI worker warm failed: 503 {"detail":"Modal can\'t be reached"}');
      };
      expect(() => studio.warm('u1', 'references')).not.toThrow();
      expect(warmed).toEqual(['references']);
      await vi.waitFor(() =>
        expect(warn).toHaveBeenCalledWith(
          expect.stringContaining('references'),
          expect.stringContaining('503'),
        ),
      );
      // And isn't asked again at once
      studio.warm('u1', 'references');
      expect(warmed).toHaveLength(1);

      // Nor when the workers throw before they even ask
      workers.warm = () => {
        throw new Error('not set up');
      };
      expect(() => studio.warm('u2', 'model')).not.toThrow();
      await vi.waitFor(() => expect(warn).toHaveBeenCalledTimes(2));

      // Workers that can't start a GPU early, and servers without workers, do nothing
      delete workers.warm;
      expect(() => studio.warm('u3', 'model')).not.toThrow();
      const none = new AIStudio({ workers: null, store: memoryStore(), storage: memoryStorage() });
      expect(() => none.warm('u1', 'model')).not.toThrow();
      expect(warmed).toHaveLength(1);
    } finally {
      warn.mockRestore();
    }
  });

  it('starts TRELLIS.2 while the pictures are drawn', async () => {
    const { studio, started, warmed, control } = setup();
    // The prompt doesn't wait for it
    control.warm = () => new Promise(() => {});
    expect(await studio.startFromPrompt('u1', 'a lamp')).toMatchObject({ status: 'drawing' });
    expect(started.map((s) => s.kind)).toEqual(['references']);
    expect(warmed).toEqual(['model']);
    // At most once every 2 minutes per user
    await studio.startFromPrompt('u1', 'a chair');
    expect(warmed).toEqual(['model']);

    // Not when the pictures couldn't be started, nor for a photo, which starts TRELLIS.2 itself
    control.startError = new Error('fetch failed');
    const failed = await studio.startFromPrompt('u2', 'a mug');
    expect(failed).toMatchObject({ status: 'failed' });
    await studio.startFromPhoto('u3', await png('#fff'));
    expect(warmed).toEqual(['model']);

    // Drawing them again does
    delete control.startError;
    expect(await studio.retry('u2', failed.id)).toMatchObject({ status: 'drawing' });
    expect(warmed).toEqual(['model', 'model']);
  });
});

describe('AIStudio with AI_MULTIVIEW=1 (the views step)', () => {
  let warn: ReturnType<typeof vi.spyOn>;
  beforeEach(() => {
    warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
  });
  afterEach(() => warn.mockRestore());

  /** The image a view was sent with */
  const viewImages = (output: ViewsOutput) =>
    output.views.map(({ file, azimuth, elevation }) => ({
      image: 'data' in file ? file.data : null,
      azimuth,
      elevation,
    }));

  it('draws the picked picture from 6 sides, then builds the preview and final from them', async () => {
    const t = setup(true, { multiview: true });
    const { studio, store, storage, views, models, started } = t;
    expect(studio.capabilities().multiview).toBe(true);
    const steps: string[] = [];
    const seen = (g: GenerationView) => {
      if (steps.at(-1) !== g.status) steps.push(g.status);
      return g;
    };

    let gen = seen(await studio.startFromPrompt('u1', 'a vintage film camera'));
    expect(started.map((s) => s.kind)).toEqual(['references']);
    t.references.set('ref-1', {
      status: 'done',
      output: { images: [{ file: { data: await png('#c84') }, seed: 3 }] },
    });
    gen = seen(await studio.get('u1', gen.id));

    // Picking a picture draws its other sides, from the picked picture's bytes
    gen = seen(await studio.pick('u1', gen.id, 0));
    expect(gen).toMatchObject({ status: 'viewing', views: [], viewsError: null, preview: null });
    const picture = storage.files.get(`ai/${gen.id}/reference-3.png`);
    expect(started.at(-1)).toEqual({ kind: 'views', input: { image: picture, requestId: gen.id } });
    // Still drawing them
    gen = seen(await studio.get('u1', gen.id));
    expect(started).toHaveLength(2);

    // Done: they are copied into storage, and the preview starts from the picture and all 6
    const output = await viewsOutput();
    views.set('views-2', { status: 'done', output });
    gen = seen(await studio.get('u1', gen.id));
    expect(gen.status).toBe('previewing');
    const keys = MOCK_VIEW_AZIMUTHS.map(
      (azimuth, i) => `ai/${gen.id}/views-views-2/${i}-${azimuth}.png`,
    );
    expect(gen.views).toEqual(
      MOCK_VIEW_AZIMUTHS.map((azimuth, i) => ({
        url: `https://files.test/${keys[i]}`,
        azimuth,
        elevation: 0,
      })),
    );
    expect(keys.map((key) => storage.files.get(key))).toEqual(
      viewImages(output).map((view) => view.image),
    );
    const previewStart = started.at(-1)!;
    expect(previewStart).toMatchObject({ kind: 'preview', input: { requestId: gen.id } });
    expect(previewStart.input['image']).toEqual(picture);
    expect(previewStart.input['seed']).toBeUndefined();
    expect(sentViews(previewStart)).toEqual(viewImages(output));
    // Their GPU time counts
    expect(store.rows.get(gen.id)!.durationMs).toBe(21_500);

    models.set('model-3', {
      status: 'done',
      output: { ...modelOutput(77, 'preview'), viewsUsed: 6 },
    });
    gen = seen(await studio.get('u1', gen.id));
    expect(gen.status).toBe('reviewing');
    expect(gen.views).toHaveLength(6);

    // The final gets the same views (read back from storage) and the preview's seed
    gen = seen(await studio.keep('u1', gen.id));
    const finalStart = started.at(-1)!;
    expect(finalStart).toMatchObject({ kind: 'final', input: { seed: 77 } });
    expect(finalStart.input['image']).toEqual(picture);
    expect(sentViews(finalStart)).toEqual(viewImages(output));

    models.set('model-4', {
      status: 'done',
      output: { ...modelOutput(77, 'final'), viewsUsed: 6 },
    });
    gen = seen(await studio.get('u1', gen.id));
    expect(gen.final).toMatchObject({ url: `https://files.test/ai/${gen.id}/final-77.glb` });

    expect(steps).toEqual([
      'drawing',
      'picking',
      'viewing',
      'previewing',
      'reviewing',
      'finishing',
      'done',
    ]);
    expect(started.map((s) => s.kind)).toEqual(['references', 'views', 'preview', 'final']);
    expect(warn).not.toHaveBeenCalled();
  });

  it('starts a photo at its other sides', async () => {
    const { studio, storage, views, started } = setup(false, { multiview: true });
    const gen = await studio.startFromPhoto('u1', await png('#ff000080'));
    expect(gen).toMatchObject({
      status: 'viewing',
      image: `https://files.test/ai/${gen.id}/input.png`,
    });
    const photo = storage.files.get(`ai/${gen.id}/input.png`)!;
    expect(started).toEqual([{ kind: 'views', input: { image: photo, requestId: gen.id } }]);

    const output = await viewsOutput();
    views.set('views-1', { status: 'done', output });
    expect(await studio.get('u1', gen.id)).toMatchObject({ status: 'previewing' });
    expect(started.at(-1)!.input['image']).toEqual(photo);
    expect(sentViews(started.at(-1)!)).toEqual(viewImages(output));
  });

  it('has no views step without the multiview worker', async () => {
    const t = setup(true, { multiview: true });
    (t.workers as { multiview: boolean }).multiview = false;
    expect(t.studio.capabilities().multiview).toBe(false);
    const gen = await picking(t);
    expect((await t.studio.pick('u1', gen.id, 0)).status).toBe('previewing');
    expect((await t.studio.startFromPhoto('u1', await png('#fff'))).status).toBe('previewing');
    expect(t.started.map((s) => s.kind)).toEqual(['references', 'preview', 'preview']);
    expect(t.started.some((s) => 'views' in s.input)).toBe(false);
    // Nor is its GPU started, even when asked
    t.studio.warm('u1', 'multiview');
    expect(t.warmed).toEqual(['model']);
  });

  it('makes the model from the picture alone when the views fail, and keeps why', async () => {
    const t = setup(true, { multiview: true });
    let gen = await picking(t);
    await t.studio.pick('u1', gen.id, 0);
    t.views.set('views-2', { status: 'failed', message: 'generation failed: CUDA out of memory' });
    gen = await t.studio.get('u1', gen.id);
    // Not a failure: the preview goes ahead without views
    expect(gen).toMatchObject({
      status: 'previewing',
      views: [],
      viewsError: 'generation failed: CUDA out of memory',
      error: null,
    });
    expect(t.started.at(-1)!.kind).toBe('preview');
    expect('views' in t.started.at(-1)!.input).toBe(false);
    expect(warn).toHaveBeenCalledWith(
      expect.stringContaining(gen.id),
      'generation failed: CUDA out of memory',
    );

    t.models.set('model-3', { status: 'done', output: modelOutput(5, 'preview') });
    expect((await t.studio.get('u1', gen.id)).status).toBe('reviewing');
    // Picking again tries the views again, without the old reason
    gen = await t.studio.pick('u1', gen.id, 0);
    expect(gen).toMatchObject({ status: 'viewing', views: [], viewsError: null });
    t.views.set('views-4', { status: 'done', output: await viewsOutput() });
    expect(await t.studio.get('u1', gen.id)).toMatchObject({
      status: 'previewing',
      viewsError: null,
    });
    expect(sentViews(t.started.at(-1)!)).toHaveLength(6);
  });

  it("goes straight to the preview when the views can't start", async () => {
    const t = setup(true, { multiview: true });
    // E.g. a job API without the multiview worker yet
    t.control.viewsStartError = new Error(
      `AI worker run failed: 404 {"detail":"unknown worker 'multiview'"}`,
    );
    const gen = await picking(t);
    expect(await t.studio.pick('u1', gen.id, 0)).toMatchObject({
      status: 'previewing',
      views: [],
      viewsError: expect.stringContaining('unknown worker'),
    });
    expect(t.started.map((s) => s.kind)).toEqual(['references', 'preview']);

    const photo = await t.studio.startFromPhoto('u1', await png('#fff'));
    expect(photo).toMatchObject({
      status: 'previewing',
      viewsError: expect.stringContaining('404'),
    });
    expect(t.started.at(-1)!.input['image']).toEqual(
      t.storage.files.get(`ai/${photo.id}/input.png`),
    );
  });

  it('goes on without views that came back broken', async () => {
    const t = setup(false, { multiview: true });
    const broken: [WorkerJobState<ViewsOutput> | Error, RegExp][] = [
      [{ status: 'done', output: { views: [], seconds: 1 } }, /no views/],
      [
        {
          status: 'done',
          output: {
            views: [{ file: { data: Buffer.from('<html>') }, azimuth: 0, elevation: 0 }],
            seconds: 1,
          },
        },
        /view 1 .* isn't a picture/,
      ],
      // A job the host forgot
      [new Error('AI worker status failed: 404 {"detail":"unknown job"}'), /unknown job/],
    ];
    for (const [state, reason] of broken) {
      const gen = await t.studio.startFromPhoto('u1', await png('#fff'));
      t.views.set([...t.views.keys()].at(-1)!, state);
      expect(await t.studio.get('u1', gen.id)).toMatchObject({
        status: 'previewing',
        views: [],
        viewsError: expect.stringMatching(reason),
      });
      expect('views' in t.started.at(-1)!.input).toBe(false);
    }
  });

  it('waits out network trouble, but stops views that take too long', async () => {
    vi.useFakeTimers({ toFake: ['Date'] });
    try {
      const start = Date.parse('2026-10-01T12:00:00Z');
      vi.setSystemTime(start);
      const t = setup(false, { multiview: true });
      const gen = await t.studio.startFromPhoto('u1', await png('#fff'));
      vi.setSystemTime(start + VIEWS_TIMEOUT_MS - 1000);
      expect((await t.studio.get('u1', gen.id)).status).toBe('viewing');
      t.views.set('views-1', new Error('fetch failed'));
      expect((await t.studio.get('u1', gen.id)).status).toBe('viewing');

      t.views.set('views-1', { status: 'running' });
      vi.setSystemTime(start + VIEWS_TIMEOUT_MS);
      expect(await t.studio.get('u1', gen.id)).toMatchObject({
        status: 'previewing',
        views: [],
        viewsError: 'the views took longer than 180 s',
      });
      // It bills while it runs, so it's stopped
      expect(t.cancelled).toEqual(['multiview:views-1']);
      expect('views' in t.started.at(-1)!.input).toBe(false);

      // Network trouble past the time limit doesn't hold the model up either
      const other = await t.studio.startFromPhoto('u1', await png('#000'));
      t.views.set('views-3', new Error('fetch failed'));
      vi.setSystemTime(start + 2 * VIEWS_TIMEOUT_MS);
      expect(await t.studio.get('u1', other.id)).toMatchObject({
        status: 'previewing',
        viewsError: 'fetch failed',
      });
    } finally {
      vi.useRealTimers();
    }
  });

  it('keeps the views when the preview fails, so Try again uses them', async () => {
    const t = setup(false, { multiview: true });
    const gen = await t.studio.startFromPhoto('u1', await png('#fff'));
    const output = await viewsOutput();
    t.views.set('views-1', { status: 'done', output });
    await t.studio.get('u1', gen.id);
    t.models.set('model-2', { status: 'failed', message: 'CUDA out of memory' });
    const failed = await t.studio.get('u1', gen.id);
    expect(failed).toMatchObject({ status: 'failed', error: 'CUDA out of memory' });
    expect(failed.views).toHaveLength(6);
    expect((await t.studio.retry('u1', gen.id)).status).toBe('previewing');
    expect(sentViews(t.started.at(-1)!)).toEqual(viewImages(output));
    // The views aren't drawn again
    expect(t.started.map((s) => s.kind)).toEqual(['views', 'preview', 'preview']);

    // Nor when the preview can't even start after them
    const other = await t.studio.startFromPhoto('u1', await png('#000'));
    t.views.set('views-4', { status: 'done', output });
    t.control.modelStartError = new Error('AI worker run failed: 400 {"detail":"bad input"}');
    expect(await t.studio.get('u1', other.id)).toMatchObject({
      status: 'failed',
      error: expect.stringContaining('bad input'),
      views: expect.arrayContaining([expect.objectContaining({ azimuth: 180 })]),
    });
    delete t.control.modelStartError;
    expect((await t.studio.retry('u1', other.id)).status).toBe('previewing');
    expect(sentViews(t.started.at(-1)!)).toEqual(viewImages(output));
  });

  it('starts one preview when two polls find the views done at once', async () => {
    const store = memoryStore();
    const storage = memoryStorage();
    const jobs = scriptedWorkers(false);
    let release: () => void = () => {};
    const downloaded = new Promise<void>((resolve) => (release = resolve));
    const studio = new AIStudio({
      workers: jobs.workers,
      store,
      storage,
      multiview: true,
      // Both polls wait here, with the views job done
      fetchImpl: async () => {
        await downloaded;
        return new Response(await png('#456'));
      },
    });
    const gen = await studio.startFromPhoto('u1', await png('#fff'));
    jobs.views.set('views-1', {
      status: 'done',
      output: {
        views: [{ file: { url: 'https://r2.test/view-0.png' }, azimuth: 0, elevation: 0 }],
        seconds: 2,
      },
    });
    const polls = [studio.get('u1', gen.id), studio.get('u1', gen.id)];
    await new Promise((resolve) => setTimeout(resolve, 10));
    release();
    for (const poll of await Promise.all(polls)) expect(poll.status).toBe('previewing');

    const previews = jobs.started.filter((s) => s.kind === 'preview');
    expect(previews).toHaveLength(2);
    // The generation keeps one; the other is stopped
    const kept = store.rows.get(gen.id)!.jobId;
    expect(['model-2', 'model-3']).toContain(kept);
    expect(jobs.cancelled).toEqual([`model:${kept === 'model-2' ? 'model-3' : 'model-2'}`]);
  });

  it('starts MV-Adapter and TRELLIS.2 while the pictures are drawn, and TRELLIS.2 for slow pickers', async () => {
    vi.useFakeTimers({ toFake: ['Date'] });
    try {
      const start = Date.parse('2026-10-01T12:00:00Z');
      vi.setSystemTime(start);
      const t = setup(true, { multiview: true });
      const gen = await picking(t);
      expect(t.warmed).toEqual(['model', 'multiview']);
      // Picked within 2 minutes: TRELLIS.2 is starting or up already
      await t.studio.pick('u1', gen.id, 0);
      expect(t.warmed).toEqual(['model', 'multiview']);
      // A photo's views start TRELLIS.2
      await t.studio.startFromPhoto('u2', await png('#fff'));
      expect(t.warmed).toEqual(['model', 'multiview', 'model']);
      // A slow picker's TRELLIS.2 has scaled down by then, so their views start it again
      const slow = await picking(t, 'u3');
      vi.setSystemTime(start + WARM_INTERVAL_MS);
      await t.studio.pick('u3', slow.id, 0);
      expect(t.warmed).toEqual(['model', 'multiview', 'model', 'model', 'multiview', 'model']);
    } finally {
      vi.useRealTimers();
    }
  });

  it('counts a generation drawing its views as one in progress', async () => {
    const { studio } = setup(false, { multiview: true });
    for (let i = 0; i < LIMITS.running; i++) {
      expect((await studio.startFromPhoto('u1', await png('#fff'))).status).toBe('viewing');
    }
    await expect(studio.startFromPhoto('u1', await png('#fff'))).rejects.toThrow(/in progress/);
  });

  it('warns when the 3D worker leaves the views out', async () => {
    const t = setup(false, { multiview: true });
    const gen = await t.studio.startFromPhoto('u1', await png('#fff'));
    t.views.set('views-1', { status: 'done', output: await viewsOutput() });
    await t.studio.get('u1', gen.id);
    // A 3D worker from before views reports no views_used
    t.models.set('model-2', { status: 'done', output: modelOutput(5, 'preview') });
    expect((await t.studio.get('u1', gen.id)).status).toBe('reviewing');
    expect(warn).toHaveBeenCalledWith(expect.stringContaining('without the 6 views'));
  });

  it('runs on the mock workers end to end (AI_WORKERS_MOCK=1)', async () => {
    const storage = memoryStorage();
    const studio = new AIStudio({
      workers: new MockWorkers(0),
      store: memoryStore(),
      storage,
      multiview: true,
    });
    let gen = await studio.startFromPrompt('u1', 'a wooden shield with a lion');
    gen = await studio.get('u1', gen.id);
    expect(gen.status).toBe('picking');
    gen = await studio.pick('u1', gen.id, gen.recommended!);
    expect(gen.status).toBe('viewing');
    gen = await studio.get('u1', gen.id);
    expect(gen).toMatchObject({ status: 'previewing', viewsError: null });
    expect(gen.views.map((view) => [view.azimuth, view.elevation])).toEqual(
      MOCK_VIEW_AZIMUTHS.map((azimuth) => [azimuth, 0]),
    );
    for (const view of gen.views) {
      const data = storage.files.get(view.url.replace('https://files.test/', ''))!;
      expect(await sharp(data).metadata()).toMatchObject({ format: 'png', hasAlpha: true });
    }
    expect((await studio.get('u1', gen.id)).status).toBe('reviewing');
    await studio.keep('u1', gen.id);
    expect((await studio.get('u1', gen.id)).status).toBe('done');
    // The mock 3D worker says it used them
    expect(warn).not.toHaveBeenCalled();

    // A photo under 128 px gets no views, and its model is made from the photo alone
    const small = await sharp({
      create: { width: 100, height: 100, channels: 3, background: '#888' },
    })
      .png()
      .toBuffer();
    let photo = await studio.startFromPhoto('u1', small);
    expect(photo.status).toBe('viewing');
    photo = await studio.get('u1', photo.id);
    expect(photo).toMatchObject({
      status: 'previewing',
      views: [],
      viewsError: expect.stringContaining('128 px'),
    });
    expect((await studio.get('u1', photo.id)).status).toBe('reviewing');
  });
});
