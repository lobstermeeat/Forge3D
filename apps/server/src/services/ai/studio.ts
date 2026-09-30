import sharp from 'sharp';
import { and, desc, eq } from 'drizzle-orm';
import type { Database } from '../../db';
import { schema } from '../../db';
import type { StorageProvider } from '../storage';
import type { FetchLike } from './providers/jobEndpoint';
import type { StudioWorkers, WorkerFile } from './types';

/**
 * The Studio's AI panel, one generation at a time:
 *
 *   prompt: drawing -> picking -> previewing -> reviewing -> finishing -> done
 *   photo:              previewing -> reviewing -> finishing -> done
 *
 * drawing: FLUX draws 4 pictures. picking: the user picks one (or another one later).
 * previewing: TRELLIS.2 makes a quick model (512³) the user can place and look at.
 * reviewing: the user keeps it, or picks another picture. finishing: the final model (1024³,
 * same seed, so it refines the shape they approved). Any job can fail; retry runs it again.
 *
 * Every job is started and then polled by `get`, so no request waits on a GPU. Pictures and
 * models are copied into the server's storage, so scenes keep working after the workers forget
 * their outputs.
 */

export type GenerationStatus =
  | 'drawing'
  | 'picking'
  | 'previewing'
  | 'reviewing'
  | 'finishing'
  | 'done'
  | 'failed';

type JobKind = 'references' | 'preview' | 'final';

export type GenerationRecord = typeof schema.aiGenerations.$inferSelect;
export type NewGeneration = typeof schema.aiGenerations.$inferInsert;

export interface GenerationStore {
  create(values: NewGeneration): Promise<GenerationRecord>;
  find(id: string, userId: string): Promise<GenerationRecord | null>;
  update(id: string, patch: Partial<NewGeneration>): Promise<GenerationRecord>;
  /**
   * Updates the generation only if `jobId` is still its job, so a slow poll can't undo a step the
   * user took meanwhile (such as Keep starting the final). Returns null when the job changed.
   */
  finishJob(
    id: string,
    jobId: string,
    patch: Partial<NewGeneration>,
  ): Promise<GenerationRecord | null>;
  recent(userId: string, limit: number): Promise<GenerationRecord[]>;
}

/** What the panel shows for a generation. */
export interface GenerationView {
  id: string;
  source: 'prompt' | 'photo';
  prompt: string | null;
  status: GenerationStatus;
  /** The pictures FLUX drew for a prompt, to pick from */
  references: { url: string; seed: number }[];
  /** The picture that went (or goes) to 3D */
  image: string | null;
  preview: { url: string; triangles: number | null } | null;
  final: { url: string; triangles: number | null } | null;
  credits: string[];
  error: string | null;
  sceneId: string | null;
  createdAt: string;
}

export class StudioError extends Error {
  constructor(
    readonly code: 'NOT_FOUND' | 'BAD_REQUEST' | 'UNAVAILABLE',
    message: string,
  ) {
    super(message);
  }
}

const RUNNING = new Set<GenerationStatus>(['drawing', 'previewing', 'finishing']);
const REFERENCE_COUNT = 4;
/** Bounds on GPU spending per user until credits exist */
export const LIMITS = { running: 3, perHour: 30 };
export const MAX_PROMPT_LENGTH = 500;
/** Photos are scaled down to this before they are stored and sent to TRELLIS.2 */
const MAX_PHOTO_SIDE = 2048;
/** Bigger photos are refused before decoding (about 8000 x 6000) */
const MAX_PHOTO_PIXELS = 50_000_000;

export interface StudioDeps {
  workers: StudioWorkers | null;
  store: GenerationStore;
  storage: StorageProvider;
  /** Downloads worker outputs stored at a URL (workers with R2) */
  fetchImpl?: FetchLike;
}

export class AIStudio {
  constructor(private readonly deps: StudioDeps) {}

  capabilities() {
    const { workers } = this.deps;
    return {
      available: workers !== null,
      prompts: workers?.prompts ?? false,
      photos: workers !== null,
      mock: workers?.name === 'mock',
    };
  }

  async startFromPrompt(userId: string, prompt: string, sceneId?: string): Promise<GenerationView> {
    const workers = this.workers();
    await this.checkLimits(userId);
    const text = prompt.trim();
    if (!text) throw new StudioError('BAD_REQUEST', 'Describe the object to make');
    if (text.length > MAX_PROMPT_LENGTH) {
      throw new StudioError(
        'BAD_REQUEST',
        `Keep the description under ${MAX_PROMPT_LENGTH} characters`,
      );
    }
    if (!workers.prompts) {
      throw new StudioError('UNAVAILABLE', 'This server makes models from photos only');
    }
    const record = await this.deps.store.create({
      userId,
      sceneId: sceneId ?? null,
      provider: workers.name,
      source: 'prompt',
      prompt: text,
      status: 'drawing',
      jobKind: 'references',
    });
    return view(await this.runJob(record, 'references'));
  }

  async startFromPhoto(userId: string, photo: Buffer, sceneId?: string): Promise<GenerationView> {
    const workers = this.workers();
    await this.checkLimits(userId);
    const image = await normalisePhoto(photo);
    const record = await this.deps.store.create({
      userId,
      sceneId: sceneId ?? null,
      provider: workers.name,
      source: 'photo',
      status: 'previewing',
      jobKind: 'preview',
    });
    const imageUrl = await this.deps.storage.write(
      `ai/${record.id}/input.${image.extension}`,
      image.data,
      image.type,
    );
    const withImage = await this.deps.store.update(record.id, { imageUrl });
    return view(await this.runJob(withImage, 'preview', image.data));
  }

  /** The generation, first moving it on if its job has finished. The panel polls this. */
  async get(userId: string, id: string): Promise<GenerationView> {
    let record = await this.find(userId, id);
    if (record.jobId && RUNNING.has(record.status as GenerationStatus)) {
      record = await this.advance(record);
    }
    return view(record);
  }

  /** Makes a preview from one of the pictures (again, if the last preview didn't work out). */
  async pick(userId: string, id: string, index: number): Promise<GenerationView> {
    const record = await this.find(userId, id);
    const reference = record.referenceImages?.[index];
    if (!reference) throw new StudioError('BAD_REQUEST', 'There is no such picture');
    if (RUNNING.has(record.status as GenerationStatus)) {
      throw new StudioError('BAD_REQUEST', 'Wait for the current step to finish');
    }
    const picked = await this.deps.store.update(record.id, {
      imageUrl: reference.url,
      previewUrl: null,
      previewTriangles: null,
      finalUrl: null,
      finalTriangles: null,
      seed: null,
    });
    return view(await this.runJob(picked, 'preview'));
  }

  /** Makes the final model from the preview the user is keeping. */
  async keep(userId: string, id: string): Promise<GenerationView> {
    const record = await this.find(userId, id);
    if (record.status !== 'reviewing' || !record.previewUrl) {
      throw new StudioError('BAD_REQUEST', 'Only a finished preview can be kept');
    }
    return view(await this.runJob(record, 'final'));
  }

  /** Runs the step that failed again. */
  async retry(userId: string, id: string): Promise<GenerationView> {
    const record = await this.find(userId, id);
    if (record.status !== 'failed' || !record.jobKind) {
      throw new StudioError('BAD_REQUEST', 'Only a failed step can be tried again');
    }
    return view(await this.runJob(record, record.jobKind as JobKind));
  }

  async recent(userId: string, limit = 12): Promise<GenerationView[]> {
    return (await this.deps.store.recent(userId, limit)).map(view);
  }

  // ─── Jobs ─────────────────────────────────────────────────────

  /** Starts a job for the generation and records it; a job that can't start fails the step. */
  private async runJob(
    record: GenerationRecord,
    kind: JobKind,
    image?: Buffer,
  ): Promise<GenerationRecord> {
    const workers = this.workers();
    const status: GenerationStatus =
      kind === 'references' ? 'drawing' : kind === 'preview' ? 'previewing' : 'finishing';
    try {
      let jobId: string;
      if (kind === 'references') {
        jobId = await workers.startReferences({
          prompt: record.prompt ?? '',
          count: REFERENCE_COUNT,
          requestId: record.id,
        });
      } else {
        jobId = await workers.startModel({
          image: image ?? (await this.readStored(record.imageUrl)),
          mode: kind,
          // The final refines the preview's shape
          seed: kind === 'final' ? (record.seed ?? undefined) : undefined,
          requestId: record.id,
        });
      }
      return await this.deps.store.update(record.id, {
        status,
        jobId,
        jobKind: kind,
        errorMessage: null,
        updatedAt: new Date(),
      });
    } catch (err) {
      return this.fail(record, kind, err);
    }
  }

  private async advance(record: GenerationRecord): Promise<GenerationRecord> {
    const workers = this.workers();
    const kind = record.jobKind as JobKind;
    const jobId = record.jobId!;
    try {
      if (kind === 'references') {
        const state = await workers.references(jobId);
        if (state.status === 'running') return record;
        if (state.status === 'failed') return this.settle(record, failure(kind, state.message));
        const references = [];
        for (const image of state.output.images) {
          const data = await this.download(image.file);
          const key = `ai/${record.id}/reference-${image.seed}.png`;
          references.push({
            url: await this.deps.storage.write(key, data, 'image/png'),
            seed: image.seed,
          });
        }
        return await this.settle(record, {
          status: 'picking',
          referenceImages: references,
          jobId: null,
          updatedAt: new Date(),
        });
      }

      const state = await workers.model(jobId);
      if (state.status === 'running') return record;
      if (state.status === 'failed') return this.settle(record, failure(kind, state.message));
      const { output } = state;
      const data = await this.download(output.file);
      const url = await this.deps.storage.write(
        `ai/${record.id}/${kind}-${output.seed}.glb`,
        data,
        'model/gltf-binary',
      );
      const common = {
        jobId: null,
        credits: output.credits,
        durationMs: (record.durationMs ?? 0) + Math.round(output.seconds * 1000),
        updatedAt: new Date(),
      };
      return await this.settle(
        record,
        kind === 'preview'
          ? {
              ...common,
              status: 'reviewing',
              previewUrl: url,
              previewTriangles: output.triangles,
              seed: output.seed,
            }
          : { ...common, status: 'done', finalUrl: url, finalTriangles: output.triangles },
      );
    } catch (err) {
      // Couldn't reach the workers: try again on the next poll
      if (isTransient(err)) return record;
      return this.settle(record, failure(kind, err));
    }
  }

  /** Records the end of the generation's current job, unless another job replaced it meanwhile. */
  private async settle(
    record: GenerationRecord,
    patch: Partial<NewGeneration>,
  ): Promise<GenerationRecord> {
    const updated = await this.deps.store.finishJob(record.id, record.jobId!, patch);
    return updated ?? (await this.find(record.userId, record.id));
  }

  private async fail(
    record: GenerationRecord,
    kind: JobKind,
    reason: unknown,
  ): Promise<GenerationRecord> {
    return this.deps.store.update(record.id, failure(kind, reason));
  }

  // ─── Files ────────────────────────────────────────────────────

  private async download(file: WorkerFile): Promise<Buffer> {
    if ('data' in file) return file.data;
    const fetchImpl = this.deps.fetchImpl ?? ((url, init) => fetch(url, init));
    const res = await fetchImpl(file.url);
    if (!res.ok) throw new Error(`Couldn't download ${file.url}: ${res.status}`);
    return Buffer.from(await res.arrayBuffer());
  }

  /** Reads back a file this service stored, from the URL it recorded. */
  private async readStored(url: string | null): Promise<Buffer> {
    const prefix = this.deps.storage.getUrl('');
    if (!url?.startsWith(prefix)) throw new Error('The picture for this model is missing');
    return this.deps.storage.read(url.slice(prefix.length));
  }

  private async checkLimits(userId: string): Promise<void> {
    const recent = await this.deps.store.recent(userId, LIMITS.perHour + LIMITS.running);
    const running = recent.filter((r) => RUNNING.has(r.status as GenerationStatus)).length;
    if (running >= LIMITS.running) {
      throw new StudioError(
        'BAD_REQUEST',
        `Wait for one of your ${running} models in progress to finish`,
      );
    }
    const hourAgo = Date.now() - 60 * 60 * 1000;
    if (recent.filter((r) => r.createdAt.getTime() > hourAgo).length >= LIMITS.perHour) {
      throw new StudioError(
        'BAD_REQUEST',
        `That's ${LIMITS.perHour} models this hour; try again later`,
      );
    }
  }

  private async find(userId: string, id: string): Promise<GenerationRecord> {
    const record = await this.deps.store.find(id, userId);
    if (!record) throw new StudioError('NOT_FOUND', 'This generation was not found');
    return record;
  }

  private workers(): StudioWorkers {
    if (!this.deps.workers) {
      throw new StudioError(
        'UNAVAILABLE',
        'AI generation is not set up on this server (see AI_WORKERS_URL in .env.example)',
      );
    }
    return this.deps.workers;
  }
}

function failure(kind: JobKind, reason: unknown): Partial<NewGeneration> {
  const message = reason instanceof Error ? reason.message : String(reason);
  return {
    status: 'failed',
    jobId: null,
    jobKind: kind,
    errorMessage: message.slice(0, 500),
    updatedAt: new Date(),
  };
}

function view(record: GenerationRecord): GenerationView {
  return {
    id: record.id,
    source: record.source === 'photo' ? 'photo' : 'prompt',
    prompt: record.prompt,
    status: record.status as GenerationStatus,
    references: record.referenceImages ?? [],
    image: record.imageUrl,
    preview: record.previewUrl
      ? { url: record.previewUrl, triangles: record.previewTriangles }
      : null,
    final: record.finalUrl ? { url: record.finalUrl, triangles: record.finalTriangles } : null,
    credits: record.credits ?? [],
    error: record.errorMessage,
    sceneId: record.sceneId,
    createdAt: record.createdAt.toISOString(),
  };
}

/** Network trouble between us and the workers, as opposed to a job that failed. */
function isTransient(err: unknown): boolean {
  const message = err instanceof Error ? err.message : '';
  return /fetch failed|ECONNRESET|ETIMEDOUT|ENOTFOUND|: 5\d\d /.test(message);
}

/**
 * Checks a photo, turns it upright (phone photos store their rotation), scales it down and keeps
 * it as PNG when it has transparency (TRELLIS.2 then skips background removal), else JPEG.
 */
export async function normalisePhoto(
  data: Buffer,
): Promise<{ data: Buffer; type: string; extension: string }> {
  let meta: sharp.Metadata;
  try {
    meta = await sharp(data, { limitInputPixels: MAX_PHOTO_PIXELS }).metadata();
  } catch {
    throw new StudioError(
      'BAD_REQUEST',
      "That file isn't a picture we can read (use PNG, JPEG or WebP)",
    );
  }
  if (!meta.width || !meta.height || !['png', 'jpeg', 'webp'].includes(meta.format ?? '')) {
    throw new StudioError(
      'BAD_REQUEST',
      "That file isn't a picture we can read (use PNG, JPEG or WebP)",
    );
  }
  const image = sharp(data, { limitInputPixels: MAX_PHOTO_PIXELS })
    .rotate()
    .resize(MAX_PHOTO_SIDE, MAX_PHOTO_SIDE, { fit: 'inside', withoutEnlargement: true });
  if (meta.hasAlpha) {
    return { data: await image.png().toBuffer(), type: 'image/png', extension: 'png' };
  }
  return {
    data: await image.jpeg({ quality: 92 }).toBuffer(),
    type: 'image/jpeg',
    extension: 'jpg',
  };
}

/** Generations kept in Postgres (the ai_generations table). */
export function drizzleGenerationStore(db: Database): GenerationStore {
  const table = schema.aiGenerations;
  return {
    async create(values) {
      const [row] = await db.insert(table).values(values).returning();
      return row!;
    },
    async find(id, userId) {
      const [row] = await db
        .select()
        .from(table)
        .where(and(eq(table.id, id), eq(table.userId, userId)));
      return row ?? null;
    },
    async update(id, patch) {
      const [row] = await db.update(table).set(patch).where(eq(table.id, id)).returning();
      if (!row) throw new StudioError('NOT_FOUND', 'This generation was not found');
      return row;
    },
    async finishJob(id, jobId, patch) {
      const [row] = await db
        .update(table)
        .set(patch)
        .where(and(eq(table.id, id), eq(table.jobId, jobId)))
        .returning();
      return row ?? null;
    },
    async recent(userId, limit) {
      return db
        .select()
        .from(table)
        .where(eq(table.userId, userId))
        .orderBy(desc(table.createdAt))
        .limit(limit);
    },
  };
}
