import sharp from 'sharp';
import { and, desc, eq, isNull } from 'drizzle-orm';
import type { Database } from '../../db';
import { schema } from '../../db';
import type { StorageProvider } from '../storage';
import type { FetchLike } from './providers/jobEndpoint';
import type {
  ModelView,
  StudioWorkers,
  TexturesOutput,
  ViewsOutput,
  WorkerFile,
  WorkerJobState,
  WorkerKind,
} from './types';

/**
 * The Studio's AI panel, one generation at a time:
 *
 *   prompt: drawing -> picking -> [viewing ->] previewing -> reviewing -> finishing -> done
 *   photo:                         [viewing ->] previewing -> reviewing -> finishing -> done
 *
 * drawing: FLUX draws 4 pictures. picking: the user picks one (or another one later).
 * viewing (only with AI_MULTIVIEW=1): MV-Adapter draws the picture from 6 sides, so the model's
 * back and sides come from those views instead of being invented. If that job fails or takes
 * longer than VIEWS_TIMEOUT_MS, the generation carries on with the picture alone and keeps the
 * reason in `viewsError`: this step never fails a generation.
 * previewing: TRELLIS.2 makes a quick model (512³) the user can place and look at.
 * reviewing: the user keeps it, or picks another picture. finishing: the final model (1024³,
 * same seed and the same views, so it refines the shape they approved). Any other job can fail;
 * retry runs it again.
 *
 * Texture options (AI_TEXTURE_OPTIONS, on by default): once the final is done, TRELLIS.2 makes
 * TEXTURE_COUNT more textures for its shape, and the user picks one of the four in the panel. On
 * the 20 test prompts the final's own texture was good enough to publish 10 times, and the best
 * of four 16 times. The final is done and usable at once: the textures job runs after it, with
 * its own state (texturesStatus), and never fails the generation.
 *
 * The judge (AI_TEXTURE_JUDGE=1, off by default): the textures job also asks a self-hosted judge
 * (Qwen3-VL-8B) which of the four textures a creator would rather use, and the panel makes its pick
 * the default, once. On a 20-object validation set its pick was publishable for 15, the final's
 * own texture for 11 (the reviewers' own best: about 16). When the judge fails, the reason is kept
 * (texturesJudgeError) and the options are offered as without it. The panel tells the server when
 * it applied the pick and which texture the creator chose (chooseTexture), so the pick never goes
 * over the creator's choice, and never goes in twice.
 *
 * Every job is started and then polled by `get`, so no request waits on a GPU. Pictures, views
 * and models are copied into the server's storage, so scenes keep working after the workers
 * forget their outputs.
 *
 * The FLUX worker rates each picture as a start for 3D (the object's size in the frame, not cut
 * off, just one), and `recommended` points out the best one. The user still picks.
 *
 * A GPU that has been idle takes a while to start (about 45 s for FLUX, 100 s for TRELLIS.2), so
 * `warm` starts one before its job: FLUX when the panel opens, TRELLIS.2 (and MV-Adapter) while
 * FLUX draws, and TRELLIS.2 again while the views are drawn, for users who took a while to pick.
 */

/** The steps that wait on a GPU job: the panel polls `get` while one runs */
export type RunningStatus = 'drawing' | 'viewing' | 'previewing' | 'finishing';
export type GenerationStatus = RunningStatus | 'picking' | 'reviewing' | 'done' | 'failed';

type JobKind = 'references' | 'views' | 'preview' | 'final';

/** The step each job is */
const STEP_OF: Record<JobKind, RunningStatus> = {
  references: 'drawing',
  views: 'viewing',
  preview: 'previewing',
  final: 'finishing',
};

export type GenerationRecord = typeof schema.aiGenerations.$inferSelect;
export type NewGeneration = typeof schema.aiGenerations.$inferInsert;
/** A picture drawn for a prompt, with the worker's rating of it (score, issues) if it gave one */
export type ReferencePicture = NonNullable<GenerationRecord['referenceImages']>[number];
/** One of the views the model is built from, as stored with the generation */
export type StoredView = NonNullable<GenerationRecord['views']>[number];
/** One more texture for the final's shape (a texture option), as stored with the generation */
export type TextureOption = NonNullable<GenerationRecord['textures']>[number];
/** Where the texture options are: their job runs, they are made, or there are none */
export type TexturesStatus = 'running' | 'done' | 'failed';
/** Who put a texture option in the scene: the creator, or the panel with the judge's pick */
export type TextureChooser = 'creator' | 'judge';

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
  /**
   * Updates the texture options of a done generation only while `jobId` is still their job (null
   * while it's still to start), so overlapping polls keep one textures job, and no textures land
   * on a final that was replaced meanwhile. Returns null when that isn't so.
   */
  finishTextures(
    id: string,
    jobId: string | null,
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
  references: ReferencePicture[];
  /** The index of the picture with the highest score, or null when none has a score */
  recommended: number | null;
  /** The picture that went (or goes) to 3D */
  image: string | null;
  /** The picture from other sides that the preview and final are built from (AI_MULTIVIEW=1) */
  views: StoredView[];
  /** Why there are no views: the multiview job failed or took too long (the model went on) */
  viewsError: string | null;
  preview: { url: string; triangles: number | null } | null;
  final: { url: string; triangles: number | null } | null;
  /**
   * Texture options: more textures for the final's shape, made after it, for the user to choose
   * from. The final's own texture is the first choice. Null when none were asked for.
   */
  textures: {
    status: TexturesStatus;
    /** How many more textures were asked for */
    count: number;
    /** The textures made so far, in order; the final's own isn't among them */
    options: TextureOption[];
    /** Why there are none, or why some of them couldn't be made */
    error: string | null;
    /**
     * The judge's pick (AI_TEXTURE_JUDGE=1), numbered as the panel numbers the textures (1 is the
     * final's own), with why, and whether the panel put it in the scene (it does so once). Null
     * when the judge wasn't asked or gave no pick.
     */
    recommended: { number: number; why: string | null; applied: boolean } | null;
    /** The texture the creator last chose in the panel; null while they haven't chosen one */
    chosen: number | null;
    /** Why the judge gave no pick, when it was asked */
    judgeError: string | null;
  } | null;
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

/**
 * Every step a GPU job runs in. A generation in one counts towards LIMITS.running, so the views
 * step is bounded like the others.
 */
const RUNNING: ReadonlySet<GenerationStatus> = new Set(Object.values(STEP_OF));
const REFERENCE_COUNT = 4;
/** Bounds on GPU spending per user until credits exist */
export const LIMITS = { running: 3, perHour: 30 };
export const MAX_PROMPT_LENGTH = 500;
/** Photos are scaled down to this before they are stored and sent to TRELLIS.2 */
const MAX_PHOTO_SIDE = 2048;
/** Bigger photos are refused before decoding (about 8000 x 6000) */
const MAX_PHOTO_PIXELS = 50_000_000;
/** How often a user may start each GPU early: about a cold start plus the minute it stays up */
export const WARM_INTERVAL_MS = 2 * 60 * 1000;
/**
 * How long the views may take, from cold (the worker's start plus the job) or stuck in a queue,
 * before the model is made from the picture alone. They only help, so don't wait long for them.
 */
export const VIEWS_TIMEOUT_MS = 3 * 60 * 1000;
/**
 * How many more textures are made for a final (texture options): with its own, 4 to choose from.
 * Each takes about 30 s of GPU, after about 30 s to make the shape again.
 */
export const TEXTURE_COUNT = 3;

/** The judge's pick and the creator's choice, before there are any */
const NO_PICK = {
  texturesPick: null,
  texturesPickWhy: null,
  texturesJudgeError: null,
  texturesPickApplied: false,
  texturesChosen: null,
} satisfies Partial<NewGeneration>;
/** The texture options' columns when a final is done and its textures job is still to start */
const TEXTURES_TO_START = {
  texturesStatus: 'running',
  texturesJobId: null,
  textures: null,
  texturesError: null,
  ...NO_PICK,
} satisfies Partial<NewGeneration>;
/** The texture options' columns when there are none */
const NO_TEXTURES = {
  texturesStatus: null,
  texturesJobId: null,
  textures: null,
  texturesError: null,
  ...NO_PICK,
} satisfies Partial<NewGeneration>;

export interface StudioDeps {
  workers: StudioWorkers | null;
  store: GenerationStore;
  storage: StorageProvider;
  /**
   * AI_MULTIVIEW=1: draw the picture from other sides before the preview, when the workers have
   * the multiview worker. Off by default.
   */
  multiview?: boolean;
  /**
   * AI_TEXTURE_OPTIONS: after the final, make TEXTURE_COUNT more textures for its shape, when the
   * workers can. On unless false.
   */
  textureOptions?: boolean;
  /**
   * AI_TEXTURE_JUDGE=1: the textures job also asks the judge which texture a creator would rather
   * use, and the panel makes its pick the default. Off by default.
   */
  textureJudge?: boolean;
  /** Downloads worker outputs stored at a URL (workers with R2) */
  fetchImpl?: FetchLike;
}

export class AIStudio {
  /** When each user last started each GPU early (key `userId:kind`), oldest first */
  private readonly warmedAt = new Map<string, number>();

  constructor(private readonly deps: StudioDeps) {}

  capabilities() {
    const { workers } = this.deps;
    return {
      available: workers !== null,
      prompts: workers?.prompts ?? false,
      photos: workers !== null,
      /** Models are built from the picture's other sides too (the views step runs) */
      multiview: this.multiview(),
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
    const kind = this.firstModelStep();
    const record = await this.deps.store.create({
      userId,
      sceneId: sceneId ?? null,
      provider: workers.name,
      source: 'photo',
      status: STEP_OF[kind],
      jobKind: kind,
    });
    const imageUrl = await this.deps.storage.write(
      `ai/${record.id}/input.${image.extension}`,
      image.data,
      image.type,
    );
    const withImage = await this.deps.store.update(record.id, { imageUrl });
    return view(await this.runJob(withImage, kind, image.data));
  }

  /**
   * The generation, first moving it on if its job has finished. The panel polls this, also while
   * the texture options are made.
   */
  async get(userId: string, id: string): Promise<GenerationView> {
    let record = await this.find(userId, id);
    if (record.jobId && RUNNING.has(record.status as GenerationStatus)) {
      record = await this.advance(record);
    }
    // The final is done and usable; its texture options follow
    if (record.status === 'done' && record.texturesStatus === 'running') {
      record = await this.advanceTextures(record);
    }
    return view(record);
  }

  /**
   * Makes a preview from one of the pictures (again, if the last preview didn't work out),
   * drawing its other sides first with AI_MULTIVIEW=1.
   */
  async pick(userId: string, id: string, index: number): Promise<GenerationView> {
    const record = await this.find(userId, id);
    const reference = record.referenceImages?.[index];
    if (!reference) throw new StudioError('BAD_REQUEST', 'There is no such picture');
    if (RUNNING.has(record.status as GenerationStatus)) {
      throw new StudioError('BAD_REQUEST', 'Wait for the current step to finish');
    }
    // The old final's textures won't be offered: stop their job (it bills while it runs)
    if (record.texturesJobId) this.cancel('model', record.texturesJobId);
    const picked = await this.deps.store.update(record.id, {
      imageUrl: reference.url,
      views: null,
      viewsError: null,
      previewUrl: null,
      previewTriangles: null,
      finalUrl: null,
      finalTriangles: null,
      finalPipeline: null,
      ...NO_TEXTURES,
      seed: null,
    });
    return view(await this.runJob(picked, this.firstModelStep()));
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

  /**
   * Records a texture option the panel put in the scene: one the creator chose (by 'creator'), or
   * the judge's pick, which the panel applies once (by 'judge'). The panel reads both back, so the
   * pick never goes over the creator's choice, nor in twice, even after a reload.
   */
  async chooseTexture(
    userId: string,
    id: string,
    number: number,
    by: TextureChooser,
  ): Promise<GenerationView> {
    const record = await this.find(userId, id);
    const count = record.textures?.length ?? 0;
    if (
      record.status !== 'done' ||
      record.texturesStatus !== 'done' ||
      !Number.isInteger(number) ||
      number < 1 ||
      number > count + 1
    ) {
      throw new StudioError('BAD_REQUEST', 'There is no such texture');
    }
    if (by === 'judge') {
      if (record.texturesPick !== number) {
        throw new StudioError('BAD_REQUEST', "That texture isn't the recommended one");
      }
      return view(await this.deps.store.update(record.id, { texturesPickApplied: true }));
    }
    return view(await this.deps.store.update(record.id, { texturesChosen: number }));
  }

  async recent(userId: string, limit = 12): Promise<GenerationView[]> {
    return (await this.deps.store.recent(userId, limit)).map(view);
  }

  /**
   * Starts a GPU before the user's job needs it, so the job skips the cold start. Returns at once
   * and never throws (only speed is at stake), and asks at most once per GPU per user every
   * WARM_INTERVAL_MS. Does nothing when the workers can't be started early, or for the multiview
   * worker when the views step doesn't run.
   */
  warm(userId: string, kind: WorkerKind): void {
    const { workers } = this.deps;
    if (!workers?.warm) return;
    if (kind === 'multiview' && !this.multiview()) return;
    const now = Date.now();
    // Keys are added as time goes on, so the expired ones are at the front
    for (const [key, at] of this.warmedAt) {
      if (now - at < WARM_INTERVAL_MS) break;
      this.warmedAt.delete(key);
    }
    const key = `${userId}:${kind}`;
    if (this.warmedAt.has(key)) return;
    this.warmedAt.set(key, now);
    void (async () => {
      try {
        await workers.warm?.(kind);
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        console.warn(`[AI] Couldn't start the ${kind} GPU early:`, message);
      }
    })();
  }

  // ─── Jobs ─────────────────────────────────────────────────────

  /**
   * Starts a job for the generation and records it. A job that can't start fails the step,
   * except the views: without them the preview starts from the picture alone.
   */
  private async runJob(
    record: GenerationRecord,
    kind: JobKind,
    image?: Buffer,
  ): Promise<GenerationRecord> {
    const workers = this.workers();
    try {
      let jobId: string;
      if (kind === 'references') {
        jobId = await workers.startReferences({
          prompt: record.prompt ?? '',
          count: REFERENCE_COUNT,
          requestId: record.id,
        });
      } else if (kind === 'views') {
        jobId = await workers.startViews({
          image: image ?? (await this.readStored(record.imageUrl)),
          requestId: record.id,
        });
      } else {
        // The preview and the final get the same views, so the final refines the same shape
        const views = await this.readViews(record.views);
        jobId = await workers.startModel({
          image: image ?? (await this.readStored(record.imageUrl)),
          ...(views ? { views } : {}),
          mode: kind,
          // The final refines the preview's shape
          seed: kind === 'final' ? (record.seed ?? undefined) : undefined,
          requestId: record.id,
        });
      }
      const started = await this.deps.store.update(record.id, {
        status: STEP_OF[kind],
        jobId,
        jobKind: kind,
        errorMessage: null,
        updatedAt: new Date(),
      });
      if (kind === 'references') {
        // The pictures take about a minute from cold, so the GPUs for the picked one start
        // meanwhile and are ready (or nearly) by the time the user picks one. TRELLIS.2 takes
        // longest to start, so it starts here even when the views come first.
        this.warm(record.userId, 'model');
        if (this.multiview()) this.warm(record.userId, 'multiview');
      } else if (kind === 'views') {
        // TRELLIS.2 is next. This starts it for users who took a while to pick (at most once
        // every WARM_INTERVAL_MS, so not again if it was started while the pictures were drawn).
        this.warm(record.userId, 'model');
      }
      return started;
    } catch (err) {
      if (kind === 'views') {
        // The picture alone still makes a model
        const updated = await this.deps.store.update(record.id, {
          views: null,
          viewsError: withoutViews(record, err),
        });
        return this.runJob(updated, 'preview', image);
      }
      return this.fail(record, kind, err);
    }
  }

  private async advance(record: GenerationRecord): Promise<GenerationRecord> {
    const workers = this.workers();
    const kind = record.jobKind as JobKind;
    const jobId = record.jobId!;
    if (kind === 'views') return this.advanceViews(record);
    try {
      if (kind === 'references') {
        const state = await workers.references(jobId);
        if (state.status === 'running') return record;
        if (state.status === 'failed') return this.settle(record, failure(kind, state.message));
        const references: ReferencePicture[] = [];
        for (const { file, seed, score, issues } of state.output.images) {
          const data = await this.download(file);
          const key = `ai/${record.id}/reference-${seed}.png`;
          references.push({
            url: await this.deps.storage.write(key, data, 'image/png'),
            seed,
            ...(score === undefined ? {} : { score }),
            ...(issues?.length ? { issues } : {}),
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
      if (record.views?.length && !output.viewsUsed) {
        // A 3D worker from before views ignores them: the model's back is invented again
        console.warn(
          `[AI] Generation ${record.id}: the ${kind} was made without the ${record.views.length} views it was sent (does the 3D worker take views yet?)`,
        );
      }
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
          : {
              ...common,
              status: 'done',
              finalUrl: url,
              finalTriangles: output.triangles,
              finalPipeline: output.pipeline ?? null,
              // More textures for its shape follow (advanceTextures starts their job)
              ...(this.textureOptions() ? TEXTURES_TO_START : NO_TEXTURES),
            },
      );
    } catch (err) {
      // Couldn't reach the workers: try again on the next poll
      if (isTransient(err)) return record;
      return this.settle(record, failure(kind, err));
    }
  }

  /**
   * Moves the views step on. Once the views are in, or the job failed or has taken longer than
   * VIEWS_TIMEOUT_MS, the preview starts: from the views, or from the picture alone.
   */
  private async advanceViews(record: GenerationRecord): Promise<GenerationRecord> {
    const workers = this.workers();
    const jobId = record.jobId!;
    // Nothing writes the row while its job runs, so updatedAt is when the views job started
    const overdue = Date.now() - record.updatedAt.getTime() >= VIEWS_TIMEOUT_MS;
    let state: WorkerJobState<ViewsOutput>;
    try {
      state = await workers.views(jobId);
    } catch (err) {
      // Couldn't reach the workers: try again on the next poll, while there's time
      if (isTransient(err) && !overdue) return record;
      return this.afterViews(record, null, err);
    }
    if (state.status === 'running') {
      if (!overdue) return record;
      // It bills while it runs, and nothing will use it now
      this.cancel('multiview', jobId);
      return this.afterViews(
        record,
        null,
        `the views took longer than ${Math.round(VIEWS_TIMEOUT_MS / 1000)} s`,
      );
    }
    if (state.status === 'failed') return this.afterViews(record, null, state.message);
    let views: DrawnViews;
    try {
      views = await this.storeViews(record, state.output);
    } catch (err) {
      if (isTransient(err) && !overdue) return record;
      return this.afterViews(record, null, err);
    }
    return this.afterViews(record, views);
  }

  /**
   * Ends the views step by starting the preview: with the views when they came, else from the
   * picture alone, logging why (kept as viewsError). Polls can overlap, so the preview job is
   * only recorded if the views job is still the generation's; the loser's preview is stopped.
   */
  private async afterViews(
    record: GenerationRecord,
    views: DrawnViews | null,
    problem?: unknown,
  ): Promise<GenerationRecord> {
    const workers = this.workers();
    const patch: Partial<NewGeneration> = {
      views: views?.stored ?? null,
      viewsError: views ? null : withoutViews(record, problem),
      durationMs: (record.durationMs ?? 0) + Math.round((views?.seconds ?? 0) * 1000),
    };
    let jobId: string;
    try {
      jobId = await workers.startModel({
        image: await this.readStored(record.imageUrl),
        ...(views ? { views: views.images } : {}),
        mode: 'preview',
        requestId: record.id,
      });
    } catch (err) {
      // The views are kept, so Try again starts the preview from them
      return this.settle(record, { ...patch, ...failure('preview', err) });
    }
    const started = await this.deps.store.finishJob(record.id, record.jobId!, {
      ...patch,
      status: 'previewing',
      jobId,
      jobKind: 'preview',
      errorMessage: null,
      updatedAt: new Date(),
    });
    if (started) return started;
    this.cancel('model', jobId);
    return this.find(record.userId, record.id);
  }

  /**
   * Moves the texture options on once the final is done: starts their job, then polls it and
   * copies the textures into storage. They only add choices, so nothing here fails the
   * generation: when they can't be made, the reason is kept as texturesError.
   */
  private async advanceTextures(record: GenerationRecord): Promise<GenerationRecord> {
    const jobId = record.texturesJobId;
    if (!jobId) return this.startTextures(record);
    let state: WorkerJobState<TexturesOutput>;
    try {
      const workers = this.workers();
      if (!workers.textures) throw new Error("This server's workers don't make texture options");
      state = await workers.textures(jobId);
    } catch (err) {
      // Couldn't reach the workers: try again on the next poll
      if (isTransient(err)) return record;
      return this.texturesFailed(record, err);
    }
    if (state.status === 'running') return record;
    if (state.status === 'failed') return this.texturesFailed(record, state.message);
    const { output } = state;
    // A final that ran out of GPU memory is made by the "512" pipeline, and so is a shape made
    // again for its textures. When the two differ, the textures fit another shape than the final.
    if (record.finalPipeline && output.pipeline && output.pipeline !== record.finalPipeline) {
      return this.texturesFailed(
        record,
        `the textures fit another shape (pipeline ${output.pipeline}; the final's is ${record.finalPipeline})`,
      );
    }
    let made: { options: TextureOption[]; problems: string[] };
    try {
      made = await this.storeTextures(record, output);
    } catch (err) {
      // Network trouble while copying them: try again on the next poll
      if (isTransient(err)) return record;
      return this.texturesFailed(record, err);
    }
    if (!made.options.length) {
      return this.texturesFailed(record, made.problems.join('; ') || 'no textures came back');
    }
    const problems = made.problems.join('; ').slice(0, 500);
    if (problems) console.warn(`[AI] Generation ${record.id}: some textures failed:`, problems);
    const pick = judgePick(output, made.options, this.textureJudge());
    if (pick.texturesJudgeError) {
      console.warn(
        `[AI] Generation ${record.id} has no recommended texture:`,
        pick.texturesJudgeError,
      );
    }
    return this.settleTextures(record, {
      texturesStatus: 'done',
      texturesJobId: null,
      textures: made.options,
      texturesError: problems || null,
      ...pick,
      durationMs: (record.durationMs ?? 0) + Math.round(output.seconds * 1000),
      updatedAt: new Date(),
    });
  }

  /**
   * Starts the textures job for a done final, from what the final was made from: the picture,
   * its seed and its views, and with AI_TEXTURE_JUDGE=1 the prompt for the judge. Overlapping
   * polls can both start one: the first is kept, the other stopped.
   */
  private async startTextures(record: GenerationRecord): Promise<GenerationRecord> {
    let jobId: string;
    try {
      const workers = this.workers();
      if (!this.textureOptions() || !workers.startTextures) {
        throw new Error('Texture options are off on this server');
      }
      if (record.seed === null) throw new Error("The final's seed is missing");
      const views = await this.readViews(record.views);
      jobId = await workers.startTextures({
        image: await this.readStored(record.imageUrl),
        ...(views ? { views } : {}),
        seed: record.seed,
        count: TEXTURE_COUNT,
        requestId: record.id,
        // What the user typed; a photo has no prompt
        ...(this.textureJudge() ? { judge: { prompt: record.prompt ?? '' } } : {}),
      });
    } catch (err) {
      // Couldn't reach the workers: try again on the next poll
      if (isTransient(err)) return record;
      return this.texturesFailed(record, err);
    }
    const started = await this.deps.store.finishTextures(record.id, null, {
      texturesJobId: jobId,
      updatedAt: new Date(),
    });
    if (started) return started;
    this.cancel('model', jobId);
    return this.find(record.userId, record.id);
  }

  /** Ends the texture options with none to choose from, logging why and keeping it. */
  private async texturesFailed(
    record: GenerationRecord,
    reason: unknown,
  ): Promise<GenerationRecord> {
    const message = messageOf(reason).slice(0, 500);
    console.warn(`[AI] Generation ${record.id} has no texture options:`, message);
    return this.settleTextures(record, {
      texturesStatus: 'failed',
      texturesJobId: null,
      texturesError: message,
      updatedAt: new Date(),
    });
  }

  /** Records how the textures job ended, unless another poll did or the final changed meanwhile. */
  private async settleTextures(
    record: GenerationRecord,
    patch: Partial<NewGeneration>,
  ): Promise<GenerationRecord> {
    const updated = await this.deps.store.finishTextures(record.id, record.texturesJobId, patch);
    return updated ?? (await this.find(record.userId, record.id));
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

  /** Stops a job nobody waits for any more (it bills while it runs). Never throws. */
  private cancel(kind: WorkerKind, jobId: string): void {
    const { workers } = this.deps;
    if (!workers?.cancel) return;
    void (async () => {
      try {
        await workers.cancel?.(kind, jobId);
      } catch (err) {
        console.warn(`[AI] Couldn't stop the ${kind} job ${jobId}:`, messageOf(err));
      }
    })();
  }

  /** Whether the views step runs: AI_MULTIVIEW=1, and a multiview worker to run it. */
  private multiview(): boolean {
    return !!this.deps.multiview && !!this.deps.workers?.multiview;
  }

  /** Whether finals get texture options: AI_TEXTURE_OPTIONS isn't off, and the workers make them. */
  private textureOptions(): boolean {
    return this.deps.textureOptions !== false && !!this.deps.workers?.startTextures;
  }

  /** Whether the textures job asks the judge for its pick: AI_TEXTURE_JUDGE=1. */
  private textureJudge(): boolean {
    return !!this.deps.textureJudge;
  }

  /** Where a chosen picture goes first: to its other sides (AI_MULTIVIEW=1), else the preview. */
  private firstModelStep(): JobKind {
    return this.multiview() ? 'views' : 'preview';
  }

  // ─── Files ────────────────────────────────────────────────────

  /**
   * Copies the views into storage, checking each is a picture. They are kept under the job's id,
   * so the views of a later pick never share a URL with these (or a browser's cached copy).
   */
  private async storeViews(record: GenerationRecord, output: ViewsOutput): Promise<DrawnViews> {
    if (!output.views.length) throw new Error('the multiview job returned no views');
    const folder = `ai/${record.id}/views-${record.jobId!.replace(/[^A-Za-z0-9_-]/g, '')}`;
    const drawn = await Promise.all(
      output.views.map(async ({ file, azimuth, elevation }, i) => {
        const data = await this.download(file);
        const format = await pictureFormat(data);
        if (!format) throw new Error(`view ${i + 1} of the multiview job isn't a picture`);
        const url = await this.deps.storage.write(
          `${folder}/${i}-${Math.round(azimuth)}.${format.extension}`,
          data,
          format.type,
        );
        return { url, azimuth, elevation, data };
      }),
    );
    return {
      stored: drawn.map(({ url, azimuth, elevation }) => ({ url, azimuth, elevation })),
      images: drawn.map(({ data, azimuth, elevation }) => ({ image: data, azimuth, elevation })),
      seconds: output.seconds,
    };
  }

  /**
   * Copies the texture options into storage, beside the final. A texture that can't be copied
   * joins the problems, with the ones the worker couldn't make; network trouble throws, so the
   * next poll tries again.
   */
  private async storeTextures(
    record: GenerationRecord,
    output: TexturesOutput,
  ): Promise<{ options: TextureOption[]; problems: string[] }> {
    const problems = output.errors.map(({ textureSeed, message }) =>
      textureSeed === null ? message : `texture ${textureSeed}: ${message}`,
    );
    const options: TextureOption[] = [];
    for (const { file, textureSeed, triangles } of output.textures) {
      try {
        const data = await this.download(file);
        const url = await this.deps.storage.write(
          `ai/${record.id}/final-${record.seed}-texture-${textureSeed}.glb`,
          data,
          'model/gltf-binary',
        );
        options.push({ url, triangles, textureSeed });
      } catch (err) {
        if (isTransient(err)) throw err;
        problems.push(`texture ${textureSeed}: ${messageOf(err)}`);
      }
    }
    return { options, problems };
  }

  /** The stored views read back for the 3D worker, or undefined when there are none. */
  private async readViews(views: StoredView[] | null): Promise<ModelView[] | undefined> {
    if (!views?.length) return undefined;
    return Promise.all(
      views.map(async ({ url, azimuth, elevation }) => ({
        image: await this.readStored(url, 'A view'),
        azimuth,
        elevation,
      })),
    );
  }

  private async download(file: WorkerFile): Promise<Buffer> {
    if ('data' in file) return file.data;
    const fetchImpl = this.deps.fetchImpl ?? ((url, init) => fetch(url, init));
    const res = await fetchImpl(file.url);
    if (!res.ok) throw new Error(`Couldn't download ${file.url}: ${res.status}`);
    return Buffer.from(await res.arrayBuffer());
  }

  /** Reads back a file this service stored, from the URL it recorded. */
  private async readStored(url: string | null, what = 'The picture'): Promise<Buffer> {
    const prefix = this.deps.storage.getUrl('');
    if (!url?.startsWith(prefix)) throw new Error(`${what} for this model is missing`);
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

/** The views as stored with the generation, and as sent to the 3D worker */
interface DrawnViews {
  stored: StoredView[];
  images: ModelView[];
  /** GPU seconds the multiview job took */
  seconds: number;
}

function messageOf(reason: unknown): string {
  return reason instanceof Error ? reason.message : String(reason);
}

function failure(kind: JobKind, reason: unknown): Partial<NewGeneration> {
  return {
    status: 'failed',
    jobId: null,
    jobKind: kind,
    errorMessage: messageOf(reason).slice(0, 500),
    updatedAt: new Date(),
  };
}

/** Logs why a generation goes on without views, and returns that to keep as its viewsError. */
function withoutViews(record: GenerationRecord, reason: unknown): string {
  const message = messageOf(reason).slice(0, 500);
  console.warn(`[AI] Generation ${record.id} goes on without views:`, message);
  return message;
}

/**
 * The judge's pick as the panel numbers the textures (1 is the final's own, then the stored ones
 * in order), with why; or why there is none. Textures that failed or couldn't be copied leave
 * gaps, so the pick is found by its texture's seed. With the judge off (not `asked`), there is
 * none, even from a job started while it was on.
 */
function judgePick(
  output: TexturesOutput,
  options: TextureOption[],
  asked: boolean,
): Pick<NewGeneration, 'texturesPick' | 'texturesPickWhy' | 'texturesJudgeError'> {
  const none = (reason: string | null) => ({
    texturesPick: null,
    texturesPickWhy: null,
    texturesJudgeError: reason?.slice(0, 500) ?? null,
  });
  if (!asked) return none(null);
  if (output.judgeError) return none(output.judgeError);
  // A 3D worker from before the judge ignores the question
  if (!output.judge) {
    return none("the judge's answer didn't come back (does the 3D worker ask the judge yet?)");
  }
  const { textureSeed, why } = output.judge;
  let number = 1;
  if (textureSeed !== null) {
    const index = options.findIndex((option) => option.textureSeed === textureSeed);
    if (index < 0) return none(`the judge picked texture ${textureSeed}, which isn't offered`);
    number = index + 2;
  }
  return {
    texturesPick: number,
    texturesPickWhy: why.slice(0, 500) || null,
    texturesJudgeError: null,
  };
}

/** A worker's picture's format, or null when it isn't a PNG, JPEG or WebP picture. */
async function pictureFormat(data: Buffer): Promise<{ type: string; extension: string } | null> {
  const meta = await sharp(data)
    .metadata()
    .catch(() => null);
  if (!meta?.width || !meta.height) return null;
  switch (meta.format) {
    case 'png':
      return { type: 'image/png', extension: 'png' };
    case 'webp':
      return { type: 'image/webp', extension: 'webp' };
    case 'jpeg':
      return { type: 'image/jpeg', extension: 'jpg' };
    default:
      return null;
  }
}

function view(record: GenerationRecord): GenerationView {
  const references = record.referenceImages ?? [];
  return {
    id: record.id,
    source: record.source === 'photo' ? 'photo' : 'prompt',
    prompt: record.prompt,
    status: record.status as GenerationStatus,
    references,
    recommended: bestPicture(references),
    image: record.imageUrl,
    views: record.views ?? [],
    viewsError: record.viewsError,
    preview: record.previewUrl
      ? { url: record.previewUrl, triangles: record.previewTriangles }
      : null,
    final: record.finalUrl ? { url: record.finalUrl, triangles: record.finalTriangles } : null,
    textures: record.texturesStatus
      ? {
          status: record.texturesStatus as TexturesStatus,
          count: TEXTURE_COUNT,
          options: record.textures ?? [],
          error: record.texturesError,
          recommended:
            record.texturesPick === null
              ? null
              : {
                  number: record.texturesPick,
                  why: record.texturesPickWhy,
                  applied: record.texturesPickApplied,
                },
          chosen: record.texturesChosen,
          judgeError: record.texturesJudgeError,
        }
      : null,
    credits: record.credits ?? [],
    error: record.errorMessage,
    sceneId: record.sceneId,
    createdAt: record.createdAt.toISOString(),
  };
}

/**
 * The picture with the highest score (the first of equals), or null when none has one, such as
 * pictures drawn before the worker rated them.
 */
function bestPicture(references: ReferencePicture[]): number | null {
  let best: number | null = null;
  let bestScore = -Infinity;
  for (const [index, { score }] of references.entries()) {
    if (typeof score === 'number' && score > bestScore) {
      best = index;
      bestScore = score;
    }
  }
  return best;
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
    async finishTextures(id, jobId, patch) {
      const [row] = await db
        .update(table)
        .set(patch)
        .where(
          and(
            eq(table.id, id),
            eq(table.status, 'done'),
            eq(table.texturesStatus, 'running'),
            jobId === null ? isNull(table.texturesJobId) : eq(table.texturesJobId, jobId),
          ),
        )
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
