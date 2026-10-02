export type GenerationType = 'text-to-3d' | 'image-to-3d';

/**
 * `preview` is the cheap pass (TRELLIS.2 at 512³, ~30k triangles) shown while the user decides;
 * `final` is the asset that gets published (1024³, ~100k triangles, 2K textures).
 */
export type GenerationQuality = 'preview' | 'final';

export interface GenerationRequest {
  type: GenerationType;
  prompt?: string;
  imageUrl?: string;
  userId: string;
  quality?: GenerationQuality;
  /** Pass the preview's seed to the final run so it refines the shape the user approved. */
  seed?: number;
  /** Stable id (letters, digits, - and _) used to name the stored outputs. */
  requestId?: string;
}

export interface GenerationResult {
  modelUrl: string;
  format: 'glb' | 'gltf' | 'obj' | 'fbx';
  thumbnailUrl?: string;
  durationMs: number;
  seed?: number;
  triangles?: number;
  bytes?: number;
  /** Attribution to show with the asset, e.g. "Built with DINOv3" (a license requirement). */
  credits?: string[];
}

export interface GenerationProgress {
  status: 'queued' | 'processing' | 'completed' | 'failed';
  progress: number; // 0-100
  message?: string;
}

export interface AIProvider {
  readonly name: string;
  readonly supportedTypes: readonly GenerationType[];

  generate(request: GenerationRequest): Promise<string>; // returns job ID
  pollStatus(jobId: string): Promise<GenerationProgress>;
  getResult(jobId: string): Promise<GenerationResult>;
}

export interface ReferenceImage {
  url: string;
  seed: number;
}

/**
 * Text prompts become a few reference images first. Picking one costs seconds of GPU;
 * throwing away a generated model costs a minute.
 */
export interface ReferenceImageProvider {
  referenceImages(
    prompt: string,
    options?: { count?: number; seed?: number; requestId?: string },
  ): Promise<ReferenceImage[]>;
}

/** A file a worker produced: at a public URL (workers with R2), or inline. */
export type WorkerFile = { url: string } | { data: Buffer };

export interface ReferencesOutput {
  images: {
    file: WorkerFile;
    seed: number;
    /** How good a start for 3D the worker rates the picture (0..1), when it rates them */
    score?: number;
    /** What makes it a worse start, e.g. "cut off at the bottom"; left out when nothing does */
    issues?: string[];
  }[];
}

export interface ModelOutput {
  file: WorkerFile;
  seed: number;
  triangles: number;
  bytes: number;
  /** GPU seconds the job took (the worker's timings added up). */
  seconds: number;
  /** Attribution to show with the model, e.g. "Built with DINOv3" (a license requirement). */
  credits: string[];
  /**
   * How many views the model was built from besides the picture; left out by 3D workers from
   * before views, which ignore them.
   */
  viewsUsed?: number;
  /**
   * The pipeline that made the shape, when the worker says: "512" for previews, "1024_cascade"
   * for finals, and "512" for a final that ran out of GPU memory (see workers/README.md).
   */
  pipeline?: string;
  /**
   * The model that made it, when the host says: "trellis2", or "pixal3d" for a final with the
   * recipe on. Orainge's job API on Modal says; RunPod and older workers leave it out.
   */
  model?: string;
}

/**
 * More textures for a final's shape (texture options): the 3D worker's "textures" job. The final's
 * own texture isn't among them.
 */
export interface TexturesOutput {
  textures: {
    file: WorkerFile;
    /** The seed the texture's noise came from: the final's seed + 1000 · k, for k = 1, 2, … */
    textureSeed: number;
    triangles: number;
    bytes: number;
  }[];
  /** The textures that failed while the others were made */
  errors: { textureSeed: number | null; message: string }[];
  /**
   * The pipeline that made the shape again, when the worker says. When it isn't the final's,
   * the textures fit another shape.
   */
  pipeline?: string;
  /** GPU seconds the job took */
  seconds: number;
}

/** Where a view of the object was drawn from, in degrees (see workers/README.md, multiview). */
export interface ViewAngle {
  /** Around the object; 0 is the picture's front */
  azimuth: number;
  /** Above the object's middle; 0 is level with it */
  elevation: number;
}

/** The picture drawn from other sides by the multiview worker (MV-Adapter). */
export interface ViewsOutput {
  views: (ViewAngle & { file: WorkerFile })[];
  /** GPU seconds the job took */
  seconds: number;
}

/** A view sent to the 3D worker with the picture, so it builds the back and sides from it. */
export type ModelView = ViewAngle & { image: Buffer };

export type WorkerJobState<T> =
  | { status: 'running' }
  | { status: 'done'; output: T }
  | { status: 'failed'; message: string };

/**
 * The GPU workers the Studio uses: FLUX for the pictures, MV-Adapter for the picture's other
 * sides, TRELLIS.2 for the models.
 */
export type WorkerKind = 'references' | 'multiview' | 'model';

/**
 * What the Studio's AI panel needs from the GPU workers. Every step is a job that is started
 * and then polled, so no request waits on a GPU (a cold start takes a minute or more).
 */
export interface StudioWorkers {
  readonly name: string;
  /** Text prompts need the reference-image worker; photos only need TRELLIS.2. */
  readonly prompts: boolean;
  /** Whether the multiview worker is there to draw the picture from other sides. */
  readonly multiview: boolean;
  /**
   * Starts a worker's GPU before a job needs it, when the host can (about 45 s for FLUX and
   * 100 s for TRELLIS.2 from cold). Resolves once it's asked, not once the GPU is ready.
   */
  warm?(kind: WorkerKind): Promise<void>;
  /** Stops a job nobody waits for any more, when the host can (it bills while it runs). */
  cancel?(kind: WorkerKind, jobId: string): Promise<void>;
  startReferences(input: { prompt: string; count: number; requestId: string }): Promise<string>;
  references(jobId: string): Promise<WorkerJobState<ReferencesOutput>>;
  /** Draws the picture from 6 sides (the multiview worker); only when `multiview` is true. */
  startViews(input: { image: Buffer; requestId: string }): Promise<string>;
  views(jobId: string): Promise<WorkerJobState<ViewsOutput>>;
  startModel(input: {
    image: Buffer;
    /** The picture from other sides, for the back and sides; the picture stays the main image */
    views?: ModelView[];
    mode: GenerationQuality;
    seed?: number;
    requestId: string;
  }): Promise<string>;
  model(jobId: string): Promise<WorkerJobState<ModelOutput>>;
  /**
   * Makes `count` more textures for a final's shape (texture options), from what the final was
   * made from: its picture, seed and views. Only workers that can make them have these.
   */
  startTextures?(input: {
    image: Buffer;
    views?: ModelView[];
    seed: number;
    count: number;
    requestId: string;
  }): Promise<string>;
  textures?(jobId: string): Promise<WorkerJobState<TexturesOutput>>;
}
