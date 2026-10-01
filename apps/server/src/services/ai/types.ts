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
}

export type WorkerJobState<T> =
  | { status: 'running' }
  | { status: 'done'; output: T }
  | { status: 'failed'; message: string };

/** The GPU workers the Studio uses: FLUX for the pictures, TRELLIS.2 for the models. */
export type WorkerKind = 'references' | 'model';

/**
 * What the Studio's AI panel needs from the GPU workers. Every step is a job that is started
 * and then polled, so no request waits on a GPU (a cold start takes a minute or more).
 */
export interface StudioWorkers {
  readonly name: string;
  /** Text prompts need the reference-image worker; photos only need TRELLIS.2. */
  readonly prompts: boolean;
  /**
   * Starts a worker's GPU before a job needs it, when the host can (about 45 s for FLUX and
   * 100 s for TRELLIS.2 from cold). Resolves once it's asked, not once the GPU is ready.
   */
  warm?(kind: WorkerKind): Promise<void>;
  startReferences(input: { prompt: string; count: number; requestId: string }): Promise<string>;
  references(jobId: string): Promise<WorkerJobState<ReferencesOutput>>;
  startModel(input: {
    image: Buffer;
    mode: GenerationQuality;
    seed?: number;
    requestId: string;
  }): Promise<string>;
  model(jobId: string): Promise<WorkerJobState<ModelOutput>>;
}
