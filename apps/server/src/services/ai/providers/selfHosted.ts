import type {
  AIProvider,
  GenerationQuality,
  GenerationType,
  GenerationProgress,
  GenerationRequest,
  GenerationResult,
  ModelOutput,
  ModelView,
  ReferenceImage,
  ReferenceImageProvider,
  ReferencesOutput,
  StudioWorkers,
  TexturesOutput,
  ViewsOutput,
  WorkerFile,
  WorkerJobState,
  WorkerKind,
} from '../types';
import { JobEndpoint, type FetchLike, type RemoteJob } from './jobEndpoint';

/** A stored file as the workers report it (see storage.py in each worker). */
interface StoredAsset {
  key: string;
  url: string | null;
  base64?: string;
}

interface Trellis2Output {
  request_id: string;
  mode: 'preview' | 'final';
  seed: number;
  glb: StoredAsset;
  bytes: number;
  raw_bytes: number;
  triangles: number;
  timings: Record<string, number>;
  credits: string[];
  /** How many views the model was built from (3D workers that take views) */
  views_used?: unknown;
  /** The pipeline that made the shape, e.g. "1024_cascade" */
  pipeline?: unknown;
  error?: string;
}

/** A "textures" job's output: texture options for a final (workers/README.md, Job contracts) */
interface TexturesJobOutput {
  request_id?: string;
  mode?: 'textures';
  seed?: unknown;
  textures?: {
    texture_seed?: unknown;
    glb?: StoredAsset;
    bytes?: unknown;
    triangles?: unknown;
  }[];
  /** The textures that failed while the others were made; left out when none did */
  texture_errors?: { texture_seed?: unknown; error?: unknown }[];
  /** The shape's pipeline: "1024_cascade", or "512" when the cascade ran out of GPU memory */
  pipeline?: unknown;
  timings?: Record<string, unknown>;
  credits?: string[];
  error?: string;
}

/** The multiview worker's output (workers/README.md, Job contracts) */
interface MultiviewOutput {
  request_id?: string;
  views?: (StoredAsset & { azimuth?: unknown; elevation?: unknown })[];
  /** The views' camera, e.g. { "type": "orthographic", … }; the 3D worker knows it */
  camera?: unknown;
  seconds?: unknown;
  error?: string;
}

interface ReferenceOutput {
  request_id: string;
  prompt: string;
  /** `score` (0..1) and `issues`: how good a start for 3D the worker rates the picture, if it does */
  images: (StoredAsset & { seed: number; score?: unknown; issues?: unknown })[];
  seconds: number;
  error?: string;
}

/**
 * Orainge's own models on serverless GPUs (Modal or RunPod, see workers/): TRELLIS.2 for
 * image-to-3D, FLUX.1 [schnell] for the reference images text prompts start from, and
 * MV-Adapter (the multiview worker) for the picture's other sides. No third-party AI APIs.
 *
 * The intended flow: `referenceImages(prompt)` -> user picks one ->
 * `generate({ type: 'image-to-3d', imageUrl, quality: 'preview' })` -> user keeps it ->
 * `generate({ ..., quality: 'final', seed: previewSeed })`. The Studio's panel runs the same
 * flow as jobs to poll (StudioWorkers), with the views step in it and texture options after the
 * final.
 */
export class SelfHostedProvider implements AIProvider, ReferenceImageProvider, StudioWorkers {
  readonly name = 'forge3d-trellis2';
  /** Text prompts need the reference-image endpoint. */
  readonly supportedTypes: readonly GenerationType[];
  readonly prompts: boolean;
  readonly multiview: boolean;

  constructor(
    private readonly trellis2: JobEndpoint,
    private readonly reference: JobEndpoint | null,
    /**
     * canWarm: the host can start a GPU ahead of a job (Orainge's job API on Modal; not RunPod).
     * multiview: the endpoint of the worker that draws a picture from 6 sides.
     */
    private readonly options: { canWarm?: boolean; multiview?: JobEndpoint | null } = {},
  ) {
    this.supportedTypes = reference ? ['image-to-3d', 'text-to-3d'] : ['image-to-3d'];
    this.prompts = reference !== null;
    this.multiview = !!options.multiview;
  }

  // The Studio's panel (StudioWorkers): start a job, then poll it

  startReferences(input: { prompt: string; count: number; requestId: string }): Promise<string> {
    if (!this.reference) throw new Error('Text prompts need the reference-image worker');
    return this.reference.run({
      prompt: input.prompt,
      count: input.count,
      request_id: input.requestId,
    });
  }

  async references(jobId: string): Promise<WorkerJobState<ReferencesOutput>> {
    if (!this.reference) throw new Error('Text prompts need the reference-image worker');
    return jobState(await this.reference.status<ReferenceOutput>(jobId), (output) => ({
      images: output.images.map((image) => ({
        file: workerFile(image),
        seed: image.seed,
        ...rating(image),
      })),
    }));
  }

  async startViews(input: { image: Buffer; requestId: string }): Promise<string> {
    return this.multiviewEndpoint().run({
      image_base64: input.image.toString('base64'),
      request_id: input.requestId,
    });
  }

  async views(jobId: string): Promise<WorkerJobState<ViewsOutput>> {
    return jobState(await this.multiviewEndpoint().status<MultiviewOutput>(jobId), (output) => ({
      views: (Array.isArray(output.views) ? output.views : []).flatMap((view) => {
        const azimuth = finite(view.azimuth);
        const elevation = finite(view.elevation);
        // A view that doesn't say where it was seen from would mislead the 3D worker
        if (azimuth === undefined || elevation === undefined) return [];
        return [{ file: workerFile(view), azimuth, elevation }];
      }),
      seconds: finite(output.seconds) ?? 0,
    }));
  }

  /**
   * The views go inline like the picture: they are the server's copies, which the workers can't
   * fetch, and outlive the multiview worker's outputs (a final may come hours later).
   */
  startModel(input: {
    image: Buffer;
    views?: ModelView[];
    mode: GenerationQuality;
    seed?: number;
    requestId: string;
  }): Promise<string> {
    return this.trellis2.run({
      image_base64: input.image.toString('base64'),
      ...(input.views?.length ? { views: inlineViews(input.views) } : {}),
      mode: input.mode,
      seed: input.seed,
      request_id: input.requestId,
    });
  }

  async model(jobId: string): Promise<WorkerJobState<ModelOutput>> {
    return jobState(await this.trellis2.status<Trellis2Output>(jobId), (output) => {
      const viewsUsed = finite(output.views_used);
      return {
        file: workerFile(output.glb),
        seed: output.seed,
        triangles: output.triangles,
        bytes: output.bytes,
        seconds: Object.values(output.timings).reduce((sum, t) => sum + t, 0),
        credits: output.credits,
        ...(viewsUsed === undefined ? {} : { viewsUsed }),
        ...(typeof output.pipeline === 'string' ? { pipeline: output.pipeline } : {}),
      };
    });
  }

  /**
   * Texture options: TRELLIS.2 makes the final's shape again from the same picture, seed and
   * views, then `count` more textures for it. Same endpoint as the preview and the final.
   */
  startTextures(input: {
    image: Buffer;
    views?: ModelView[];
    seed: number;
    count: number;
    requestId: string;
  }): Promise<string> {
    return this.trellis2.run({
      image_base64: input.image.toString('base64'),
      ...(input.views?.length ? { views: inlineViews(input.views) } : {}),
      mode: 'textures',
      seed: input.seed,
      count: input.count,
      request_id: input.requestId,
    });
  }

  async textures(jobId: string): Promise<WorkerJobState<TexturesOutput>> {
    return jobState(await this.trellis2.status<TexturesJobOutput>(jobId), (output) => ({
      textures: (Array.isArray(output.textures) ? output.textures : []).flatMap((texture) => {
        const textureSeed = finite(texture.texture_seed);
        // Without its file or its seed, a texture can't be offered
        if (textureSeed === undefined || !texture.glb) return [];
        return [
          {
            file: workerFile(texture.glb),
            textureSeed,
            triangles: finite(texture.triangles) ?? 0,
            bytes: finite(texture.bytes) ?? 0,
          },
        ];
      }),
      errors: (Array.isArray(output.texture_errors) ? output.texture_errors : []).map((failed) => ({
        textureSeed: finite(failed.texture_seed) ?? null,
        message: typeof failed.error === 'string' ? failed.error : 'failed',
      })),
      ...(typeof output.pipeline === 'string' ? { pipeline: output.pipeline } : {}),
      seconds: Object.values(output.timings ?? {}).reduce<number>(
        (sum, t) => sum + (finite(t) ?? 0),
        0,
      ),
    }));
  }

  /** On Modal, starts the worker's GPU ahead of its job. RunPod has no route for it: no-op. */
  async warm(kind: WorkerKind): Promise<void> {
    if (!this.options.canWarm) return;
    await this.endpoint(kind)?.warm();
  }

  /** Stops a job, such as views that took too long. */
  async cancel(kind: WorkerKind, jobId: string): Promise<void> {
    await this.endpoint(kind)?.cancel(jobId);
  }

  private endpoint(kind: WorkerKind): JobEndpoint | null {
    if (kind === 'model') return this.trellis2;
    if (kind === 'references') return this.reference;
    return this.options.multiview ?? null;
  }

  private multiviewEndpoint(): JobEndpoint {
    if (!this.options.multiview) throw new Error('The views need the multiview worker');
    return this.options.multiview;
  }

  async referenceImages(
    prompt: string,
    options: { count?: number; seed?: number; requestId?: string } = {},
  ): Promise<ReferenceImage[]> {
    if (!this.reference)
      throw new Error('Reference images need the reference worker (see workers/README.md)');
    const job = await this.reference.runSync<ReferenceOutput>({
      prompt,
      count: options.count ?? 4,
      seed: options.seed,
      request_id: options.requestId,
    });
    return outputOf(job).images.map((image) => ({
      url: assetUrl(image, 'image/png'),
      seed: image.seed,
    }));
  }

  async generate(request: GenerationRequest): Promise<string> {
    let imageUrl = request.imageUrl;
    if (!imageUrl) {
      if (request.type !== 'text-to-3d' || !request.prompt) {
        throw new Error('image-to-3d needs an imageUrl');
      }
      // No user pick: take a single reference image
      const [first] = await this.referenceImages(request.prompt, {
        count: 1,
        requestId: request.requestId,
      });
      if (!first) throw new Error('No reference image was generated');
      imageUrl = first.url;
    }
    // Workers without R2 return images inline; send those as data, not as a URL to fetch
    const image = imageUrl.startsWith('data:')
      ? { image_base64: imageUrl.slice(imageUrl.indexOf(',') + 1) }
      : { image_url: imageUrl };
    return this.trellis2.run({
      ...image,
      mode: request.quality ?? 'final',
      seed: request.seed,
      request_id: request.requestId,
    });
  }

  async pollStatus(jobId: string): Promise<GenerationProgress> {
    const job = await this.trellis2.status<Trellis2Output>(jobId);
    switch (job.status) {
      case 'IN_QUEUE':
        return { status: 'queued', progress: 0 };
      case 'IN_PROGRESS':
        return { status: 'processing', progress: 50 };
      case 'COMPLETED':
        return job.output?.error
          ? { status: 'failed', progress: 100, message: job.output.error }
          : { status: 'completed', progress: 100 };
      default:
        return {
          status: 'failed',
          progress: 100,
          message: job.error ?? `Job ${job.status.toLowerCase().replace('_', ' ')}`,
        };
    }
  }

  async getResult(jobId: string): Promise<GenerationResult> {
    const output = outputOf(await this.trellis2.status<Trellis2Output>(jobId));
    const seconds = Object.values(output.timings).reduce((sum, t) => sum + t, 0);
    return {
      modelUrl: assetUrl(output.glb, 'model/gltf-binary'),
      format: 'glb',
      durationMs: Math.round(seconds * 1000),
      seed: output.seed,
      triangles: output.triangles,
      bytes: output.bytes,
      credits: output.credits,
    };
  }
}

function jobState<T extends { error?: string }, R>(
  job: RemoteJob<T>,
  map: (output: T) => R,
): WorkerJobState<R> {
  switch (job.status) {
    case 'IN_QUEUE':
    case 'IN_PROGRESS':
      return { status: 'running' };
    case 'COMPLETED':
      if (!job.output)
        return { status: 'failed', message: `Job ${job.id} finished without output` };
      if (job.output.error) return { status: 'failed', message: job.output.error };
      return { status: 'done', output: map(job.output) };
    default:
      return {
        status: 'failed',
        message: job.error ?? `The job was ${job.status.toLowerCase().replace('_', ' ')}`,
      };
  }
}

/**
 * How the worker rated a picture as a start for 3D, keeping only what it could read: workers from
 * before ratings send neither field.
 */
function rating(image: { score?: unknown; issues?: unknown }): {
  score?: number;
  issues?: string[];
} {
  const score =
    typeof image.score === 'number' && Number.isFinite(image.score) ? image.score : undefined;
  const issues = Array.isArray(image.issues)
    ? image.issues
        .filter((issue): issue is string => typeof issue === 'string')
        .map((issue) => issue.trim())
        .filter(Boolean)
    : [];
  return {
    ...(score === undefined ? {} : { score }),
    ...(issues.length ? { issues } : {}),
  };
}

function finite(value: unknown): number | undefined {
  return typeof value === 'number' && Number.isFinite(value) ? value : undefined;
}

/** Views as the 3D worker takes them: inline, like the picture */
function inlineViews(views: ModelView[]) {
  return views.map(({ image, azimuth, elevation }) => ({
    image_base64: image.toString('base64'),
    azimuth,
    elevation,
  }));
}

function workerFile(asset: StoredAsset): WorkerFile {
  if (asset.url) return { url: asset.url };
  if (asset.base64) return { data: Buffer.from(asset.base64, 'base64') };
  throw new Error(`Worker stored ${asset.key} without a public URL; set R2_PUBLIC_BASE_URL on it`);
}

function outputOf<T extends { error?: string }>(job: RemoteJob<T>): T {
  if (job.status !== 'COMPLETED' || !job.output) {
    throw new Error(job.error ?? `Job ${job.id} is ${job.status}`);
  }
  if (job.output.error) throw new Error(job.output.error);
  return job.output;
}

function assetUrl(asset: StoredAsset, mimeType: string): string {
  if (asset.url) return asset.url;
  // Local development: workers without R2 return small files inline
  if (asset.base64) return `data:${mimeType};base64,${asset.base64}`;
  throw new Error(`Worker stored ${asset.key} without a public URL; set R2_PUBLIC_BASE_URL on it`);
}

/**
 * Built from the environment. Either the job API on Modal (workers/modal_app.py):
 * AI_WORKERS_URL and AI_WORKERS_TOKEN; or RunPod: RUNPOD_API_KEY, RUNPOD_TRELLIS2_ENDPOINT_ID and
 * (optional) RUNPOD_REFERENCE_ENDPOINT_ID and RUNPOD_MULTIVIEW_ENDPOINT_ID. AI_WORKERS_URL wins
 * when both are set. The multiview worker is only used with AI_MULTIVIEW=1 (see ../index.ts).
 */
export function createSelfHostedProvider(
  env: Record<string, string | undefined> = process.env,
  fetchImpl?: FetchLike,
): SelfHostedProvider | null {
  const workersUrl = env['AI_WORKERS_URL']?.trim().replace(/\/+$/, '');
  if (workersUrl) {
    // Trimmed like the job API trims its copy: a pasted token often ends in a line break
    const token = env['AI_WORKERS_TOKEN']?.trim();
    if (!token) throw new Error('AI_WORKERS_URL is set but AI_WORKERS_TOKEN is not');
    return new SelfHostedProvider(
      new JobEndpoint(`${workersUrl}/trellis2`, token, fetchImpl),
      new JobEndpoint(`${workersUrl}/reference`, token, fetchImpl),
      { canWarm: true, multiview: new JobEndpoint(`${workersUrl}/multiview`, token, fetchImpl) },
    );
  }
  const apiKey = env['RUNPOD_API_KEY'];
  const trellis2 = env['RUNPOD_TRELLIS2_ENDPOINT_ID'];
  if (!apiKey || !trellis2) return null;
  const reference = env['RUNPOD_REFERENCE_ENDPOINT_ID'];
  const multiview = env['RUNPOD_MULTIVIEW_ENDPOINT_ID'];
  return new SelfHostedProvider(
    JobEndpoint.runPod(trellis2, apiKey, fetchImpl),
    reference ? JobEndpoint.runPod(reference, apiKey, fetchImpl) : null,
    { multiview: multiview ? JobEndpoint.runPod(multiview, apiKey, fetchImpl) : null },
  );
}
