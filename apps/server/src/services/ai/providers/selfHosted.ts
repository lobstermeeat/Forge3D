import type {
  AIProvider,
  GenerationQuality,
  GenerationType,
  GenerationProgress,
  GenerationRequest,
  GenerationResult,
  ModelOutput,
  ReferenceImage,
  ReferenceImageProvider,
  ReferencesOutput,
  StudioWorkers,
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
 * image-to-3D and FLUX.1 [schnell] for the reference images text prompts start from. No
 * third-party AI APIs.
 *
 * The intended flow: `referenceImages(prompt)` -> user picks one ->
 * `generate({ type: 'image-to-3d', imageUrl, quality: 'preview' })` -> user keeps it ->
 * `generate({ ..., quality: 'final', seed: previewSeed })`.
 */
export class SelfHostedProvider implements AIProvider, ReferenceImageProvider, StudioWorkers {
  readonly name = 'forge3d-trellis2';
  /** Text prompts need the reference-image endpoint. */
  readonly supportedTypes: readonly GenerationType[];
  readonly prompts: boolean;

  constructor(
    private readonly trellis2: JobEndpoint,
    private readonly reference: JobEndpoint | null,
    /** canWarm: the host can start a GPU ahead of a job (Orainge's job API on Modal; not RunPod) */
    private readonly options: { canWarm?: boolean } = {},
  ) {
    this.supportedTypes = reference ? ['image-to-3d', 'text-to-3d'] : ['image-to-3d'];
    this.prompts = reference !== null;
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

  startModel(input: {
    image: Buffer;
    mode: GenerationQuality;
    seed?: number;
    requestId: string;
  }): Promise<string> {
    return this.trellis2.run({
      image_base64: input.image.toString('base64'),
      mode: input.mode,
      seed: input.seed,
      request_id: input.requestId,
    });
  }

  async model(jobId: string): Promise<WorkerJobState<ModelOutput>> {
    return jobState(await this.trellis2.status<Trellis2Output>(jobId), (output) => ({
      file: workerFile(output.glb),
      seed: output.seed,
      triangles: output.triangles,
      bytes: output.bytes,
      seconds: Object.values(output.timings).reduce((sum, t) => sum + t, 0),
      credits: output.credits,
    }));
  }

  /** On Modal, starts the worker's GPU ahead of its job. RunPod has no route for it: no-op. */
  async warm(kind: WorkerKind): Promise<void> {
    if (!this.options.canWarm) return;
    await (kind === 'model' ? this.trellis2 : this.reference)?.warm();
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
 * (optional) RUNPOD_REFERENCE_ENDPOINT_ID. AI_WORKERS_URL wins when both are set.
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
      { canWarm: true },
    );
  }
  const apiKey = env['RUNPOD_API_KEY'];
  const trellis2 = env['RUNPOD_TRELLIS2_ENDPOINT_ID'];
  if (!apiKey || !trellis2) return null;
  const reference = env['RUNPOD_REFERENCE_ENDPOINT_ID'];
  return new SelfHostedProvider(
    JobEndpoint.runPod(trellis2, apiKey, fetchImpl),
    reference ? JobEndpoint.runPod(reference, apiKey, fetchImpl) : null,
  );
}
