import { AIOrchestrator } from './orchestrator';
import { createSelfHostedProvider } from './providers/selfHosted';
import { MockWorkers } from './providers/mock';
import { AIStudio, drizzleGenerationStore } from './studio';
import type { StudioWorkers } from './types';
import { db } from '../../db';
import { getStorage } from '../storage';

export { AIOrchestrator } from './orchestrator';
export { SelfHostedProvider, createSelfHostedProvider } from './providers/selfHosted';
export { JobEndpoint } from './providers/jobEndpoint';
export { MockWorkers } from './providers/mock';
export { AIStudio, StudioError } from './studio';
export type {
  GenerationView,
  GenerationStatus,
  RunningStatus,
  TextureChooser,
  TextureOption,
  TexturesStatus,
} from './studio';
export type {
  AIProvider,
  GenerationRequest,
  GenerationResult,
  GenerationProgress,
  GenerationQuality,
  ReferenceImage,
  ReferenceImageProvider,
} from './types';

/** The orchestrator with every configured provider. Orainge only runs its own models. */
export function createAIOrchestrator(
  env: Record<string, string | undefined> = process.env,
): AIOrchestrator {
  const orchestrator = new AIOrchestrator();
  const selfHosted = createSelfHostedProvider(env);
  if (selfHosted) orchestrator.registerProvider(selfHosted);
  return orchestrator;
}

/**
 * The workers the Studio's AI panel uses: the self-hosted ones from the environment, or with
 * AI_WORKERS_MOCK=1 (development only) stand-ins that need no GPU.
 */
export function createStudioWorkers(
  env: Record<string, string | undefined> = process.env,
): StudioWorkers | null {
  if (env['AI_WORKERS_MOCK'] === '1') {
    if (env['NODE_ENV'] === 'production')
      throw new Error('AI_WORKERS_MOCK is for development only');
    return new MockWorkers(Number(env['AI_WORKERS_MOCK_DELAY_MS'] ?? 1500));
  }
  return createSelfHostedProvider(env);
}

/**
 * AI_MULTIVIEW=1: the multiview worker draws the chosen picture from 6 sides, and the preview and
 * final are built from those views too, so the model's back isn't invented. Off by default.
 */
export function multiviewEnabled(env: Record<string, string | undefined> = process.env): boolean {
  return env['AI_MULTIVIEW']?.trim() === '1';
}

/**
 * Texture options: after each final, TRELLIS.2 makes 3 more textures for its shape and the
 * creator picks one in the panel. On by default; AI_TEXTURE_OPTIONS=0 (or false) turns it off.
 */
export function textureOptionsEnabled(
  env: Record<string, string | undefined> = process.env,
): boolean {
  const value = env['AI_TEXTURE_OPTIONS']?.trim().toLowerCase();
  return value !== '0' && value !== 'false';
}

/**
 * AI_TEXTURE_JUDGE=1 (or true): the textures job also asks the judge (Qwen3-VL-8B on the workers)
 * which of the four textures a creator would rather use, and the panel makes that one the default.
 * Off by default: the workers need the judge's weights (`download_models --which judge8b`).
 */
export function textureJudgeEnabled(
  env: Record<string, string | undefined> = process.env,
): boolean {
  const value = env['AI_TEXTURE_JUDGE']?.trim().toLowerCase();
  return value === '1' || value === 'true';
}

let studio: AIStudio | null = null;

export function getStudio(): AIStudio {
  if (!studio) {
    const workers = createStudioWorkers();
    const multiview = multiviewEnabled();
    if (multiview && workers && !workers.multiview) {
      console.warn(
        '[AI] AI_MULTIVIEW=1, but there is no multiview worker (on RunPod, set RUNPOD_MULTIVIEW_ENDPOINT_ID): models are made from the picture alone',
      );
    }
    const textureOptions = textureOptionsEnabled();
    const textureJudge = textureJudgeEnabled();
    if (textureJudge && !textureOptions) {
      console.warn(
        '[AI] AI_TEXTURE_JUDGE=1, but texture options are off (AI_TEXTURE_OPTIONS): there are no textures to judge',
      );
    }
    studio = new AIStudio({
      workers,
      store: drizzleGenerationStore(db),
      storage: getStorage(),
      multiview,
      textureOptions,
      textureJudge,
    });
  }
  return studio;
}
