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
export type { GenerationView, GenerationStatus } from './studio';
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

let studio: AIStudio | null = null;

export function getStudio(): AIStudio {
  studio ??= new AIStudio({
    workers: createStudioWorkers(),
    store: drizzleGenerationStore(db),
    storage: getStorage(),
  });
  return studio;
}
