import { AIOrchestrator } from './orchestrator';
import { createSelfHostedProvider } from './providers/selfHosted';

export { AIOrchestrator } from './orchestrator';
export { SelfHostedProvider, createSelfHostedProvider } from './providers/selfHosted';
export { JobEndpoint } from './providers/jobEndpoint';
export type {
  AIProvider,
  GenerationRequest,
  GenerationResult,
  GenerationProgress,
  GenerationQuality,
  ReferenceImage,
  ReferenceImageProvider,
} from './types';

/** The orchestrator with every configured provider. FORGE 3D only runs its own models. */
export function createAIOrchestrator(
  env: Record<string, string | undefined> = process.env,
): AIOrchestrator {
  const orchestrator = new AIOrchestrator();
  const selfHosted = createSelfHostedProvider(env);
  if (selfHosted) orchestrator.registerProvider(selfHosted);
  return orchestrator;
}
