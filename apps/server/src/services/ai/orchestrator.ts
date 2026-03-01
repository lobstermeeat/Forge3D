import type { AIProvider, GenerationRequest, GenerationProgress, GenerationResult } from './types';

const POLL_INTERVAL_MS = 3000;
const MAX_POLL_ATTEMPTS = 200; // ~10 minutes at 3s intervals
const RETRY_COUNT = 3;
const RETRY_BASE_DELAY_MS = 1000;

export class AIOrchestrator {
  private providers: Map<string, AIProvider> = new Map();

  registerProvider(provider: AIProvider): void {
    this.providers.set(provider.name, provider);
  }

  getProvider(name: string): AIProvider | undefined {
    return this.providers.get(name);
  }

  getAvailableProviders(type: 'text-to-3d' | 'image-to-3d'): AIProvider[] {
    return Array.from(this.providers.values()).filter((p) =>
      p.supportedTypes.includes(type),
    );
  }

  selectProvider(type: 'text-to-3d' | 'image-to-3d', preferred?: string): AIProvider {
    if (preferred) {
      const provider = this.providers.get(preferred);
      if (provider && provider.supportedTypes.includes(type)) {
        return provider;
      }
    }

    const available = this.getAvailableProviders(type);
    if (available.length === 0) {
      throw new Error(`No providers available for ${type}`);
    }

    // Default: pick the first available provider
    return available[0]!;
  }

  async generate(
    request: GenerationRequest,
    preferredProvider?: string,
  ): Promise<{ provider: string; jobId: string }> {
    const provider = this.selectProvider(request.type, preferredProvider);

    const jobId = await this.withRetry(() => provider.generate(request));

    return { provider: provider.name, jobId };
  }

  async pollUntilComplete(
    providerName: string,
    jobId: string,
    onProgress?: (progress: GenerationProgress) => void,
  ): Promise<GenerationResult> {
    const provider = this.providers.get(providerName);
    if (!provider) throw new Error(`Provider ${providerName} not found`);

    for (let attempt = 0; attempt < MAX_POLL_ATTEMPTS; attempt++) {
      const progress = await provider.pollStatus(jobId);
      onProgress?.(progress);

      if (progress.status === 'completed') {
        return provider.getResult(jobId);
      }

      if (progress.status === 'failed') {
        throw new Error(progress.message ?? 'Generation failed');
      }

      await this.sleep(POLL_INTERVAL_MS);
    }

    throw new Error('Generation timed out');
  }

  private async withRetry<T>(fn: () => Promise<T>): Promise<T> {
    let lastError: Error | undefined;

    for (let attempt = 0; attempt < RETRY_COUNT; attempt++) {
      try {
        return await fn();
      } catch (error) {
        lastError = error instanceof Error ? error : new Error(String(error));
        if (attempt < RETRY_COUNT - 1) {
          const delay = RETRY_BASE_DELAY_MS * Math.pow(2, attempt);
          await this.sleep(delay);
        }
      }
    }

    throw lastError;
  }

  private sleep(ms: number): Promise<void> {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }
}
