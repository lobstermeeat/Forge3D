import type { AIProvider, GenerationRequest, GenerationProgress, GenerationResult } from '../types';

const MESHY_API_URL = 'https://api.meshy.ai/v2';

export class MeshyProvider implements AIProvider {
  readonly name = 'meshy';
  readonly supportedTypes = ['text-to-3d', 'image-to-3d'] as const;
  private apiKey: string;

  constructor(apiKey: string) {
    this.apiKey = apiKey;
  }

  async generate(request: GenerationRequest): Promise<string> {
    const endpoint =
      request.type === 'image-to-3d'
        ? `${MESHY_API_URL}/image-to-3d`
        : `${MESHY_API_URL}/text-to-3d`;

    const body: Record<string, unknown> = {};
    if (request.prompt) body['prompt'] = request.prompt;
    if (request.imageUrl) body['image_url'] = request.imageUrl;

    const res = await fetch(endpoint, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${this.apiKey}`,
      },
      body: JSON.stringify(body),
    });

    if (!res.ok) {
      throw new Error(`Meshy API error: ${res.status} ${await res.text()}`);
    }

    const data = (await res.json()) as { result: string };
    return data.result;
  }

  async pollStatus(jobId: string): Promise<GenerationProgress> {
    const res = await fetch(`${MESHY_API_URL}/image-to-3d/${jobId}`, {
      headers: { Authorization: `Bearer ${this.apiKey}` },
    });

    if (!res.ok) {
      throw new Error(`Meshy poll error: ${res.status}`);
    }

    const data = (await res.json()) as {
      status: string;
      progress: number;
    };

    const statusMap: Record<string, GenerationProgress['status']> = {
      PENDING: 'queued',
      IN_PROGRESS: 'processing',
      SUCCEEDED: 'completed',
      FAILED: 'failed',
    };

    return {
      status: statusMap[data.status] ?? 'processing',
      progress: data.progress ?? 0,
    };
  }

  async getResult(jobId: string): Promise<GenerationResult> {
    const res = await fetch(`${MESHY_API_URL}/image-to-3d/${jobId}`, {
      headers: { Authorization: `Bearer ${this.apiKey}` },
    });

    if (!res.ok) {
      throw new Error(`Meshy result error: ${res.status}`);
    }

    const data = (await res.json()) as {
      model_urls: { glb: string };
      thumbnail_url?: string;
    };

    return {
      modelUrl: data.model_urls.glb,
      format: 'glb',
      thumbnailUrl: data.thumbnail_url,
      durationMs: 0,
    };
  }
}
