import type { AIProvider, GenerationRequest, GenerationProgress, GenerationResult } from '../types';

const TRIPO_API_URL = 'https://api.tripo3d.ai/v2/openapi';

export class TripoProvider implements AIProvider {
  readonly name = 'tripo';
  readonly supportedTypes = ['text-to-3d', 'image-to-3d'] as const;
  private apiKey: string;

  constructor(apiKey: string) {
    this.apiKey = apiKey;
  }

  async generate(request: GenerationRequest): Promise<string> {
    const body: Record<string, unknown> = { type: request.type };
    if (request.prompt) body['prompt'] = request.prompt;
    if (request.imageUrl) body['file'] = { url: request.imageUrl };

    const res = await fetch(`${TRIPO_API_URL}/task`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${this.apiKey}`,
      },
      body: JSON.stringify(body),
    });

    if (!res.ok) {
      throw new Error(`Tripo API error: ${res.status} ${await res.text()}`);
    }

    const data = (await res.json()) as { data: { task_id: string } };
    return data.data.task_id;
  }

  async pollStatus(jobId: string): Promise<GenerationProgress> {
    const res = await fetch(`${TRIPO_API_URL}/task/${jobId}`, {
      headers: { Authorization: `Bearer ${this.apiKey}` },
    });

    if (!res.ok) {
      throw new Error(`Tripo poll error: ${res.status}`);
    }

    const data = (await res.json()) as {
      data: { status: string; progress: number };
    };

    const statusMap: Record<string, GenerationProgress['status']> = {
      queued: 'queued',
      running: 'processing',
      success: 'completed',
      failed: 'failed',
    };

    return {
      status: statusMap[data.data.status] ?? 'processing',
      progress: data.data.progress ?? 0,
    };
  }

  async getResult(jobId: string): Promise<GenerationResult> {
    const res = await fetch(`${TRIPO_API_URL}/task/${jobId}`, {
      headers: { Authorization: `Bearer ${this.apiKey}` },
    });

    if (!res.ok) {
      throw new Error(`Tripo result error: ${res.status}`);
    }

    const data = (await res.json()) as {
      data: {
        output: { model: string; rendered_image?: string };
        running_left_time?: number;
      };
    };

    return {
      modelUrl: data.data.output.model,
      format: 'glb',
      thumbnailUrl: data.data.output.rendered_image,
      durationMs: 0,
    };
  }
}
