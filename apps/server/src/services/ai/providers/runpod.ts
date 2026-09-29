/** Minimal client for a RunPod serverless endpoint (https://docs.runpod.io/serverless/endpoints). */

export type RunPodStatus =
  | 'IN_QUEUE'
  | 'IN_PROGRESS'
  | 'COMPLETED'
  | 'FAILED'
  | 'CANCELLED'
  | 'TIMED_OUT';

export interface RunPodJob<T = unknown> {
  id: string;
  status: RunPodStatus;
  output?: T;
  error?: string;
}

export type FetchLike = (url: string, init?: RequestInit) => Promise<Response>;

export class RunPodEndpoint {
  constructor(
    private readonly endpointId: string,
    private readonly apiKey: string,
    private readonly fetchImpl: FetchLike = fetch,
    private readonly baseUrl = 'https://api.runpod.ai/v2',
  ) {}

  /** Queue a job; returns its id. */
  async run(input: Record<string, unknown>): Promise<string> {
    const job = await this.request<RunPodJob>('run', {
      method: 'POST',
      body: JSON.stringify({ input }),
    });
    return job.id;
  }

  /** Run and wait (RunPod holds the request open for up to ~90 s, then we poll). */
  async runSync<T>(input: Record<string, unknown>, timeoutMs = 180_000): Promise<RunPodJob<T>> {
    const started = Date.now();
    let job = await this.request<RunPodJob<T>>('runsync', {
      method: 'POST',
      body: JSON.stringify({ input }),
    });
    while (job.status === 'IN_QUEUE' || job.status === 'IN_PROGRESS') {
      if (Date.now() - started > timeoutMs) {
        // Don't leave it running (and billing) when a retry will queue it again
        await this.cancel(job.id).catch(() => {});
        throw new Error(`RunPod job ${job.id} timed out`);
      }
      await new Promise((resolve) => setTimeout(resolve, 2000));
      job = await this.status<T>(job.id);
    }
    return job;
  }

  status<T>(jobId: string): Promise<RunPodJob<T>> {
    return this.request<RunPodJob<T>>(`status/${encodeURIComponent(jobId)}`, { method: 'GET' });
  }

  async cancel(jobId: string): Promise<void> {
    await this.request(`cancel/${encodeURIComponent(jobId)}`, { method: 'POST' });
  }

  private async request<T>(path: string, init: RequestInit): Promise<T> {
    const res = await this.fetchImpl(`${this.baseUrl}/${this.endpointId}/${path}`, {
      ...init,
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${this.apiKey}`,
      },
    });
    if (!res.ok) {
      throw new Error(`RunPod ${path.split('/')[0]} failed: ${res.status} ${await res.text()}`);
    }
    return (await res.json()) as T;
  }
}
