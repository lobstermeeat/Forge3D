/**
 * Client for a RunPod-style job API. Two hosts speak it: RunPod serverless endpoints
 * (https://docs.runpod.io/serverless/endpoints) and Orainge's own job API on Modal
 * (workers/job_api.py).
 */

export type JobStatus =
  | 'IN_QUEUE'
  | 'IN_PROGRESS'
  | 'COMPLETED'
  | 'FAILED'
  | 'CANCELLED'
  | 'TIMED_OUT';

export interface RemoteJob<T = unknown> {
  id: string;
  status: JobStatus;
  output?: T;
  error?: string;
}

export type FetchLike = (url: string, init?: RequestInit) => Promise<Response>;

export class JobEndpoint {
  /**
   * @param url The endpoint's base URL, e.g. `https://api.runpod.ai/v2/<endpoint id>` or
   *   `https://<workspace>--orainge-ai-api.modal.run/trellis2`.
   * @param token Sent as a bearer token: the RunPod API key, or ORAINGE_WORKER_TOKEN on Modal.
   */
  constructor(
    private readonly url: string,
    private readonly token: string,
    private readonly fetchImpl: FetchLike = (input, init) => fetch(input, init),
  ) {}

  static runPod(endpointId: string, apiKey: string, fetchImpl?: FetchLike): JobEndpoint {
    return new JobEndpoint(`https://api.runpod.ai/v2/${endpointId}`, apiKey, fetchImpl);
  }

  /** Queue a job; returns its id. */
  async run(input: Record<string, unknown>): Promise<string> {
    const job = await this.request<RemoteJob>('run', {
      method: 'POST',
      body: JSON.stringify({ input }),
    });
    return job.id;
  }

  /** Run and wait. The host holds the request open for a while (RunPod ~90 s, Modal 60 s), then we poll. */
  async runSync<T>(input: Record<string, unknown>, timeoutMs = 180_000): Promise<RemoteJob<T>> {
    const started = Date.now();
    let job = await this.request<RemoteJob<T>>('runsync', {
      method: 'POST',
      body: JSON.stringify({ input }),
    });
    while (job.status === 'IN_QUEUE' || job.status === 'IN_PROGRESS') {
      if (Date.now() - started > timeoutMs) {
        // Don't leave it running (and billing) when a retry will queue it again
        await this.cancel(job.id).catch(() => {});
        throw new Error(`Job ${job.id} timed out`);
      }
      await new Promise((resolve) => setTimeout(resolve, 2000));
      job = await this.status<T>(job.id);
    }
    return job;
  }

  status<T>(jobId: string): Promise<RemoteJob<T>> {
    return this.request<RemoteJob<T>>(`status/${encodeURIComponent(jobId)}`, { method: 'GET' });
  }

  async cancel(jobId: string): Promise<void> {
    await this.request(`cancel/${encodeURIComponent(jobId)}`, { method: 'POST' });
  }

  /**
   * Starts one of the worker's containers without waiting for it, so a job that follows soon
   * skips the cold start. Only Orainge's job API on Modal has this route; RunPod doesn't.
   */
  async warm(): Promise<void> {
    await this.request('warm', { method: 'POST' });
  }

  private async request<T>(path: string, init: RequestInit): Promise<T> {
    const res = await this.fetchImpl(`${this.url}/${path}`, {
      ...init,
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${this.token}`,
      },
    });
    if (!res.ok) {
      throw new Error(`AI worker ${path.split('/')[0]} failed: ${res.status} ${await res.text()}`);
    }
    return (await res.json()) as T;
  }
}
