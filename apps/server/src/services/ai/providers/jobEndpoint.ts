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

/**
 * How long a request to the job API may take before it is given up: run, status, cancel and warm.
 * Each answers at once; the longest is a run that uploads a picture and its views (a few MB).
 */
export const REQUEST_TIMEOUT_MS = 30_000;
/** runsync is held open by the host before it answers (RunPod ~90 s, Modal 60 s; Modal cuts at 150 s) */
export const RUNSYNC_TIMEOUT_MS = 150_000;

export class JobEndpoint {
  /**
   * @param url The endpoint's base URL, e.g. `https://api.runpod.ai/v2/<endpoint id>` or
   *   `https://<workspace>--orainge-ai-api.modal.run/trellis2`.
   * @param token Sent as a bearer token: the RunPod API key, or ORAINGE_WORKER_TOKEN on Modal.
   * @param timeoutMs How long each request but runsync may take (REQUEST_TIMEOUT_MS).
   */
  constructor(
    private readonly url: string,
    private readonly token: string,
    private readonly fetchImpl: FetchLike = (input, init) => fetch(input, init),
    private readonly timeoutMs = REQUEST_TIMEOUT_MS,
  ) {}

  static runPod(endpointId: string, apiKey: string, fetchImpl?: FetchLike): JobEndpoint {
    return new JobEndpoint(`https://api.runpod.ai/v2/${endpointId}`, apiKey, fetchImpl);
  }

  /**
   * Queue a job; returns its id. Not idempotent: a run that fails after it was sent may still have
   * queued a job (see neverSent).
   */
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
    let job = await this.request<RemoteJob<T>>(
      'runsync',
      { method: 'POST', body: JSON.stringify({ input }) },
      Math.max(this.timeoutMs, RUNSYNC_TIMEOUT_MS),
    );
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

  private async request<T>(path: string, init: RequestInit, timeoutMs = this.timeoutMs): Promise<T> {
    const route = path.split('/')[0];
    try {
      const res = await this.fetchImpl(`${this.url}/${path}`, {
        ...init,
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${this.token}`,
        },
        // Covers reading the answer too: a host that stops halfway can't hold a poll up
        signal: AbortSignal.timeout(timeoutMs),
      });
      if (!res.ok) {
        throw new Error(`AI worker ${route} failed: ${res.status} ${await res.text()}`);
      }
      return (await res.json()) as T;
    } catch (err) {
      if (err instanceof Error && err.name === 'TimeoutError') {
        const seconds = timeoutMs / 1000;
        throw new Error(`AI worker ${route} timed out after ${seconds} s`, { cause: err });
      }
      throw err;
    }
  }
}

/**
 * The codes fetch gives (on the error's cause) when it couldn't connect, so no byte of the request
 * left: the connection was refused, the host wasn't found (or DNS failed for now), or connecting
 * timed out.
 */
const NOT_CONNECTED = new Set(['ECONNREFUSED', 'ENOTFOUND', 'EAI_AGAIN', 'UND_ERR_CONNECT_TIMEOUT']);

/**
 * Whether a request failed before it reached the job API, so the API never saw it. Only then is a
 * run safe to send again: one that may have arrived (a 5xx, a connection reset, a timeout) may have
 * queued a job that nobody would record or stop.
 */
export function neverSent(err: unknown): boolean {
  let cause = err;
  // fetch's own error is "fetch failed"; the reason is its cause (or that error's cause)
  for (let depth = 0; depth < 4 && cause instanceof Error; depth++) {
    const code = (cause as { code?: unknown }).code;
    if (typeof code === 'string' && NOT_CONNECTED.has(code)) return true;
    cause = cause.cause;
  }
  return false;
}
