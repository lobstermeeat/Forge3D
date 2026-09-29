import { describe, expect, it } from 'vitest';
import { createAIOrchestrator } from '../index';
import { RunPodEndpoint, type FetchLike } from './runpod';
import { SelfHostedProvider, createSelfHostedProvider } from './selfHosted';

interface Call {
  url: string;
  method: string;
  body: unknown;
  auth: string | null;
}

/** A fake RunPod API: answers each request from a queue of JSON bodies. */
function fakeRunPod(responses: unknown[]) {
  const calls: Call[] = [];
  const fetchImpl: FetchLike = async (url, init) => {
    const headers = new Headers(init?.headers);
    calls.push({
      url,
      method: init?.method ?? 'GET',
      body: init?.body ? JSON.parse(String(init.body)) : undefined,
      auth: headers.get('Authorization'),
    });
    const next = responses.shift();
    if (next === undefined) throw new Error(`unexpected request to ${url}`);
    return new Response(JSON.stringify(next), { status: 200 });
  };
  return { calls, fetchImpl };
}

const trellisOutput = {
  request_id: 'gen_1',
  mode: 'final',
  seed: 77,
  glb: { key: 'ai/gen_1/final.glb', url: 'https://assets.forge3d.app/ai/gen_1/final.glb' },
  bytes: 2_400_000,
  raw_bytes: 9_000_000,
  triangles: 100_000,
  timings: { generate_s: 41.2, export_s: 12.5, compress_s: 6.1, upload_s: 0.7 },
  credits: ['Built with DINOv3', '3D generation: TRELLIS.2 (Microsoft, MIT)'],
};

describe('SelfHostedProvider', () => {
  it('queues an image-to-3d job with the preview seed and maps the result', async () => {
    const { calls, fetchImpl } = fakeRunPod([
      { id: 'job-1', status: 'IN_QUEUE' },
      { id: 'job-1', status: 'IN_PROGRESS' },
      { id: 'job-1', status: 'COMPLETED', output: trellisOutput },
      { id: 'job-1', status: 'COMPLETED', output: trellisOutput },
    ]);
    const provider = new SelfHostedProvider(
      new RunPodEndpoint('trellis-ep', 'rp-key', fetchImpl),
      null,
    );

    const jobId = await provider.generate({
      type: 'image-to-3d',
      imageUrl: 'https://assets.forge3d.app/uploads/cat.png',
      userId: 'u1',
      quality: 'final',
      seed: 77,
      requestId: 'gen_1',
    });
    expect(jobId).toBe('job-1');
    expect(calls[0]).toEqual({
      url: 'https://api.runpod.ai/v2/trellis-ep/run',
      method: 'POST',
      body: {
        input: {
          image_url: 'https://assets.forge3d.app/uploads/cat.png',
          mode: 'final',
          seed: 77,
          request_id: 'gen_1',
        },
      },
      auth: 'Bearer rp-key',
    });

    expect(await provider.pollStatus('job-1')).toEqual({ status: 'processing', progress: 50 });
    expect(await provider.pollStatus('job-1')).toEqual({ status: 'completed', progress: 100 });
    expect(calls[2]!.url).toBe('https://api.runpod.ai/v2/trellis-ep/status/job-1');

    const result = await provider.getResult('job-1');
    expect(result).toEqual({
      modelUrl: 'https://assets.forge3d.app/ai/gen_1/final.glb',
      format: 'glb',
      durationMs: 60_500,
      seed: 77,
      triangles: 100_000,
      bytes: 2_400_000,
      credits: ['Built with DINOv3', '3D generation: TRELLIS.2 (Microsoft, MIT)'],
    });
  });

  it('defaults to the final quality', async () => {
    const { calls, fetchImpl } = fakeRunPod([{ id: 'job-2', status: 'IN_QUEUE' }]);
    const provider = new SelfHostedProvider(new RunPodEndpoint('ep', 'k', fetchImpl), null);
    await provider.generate({ type: 'image-to-3d', imageUrl: 'https://x/y.png', userId: 'u' });
    expect((calls[0]!.body as { input: { mode: string } }).input.mode).toBe('final');
  });

  it('reports worker failures', async () => {
    const { fetchImpl } = fakeRunPod([
      {
        id: 'job-3',
        status: 'FAILED',
        error: 'generation failed: RuntimeError: CUDA out of memory',
      },
      { id: 'job-3', status: 'COMPLETED', output: { error: 'invalid input: mode must be one of' } },
    ]);
    const provider = new SelfHostedProvider(new RunPodEndpoint('ep', 'k', fetchImpl), null);
    expect(await provider.pollStatus('job-3')).toEqual({
      status: 'failed',
      progress: 100,
      message: 'generation failed: RuntimeError: CUDA out of memory',
    });
    await expect(provider.getResult('job-3')).rejects.toThrow('invalid input: mode must be one of');
  });

  it('turns a prompt into reference images, and uses the first when there is no pick', async () => {
    const reference = fakeRunPod([
      {
        id: 'ref-1',
        status: 'COMPLETED',
        output: {
          request_id: 'gen_9',
          prompt: 'a teapot. A single object…',
          images: [
            {
              key: 'ai/gen_9/reference-0.png',
              url: 'https://assets.forge3d.app/ai/gen_9/reference-0.png',
              seed: 5,
            },
            { key: 'ai/gen_9/reference-1.png', url: null, base64: 'iVBORw0KGgo=', seed: 6 },
          ],
          seconds: 3.2,
        },
      },
      {
        id: 'ref-2',
        status: 'COMPLETED',
        output: {
          request_id: 'gen_9',
          prompt: 'a teapot. A single object…',
          images: [
            {
              key: 'ai/gen_9/reference-0.png',
              url: 'https://assets.forge3d.app/ai/gen_9/reference-0.png',
              seed: 5,
            },
          ],
          seconds: 1.1,
        },
      },
    ]);
    const trellis = fakeRunPod([{ id: 'job-9', status: 'IN_QUEUE' }]);
    const provider = new SelfHostedProvider(
      new RunPodEndpoint('trellis-ep', 'k', trellis.fetchImpl),
      new RunPodEndpoint('ref-ep', 'k', reference.fetchImpl),
    );

    const images = await provider.referenceImages('a teapot', { requestId: 'gen_9', seed: 5 });
    expect(images).toEqual([
      { url: 'https://assets.forge3d.app/ai/gen_9/reference-0.png', seed: 5 },
      { url: 'data:image/png;base64,iVBORw0KGgo=', seed: 6 },
    ]);
    expect(reference.calls[0]!.url).toBe('https://api.runpod.ai/v2/ref-ep/runsync');
    expect(reference.calls[0]!.body).toEqual({
      input: { prompt: 'a teapot', count: 4, seed: 5, request_id: 'gen_9' },
    });

    await provider.generate({
      type: 'text-to-3d',
      prompt: 'a teapot',
      userId: 'u',
      requestId: 'gen_9',
      quality: 'preview',
    });
    expect((reference.calls[1]!.body as { input: { count: number } }).input.count).toBe(1);
    expect(trellis.calls[0]!.body).toEqual({
      input: {
        image_url: 'https://assets.forge3d.app/ai/gen_9/reference-0.png',
        mode: 'preview',
        request_id: 'gen_9',
      },
    });
  });

  it('is only registered when RunPod is configured', () => {
    expect(createSelfHostedProvider({})).toBeNull();
    expect(createAIOrchestrator({}).getAvailableProviders('image-to-3d')).toEqual([]);
    const orchestrator = createAIOrchestrator({
      RUNPOD_API_KEY: 'k',
      RUNPOD_TRELLIS2_ENDPOINT_ID: 'ep',
    });
    expect(orchestrator.selectProvider('image-to-3d').name).toBe('forge3d-trellis2');
  });
});
