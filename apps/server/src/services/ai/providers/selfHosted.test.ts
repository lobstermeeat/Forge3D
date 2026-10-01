import { describe, expect, it } from 'vitest';
import { createAIOrchestrator } from '../index';
import { JobEndpoint, type FetchLike } from './jobEndpoint';
import { SelfHostedProvider, createSelfHostedProvider } from './selfHosted';

interface Call {
  url: string;
  method: string;
  body: unknown;
  auth: string | null;
}

/** A fake job API (RunPod or Modal): answers each request from a queue of JSON bodies. */
function fakeJobApi(responses: unknown[]) {
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
    const { calls, fetchImpl } = fakeJobApi([
      { id: 'job-1', status: 'IN_QUEUE' },
      { id: 'job-1', status: 'IN_PROGRESS' },
      { id: 'job-1', status: 'COMPLETED', output: trellisOutput },
      { id: 'job-1', status: 'COMPLETED', output: trellisOutput },
    ]);
    const provider = new SelfHostedProvider(
      JobEndpoint.runPod('trellis-ep', 'rp-key', fetchImpl),
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
    const { calls, fetchImpl } = fakeJobApi([{ id: 'job-2', status: 'IN_QUEUE' }]);
    const provider = new SelfHostedProvider(JobEndpoint.runPod('ep', 'k', fetchImpl), null);
    await provider.generate({ type: 'image-to-3d', imageUrl: 'https://x/y.png', userId: 'u' });
    expect((calls[0]!.body as { input: { mode: string } }).input.mode).toBe('final');
  });

  it('reports worker failures', async () => {
    const { fetchImpl } = fakeJobApi([
      {
        id: 'job-3',
        status: 'FAILED',
        error: 'generation failed: RuntimeError: CUDA out of memory',
      },
      { id: 'job-3', status: 'COMPLETED', output: { error: 'invalid input: mode must be one of' } },
    ]);
    const provider = new SelfHostedProvider(JobEndpoint.runPod('ep', 'k', fetchImpl), null);
    expect(await provider.pollStatus('job-3')).toEqual({
      status: 'failed',
      progress: 100,
      message: 'generation failed: RuntimeError: CUDA out of memory',
    });
    await expect(provider.getResult('job-3')).rejects.toThrow('invalid input: mode must be one of');
  });

  it('turns a prompt into reference images, and uses the first when there is no pick', async () => {
    const reference = fakeJobApi([
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
    const trellis = fakeJobApi([{ id: 'job-9', status: 'IN_QUEUE' }]);
    const provider = new SelfHostedProvider(
      JobEndpoint.runPod('trellis-ep', 'k', trellis.fetchImpl),
      JobEndpoint.runPod('ref-ep', 'k', reference.fetchImpl),
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

  it('sends inline reference images as data instead of a URL', async () => {
    const { calls, fetchImpl } = fakeJobApi([{ id: 'job-5', status: 'IN_QUEUE' }]);
    const provider = new SelfHostedProvider(JobEndpoint.runPod('ep', 'k', fetchImpl), null);
    await provider.generate({
      type: 'image-to-3d',
      imageUrl: 'data:image/png;base64,iVBORw0KGgo=',
      userId: 'u',
      quality: 'preview',
    });
    expect(calls[0]!.body).toEqual({ input: { image_base64: 'iVBORw0KGgo=', mode: 'preview' } });
  });

  it('cancels a reference job that outlives the timeout', async () => {
    const { calls, fetchImpl } = fakeJobApi([{ id: 'ref-slow', status: 'IN_PROGRESS' }, {}]);
    const endpoint = JobEndpoint.runPod('ref-ep', 'k', fetchImpl);
    await expect(endpoint.runSync({ prompt: 'x' }, -1)).rejects.toThrow('Job ref-slow timed out');
    expect(calls[1]).toMatchObject({
      url: 'https://api.runpod.ai/v2/ref-ep/cancel/ref-slow',
      method: 'POST',
    });
  });

  it('is only registered when workers are configured', () => {
    expect(createSelfHostedProvider({})).toBeNull();
    expect(createAIOrchestrator({}).getAvailableProviders('image-to-3d')).toEqual([]);
    const imageOnly = createAIOrchestrator({
      RUNPOD_API_KEY: 'k',
      RUNPOD_TRELLIS2_ENDPOINT_ID: 'ep',
    });
    expect(imageOnly.selectProvider('image-to-3d').name).toBe('forge3d-trellis2');
    // Text prompts need the reference-image endpoint
    expect(() => imageOnly.selectProvider('text-to-3d')).toThrow(
      'No providers available for text-to-3d',
    );
    const both = createAIOrchestrator({
      RUNPOD_API_KEY: 'k',
      RUNPOD_TRELLIS2_ENDPOINT_ID: 'ep',
      RUNPOD_REFERENCE_ENDPOINT_ID: 'ref',
    });
    expect(both.selectProvider('text-to-3d').name).toBe('forge3d-trellis2');
  });

  it('uses the job API on Modal when AI_WORKERS_URL is set', async () => {
    const api = fakeJobApi([
      {
        id: 'fc-01K6REF',
        status: 'COMPLETED',
        output: {
          request_id: 'gen_3',
          prompt: 'x',
          images: [{ key: 'ai/gen_3/reference-8.png', url: null, base64: 'iVBORw0KGgo=', seed: 8 }],
          seconds: 2,
        },
      },
      { id: 'fc-01K6GLB', status: 'IN_QUEUE' },
      { id: 'fc-01K6GLB', status: 'IN_PROGRESS' },
    ]);
    const provider = createSelfHostedProvider(
      {
        // A trailing slash, or a line break after the token, is fine; Modal wins over RunPod
        AI_WORKERS_URL: 'https://orainge--orainge-ai-api.modal.run/',
        AI_WORKERS_TOKEN: 'worker-token\n',
        RUNPOD_API_KEY: 'k',
        RUNPOD_TRELLIS2_ENDPOINT_ID: 'ep',
      },
      api.fetchImpl,
    )!;
    expect(provider.supportedTypes).toEqual(['image-to-3d', 'text-to-3d']);

    const jobId = await provider.generate({
      type: 'text-to-3d',
      prompt: 'a teapot',
      userId: 'u',
      requestId: 'gen_3',
      quality: 'preview',
    });
    expect(jobId).toBe('fc-01K6GLB');
    expect(await provider.pollStatus(jobId)).toEqual({ status: 'processing', progress: 50 });
    expect(api.calls.map((call) => [call.method, call.url, call.auth])).toEqual([
      [
        'POST',
        'https://orainge--orainge-ai-api.modal.run/reference/runsync',
        'Bearer worker-token',
      ],
      ['POST', 'https://orainge--orainge-ai-api.modal.run/trellis2/run', 'Bearer worker-token'],
      [
        'GET',
        'https://orainge--orainge-ai-api.modal.run/trellis2/status/fc-01K6GLB',
        'Bearer worker-token',
      ],
    ]);
    // Without R2 the reference image comes back inline and goes to TRELLIS.2 as data
    expect(api.calls[1]!.body).toEqual({
      input: { image_base64: 'iVBORw0KGgo=', mode: 'preview', request_id: 'gen_3' },
    });
  });

  it('refuses AI_WORKERS_URL without a token', () => {
    expect(() =>
      createSelfHostedProvider({ AI_WORKERS_URL: 'https://orainge--orainge-ai-api.modal.run' }),
    ).toThrow('AI_WORKERS_URL is set but AI_WORKERS_TOKEN is not');
  });

  it('runs the Studio panel as jobs to poll (StudioWorkers)', async () => {
    const api = fakeJobApi([
      { id: 'fc-ref', status: 'IN_QUEUE' },
      { id: 'fc-ref', status: 'IN_PROGRESS' },
      {
        id: 'fc-ref',
        status: 'COMPLETED',
        output: {
          request_id: 'gen-4',
          prompt: 'a lamp',
          seconds: 3.2,
          images: [
            { key: 'ai/gen-4/reference-7.png', url: null, base64: 'iVBORw0KGgo=', seed: 7 },
            {
              key: 'ai/gen-4/reference-8.png',
              url: 'https://r2.test/ai/gen-4/reference-8.png',
              seed: 8,
            },
          ],
        },
      },
      { id: 'fc-glb', status: 'IN_QUEUE' },
      { id: 'fc-glb', status: 'COMPLETED', output: trellisOutput },
      {
        id: 'fc-bad',
        status: 'COMPLETED',
        output: { ...trellisOutput, error: 'invalid input: no object found' },
      },
      { id: 'fc-gone', status: 'CANCELLED', error: undefined },
    ]);
    const provider = createSelfHostedProvider(
      { AI_WORKERS_URL: 'https://w.modal.run', AI_WORKERS_TOKEN: 'worker-token' },
      api.fetchImpl,
    )!;
    expect(provider.prompts).toBe(true);

    expect(await provider.startReferences({ prompt: 'a lamp', count: 4, requestId: 'gen-4' })).toBe(
      'fc-ref',
    );
    expect(await provider.references('fc-ref')).toEqual({ status: 'running' });
    expect(await provider.references('fc-ref')).toEqual({
      status: 'done',
      output: {
        images: [
          { file: { data: Buffer.from('iVBORw0KGgo=', 'base64') }, seed: 7 },
          { file: { url: 'https://r2.test/ai/gen-4/reference-8.png' }, seed: 8 },
        ],
      },
    });

    const image = Buffer.from('picture');
    expect(await provider.startModel({ image, mode: 'final', seed: 77, requestId: 'gen-4' })).toBe(
      'fc-glb',
    );
    expect(await provider.model('fc-glb')).toEqual({
      status: 'done',
      output: {
        file: { url: trellisOutput.glb.url },
        seed: 77,
        triangles: 100_000,
        bytes: 2_400_000,
        seconds: expect.closeTo(60.5, 5),
        credits: trellisOutput.credits,
      },
    });
    expect(await provider.model('fc-bad')).toEqual({
      status: 'failed',
      message: 'invalid input: no object found',
    });
    expect(await provider.model('fc-gone')).toEqual({
      status: 'failed',
      message: 'The job was cancelled',
    });

    expect(
      api.calls.map((call) => `${call.method} ${call.url.replace('https://w.modal.run', '')}`),
    ).toEqual([
      'POST /reference/run',
      'GET /reference/status/fc-ref',
      'GET /reference/status/fc-ref',
      'POST /trellis2/run',
      'GET /trellis2/status/fc-glb',
      'GET /trellis2/status/fc-bad',
      'GET /trellis2/status/fc-gone',
    ]);
    expect(api.calls[0]!.body).toEqual({
      input: { prompt: 'a lamp', count: 4, request_id: 'gen-4' },
    });
    expect(api.calls[3]!.body).toEqual({
      input: {
        image_base64: image.toString('base64'),
        mode: 'final',
        seed: 77,
        request_id: 'gen-4',
      },
    });
  });

  it('starts a GPU early through the job API on Modal, and not on RunPod', async () => {
    const modal = fakeJobApi([{ status: 'WARMING' }, { status: 'WARMING' }]);
    const provider = createSelfHostedProvider(
      { AI_WORKERS_URL: 'https://w.modal.run', AI_WORKERS_TOKEN: 'worker-token' },
      modal.fetchImpl,
    )!;
    await provider.warm('references');
    await provider.warm('model');
    expect(modal.calls).toEqual([
      {
        url: 'https://w.modal.run/reference/warm',
        method: 'POST',
        body: undefined,
        auth: 'Bearer worker-token',
      },
      {
        url: 'https://w.modal.run/trellis2/warm',
        method: 'POST',
        body: undefined,
        auth: 'Bearer worker-token',
      },
    ]);

    // RunPod endpoints have no such route, so nothing is sent
    const runPod = fakeJobApi([]);
    const onRunPod = createSelfHostedProvider(
      {
        RUNPOD_API_KEY: 'k',
        RUNPOD_TRELLIS2_ENDPOINT_ID: 'ep',
        RUNPOD_REFERENCE_ENDPOINT_ID: 'ref',
      },
      runPod.fetchImpl,
    )!;
    await onRunPod.warm('references');
    await onRunPod.warm('model');
    expect(runPod.calls).toEqual([]);
  });

  it('passes on how the worker rated each picture, when it did', async () => {
    const picture = (seed: number) => ({
      key: `ai/gen-5/reference-${seed}.png`,
      url: `https://r2.test/ai/gen-5/reference-${seed}.png`,
      seed,
    });
    const api = fakeJobApi([
      {
        id: 'fc-ref',
        status: 'COMPLETED',
        output: {
          request_id: 'gen-5',
          prompt: 'a lamp',
          seconds: 3.2,
          images: [
            { ...picture(7), score: 0.35, issues: ['cut off at the bottom', ' two objects '] },
            { ...picture(8), score: 0.88, issues: [] },
            picture(9), // a worker from before ratings
            // What can't be read is left out
            { ...picture(10), score: null, issues: ['', 7, null, 'small in the frame'] },
            { ...picture(11), score: '0.9', issues: 'cut off' },
          ],
        },
      },
    ]);
    const provider = createSelfHostedProvider(
      { AI_WORKERS_URL: 'https://w.modal.run', AI_WORKERS_TOKEN: 'worker-token' },
      api.fetchImpl,
    )!;
    const file = (seed: number) => ({ url: `https://r2.test/ai/gen-5/reference-${seed}.png` });
    expect(await provider.references('fc-ref')).toStrictEqual({
      status: 'done',
      output: {
        images: [
          { file: file(7), seed: 7, score: 0.35, issues: ['cut off at the bottom', 'two objects'] },
          { file: file(8), seed: 8, score: 0.88 },
          { file: file(9), seed: 9 },
          { file: file(10), seed: 10, issues: ['small in the frame'] },
          { file: file(11), seed: 11 },
        ],
      },
    });
  });
});
