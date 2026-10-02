import { describe, expect, it } from 'vitest';
import {
  createAIOrchestrator,
  createStudioWorkers,
  multiviewEnabled,
  textureJudgeEnabled,
  textureOptionsEnabled,
} from '../index';
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

  it('draws the views on the multiview worker and sends them with the picture to TRELLIS.2', async () => {
    const view = (azimuth: unknown, more: object) => ({
      key: `ai/gen-6/view-${String(azimuth)}.png`,
      azimuth,
      ...more,
    });
    const api = fakeJobApi([
      { id: 'fc-mv', status: 'IN_QUEUE' },
      { id: 'fc-mv', status: 'IN_PROGRESS' },
      {
        id: 'fc-mv',
        status: 'COMPLETED',
        output: {
          request_id: 'gen-6',
          views: [
            view(0, { elevation: 0, url: 'https://r2.test/ai/gen-6/view-0.png' }),
            view(45, { elevation: 0, url: null, base64: 'iVBORw0KGgo=' }),
            // A view that doesn't say where it was seen from is left out
            view('90', { elevation: 0, url: 'https://r2.test/ai/gen-6/view-90.png' }),
            view(180, { url: 'https://r2.test/ai/gen-6/view-180.png' }),
          ],
          camera: { type: 'orthographic' },
          seconds: 18.4,
        },
      },
      { id: 'fc-glb', status: 'IN_QUEUE' },
      { id: 'fc-glb', status: 'COMPLETED', output: { ...trellisOutput, views_used: 2 } },
      // A 3D worker from before views doesn't report them
      { id: 'fc-old', status: 'COMPLETED', output: trellisOutput },
      { id: 'fc-oom', status: 'COMPLETED', output: { error: 'generation failed: out of memory' } },
      {
        id: 'fc-r2',
        status: 'COMPLETED',
        output: { views: [view(0, { elevation: 0, url: null })], seconds: 1 },
      },
      { status: 'WARMING' },
      { id: 'fc-mv', status: 'CANCELLED' },
    ]);
    const provider = createSelfHostedProvider(
      { AI_WORKERS_URL: 'https://w.modal.run', AI_WORKERS_TOKEN: 'worker-token' },
      api.fetchImpl,
    )!;
    expect(provider.multiview).toBe(true);

    const picture = Buffer.from('picture');
    expect(await provider.startViews({ image: picture, requestId: 'gen-6' })).toBe('fc-mv');
    expect(await provider.views('fc-mv')).toEqual({ status: 'running' });
    expect(await provider.views('fc-mv')).toStrictEqual({
      status: 'done',
      output: {
        views: [
          { file: { url: 'https://r2.test/ai/gen-6/view-0.png' }, azimuth: 0, elevation: 0 },
          { file: { data: Buffer.from('iVBORw0KGgo=', 'base64') }, azimuth: 45, elevation: 0 },
        ],
        seconds: 18.4,
      },
    });

    const views = [
      { image: Buffer.from('front'), azimuth: 0, elevation: 0 },
      { image: Buffer.from('side'), azimuth: 45, elevation: 0 },
    ];
    expect(
      await provider.startModel({ image: picture, views, mode: 'preview', requestId: 'gen-6' }),
    ).toBe('fc-glb');
    expect(await provider.model('fc-glb')).toMatchObject({
      status: 'done',
      output: { seed: 77, viewsUsed: 2 },
    });
    const old = await provider.model('fc-old');
    expect(old.status === 'done' && 'viewsUsed' in old.output).toBe(false);
    expect(await provider.views('fc-oom')).toEqual({
      status: 'failed',
      message: 'generation failed: out of memory',
    });
    await expect(provider.views('fc-r2')).rejects.toThrow(/without a public URL/);
    await provider.warm('multiview');
    await provider.cancel('multiview', 'fc-mv');

    expect(
      api.calls.map((call) => `${call.method} ${call.url.replace('https://w.modal.run', '')}`),
    ).toEqual([
      'POST /multiview/run',
      'GET /multiview/status/fc-mv',
      'GET /multiview/status/fc-mv',
      'POST /trellis2/run',
      'GET /trellis2/status/fc-glb',
      'GET /trellis2/status/fc-old',
      'GET /multiview/status/fc-oom',
      'GET /multiview/status/fc-r2',
      'POST /multiview/warm',
      'POST /multiview/cancel/fc-mv',
    ]);
    expect(api.calls.every((call) => call.auth === 'Bearer worker-token')).toBe(true);
    expect(api.calls[0]!.body).toEqual({
      input: { image_base64: picture.toString('base64'), request_id: 'gen-6' },
    });
    // The views go inline, like the picture
    expect(api.calls[3]!.body).toEqual({
      input: {
        image_base64: picture.toString('base64'),
        views: [
          { image_base64: Buffer.from('front').toString('base64'), azimuth: 0, elevation: 0 },
          { image_base64: Buffer.from('side').toString('base64'), azimuth: 45, elevation: 0 },
        ],
        mode: 'preview',
        request_id: 'gen-6',
      },
    });
  });

  it('has the multiview worker on RunPod only with RUNPOD_MULTIVIEW_ENDPOINT_ID', async () => {
    const env = { RUNPOD_API_KEY: 'k', RUNPOD_TRELLIS2_ENDPOINT_ID: 'ep' };
    const image = Buffer.from('picture');
    const without = createSelfHostedProvider(env, fakeJobApi([]).fetchImpl)!;
    expect(without.multiview).toBe(false);
    await expect(without.startViews({ image, requestId: 'g' })).rejects.toThrow(
      'The views need the multiview worker',
    );

    const api = fakeJobApi([
      { id: 'mv-1', status: 'IN_QUEUE' },
      { id: 'mv-1', status: 'CANCELLED' },
    ]);
    const provider = createSelfHostedProvider(
      { ...env, RUNPOD_MULTIVIEW_ENDPOINT_ID: 'mv-ep' },
      api.fetchImpl,
    )!;
    expect(provider.multiview).toBe(true);
    expect(await provider.startViews({ image, requestId: 'g' })).toBe('mv-1');
    // RunPod can't start a GPU early, but can stop a job
    await provider.warm('multiview');
    await provider.cancel('multiview', 'mv-1');
    expect(api.calls.map((call) => `${call.method} ${call.url}`)).toEqual([
      'POST https://api.runpod.ai/v2/mv-ep/run',
      'POST https://api.runpod.ai/v2/mv-ep/cancel/mv-1',
    ]);
  });

  it('asks TRELLIS.2 for texture options and passes on what it made', async () => {
    const texture = (k: number, more: object = {}) => ({
      texture_seed: 77 + 1000 * k,
      glb: {
        key: `ai/gen-7/final-77-texture-${k}.glb`,
        url: `https://r2.test/ai/gen-7/final-77-texture-${k}.glb`,
      },
      bytes: 1_550_172,
      raw_bytes: 8_915_432,
      triangles: 96_205,
      projection: { applied: true, reason: 'applied', iou: 0.9755 },
      ...more,
    });
    const api = fakeJobApi([
      { id: 'fc-tex', status: 'IN_QUEUE' },
      { id: 'fc-tex', status: 'IN_PROGRESS' },
      {
        id: 'fc-tex',
        status: 'COMPLETED',
        output: {
          request_id: 'gen-7',
          mode: 'textures',
          seed: 77,
          textures: [
            texture(1),
            texture(3, {
              glb: { key: 'ai/gen-7/final-77-texture-3.glb', url: null, base64: 'Z2xURg==' },
            }),
            // Without its file or its seed, a texture can't be offered
            texture(4, { glb: undefined }),
            texture(5, { texture_seed: '5077' }),
          ],
          texture_errors: [{ texture_seed: 2077, error: 'ConnectionError: R2 unreachable' }],
          pipeline: '1024_cascade',
          model: 'trellis2',
          views_used: 1,
          timings: {
            generate_s: 27.7,
            retexture_s: 30.6,
            export_s: 70.8,
            pack_s: 4.5,
            upload_s: 0.9,
          },
          credits: ['Built with DINOv3', '3D generation: TRELLIS.2 (Microsoft, MIT)'],
        },
      },
      {
        id: 'fc-512',
        status: 'COMPLETED',
        output: {
          request_id: 'gen-7',
          mode: 'textures',
          seed: 77,
          textures: [texture(1)],
          pipeline: '512',
          timings: { generate_s: 10, retexture_s: 9.5 },
        },
      },
      // Where the finals are Pixal3D's (the recipe)
      {
        id: 'fc-pixal',
        status: 'COMPLETED',
        output: { error: 'texture options need TRELLIS.2 finals' },
      },
      // A final says which pipeline made it
      { id: 'fc-final', status: 'COMPLETED', output: { ...trellisOutput, pipeline: '512' } },
      { id: 'fc-tex-2', status: 'IN_QUEUE' },
    ]);
    const provider = createSelfHostedProvider(
      { AI_WORKERS_URL: 'https://w.modal.run', AI_WORKERS_TOKEN: 'worker-token' },
      api.fetchImpl,
    )!;
    const picture = Buffer.from('picture');
    const views = [{ image: Buffer.from('back'), azimuth: 180, elevation: 0 }];
    expect(
      await provider.startTextures({
        image: picture,
        views,
        seed: 77,
        count: 3,
        requestId: 'gen-7',
      }),
    ).toBe('fc-tex');
    expect(await provider.textures('fc-tex')).toEqual({ status: 'running' });
    expect(await provider.textures('fc-tex')).toStrictEqual({
      status: 'done',
      output: {
        textures: [
          {
            file: { url: 'https://r2.test/ai/gen-7/final-77-texture-1.glb' },
            textureSeed: 1077,
            triangles: 96_205,
            bytes: 1_550_172,
          },
          {
            file: { data: Buffer.from('Z2xURg==', 'base64') },
            textureSeed: 3077,
            triangles: 96_205,
            bytes: 1_550_172,
          },
        ],
        errors: [{ textureSeed: 2077, message: 'ConnectionError: R2 unreachable' }],
        pipeline: '1024_cascade',
        seconds: expect.closeTo(134.5, 5),
      },
    });
    expect(await provider.textures('fc-512')).toMatchObject({
      status: 'done',
      output: { errors: [], pipeline: '512', seconds: expect.closeTo(19.5, 5) },
    });
    expect(await provider.textures('fc-pixal')).toEqual({
      status: 'failed',
      message: 'texture options need TRELLIS.2 finals',
    });
    expect(await provider.model('fc-final')).toMatchObject({
      status: 'done',
      output: { seed: 77, pipeline: '512' },
    });
    // Without views, none are sent
    await provider.startTextures({ image: picture, seed: 77, count: 1, requestId: 'gen-7' });

    expect(
      api.calls.map((call) => `${call.method} ${call.url.replace('https://w.modal.run', '')}`),
    ).toEqual([
      'POST /trellis2/run',
      'GET /trellis2/status/fc-tex',
      'GET /trellis2/status/fc-tex',
      'GET /trellis2/status/fc-512',
      'GET /trellis2/status/fc-pixal',
      'GET /trellis2/status/fc-final',
      'POST /trellis2/run',
    ]);
    // The final's picture, seed and views, inline as for the final
    expect(api.calls[0]!.body).toEqual({
      input: {
        image_base64: picture.toString('base64'),
        views: [
          { image_base64: Buffer.from('back').toString('base64'), azimuth: 180, elevation: 0 },
        ],
        mode: 'textures',
        seed: 77,
        count: 3,
        request_id: 'gen-7',
      },
    });
    expect(api.calls[6]!.body).toEqual({
      input: {
        image_base64: picture.toString('base64'),
        mode: 'textures',
        seed: 77,
        count: 1,
        request_id: 'gen-7',
      },
    });
  });

  it("asks for the judge's pick with the textures, and reads it by the texture's seed", async () => {
    const texture = (k: number, more: object = {}) => ({
      texture_seed: 77 + 1000 * k,
      glb: {
        key: `ai/gen-8/final-77-texture-${k}.glb`,
        url: `https://r2.test/ai/gen-8/final-77-texture-${k}.glb`,
      },
      bytes: 1_550_172,
      triangles: 96_205,
      ...more,
    });
    // Texture 2 failed, so the worker's list goes 1, 3, 4, 5; 4 came back without its file, and
    // 5 without a seed the server can read
    const textures = [
      texture(1),
      texture(3),
      texture(4, { glb: undefined }),
      texture(5, { texture_seed: '5077' }),
    ];
    const done = (more: object) => ({
      id: 'fc-judged',
      status: 'COMPLETED',
      output: {
        request_id: 'gen-8',
        mode: 'textures',
        seed: 77,
        textures,
        texture_errors: [{ texture_seed: 2077, error: 'CUDA out of memory' }],
        pipeline: '1024_cascade',
        timings: { generate_s: 27.7, retexture_s: 40.8 },
        ...more,
      },
    });
    const verdicts = ['edits', 'reject', 'publish', 'edits', 'reject'];
    const answers: [object, object][] = [
      // pick k is the k-th entry of the worker's own list: the second is texture 3
      [
        { judge: { pick: 2, verdicts, why: ' The back stays clean. ', model: '8b', seconds: 9.1 } },
        { judge: { textureSeed: 3077, why: 'The back stays clean.' } },
      ],
      // 0 is the final's own texture
      [
        { judge: { pick: 0, verdicts, why: 'The final is fine as it is.' } },
        { judge: { textureSeed: null, why: 'The final is fine as it is.' } },
      ],
      // A texture the server won't offer is still named by its seed (the studio drops the pick)
      [{ judge: { pick: 3 } }, { judge: { textureSeed: 4077, why: '' } }],
      [
        { judge: { pick: 4 } },
        { judgeError: 'the judge picked texture 4, which came back without its seed' },
      ],
      [{ judge: { pick: 5 } }, { judgeError: "the judge's pick, 5, isn't a texture" }],
      [{ judge: { pick: -1 } }, { judgeError: "the judge's pick, -1, isn't a texture" }],
      [{ judge: { pick: 1.5 } }, { judgeError: "the judge's pick, 1.5, isn't a texture" }],
      [{ judge: { pick: '1' } }, { judgeError: `the judge's pick, "1", isn't a texture` }],
      [{ judge: true }, { judgeError: "the judge's pick, none, isn't a texture" }],
      // The judge failed: the textures are there, without a pick
      [
        { judge_error: ' judge failed: CUDA out of memory ' },
        { judgeError: 'judge failed: CUDA out of memory' },
      ],
      [
        { judge: { pick: 1 }, judge_error: 'judge failed: timed out' },
        { judgeError: 'judge failed: timed out' },
      ],
      [{ judge_error: { detail: 'odd' } }, { judgeError: 'the judge failed' }],
      // Not asked, or a worker from before the judge
      [{}, {}],
      [{ judge: null }, {}],
    ];
    const api = fakeJobApi([
      { id: 'fc-judged', status: 'IN_QUEUE' },
      ...answers.map(([answer]) => done(answer)),
    ]);
    const provider = createSelfHostedProvider(
      { AI_WORKERS_URL: 'https://w.modal.run', AI_WORKERS_TOKEN: 'worker-token' },
      api.fetchImpl,
    )!;
    const picture = Buffer.from('picture');
    expect(
      await provider.startTextures({
        image: picture,
        seed: 77,
        count: 4,
        requestId: 'gen-8',
        judge: { prompt: 'a brass desk lamp' },
      }),
    ).toBe('fc-judged');
    // The judge and what the user typed go with the textures job; without `judge`, neither does
    // (see the test above)
    expect(api.calls[0]!.body).toEqual({
      input: {
        image_base64: picture.toString('base64'),
        mode: 'textures',
        seed: 77,
        count: 4,
        judge: true,
        prompt: 'a brass desk lamp',
        request_id: 'gen-8',
      },
    });

    for (const [answer, expected] of answers) {
      const state = await provider.textures('fc-judged');
      if (state.status !== 'done') throw new Error(`the job is ${state.status}`);
      const { judge, judgeError } = state.output;
      const read = { ...(judge ? { judge } : {}), ...(judgeError ? { judgeError } : {}) };
      expect(read, JSON.stringify(answer)).toEqual(expected);
      // The textures are read as without the judge
      expect(state.output.textures.map((t) => t.textureSeed)).toEqual([1077, 3077]);
    }
  });

  it('asks the judge only with AI_TEXTURE_JUDGE=1 (or true)', () => {
    expect(textureJudgeEnabled({})).toBe(false);
    expect(textureJudgeEnabled({ AI_TEXTURE_JUDGE: '' })).toBe(false);
    expect(textureJudgeEnabled({ AI_TEXTURE_JUDGE: '0' })).toBe(false);
    expect(textureJudgeEnabled({ AI_TEXTURE_JUDGE: 'false' })).toBe(false);
    expect(textureJudgeEnabled({ AI_TEXTURE_JUDGE: 'yes' })).toBe(false);
    expect(textureJudgeEnabled({ AI_TEXTURE_JUDGE: '1' })).toBe(true);
    expect(textureJudgeEnabled({ AI_TEXTURE_JUDGE: ' 1\n' })).toBe(true);
    expect(textureJudgeEnabled({ AI_TEXTURE_JUDGE: 'true' })).toBe(true);
    expect(textureJudgeEnabled({ AI_TEXTURE_JUDGE: 'TRUE' })).toBe(true);
  });

  it('makes texture options unless AI_TEXTURE_OPTIONS is 0 or false', () => {
    expect(textureOptionsEnabled({})).toBe(true);
    expect(textureOptionsEnabled({ AI_TEXTURE_OPTIONS: '' })).toBe(true);
    expect(textureOptionsEnabled({ AI_TEXTURE_OPTIONS: '1' })).toBe(true);
    expect(textureOptionsEnabled({ AI_TEXTURE_OPTIONS: 'true' })).toBe(true);
    expect(textureOptionsEnabled({ AI_TEXTURE_OPTIONS: '0' })).toBe(false);
    expect(textureOptionsEnabled({ AI_TEXTURE_OPTIONS: ' 0\n' })).toBe(false);
    expect(textureOptionsEnabled({ AI_TEXTURE_OPTIONS: 'false' })).toBe(false);
    expect(textureOptionsEnabled({ AI_TEXTURE_OPTIONS: 'FALSE' })).toBe(false);
    // The mock workers make them too, so development works end to end
    expect(createStudioWorkers({ AI_WORKERS_MOCK: '1' })!.startTextures).toBeTypeOf('function');
  });

  it('turns the views step on only with AI_MULTIVIEW=1', () => {
    expect(multiviewEnabled({})).toBe(false);
    expect(multiviewEnabled({ AI_MULTIVIEW: '' })).toBe(false);
    expect(multiviewEnabled({ AI_MULTIVIEW: '0' })).toBe(false);
    expect(multiviewEnabled({ AI_MULTIVIEW: 'true' })).toBe(false);
    expect(multiviewEnabled({ AI_MULTIVIEW: '1' })).toBe(true);
    expect(multiviewEnabled({ AI_MULTIVIEW: ' 1\n' })).toBe(true);
    // The mock workers have one, so development works end to end
    expect(createStudioWorkers({ AI_WORKERS_MOCK: '1' })!.multiview).toBe(true);
  });
});
