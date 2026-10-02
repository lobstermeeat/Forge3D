import { describe, expect, it } from 'vitest';
import sharp from 'sharp';
import { MOCK_VIEW_AZIMUTHS, MockWorkers, mockGlb } from './mock';

function readGlb(data: Buffer) {
  expect(data.toString('ascii', 0, 4)).toBe('glTF');
  expect(data.readUInt32LE(4)).toBe(2);
  expect(data.readUInt32LE(8)).toBe(data.length);
  const jsonLength = data.readUInt32LE(12);
  expect(data.toString('ascii', 16, 20)).toBe('JSON');
  const json = JSON.parse(data.toString('utf8', 20, 20 + jsonLength));
  const binStart = 20 + jsonLength;
  expect(data.toString('ascii', binStart + 4, binStart + 8)).toBe('BIN\0');
  expect(data.readUInt32LE(binStart)).toBeGreaterThanOrEqual(json.buffers[0].byteLength);
  return json;
}

describe('mock workers', () => {
  it('build a valid GLB: a small house, grey-blue as a preview and coloured as a final', () => {
    const preview = readGlb(mockGlb(7, 'preview'));
    const final = readGlb(mockGlb(7, 'final'));
    expect(preview.accessors[0].count).toBe(14 * 3);
    expect(preview.accessors[0].max).toEqual([0.4, 0.9, 0.4]);
    expect(preview.materials[0].pbrMetallicRoughness.baseColorFactor).not.toEqual(
      final.materials[0].pbrMetallicRoughness.baseColorFactor,
    );
  });

  it('draw four pictures, then make models, with the failures the panel shows', async () => {
    const workers = new MockWorkers(0);
    const refs = await workers.startReferences({ prompt: 'a <lamp>', count: 4 });
    const drawn = await workers.references(refs);
    expect(drawn.status).toBe('done');
    const images = drawn.status === 'done' ? drawn.output.images : [];
    expect(images).toHaveLength(4);
    const first = images[0]!.file;
    expect('data' in first && (await sharp(first.data).metadata()).width).toBe(512);

    // Rated like the FLUX worker rates its pictures, with the second clearly the best and the
    // others saying what's wrong with them
    const scores = images.map((image) => image.score!);
    expect(scores.every((score) => score >= 0 && score <= 1)).toBe(true);
    const [best, runnerUp] = [...scores].sort((a, b) => b - a);
    expect(scores.indexOf(best!)).toBe(1);
    expect(best! - runnerUp!).toBeGreaterThanOrEqual(0.2);
    expect(images.map((image) => image.issues?.length ?? 0)).toEqual([1, 0, 2, 1]);
    const again = await workers.references(
      await workers.startReferences({ prompt: 'a chair', count: 4 }),
    );
    expect(again.status === 'done' && again.output.images.map((image) => image.score)).toEqual(
      scores,
    );

    const failing = await workers.startReferences({ prompt: 'please fail', count: 4 });
    expect(await workers.references(failing)).toMatchObject({ status: 'failed' });

    const photo = await sharp({
      create: { width: 256, height: 256, channels: 3, background: '#888' },
    })
      .png()
      .toBuffer();
    const job = await workers.startModel({ image: photo, mode: 'final', seed: 99 });
    expect(await workers.model(job)).toMatchObject({ status: 'done', output: { seed: 99 } });

    const tiny = await sharp({ create: { width: 16, height: 16, channels: 3, background: '#888' } })
      .png()
      .toBuffer();
    const noObject = await workers.startModel({ image: tiny, mode: 'preview' });
    expect(await workers.model(noObject)).toMatchObject({
      status: 'failed',
      message: expect.stringMatching(/no object found/),
    });
    expect(await workers.model('mock-404')).toMatchObject({ status: 'failed' });
  });

  it('draw 6 views around the picture as cutouts, like the multiview worker', async () => {
    const workers = new MockWorkers(0);
    expect(workers.multiview).toBe(true);
    const picture = await sharp({
      create: { width: 512, height: 512, channels: 3, background: '#888' },
    })
      .png()
      .toBuffer();
    const drawn = await workers.views(await workers.startViews({ image: picture }));
    expect(drawn.status).toBe('done');
    const views = drawn.status === 'done' ? drawn.output.views : [];
    expect(views.map(({ azimuth, elevation }) => [azimuth, elevation])).toEqual(
      [0, 45, 90, 180, 270, 315].map((azimuth) => [azimuth, 0]),
    );
    expect(MOCK_VIEW_AZIMUTHS).toEqual([0, 45, 90, 180, 270, 315]);
    const sizes = new Set<number>();
    for (const { file } of views) {
      const data = 'data' in file ? file.data : Buffer.alloc(0);
      expect(await sharp(data).metadata()).toMatchObject({
        format: 'png',
        width: 256,
        height: 256,
        hasAlpha: true,
      });
      // Cut out: the corners are transparent, the object isn't
      const { data: pixels, info } = await sharp(data).raw().toBuffer({ resolveWithObject: true });
      const alpha = (x: number, y: number) => pixels[(y * info.width + x) * info.channels + 3];
      expect(alpha(0, 0)).toBe(0);
      expect(alpha(128, 160)).toBe(255);
      sizes.add(data.length);
    }
    // Each side looks different
    expect(sizes.size).toBeGreaterThan(1);

    // A small photo gets none, so the fallback can be tried
    const small = await sharp({
      create: { width: 100, height: 100, channels: 3, background: '#888' },
    })
      .png()
      .toBuffer();
    expect(await workers.views(await workers.startViews({ image: small }))).toMatchObject({
      status: 'failed',
      message: expect.stringContaining('128 px'),
    });
    expect(await workers.views('mock-404')).toMatchObject({ status: 'failed' });
  });

  it('build models from the views they are sent, and stop jobs', async () => {
    const workers = new MockWorkers(0);
    const picture = await sharp({
      create: { width: 256, height: 256, channels: 3, background: '#888' },
    })
      .png()
      .toBuffer();
    const views = [0, 90, 180].map((azimuth) => ({ image: picture, azimuth, elevation: 0 }));
    const withViews = await workers.startModel({ image: picture, views, mode: 'preview' });
    expect(await workers.model(withViews)).toMatchObject({
      status: 'done',
      output: { viewsUsed: 3 },
    });
    const alone = await workers.startModel({ image: picture, mode: 'preview' });
    expect(await workers.model(alone)).toMatchObject({ status: 'done', output: { viewsUsed: 0 } });

    const job = await workers.startViews({ image: picture });
    await workers.cancel('multiview', job);
    expect(await workers.views(job)).toMatchObject({
      status: 'failed',
      message: `Unknown job ${job}`,
    });
  });
});
