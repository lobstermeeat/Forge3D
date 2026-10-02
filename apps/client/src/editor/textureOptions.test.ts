import { describe, expect, it } from 'vitest';
import {
  chosenTexture,
  recommendationStep,
  recommendedTexture,
  textureChoices,
  texturesRunning,
  type TextureGeneration,
} from './textureOptions';

type Textures = NonNullable<TextureGeneration['textures']>;

const final = { url: 'https://files.test/ai/g1/final-42.glb', triangles: 100_000 };
const option = (textureSeed: number) => ({
  url: `https://files.test/ai/g1/final-42-texture-${textureSeed}.glb`,
  triangles: 96_000,
  textureSeed,
});
const textures = (
  status: 'running' | 'done' | 'failed',
  options: ReturnType<typeof option>[] = [],
  more: Partial<Textures> = {},
): Textures => ({
  status,
  count: 3,
  options,
  error: status === 'failed' ? 'CUDA out of memory' : null,
  recommended: null,
  chosen: null,
  judgeError: null,
  ...more,
});

describe('texture options', () => {
  it("number the final's own texture 1, then the others in order", () => {
    const g = {
      status: 'done' as const,
      final,
      textures: textures('done', [option(1042), option(3042)]),
    };
    expect(textureChoices(g)).toEqual([
      { number: 1, url: final.url },
      { number: 2, url: option(1042).url },
      { number: 3, url: option(3042).url },
    ]);
  });

  it('offer nothing to choose until the others are made, or when none could be', () => {
    expect(textureChoices({ status: 'done', final, textures: null })).toEqual([]);
    expect(textureChoices({ status: 'done', final, textures: textures('running') })).toEqual([]);
    expect(textureChoices({ status: 'done', final, textures: textures('failed') })).toEqual([]);
    expect(textureChoices({ status: 'done', final, textures: textures('done') })).toEqual([]);
    // Another picture picked since: the old textures don't belong to the new preview
    expect(
      textureChoices({
        status: 'previewing',
        final: null,
        textures: textures('done', [option(1)]),
      }),
    ).toEqual([]);
  });

  it("show which one the scene has, by its model's URL", () => {
    const choices = textureChoices({
      status: 'done',
      final,
      textures: textures('done', [option(1042), option(2042), option(3042)]),
    });
    expect(chosenTexture(choices, final.url)).toBe(1);
    expect(chosenTexture(choices, option(3042).url)).toBe(4);
    // Not in the scene, or showing something else (the preview)
    expect(chosenTexture(choices, null)).toBeNull();
    expect(chosenTexture(choices, 'https://files.test/ai/g1/preview-42.glb')).toBeNull();
    expect(chosenTexture([], final.url)).toBeNull();
  });

  it('keep the panel asking while they are made', () => {
    expect(texturesRunning({ status: 'done', final, textures: textures('running') })).toBe(true);
    expect(texturesRunning({ status: 'done', final, textures: textures('done') })).toBe(false);
    expect(texturesRunning({ status: 'done', final, textures: textures('failed') })).toBe(false);
    expect(texturesRunning({ status: 'done', final, textures: null })).toBe(false);
    expect(texturesRunning(undefined)).toBe(false);
  });
});

describe('the recommended texture (AI_TEXTURE_JUDGE=1)', () => {
  const why = 'The back keeps the brass colour.';
  const three = [option(1042), option(2042), option(3042)];
  /** A done final whose judge picked `number` (1 is the final's own texture) */
  const judged = (number: number, more: Partial<Textures> = {}): TextureGeneration => ({
    status: 'done',
    final,
    textures: textures('done', three, {
      recommended: { number, why, applied: false },
      ...more,
    }),
  });
  const unjudged = (more: Partial<Textures> = {}): TextureGeneration => ({
    status: 'done',
    final,
    textures: textures('done', three, more),
  });
  const inScene = { sceneUrl: final.url, playing: false };

  it("is the judge's pick among the choices, unless it is the final's own", () => {
    expect(recommendedTexture(judged(3))).toEqual({ number: 3, url: option(2042).url, why });
    // The final's own texture is in the scene already: as without the judge
    expect(recommendedTexture(judged(1))).toBeNull();
    // No pick (the judge is off, or failed), or one that isn't a choice
    expect(recommendedTexture(unjudged())).toBeNull();
    expect(recommendedTexture(unjudged({ judgeError: 'judge failed: timed out' }))).toBeNull();
    expect(recommendedTexture(judged(5))).toBeNull();
    // None while the textures are made
    expect(
      recommendedTexture({
        status: 'done',
        final,
        textures: textures('running', [], { recommended: { number: 3, why, applied: false } }),
      }),
    ).toBeNull();
  });

  it("goes in once, over the final's own texture", () => {
    expect(recommendationStep(judged(3), inScene)).toBe('apply');
    // Once applied (on this page, or as the server says after a reload), the panel says so
    // while the scene shows it, and never applies it again, even after Undo took it out
    const scenePick = { sceneUrl: option(2042).url, playing: false };
    expect(recommendationStep(judged(3), { ...scenePick, applied: true })).toBe('applied');
    expect(recommendationStep(judged(3), { ...inScene, applied: true })).toBe('none');
    const appliedBefore = judged(3, { recommended: { number: 3, why, applied: true } });
    expect(recommendationStep(appliedBefore, scenePick)).toBe('applied');
    expect(recommendationStep(appliedBefore, inScene)).toBe('none');
  });

  it("never goes over the creator's choice", () => {
    // Chosen on this page, or recorded on the server (here they went back to 1)
    expect(recommendationStep(judged(3), { ...inScene, chosen: true })).toBe('none');
    expect(recommendationStep(judged(3, { chosen: 1 }), inScene)).toBe('none');
    // Choosing after the switch ends the note, even choosing the same texture
    const scenePick = { sceneUrl: option(2042).url, playing: false, applied: true };
    expect(recommendationStep(judged(3, { chosen: 3 }), scenePick)).toBe('none');
    expect(recommendationStep(judged(3), { ...scenePick, chosen: true })).toBe('none');
    // The scene shows another texture or the preview, or the model was removed
    expect(recommendationStep(judged(3), { ...inScene, sceneUrl: option(1042).url })).toBe('none');
    expect(
      recommendationStep(judged(3), {
        ...inScene,
        sceneUrl: 'https://files.test/ai/g1/preview-42.glb',
      }),
    ).toBe('none');
    expect(recommendationStep(judged(3), { ...inScene, sceneUrl: null })).toBe('none');
  });

  it('waits while the scene plays', () => {
    expect(recommendationStep(judged(3), { ...inScene, playing: true })).toBe('wait');
    expect(recommendationStep(judged(3), { ...inScene, playing: true, applied: true })).toBe(
      'none',
    );
  });

  it("does nothing without a pick, or when the pick is the final's own", () => {
    expect(recommendationStep(judged(1), inScene)).toBe('none');
    expect(recommendationStep(unjudged(), inScene)).toBe('none');
    expect(recommendationStep(unjudged({ judgeError: 'judge failed' }), inScene)).toBe('none');
    expect(recommendationStep({ status: 'done', final, textures: null }, inScene)).toBe('none');
  });
});
