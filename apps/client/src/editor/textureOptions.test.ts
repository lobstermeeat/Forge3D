import { describe, expect, it } from 'vitest';
import { chosenTexture, textureChoices, texturesRunning } from './textureOptions';

const final = { url: 'https://files.test/ai/g1/final-42.glb', triangles: 100_000 };
const option = (textureSeed: number) => ({
  url: `https://files.test/ai/g1/final-42-texture-${textureSeed}.glb`,
  triangles: 96_000,
  textureSeed,
});
const textures = (
  status: 'running' | 'done' | 'failed',
  options: ReturnType<typeof option>[] = [],
) => ({ status, count: 3, options, error: status === 'failed' ? 'CUDA out of memory' : null });

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
