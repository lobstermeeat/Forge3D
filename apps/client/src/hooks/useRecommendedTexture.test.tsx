// @vitest-environment jsdom
import { StrictMode } from 'react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { cleanup, renderHook } from '@testing-library/react';
import type { RecommendedTexture, TextureGeneration } from '@/editor/textureOptions';
import { useAIStore } from '@/stores/aiStore';
import { useRecommendedTexture } from './useRecommendedTexture';

const final = { url: 'https://files.test/ai/g1/final-42.glb', triangles: 100_000 };
const optionUrl = (k: number) => `https://files.test/ai/g1/final-42-texture-${42 + 1000 * k}.glb`;
const why = 'The back keeps the brass colour, with no smudges.';

/**
 * A done final with 3 more textures, whose judge picked `pick` (1 is the final's own texture;
 * null: no pick), and what the server recorded: the pick applied, the texture the creator chose
 */
function judged(
  pick: number | null,
  server: { applied?: boolean; chosen?: number | null } = {},
): TextureGeneration {
  return {
    status: 'done',
    final,
    textures: {
      status: 'done',
      count: 3,
      options: [1, 2, 3].map((k) => ({
        url: optionUrl(k),
        triangles: 96_000,
        textureSeed: 42 + 1000 * k,
      })),
      error: null,
      recommended: pick === null ? null : { number: pick, why, applied: server.applied ?? false },
      chosen: server.chosen ?? null,
      judgeError: null,
    },
  };
}

interface Props {
  g: TextureGeneration | undefined;
  sceneUrl: string | null;
  playing: boolean;
}

/** The hook as the AI panel runs it, re-rendered with new props as the panel would be */
function mount(initialProps: Props, { strict = false } = {}) {
  const apply = vi.fn<(texture: RecommendedTexture) => void>();
  const hook = renderHook(
    ({ g, sceneUrl, playing }: Props) => useRecommendedTexture(g, { sceneUrl, playing, apply }),
    { initialProps, ...(strict ? { wrapper: StrictMode } : {}) },
  );
  return { ...hook, apply };
}

/** A new page: this page's memory of the textures goes */
function reload() {
  useAIStore.setState({ recommendedApplied: {}, textureChosen: {} });
}

beforeEach(reload);
afterEach(cleanup);

describe('useRecommendedTexture', () => {
  it('switches the scene to the recommended texture once, however often the panel renders', () => {
    const { result, rerender, apply } = mount({
      g: judged(3),
      sceneUrl: final.url,
      playing: false,
    });
    expect(apply).toHaveBeenCalledTimes(1);
    expect(apply).toHaveBeenCalledWith({ number: 3, url: optionUrl(2), why });
    expect(useAIStore.getState().recommendedApplied).toEqual({ [final.url]: true });

    // The scene shows it: the panel says it switched
    rerender({ g: judged(3), sceneUrl: optionUrl(2), playing: false });
    expect(result.current).toBe('applied');
    // The server's record comes back with the generation
    rerender({ g: judged(3, { applied: true }), sceneUrl: optionUrl(2), playing: false });
    expect(result.current).toBe('applied');
    // Undo puts the final's own texture back: the pick doesn't go in again
    rerender({ g: judged(3, { applied: true }), sceneUrl: final.url, playing: false });
    expect(result.current).toBe('none');
    rerender({ g: judged(3), sceneUrl: final.url, playing: false });
    expect(apply).toHaveBeenCalledTimes(1);
  });

  it('applies it once when the panel closes and opens again, and under StrictMode', () => {
    // StrictMode runs effects twice
    const first = mount({ g: judged(3), sceneUrl: final.url, playing: false }, { strict: true });
    expect(first.apply).toHaveBeenCalledTimes(1);
    first.unmount();

    // Another Library tab and back: a new panel on the same page, the scene back on texture 1
    const again = mount({ g: judged(3), sceneUrl: final.url, playing: false });
    expect(again.apply).not.toHaveBeenCalled();
    expect(again.result.current).toBe('none');
  });

  it('remembers it across a reload through what the server recorded', () => {
    // The panel applied it, then Undo took it out; the page reloads
    reload();
    const reloaded = mount({
      g: judged(3, { applied: true }),
      sceneUrl: final.url,
      playing: false,
    });
    expect(reloaded.apply).not.toHaveBeenCalled();
    expect(reloaded.result.current).toBe('none');
    // While the scene shows it, the panel still says it switched
    reloaded.rerender({ g: judged(3, { applied: true }), sceneUrl: optionUrl(2), playing: false });
    expect(reloaded.result.current).toBe('applied');
  });

  it("never goes over the creator's choice", () => {
    // Chosen on this page
    useAIStore.getState().markTextureChosen(final.url);
    const here = mount({ g: judged(3), sceneUrl: final.url, playing: false });
    expect(here.apply).not.toHaveBeenCalled();
    expect(here.result.current).toBe('none');
    here.unmount();

    // Recorded on the server before a reload: they chose another, then texture 1 again
    reload();
    const recorded = mount({ g: judged(3, { chosen: 1 }), sceneUrl: final.url, playing: false });
    expect(recorded.apply).not.toHaveBeenCalled();
    recorded.unmount();

    // The scene shows another texture, or the model was removed
    for (const sceneUrl of [optionUrl(1), null]) {
      const other = mount({ g: judged(3), sceneUrl, playing: false });
      expect(other.apply).not.toHaveBeenCalled();
      expect(other.result.current).toBe('none');
      other.unmount();
    }
    expect(useAIStore.getState().recommendedApplied).toEqual({});
  });

  it('waits while the scene plays, and switches when it stops', () => {
    const { result, rerender, apply } = mount({ g: judged(3), sceneUrl: final.url, playing: true });
    expect(result.current).toBe('wait');
    expect(apply).not.toHaveBeenCalled();
    rerender({ g: judged(3), sceneUrl: final.url, playing: true });
    expect(apply).not.toHaveBeenCalled();

    rerender({ g: judged(3), sceneUrl: final.url, playing: false });
    expect(apply).toHaveBeenCalledTimes(1);
    expect(apply).toHaveBeenCalledWith({ number: 3, url: optionUrl(2), why });
  });

  it("does nothing without a pick, with the final's own as the pick, or before the options", () => {
    const running: TextureGeneration = {
      ...judged(3),
      textures: { ...judged(3).textures!, status: 'running', options: [] },
    };
    for (const g of [judged(null), judged(1), running, undefined]) {
      const { result, apply, unmount } = mount({ g, sceneUrl: final.url, playing: false });
      expect(result.current).toBe('none');
      expect(apply).not.toHaveBeenCalled();
      unmount();
    }
    expect(useAIStore.getState().recommendedApplied).toEqual({});
  });
});
