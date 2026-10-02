import { useEffect, useRef } from 'react';
import {
  recommendationStep,
  recommendedTexture,
  type RecommendationStep,
  type RecommendedTexture,
  type TextureGeneration,
} from '@/editor/textureOptions';
import { useAIStore } from '@/stores/aiStore';

/**
 * Makes the judge's pick the default texture (AI_TEXTURE_JUDGE=1 on the server). When a final's
 * texture options arrive with a recommended texture, `apply` swaps it into the scene in place of
 * the final's own, once:
 * - never twice: the AI store remembers it (it outlives the panel, and the page's re-renders),
 *   and the server does after a reload;
 * - never after the creator chose a texture (the same two records);
 * - not while the scene plays: it waits, and applies it when the scene stops.
 *
 * Returns where the recommendation stands, so the panel can say it switched.
 */
export function useRecommendedTexture(
  g: TextureGeneration | undefined,
  {
    sceneUrl,
    playing,
    apply,
  }: {
    /** The model file the scene shows for this generation, or null when it isn't in the scene */
    sceneUrl: string | null | undefined;
    playing: boolean;
    apply: (texture: RecommendedTexture) => void;
  },
): RecommendationStep {
  const finalUrl = g?.final?.url;
  const applied = useAIStore((s) => !!finalUrl && !!s.recommendedApplied[finalUrl]);
  const chosen = useAIStore((s) => !!finalUrl && !!s.textureChosen[finalUrl]);
  const step = g ? recommendationStep(g, { sceneUrl, playing, applied, chosen }) : 'none';

  const latest = useRef({ g, apply });
  latest.current = { g, apply };
  useEffect(() => {
    if (step !== 'apply' || !finalUrl) return;
    const { g: current, apply: swap } = latest.current;
    const texture = current ? recommendedTexture(current) : null;
    const memory = useAIStore.getState();
    if (!texture || memory.recommendedApplied[finalUrl] || memory.textureChosen[finalUrl]) return;
    // Remembered before the swap, so no render (or StrictMode's second run) applies it again
    memory.markRecommendedApplied(finalUrl);
    swap(texture);
  }, [step, finalUrl]);

  return step;
}
