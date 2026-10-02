// Type only, like AppRouter in api/trpc.ts: no server code is bundled
import type { GenerationView } from '../../../../apps/server/src/services/ai/studio';

/**
 * Texture options: after the final, the server makes more textures for its shape, and the AI
 * panel lets the creator swap the model in their scene between them. Option 1 is the final's
 * own texture. With AI_TEXTURE_JUDGE=1 on the server, a judge recommends one, and the panel
 * makes it the default.
 */
export interface TextureChoice {
  /** 1 for the final's own texture, then 2, 3, … as the panel numbers them */
  number: number;
  url: string;
}

/** The texture the judge recommends, and why, in its one sentence */
export interface RecommendedTexture extends TextureChoice {
  why: string | null;
}

/**
 * What the panel does with the recommended texture now:
 * - 'apply': put it in the scene in place of the final's own texture (once);
 * - 'wait': the same, once the scene stops playing;
 * - 'applied': the panel put it in, the scene still shows it and the creator hasn't chosen
 *   another, so the panel says it switched;
 * - 'none': nothing more.
 */
export type RecommendationStep = 'apply' | 'wait' | 'applied' | 'none';

/** How often the panel asks about the texture options while they are made (about 2 minutes) */
export const TEXTURES_POLL_MS = 5000;

/** What texture options need of a generation */
export type TextureGeneration = Pick<GenerationView, 'status' | 'final' | 'textures'>;

/** Whether the generation's texture options are still being made. */
export function texturesRunning(g: TextureGeneration | undefined): boolean {
  return g?.status === 'done' && g.textures?.status === 'running';
}

/**
 * The textures to choose from, the final's own first, once the others are made; none before
 * then, or when none could be made.
 */
export function textureChoices(g: TextureGeneration): TextureChoice[] {
  if (g.status !== 'done' || !g.final || g.textures?.status !== 'done') return [];
  if (!g.textures.options.length) return [];
  return [
    { number: 1, url: g.final.url },
    ...g.textures.options.map((option, i) => ({ number: i + 2, url: option.url })),
  ];
}

/** Which choice the scene shows, by its model's URL, or null when it shows none of them. */
export function chosenTexture(
  choices: TextureChoice[],
  sceneUrl: string | null | undefined,
): number | null {
  return choices.find((choice) => choice.url === sceneUrl)?.number ?? null;
}

/**
 * The texture the judge recommends, once the options are made. None when it picked the final's
 * own texture: the scene has that one already, so the panel stays as it is without the judge.
 */
export function recommendedTexture(g: TextureGeneration): RecommendedTexture | null {
  const pick = g.textures?.recommended;
  if (!pick || pick.number === 1) return null;
  const choice = textureChoices(g).find(({ number }) => number === pick.number);
  return choice ? { ...choice, why: pick.why } : null;
}

/**
 * Where the recommended texture stands (see RecommendationStep). The panel applies it once, only
 * in place of the final's own texture, never after the creator chose a texture, and never again
 * after it was applied, even when Undo took it out. `applied` and `chosen` are what this page
 * remembers; the server's record of both comes with the generation.
 */
export function recommendationStep(
  g: TextureGeneration,
  {
    sceneUrl,
    playing,
    applied = false,
    chosen = false,
  }: {
    /** The model file the scene shows for this generation, or null when it isn't in the scene */
    sceneUrl: string | null | undefined;
    playing: boolean;
    applied?: boolean;
    chosen?: boolean;
  },
): RecommendationStep {
  const texture = recommendedTexture(g);
  if (!texture || !g.final) return 'none';
  // The creator's choice stands
  if (chosen || g.textures?.chosen != null) return 'none';
  if (applied || g.textures?.recommended?.applied) {
    return sceneUrl === texture.url ? 'applied' : 'none';
  }
  // Only over the final's own texture: not another model, and not when the model was removed
  if (sceneUrl !== g.final.url) return 'none';
  return playing ? 'wait' : 'apply';
}
