// Type only, like AppRouter in api/trpc.ts: no server code is bundled
import type { GenerationView } from '../../../../apps/server/src/services/ai/studio';

/**
 * Texture options: after the final, the server makes more textures for its shape, and the AI
 * panel lets the creator swap the model in their scene between them. Option 1 is the final's
 * own texture.
 */
export interface TextureChoice {
  /** 1 for the final's own texture, then 2, 3, … as the panel numbers them */
  number: number;
  url: string;
}

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
