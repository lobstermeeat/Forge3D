import { z } from 'zod';
import { eq } from 'drizzle-orm';
import { TRPCError } from '@trpc/server';
import { router, publicProcedure, protectedProcedure } from '../trpc';
import { schema } from '../../db';
import { getStudio, StudioError } from '../../services/ai';
import { MAX_PROMPT_LENGTH, TEXTURE_COUNT } from '../../services/ai/studio';

/** Photos arrive as data URLs; the client scales them to 2048 px first, so this is generous. */
const MAX_PHOTO_DATA_URL = 12 * 1024 * 1024;
const PHOTO_DATA_URL = /^data:image\/(png|jpeg|webp);base64,/;

const id = z.object({ id: z.string().uuid() });

/**
 * The Studio's AI panel: prompt or photo -> pick a picture -> preview -> keep -> final, then
 * texture options for the final (with AI_TEXTURE_JUDGE=1, the judge's pick among them).
 */
export const aiRouter = router({
  capabilities: publicProcedure.query(() => getStudio().capabilities()),

  start: protectedProcedure
    .input(
      z.union([
        z.object({
          prompt: z.string().min(1).max(MAX_PROMPT_LENGTH),
          sceneId: z.string().uuid().optional(),
        }),
        z.object({
          photo: z
            .string()
            .max(MAX_PHOTO_DATA_URL)
            .regex(PHOTO_DATA_URL, 'Use a PNG, JPEG or WebP picture'),
          sceneId: z.string().uuid().optional(),
        }),
      ]),
    )
    .mutation(async ({ ctx, input }) => {
      // The scene is only recorded for reference; an unknown one is ignored
      let sceneId: string | undefined;
      if (input.sceneId) {
        const [scene] = await ctx.db
          .select({ id: schema.scenes.id })
          .from(schema.scenes)
          .where(eq(schema.scenes.id, input.sceneId));
        sceneId = scene?.id;
      }
      return studioCall(() =>
        'prompt' in input
          ? getStudio().startFromPrompt(ctx.user.id, input.prompt, sceneId)
          : getStudio().startFromPhoto(
              ctx.user.id,
              Buffer.from(input.photo.slice(input.photo.indexOf(',') + 1), 'base64'),
              sceneId,
            ),
      );
    }),

  /**
   * Polled while a step runs, and while a done final's texture options are made: moves the
   * generation on when its job has finished. The options come back in `textures`.
   */
  get: protectedProcedure
    .input(id)
    .query(({ ctx, input }) => studioCall(() => getStudio().get(ctx.user.id, input.id))),

  pick: protectedProcedure
    .input(id.extend({ index: z.number().int().min(0).max(3) }))
    .mutation(({ ctx, input }) =>
      studioCall(() => getStudio().pick(ctx.user.id, input.id, input.index)),
    ),

  keep: protectedProcedure
    .input(id)
    .mutation(({ ctx, input }) => studioCall(() => getStudio().keep(ctx.user.id, input.id))),

  retry: protectedProcedure
    .input(id)
    .mutation(({ ctx, input }) => studioCall(() => getStudio().retry(ctx.user.id, input.id))),

  /**
   * Records a texture option the panel put in the scene (1 is the final's own): one the creator
   * chose, or the judge's pick, which the panel applies once. Returns the generation.
   */
  chooseTexture: protectedProcedure
    .input(
      id.extend({
        number: z
          .number()
          .int()
          .min(1)
          .max(TEXTURE_COUNT + 1),
        by: z.enum(['creator', 'judge']),
      }),
    )
    .mutation(({ ctx, input }) =>
      studioCall(() => getStudio().chooseTexture(ctx.user.id, input.id, input.number, input.by)),
    ),

  recent: protectedProcedure.query(({ ctx }) => studioCall(() => getStudio().recent(ctx.user.id))),

  /**
   * Starts a GPU before its job is sent, without waiting for it: FLUX while the user types, and
   * the multiview worker (AI_MULTIVIEW=1) while they choose a photo.
   */
  warm: protectedProcedure
    .input(z.object({ worker: z.enum(['references', 'multiview', 'model']) }))
    .mutation(({ ctx, input }) => {
      getStudio().warm(ctx.user.id, input.worker);
    }),
});

async function studioCall<T>(call: () => Promise<T>): Promise<T> {
  try {
    return await call();
  } catch (err) {
    if (err instanceof StudioError) {
      const code =
        err.code === 'NOT_FOUND'
          ? 'NOT_FOUND'
          : err.code === 'UNAVAILABLE'
            ? 'PRECONDITION_FAILED'
            : 'BAD_REQUEST';
      throw new TRPCError({ code, message: err.message });
    }
    throw err;
  }
}
