import { z } from 'zod';
import { eq } from 'drizzle-orm';
import { router, protectedProcedure } from '../trpc';
import { schema } from '../../db';
import { SCENE_VERSION } from '@forge3d/shared';

export const sceneRouter = router({
  getById: protectedProcedure
    .input(z.object({ id: z.string().uuid() }))
    .query(async ({ ctx, input }) => {
      const [scene] = await ctx.db
        .select()
        .from(schema.scenes)
        .where(eq(schema.scenes.id, input.id));

      return scene ?? null;
    }),

  create: protectedProcedure
    .input(
      z.object({
        projectId: z.string().uuid(),
        name: z.string().min(1).max(200),
      }),
    )
    .mutation(async ({ ctx, input }) => {
      const [scene] = await ctx.db
        .insert(schema.scenes)
        .values({
          projectId: input.projectId,
          name: input.name,
          version: SCENE_VERSION,
          data: { version: SCENE_VERSION, entities: [], materials: [] },
        })
        .returning();

      return scene;
    }),

  save: protectedProcedure
    .input(
      z.object({
        id: z.string().uuid(),
        data: z.record(z.unknown()),
      }),
    )
    .mutation(async ({ ctx, input }) => {
      const [updated] = await ctx.db
        .update(schema.scenes)
        .set({ data: input.data, updatedAt: new Date() })
        .where(eq(schema.scenes.id, input.id))
        .returning();

      return updated;
    }),

  delete: protectedProcedure
    .input(z.object({ id: z.string().uuid() }))
    .mutation(async ({ ctx, input }) => {
      await ctx.db.delete(schema.scenes).where(eq(schema.scenes.id, input.id));
      return { success: true };
    }),
});
