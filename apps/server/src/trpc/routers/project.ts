import { z } from 'zod';
import { eq, desc } from 'drizzle-orm';
import { router, protectedProcedure } from '../trpc';
import { schema } from '../../db';
import { createProjectSchema, updateProjectSchema } from '@forge3d/shared';

export const projectRouter = router({
  list: protectedProcedure.query(async ({ ctx }) => {
    return ctx.db
      .select()
      .from(schema.projects)
      .where(eq(schema.projects.userId, ctx.user.id))
      .orderBy(desc(schema.projects.updatedAt));
  }),

  getById: protectedProcedure
    .input(z.object({ id: z.string().uuid() }))
    .query(async ({ ctx, input }) => {
      const [project] = await ctx.db
        .select()
        .from(schema.projects)
        .where(eq(schema.projects.id, input.id));

      if (!project || project.userId !== ctx.user.id) {
        return null;
      }

      const projectScenes = await ctx.db
        .select()
        .from(schema.scenes)
        .where(eq(schema.scenes.projectId, input.id))
        .orderBy(desc(schema.scenes.updatedAt));

      return { ...project, scenes: projectScenes };
    }),

  create: protectedProcedure
    .input(createProjectSchema)
    .mutation(async ({ ctx, input }) => {
      const [project] = await ctx.db
        .insert(schema.projects)
        .values({
          userId: ctx.user.id,
          name: input.name,
          description: input.description,
        })
        .returning();

      return project;
    }),

  update: protectedProcedure
    .input(z.object({ id: z.string().uuid() }).merge(updateProjectSchema))
    .mutation(async ({ ctx, input }) => {
      const { id, ...data } = input;
      const [updated] = await ctx.db
        .update(schema.projects)
        .set({ ...data, updatedAt: new Date() })
        .where(eq(schema.projects.id, id))
        .returning();

      return updated;
    }),

  delete: protectedProcedure
    .input(z.object({ id: z.string().uuid() }))
    .mutation(async ({ ctx, input }) => {
      await ctx.db.delete(schema.projects).where(eq(schema.projects.id, input.id));
      return { success: true };
    }),
});
