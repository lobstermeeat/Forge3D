import { z } from 'zod';
import { eq, and, sql } from 'drizzle-orm';
import { router, publicProcedure, protectedProcedure } from '../trpc';
import { schema } from '../../db';
import { profileSetupSchema, profileUpdateSchema } from '@forge3d/shared';
import type { CreatorProfile } from '@forge3d/shared';

export const profileRouter = router({
  /** Public: get a creator profile by username */
  getByUsername: publicProcedure
    .input(z.object({ username: z.string().min(1) }))
    .query(async ({ ctx, input }) => {
      const [row] = await ctx.db
        .select({
          id: schema.profiles.id,
          username: schema.profiles.username,
          displayName: schema.profiles.displayName,
          bio: schema.profiles.bio,
          websiteUrl: schema.profiles.websiteUrl,
          avatarUrl: schema.profiles.avatarUrl,
          headerUrl: schema.profiles.headerUrl,
          followerCount: schema.profiles.followerCount,
          followingCount: schema.profiles.followingCount,
        })
        .from(schema.profiles)
        .where(eq(schema.profiles.username, input.username));

      if (!row) return null;

      // Check if caller follows this user
      let isFollowing = false;
      if (ctx.user) {
        const [follow] = await ctx.db
          .select({ followerId: schema.follows.followerId })
          .from(schema.follows)
          .where(
            and(
              eq(schema.follows.followerId, ctx.user.id),
              eq(schema.follows.followingId, row.id),
            ),
          );
        isFollowing = !!follow;
      }

      const profile: CreatorProfile = {
        id: row.id,
        username: row.username,
        displayName: row.displayName ?? row.username,
        bio: row.bio,
        websiteUrl: row.websiteUrl,
        avatarUrl: row.avatarUrl,
        headerUrl: row.headerUrl,
        followerCount: row.followerCount,
        followingCount: row.followingCount,
        isFollowing,
      };

      return profile;
    }),

  /** Auth: update own profile */
  update: protectedProcedure
    .input(profileUpdateSchema)
    .mutation(async ({ ctx, input }) => {
      // Upsert: update if exists, insert if not
      const [existing] = await ctx.db
        .select({ id: schema.profiles.id })
        .from(schema.profiles)
        .where(eq(schema.profiles.id, ctx.user.id));

      if (existing) {
        const [updated] = await ctx.db
          .update(schema.profiles)
          .set({
            ...(input.displayName !== undefined && { displayName: input.displayName }),
            ...(input.bio !== undefined && { bio: input.bio || null }),
            ...(input.websiteUrl !== undefined && { websiteUrl: input.websiteUrl || null }),
            ...(input.avatarUrl !== undefined && { avatarUrl: input.avatarUrl || null }),
            updatedAt: new Date(),
          })
          .where(eq(schema.profiles.id, ctx.user.id))
          .returning();
        return updated;
      }

      // No profile yet — create with a temporary username
      const tempUsername = `user-${ctx.user.id.slice(0, 8)}`;
      const [created] = await ctx.db
        .insert(schema.profiles)
        .values({
          id: ctx.user.id,
          username: tempUsername,
          displayName: input.displayName ?? null,
          bio: input.bio ?? null,
          websiteUrl: input.websiteUrl ?? null,
          avatarUrl: input.avatarUrl ?? null,
        })
        .returning();
      return created;
    }),

  /** Auth: set or update username */
  setupUsername: protectedProcedure
    .input(profileSetupSchema)
    .mutation(async ({ ctx, input }) => {
      // Check uniqueness
      const [taken] = await ctx.db
        .select({ id: schema.profiles.id })
        .from(schema.profiles)
        .where(eq(schema.profiles.username, input.username));

      if (taken && taken.id !== ctx.user.id) {
        throw new Error('Username is already taken');
      }

      // Upsert profile with username
      const [existing] = await ctx.db
        .select({ id: schema.profiles.id })
        .from(schema.profiles)
        .where(eq(schema.profiles.id, ctx.user.id));

      if (existing) {
        const [updated] = await ctx.db
          .update(schema.profiles)
          .set({ username: input.username, updatedAt: new Date() })
          .where(eq(schema.profiles.id, ctx.user.id))
          .returning();
        return updated;
      }

      const [created] = await ctx.db
        .insert(schema.profiles)
        .values({
          id: ctx.user.id,
          username: input.username,
        })
        .returning();
      return created;
    }),

  /** Auth: get own profile */
  me: protectedProcedure.query(async ({ ctx }) => {
    const [profile] = await ctx.db
      .select()
      .from(schema.profiles)
      .where(eq(schema.profiles.id, ctx.user.id));

    return profile ?? null;
  }),
});
