import { eq, and, desc, sql } from 'drizzle-orm';
import { router, publicProcedure, protectedProcedure } from '../trpc';
import { schema } from '../../db';
import { followInputSchema, followerListSchema } from '@forge3d/shared';

export const socialRouter = router({
  /** Auth: follow a user */
  follow: protectedProcedure
    .input(followInputSchema)
    .mutation(async ({ ctx, input }) => {
      if (input.userId === ctx.user.id) {
        throw new Error('Cannot follow yourself');
      }

      // Insert follow (ignore if already following)
      const inserted = await ctx.db
        .insert(schema.follows)
        .values({
          followerId: ctx.user.id,
          followingId: input.userId,
        })
        .onConflictDoNothing()
        .returning();

      // Only increment counters if a row was actually inserted
      if (inserted.length > 0) {
        // Increment follower count on target
        await ctx.db
          .update(schema.profiles)
          .set({
            followerCount: sql`${schema.profiles.followerCount} + 1`,
            updatedAt: new Date(),
          })
          .where(eq(schema.profiles.id, input.userId));

        // Increment following count on self
        await ctx.db
          .update(schema.profiles)
          .set({
            followingCount: sql`${schema.profiles.followingCount} + 1`,
            updatedAt: new Date(),
          })
          .where(eq(schema.profiles.id, ctx.user.id));
      }

      return { following: true };
    }),

  /** Auth: unfollow a user */
  unfollow: protectedProcedure
    .input(followInputSchema)
    .mutation(async ({ ctx, input }) => {
      const deleted = await ctx.db
        .delete(schema.follows)
        .where(
          and(
            eq(schema.follows.followerId, ctx.user.id),
            eq(schema.follows.followingId, input.userId),
          ),
        )
        .returning();

      // Only decrement counters if a row was actually deleted
      if (deleted.length > 0) {
        await ctx.db
          .update(schema.profiles)
          .set({
            followerCount: sql`GREATEST(${schema.profiles.followerCount} - 1, 0)`,
            updatedAt: new Date(),
          })
          .where(eq(schema.profiles.id, input.userId));

        await ctx.db
          .update(schema.profiles)
          .set({
            followingCount: sql`GREATEST(${schema.profiles.followingCount} - 1, 0)`,
            updatedAt: new Date(),
          })
          .where(eq(schema.profiles.id, ctx.user.id));
      }

      return { following: false };
    }),

  /** Public: list followers of a user */
  getFollowers: publicProcedure
    .input(followerListSchema)
    .query(async ({ ctx, input }) => {
      const results = await ctx.db
        .select({
          id: schema.profiles.id,
          username: schema.profiles.username,
          displayName: schema.profiles.displayName,
          avatarUrl: schema.profiles.avatarUrl,
          followedAt: schema.follows.createdAt,
        })
        .from(schema.follows)
        .innerJoin(schema.profiles, eq(schema.follows.followerId, schema.profiles.id))
        .where(eq(schema.follows.followingId, input.userId))
        .orderBy(desc(schema.follows.createdAt))
        .limit(input.limit + 1);

      const hasMore = results.length > input.limit;
      const items = hasMore ? results.slice(0, input.limit) : results;

      return {
        items: items.map((r) => ({
          id: r.id,
          username: r.username,
          displayName: r.displayName ?? r.username,
          avatarUrl: r.avatarUrl,
        })),
        nextCursor: hasMore ? items[items.length - 1]?.id : null,
        hasMore,
      };
    }),

  /** Public: list who a user is following */
  getFollowing: publicProcedure
    .input(followerListSchema)
    .query(async ({ ctx, input }) => {
      const results = await ctx.db
        .select({
          id: schema.profiles.id,
          username: schema.profiles.username,
          displayName: schema.profiles.displayName,
          avatarUrl: schema.profiles.avatarUrl,
          followedAt: schema.follows.createdAt,
        })
        .from(schema.follows)
        .innerJoin(schema.profiles, eq(schema.follows.followingId, schema.profiles.id))
        .where(eq(schema.follows.followerId, input.userId))
        .orderBy(desc(schema.follows.createdAt))
        .limit(input.limit + 1);

      const hasMore = results.length > input.limit;
      const items = hasMore ? results.slice(0, input.limit) : results;

      return {
        items: items.map((r) => ({
          id: r.id,
          username: r.username,
          displayName: r.displayName ?? r.username,
          avatarUrl: r.avatarUrl,
        })),
        nextCursor: hasMore ? items[items.length - 1]?.id : null,
        hasMore,
      };
    }),
});
