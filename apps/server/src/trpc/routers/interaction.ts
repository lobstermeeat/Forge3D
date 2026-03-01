import { z } from 'zod';
import { eq, and, desc, sql, isNull, inArray } from 'drizzle-orm';
import { router, publicProcedure, protectedProcedure } from '../trpc';
import { schema } from '../../db';
import {
  likeInputSchema,
  commentInputSchema,
  commentListSchema,
  deleteCommentSchema,
} from '@forge3d/shared';
import type { Comment } from '@forge3d/shared';

export const interactionRouter = router({
  /** Auth: like an experience */
  like: protectedProcedure
    .input(likeInputSchema)
    .mutation(async ({ ctx, input }) => {
      const inserted = await ctx.db
        .insert(schema.likes)
        .values({
          userId: ctx.user.id,
          experienceId: input.experienceId,
        })
        .onConflictDoNothing()
        .returning();

      if (inserted.length > 0) {
        await ctx.db
          .update(schema.experiences)
          .set({ likeCount: sql`${schema.experiences.likeCount} + 1` })
          .where(eq(schema.experiences.id, input.experienceId));
      }

      return { liked: true };
    }),

  /** Auth: unlike an experience */
  unlike: protectedProcedure
    .input(likeInputSchema)
    .mutation(async ({ ctx, input }) => {
      const deleted = await ctx.db
        .delete(schema.likes)
        .where(
          and(
            eq(schema.likes.userId, ctx.user.id),
            eq(schema.likes.experienceId, input.experienceId),
          ),
        )
        .returning();

      if (deleted.length > 0) {
        await ctx.db
          .update(schema.experiences)
          .set({
            likeCount: sql`GREATEST(${schema.experiences.likeCount} - 1, 0)`,
          })
          .where(eq(schema.experiences.id, input.experienceId));
      }

      return { liked: false };
    }),

  /** Auth: check if user liked an experience */
  isLiked: protectedProcedure
    .input(likeInputSchema)
    .query(async ({ ctx, input }) => {
      const [row] = await ctx.db
        .select({ userId: schema.likes.userId })
        .from(schema.likes)
        .where(
          and(
            eq(schema.likes.userId, ctx.user.id),
            eq(schema.likes.experienceId, input.experienceId),
          ),
        );

      return { liked: !!row };
    }),

  /** Auth: post a comment */
  comment: protectedProcedure
    .input(commentInputSchema)
    .mutation(async ({ ctx, input }) => {
      const rows = await ctx.db
        .insert(schema.comments)
        .values({
          experienceId: input.experienceId,
          userId: ctx.user.id,
          body: input.body,
          parentId: input.parentId ?? null,
        })
        .returning();

      const newComment = rows[0]!;

      // Increment comment count on experience
      await ctx.db
        .update(schema.experiences)
        .set({ commentCount: sql`${schema.experiences.commentCount} + 1` })
        .where(eq(schema.experiences.id, input.experienceId));

      // Fetch user info for response
      const [user] = await ctx.db
        .select({
          name: schema.users.name,
          avatarUrl: schema.users.avatarUrl,
        })
        .from(schema.users)
        .where(eq(schema.users.id, ctx.user.id));

      const result: Comment = {
        id: newComment.id,
        body: newComment.body,
        userId: newComment.userId,
        userName: user?.name ?? 'Unknown',
        userAvatar: user?.avatarUrl ?? null,
        parentId: newComment.parentId,
        likeCount: 0,
        createdAt: newComment.createdAt.toISOString(),
      };

      return result;
    }),

  /** Auth: delete a comment (own or admin) */
  deleteComment: protectedProcedure
    .input(deleteCommentSchema)
    .mutation(async ({ ctx, input }) => {
      // Fetch the comment
      const [comment] = await ctx.db
        .select()
        .from(schema.comments)
        .where(eq(schema.comments.id, input.commentId));

      if (!comment) {
        throw new Error('Comment not found');
      }

      // Check ownership or admin role
      const [user] = await ctx.db
        .select({ role: schema.users.role })
        .from(schema.users)
        .where(eq(schema.users.id, ctx.user.id));

      if (comment.userId !== ctx.user.id && user?.role !== 'admin') {
        throw new Error('Not authorized to delete this comment');
      }

      // Soft-delete: set status to 'removed'
      await ctx.db
        .update(schema.comments)
        .set({ status: 'removed', updatedAt: new Date() })
        .where(eq(schema.comments.id, input.commentId));

      // Decrement comment count
      await ctx.db
        .update(schema.experiences)
        .set({
          commentCount: sql`GREATEST(${schema.experiences.commentCount} - 1, 0)`,
        })
        .where(eq(schema.experiences.id, comment.experienceId));

      return { success: true };
    }),

  /** Public: list comments for an experience */
  listComments: publicProcedure
    .input(commentListSchema)
    .query(async ({ ctx, input }) => {
      const orderCol =
        input.sort === 'popular'
          ? desc(schema.comments.likeCount)
          : desc(schema.comments.createdAt);

      // Fetch top-level comments
      const topLevel = await ctx.db
        .select({
          id: schema.comments.id,
          body: schema.comments.body,
          userId: schema.comments.userId,
          parentId: schema.comments.parentId,
          likeCount: schema.comments.likeCount,
          createdAt: schema.comments.createdAt,
          userName: schema.users.name,
          userAvatar: schema.users.avatarUrl,
        })
        .from(schema.comments)
        .innerJoin(schema.users, eq(schema.comments.userId, schema.users.id))
        .where(
          and(
            eq(schema.comments.experienceId, input.experienceId),
            isNull(schema.comments.parentId),
            eq(schema.comments.status, 'visible'),
          ),
        )
        .orderBy(orderCol)
        .limit(input.limit + 1);

      const hasMore = topLevel.length > input.limit;
      const items = hasMore ? topLevel.slice(0, input.limit) : topLevel;

      if (items.length === 0) {
        return { items: [], nextCursor: null, hasMore: false };
      }

      // Fetch replies for these top-level comments
      const parentIds = items.map((c) => c.id);
      const replies = await ctx.db
        .select({
          id: schema.comments.id,
          body: schema.comments.body,
          userId: schema.comments.userId,
          parentId: schema.comments.parentId,
          likeCount: schema.comments.likeCount,
          createdAt: schema.comments.createdAt,
          userName: schema.users.name,
          userAvatar: schema.users.avatarUrl,
        })
        .from(schema.comments)
        .innerJoin(schema.users, eq(schema.comments.userId, schema.users.id))
        .where(
          and(
            eq(schema.comments.experienceId, input.experienceId),
            eq(schema.comments.status, 'visible'),
            inArray(schema.comments.parentId, parentIds),
          ),
        )
        .orderBy(desc(schema.comments.createdAt));

      // Group replies by parentId
      const replyMap = new Map<string, Comment[]>();
      for (const r of replies) {
        const comment: Comment = {
          id: r.id,
          body: r.body,
          userId: r.userId,
          userName: r.userName,
          userAvatar: r.userAvatar,
          parentId: r.parentId,
          likeCount: r.likeCount,
          createdAt: r.createdAt.toISOString(),
        };
        const existing = replyMap.get(r.parentId!) ?? [];
        existing.push(comment);
        replyMap.set(r.parentId!, existing);
      }

      // Build final comment list
      const comments: Comment[] = items.map((c) => ({
        id: c.id,
        body: c.body,
        userId: c.userId,
        userName: c.userName,
        userAvatar: c.userAvatar,
        parentId: c.parentId,
        likeCount: c.likeCount,
        createdAt: c.createdAt.toISOString(),
        replies: replyMap.get(c.id) ?? [],
      }));

      return {
        items: comments,
        nextCursor: hasMore ? items[items.length - 1]?.id : null,
        hasMore,
      };
    }),
});
