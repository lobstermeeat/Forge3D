import { eq, and, desc, sql } from 'drizzle-orm';
import { TRPCError } from '@trpc/server';
import { router, protectedProcedure, adminProcedure } from '../trpc';
import { schema } from '../../db';
import { reportInputSchema, reportListSchema, reviewReportSchema } from '@forge3d/shared';
import { removeExperience } from '../../services/search';
import type { Report } from '@forge3d/shared';

export const moderationRouter = router({
  report: protectedProcedure
    .input(reportInputSchema)
    .mutation(async ({ ctx, input }) => {
      // Prevent duplicate reports from same user on same target
      const [existing] = await ctx.db
        .select({ id: schema.reports.id })
        .from(schema.reports)
        .where(
          and(
            eq(schema.reports.reporterId, ctx.user.id),
            eq(schema.reports.targetType, input.targetType),
            eq(schema.reports.targetId, input.targetId),
          ),
        );

      if (existing) {
        return { reported: false, message: 'Already reported' };
      }

      await ctx.db.insert(schema.reports).values({
        reporterId: ctx.user.id,
        targetType: input.targetType,
        targetId: input.targetId,
        reason: input.reason,
      });

      return { reported: true };
    }),

  listReports: adminProcedure
    .input(reportListSchema)
    .query(async ({ ctx, input }) => {
      const conditions = [];
      if (input.status) {
        conditions.push(eq(schema.reports.status, input.status));
      }

      const results = await ctx.db
        .select({
          id: schema.reports.id,
          reporterId: schema.reports.reporterId,
          reporterName: schema.users.name,
          targetType: schema.reports.targetType,
          targetId: schema.reports.targetId,
          reason: schema.reports.reason,
          status: schema.reports.status,
          createdAt: schema.reports.createdAt,
          reviewedAt: schema.reports.reviewedAt,
        })
        .from(schema.reports)
        .leftJoin(schema.users, eq(schema.reports.reporterId, schema.users.id))
        .where(conditions.length > 0 ? and(...conditions) : undefined)
        .orderBy(desc(schema.reports.createdAt))
        .limit(input.limit + 1);

      const hasMore = results.length > input.limit;
      const items = hasMore ? results.slice(0, input.limit) : results;

      const reports: Report[] = items.map((r) => ({
        id: r.id,
        reporterId: r.reporterId,
        reporterName: r.reporterName ?? 'Unknown',
        targetType: r.targetType as Report['targetType'],
        targetId: r.targetId,
        reason: r.reason,
        status: r.status as Report['status'],
        createdAt: r.createdAt.toISOString(),
        reviewedAt: r.reviewedAt?.toISOString(),
      }));

      return {
        items: reports,
        nextCursor: hasMore ? items[items.length - 1]?.id : null,
        hasMore,
      };
    }),

  reviewReport: adminProcedure
    .input(reviewReportSchema)
    .mutation(async ({ ctx, input }) => {
      const [report] = await ctx.db
        .select()
        .from(schema.reports)
        .where(eq(schema.reports.id, input.reportId));

      if (!report) {
        throw new TRPCError({ code: 'NOT_FOUND', message: 'Report not found' });
      }

      const newStatus = input.action === 'resolve' ? 'resolved' : 'dismissed';

      await ctx.db
        .update(schema.reports)
        .set({
          status: newStatus,
          reviewedBy: ctx.user.id,
          reviewedAt: new Date(),
        })
        .where(eq(schema.reports.id, input.reportId));

      // If resolving, remove the offending content
      if (input.action === 'resolve') {
        if (report.targetType === 'experience') {
          await ctx.db
            .update(schema.experiences)
            .set({ status: 'removed', updatedAt: new Date() })
            .where(eq(schema.experiences.id, report.targetId));

          // Remove from search index
          await removeExperience(report.targetId);
        } else if (report.targetType === 'comment') {
          await ctx.db
            .update(schema.comments)
            .set({ status: 'removed', updatedAt: new Date() })
            .where(eq(schema.comments.id, report.targetId));
        }
      }

      return { status: newStatus };
    }),

  getStats: adminProcedure.query(async ({ ctx }) => {
    const [stats] = await ctx.db
      .select({
        pending: sql<number>`count(*) filter (where ${schema.reports.status} = 'pending')::int`,
        resolved: sql<number>`count(*) filter (where ${schema.reports.status} = 'resolved')::int`,
        dismissed: sql<number>`count(*) filter (where ${schema.reports.status} = 'dismissed')::int`,
        total: sql<number>`count(*)::int`,
      })
      .from(schema.reports);

    return stats ?? { pending: 0, resolved: 0, dismissed: 0, total: 0 };
  }),
});
