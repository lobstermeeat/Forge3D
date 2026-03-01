import { eq, and, or, sql, desc, ilike, inArray } from 'drizzle-orm';
import { router, publicProcedure } from '../trpc';
import { db as dbInstance, schema } from '../../db';
import { searchExperiencesSchema } from '@forge3d/shared';
import { searchExperiences as meiliSearch } from '../../services/search';
import type { ExperienceMeta } from '@forge3d/shared';

function buildCreatorMeta(
  exp: typeof schema.experiences.$inferSelect,
  creatorMap: Map<string, { id: string; username: string | null; displayName: string | null; avatarUrl: string | null }>,
): ExperienceMeta {
  const creator = creatorMap.get(exp.creatorId);
  return {
    id: exp.id,
    title: exp.title,
    description: exp.description,
    slug: exp.slug,
    status: exp.status as ExperienceMeta['status'],
    formatType: exp.formatType as ExperienceMeta['formatType'],
    thumbnailUrl: exp.thumbnailUrl,
    previewUrl: exp.previewUrl,
    viewCount: exp.viewCount,
    likeCount: exp.likeCount,
    commentCount: exp.commentCount,
    polyCount: exp.polyCount,
    fileSize: exp.fileSize,
    creator: {
      id: creator?.id ?? exp.creatorId,
      username: creator?.username ?? 'unknown',
      displayName: creator?.displayName ?? creator?.username ?? 'Unknown',
      avatarUrl: creator?.avatarUrl ?? null,
    },
    categoryId: exp.categoryId,
    publishedAt: exp.publishedAt?.toISOString() ?? null,
    createdAt: exp.createdAt.toISOString(),
  };
}

async function fetchCreatorMap(
  db: typeof dbInstance,
  creatorIds: string[],
) {
  if (creatorIds.length === 0) return new Map<string, { id: string; username: string | null; displayName: string | null; avatarUrl: string | null }>();

  const creators = await db
    .select({
      id: schema.users.id,
      username: schema.profiles.username,
      displayName: schema.profiles.displayName,
      avatarUrl: schema.profiles.avatarUrl,
    })
    .from(schema.users)
    .leftJoin(schema.profiles, eq(schema.users.id, schema.profiles.id))
    .where(inArray(schema.users.id, creatorIds));

  return new Map(creators.map((c) => [c.id, c]));
}

export const searchRouter = router({
  experiences: publicProcedure
    .input(searchExperiencesSchema)
    .query(async ({ ctx, input }) => {
      // Try Meilisearch first
      const meiliResult = await meiliSearch({
        query: input.query,
        categoryId: input.categoryId,
        formatType: input.formatType,
        sort: input.sort,
        limit: input.limit,
        offset: input.offset,
      });

      if (meiliResult.hits.length > 0) {
        const hitIds = meiliResult.hits.map((h) => h.id);
        const experiences = await ctx.db
          .select()
          .from(schema.experiences)
          .where(
            and(
              eq(schema.experiences.status, 'published'),
              inArray(schema.experiences.id, hitIds),
            ),
          );

        const creatorIds = [...new Set(experiences.map((e) => e.creatorId))];
        const creatorMap = await fetchCreatorMap(ctx.db, creatorIds);

        // Preserve Meilisearch ordering
        const idOrder = new Map(hitIds.map((id, i) => [id, i]));
        experiences.sort((a, b) => (idOrder.get(a.id) ?? 0) - (idOrder.get(b.id) ?? 0));

        const items = experiences.map((exp) => buildCreatorMeta(exp, creatorMap));
        return { items, totalHits: meiliResult.totalHits, query: input.query };
      }

      // Fallback: SQL ILIKE search
      const conditions = [
        eq(schema.experiences.status, 'published'),
        or(
          ilike(schema.experiences.title, `%${input.query}%`),
          ilike(schema.experiences.description, `%${input.query}%`),
        ),
      ];

      if (input.categoryId) {
        conditions.push(eq(schema.experiences.categoryId, input.categoryId));
      }

      let orderBy;
      if (input.sort === 'newest') orderBy = desc(schema.experiences.publishedAt);
      else if (input.sort === 'popular') orderBy = desc(schema.experiences.likeCount);
      else orderBy = desc(schema.experiences.publishedAt);

      const results = await ctx.db
        .select()
        .from(schema.experiences)
        .where(and(...conditions))
        .orderBy(orderBy)
        .limit(input.limit)
        .offset(input.offset);

      // Count total
      const [countResult] = await ctx.db
        .select({ count: sql<number>`count(*)::int` })
        .from(schema.experiences)
        .where(and(...conditions));

      const creatorIds = [...new Set(results.map((e) => e.creatorId))];
      const creatorMap = await fetchCreatorMap(ctx.db, creatorIds);

      const items = results.map((exp) => buildCreatorMeta(exp, creatorMap));
      return { items, totalHits: countResult?.count ?? 0, query: input.query };
    }),
});
