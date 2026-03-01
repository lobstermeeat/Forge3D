import { eq, and, desc, sql, gte } from 'drizzle-orm';
import type { Database } from '../db';
import { schema } from '../db';

export async function getTrending(db: Database, limit: number = 20) {
  const thirtyDaysAgo = new Date();
  thirtyDaysAgo.setDate(thirtyDaysAgo.getDate() - 30);

  const results = await db
    .select({
      id: schema.experiences.id,
      title: schema.experiences.title,
      description: schema.experiences.description,
      slug: schema.experiences.slug,
      status: schema.experiences.status,
      formatType: schema.experiences.formatType,
      thumbnailUrl: schema.experiences.thumbnailUrl,
      previewUrl: schema.experiences.previewUrl,
      viewCount: schema.experiences.viewCount,
      likeCount: schema.experiences.likeCount,
      commentCount: schema.experiences.commentCount,
      polyCount: schema.experiences.polyCount,
      fileSize: schema.experiences.fileSize,
      creatorId: schema.experiences.creatorId,
      categoryId: schema.experiences.categoryId,
      publishedAt: schema.experiences.publishedAt,
      createdAt: schema.experiences.createdAt,
      score: sql<number>`(${schema.experiences.viewCount} + ${schema.experiences.likeCount} * 3 + ${schema.experiences.commentCount} * 5)`.as('score'),
    })
    .from(schema.experiences)
    .where(
      and(
        eq(schema.experiences.status, 'published'),
        gte(schema.experiences.publishedAt, thirtyDaysAgo),
      ),
    )
    .orderBy(desc(sql`score`))
    .limit(limit);

  return results;
}
