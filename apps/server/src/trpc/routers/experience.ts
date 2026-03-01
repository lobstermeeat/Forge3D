import { z } from 'zod';
import { eq, desc, and, sql, inArray } from 'drizzle-orm';
import { nanoid } from 'nanoid';
import { router, publicProcedure, protectedProcedure } from '../trpc';
import { schema } from '../../db';
import { publishExperienceSchema, experienceListSchema } from '@forge3d/shared';
import { getStorage } from '../../services/storage';
import { processThumbnail } from '../../services/thumbnail';
import { indexExperience, removeExperience } from '../../services/search';
import { getTrending } from '../../services/trending';
import type { ExperienceMeta, ExperienceData } from '@forge3d/shared';

export const experienceRouter = router({
  /** Public: get a published experience by slug */
  getBySlug: publicProcedure
    .input(z.object({ slug: z.string().min(1) }))
    .query(async ({ ctx, input }) => {
      const [exp] = await ctx.db
        .select()
        .from(schema.experiences)
        .where(
          and(
            eq(schema.experiences.slug, input.slug),
            eq(schema.experiences.status, 'published'),
          ),
        );

      if (!exp) return null;

      // Fetch creator profile
      const [creator] = await ctx.db
        .select({
          id: schema.users.id,
          username: schema.profiles.username,
          displayName: schema.profiles.displayName,
          avatarUrl: schema.profiles.avatarUrl,
        })
        .from(schema.users)
        .leftJoin(schema.profiles, eq(schema.users.id, schema.profiles.id))
        .where(eq(schema.users.id, exp.creatorId));

      const meta: ExperienceMeta = {
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

      return {
        meta,
        experienceData: exp.experienceData as unknown as ExperienceData,
        sceneUrl: exp.sceneUrl,
      };
    }),

  /** Public: increment view count (fire-and-forget, called once per page load) */
  recordView: publicProcedure
    .input(z.object({ id: z.string().uuid() }))
    .mutation(async ({ ctx, input }) => {
      await ctx.db
        .update(schema.experiences)
        .set({ viewCount: sql`${schema.experiences.viewCount} + 1` })
        .where(eq(schema.experiences.id, input.id));
      return { ok: true };
    }),

  /** Auth: publish a scene as an experience */
  publish: protectedProcedure
    .input(publishExperienceSchema)
    .mutation(async ({ ctx, input }) => {
      // Fetch scene data
      const [scene] = await ctx.db
        .select()
        .from(schema.scenes)
        .where(eq(schema.scenes.id, input.sceneId));

      if (!scene) {
        throw new Error('Scene not found');
      }

      // Generate slug
      const slug = nanoid(10);
      const id = crypto.randomUUID();

      // Build experience data wrapper
      const sceneData = scene.data as Record<string, unknown>;
      const experienceData: ExperienceData = {
        version: 1,
        scene: sceneData as unknown as ExperienceData['scene'],
        camera: {
          position: [5, 5, 5],
          target: [0, 0, 0],
          fov: 60,
        },
        environment: {
          backgroundColor: [0.067, 0.067, 0.106], // #11111b
          ambientIntensity: 0.4,
        },
        format: input.formatConfig ?? { type: 'turntable', speed: 1, axis: 'y' },
      };

      // Store scene bundle
      const storage = getStorage();
      const sceneJson = JSON.stringify(experienceData);
      const sceneBuffer = Buffer.from(sceneJson, 'utf-8');
      const sceneUrl = await storage.write(
        `experiences/${id}/scene.json`,
        sceneBuffer,
        'application/json',
      );

      // Process thumbnail if provided
      let thumbnailUrl: string | null = null;
      let previewUrl: string | null = null;
      if (input.thumbnail) {
        const result = await processThumbnail(input.thumbnail, id);
        thumbnailUrl = result.thumbnailUrl;
        previewUrl = result.previewUrl;
      }

      // Count polygons (rough estimate from entity count)
      const entities = (sceneData as { entities?: unknown[] }).entities ?? [];
      const polyCount = entities.length * 1000; // rough estimate

      // Get project ID from scene
      const projectId = scene.projectId;

      // Create experience record
      const [experience] = await ctx.db
        .insert(schema.experiences)
        .values({
          id,
          creatorId: ctx.user.id,
          projectId,
          sceneId: input.sceneId,
          title: input.title,
          description: input.description ?? null,
          slug,
          status: 'published',
          formatType: input.formatType,
          sceneUrl,
          thumbnailUrl,
          previewUrl,
          experienceData: experienceData as unknown as Record<string, unknown>,
          viewCount: 0,
          likeCount: 0,
          commentCount: 0,
          polyCount,
          fileSize: sceneBuffer.length,
          categoryId: input.categoryId ?? null,
          publishedAt: new Date(),
        })
        .returning();

      // Store tags
      if (input.tags.length > 0) {
        await ctx.db.insert(schema.experienceTags).values(
          input.tags.map((tag) => ({
            experienceId: id,
            tag,
          })),
        );
      }

      // Fan-out to followers' feeds
      const followers = await ctx.db
        .select({ followerId: schema.follows.followerId })
        .from(schema.follows)
        .where(eq(schema.follows.followingId, ctx.user.id));

      if (followers.length > 0) {
        await ctx.db.insert(schema.feedEntries).values(
          followers.map((f) => ({
            userId: f.followerId,
            experienceId: id,
          })),
        );
      }

      // Index in Meilisearch (fire-and-forget)
      const [profile] = await ctx.db
        .select({ displayName: schema.profiles.displayName, username: schema.profiles.username })
        .from(schema.profiles)
        .where(eq(schema.profiles.id, ctx.user.id));

      indexExperience({
        id,
        title: input.title,
        description: input.description ?? null,
        tags: input.tags,
        creatorId: ctx.user.id,
        creatorName: profile?.displayName ?? profile?.username ?? 'Unknown',
        categoryId: input.categoryId ?? null,
        formatType: input.formatType,
        publishedAt: new Date().toISOString(),
        viewCount: 0,
        likeCount: 0,
      }).catch(() => {});

      return { slug, id, experience };
    }),

  /** Auth: unpublish an experience */
  unpublish: protectedProcedure
    .input(z.object({ id: z.string().uuid() }))
    .mutation(async ({ ctx, input }) => {
      const [updated] = await ctx.db
        .update(schema.experiences)
        .set({ status: 'unpublished', updatedAt: new Date() })
        .where(
          and(
            eq(schema.experiences.id, input.id),
            eq(schema.experiences.creatorId, ctx.user.id),
          ),
        )
        .returning();

      if (updated) {
        removeExperience(input.id).catch(() => {});
      }

      return updated ?? null;
    }),

  /** Public: list experiences by creator */
  listByCreator: publicProcedure
    .input(
      z.object({
        userId: z.string().uuid(),
        cursor: z.string().optional(),
        limit: z.number().int().min(1).max(50).default(20),
      }),
    )
    .query(async ({ ctx, input }) => {
      const results = await ctx.db
        .select()
        .from(schema.experiences)
        .where(
          and(
            eq(schema.experiences.creatorId, input.userId),
            eq(schema.experiences.status, 'published'),
          ),
        )
        .orderBy(desc(schema.experiences.publishedAt))
        .limit(input.limit + 1);

      const hasMore = results.length > input.limit;
      const items = hasMore ? results.slice(0, input.limit) : results;

      return {
        items,
        nextCursor: hasMore ? items[items.length - 1]?.id : null,
        hasMore,
      };
    }),

  /** Public: explore all published experiences */
  listExplore: publicProcedure
    .input(experienceListSchema)
    .query(async ({ ctx, input }) => {
      const conditions = [eq(schema.experiences.status, 'published')];

      if (input.categoryId) {
        conditions.push(eq(schema.experiences.categoryId, input.categoryId));
      }

      const results = await ctx.db
        .select()
        .from(schema.experiences)
        .where(and(...conditions))
        .orderBy(desc(schema.experiences.publishedAt))
        .limit(input.limit + 1);

      const hasMore = results.length > input.limit;
      const items = hasMore ? results.slice(0, input.limit) : results;

      return {
        items,
        nextCursor: hasMore ? items[items.length - 1]?.id : null,
        hasMore,
      };
    }),

  /** Auth: list experiences from followed creators */
  listFollowing: protectedProcedure
    .input(experienceListSchema)
    .query(async ({ ctx, input }) => {
      const results = await ctx.db
        .select({
          id: schema.feedEntries.id,
          experienceId: schema.feedEntries.experienceId,
          createdAt: schema.feedEntries.createdAt,
        })
        .from(schema.feedEntries)
        .where(eq(schema.feedEntries.userId, ctx.user.id))
        .orderBy(desc(schema.feedEntries.createdAt))
        .limit(input.limit + 1);

      const hasMore = results.length > input.limit;
      const items = hasMore ? results.slice(0, input.limit) : results;

      // Fetch full experience data for these IDs
      if (items.length === 0) {
        return { items: [], nextCursor: null, hasMore: false };
      }

      const experienceIds = items.map((i) => i.experienceId);
      const experiences = await ctx.db
        .select()
        .from(schema.experiences)
        .where(
          and(
            eq(schema.experiences.status, 'published'),
            inArray(schema.experiences.id, experienceIds),
          ),
        );

      return {
        items: experiences,
        nextCursor: hasMore ? items[items.length - 1]?.id : null,
        hasMore,
      };
    }),

  /** Public: list trending experiences */
  listTrending: publicProcedure
    .input(z.object({ limit: z.number().int().min(1).max(50).default(20) }))
    .query(async ({ ctx, input }) => {
      const results = await getTrending(ctx.db, input.limit);

      // Fetch creator info
      const creatorIds = [...new Set(results.map((e) => e.creatorId))];
      const creators = creatorIds.length > 0
        ? await ctx.db
            .select({
              id: schema.users.id,
              username: schema.profiles.username,
              displayName: schema.profiles.displayName,
              avatarUrl: schema.profiles.avatarUrl,
            })
            .from(schema.users)
            .leftJoin(schema.profiles, eq(schema.users.id, schema.profiles.id))
            .where(inArray(schema.users.id, creatorIds))
        : [];

      const creatorMap = new Map(creators.map((c) => [c.id, c]));

      const items: ExperienceMeta[] = results.map((exp) => {
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
      });

      return { items };
    }),
});
