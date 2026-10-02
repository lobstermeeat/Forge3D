import { pgTable, text, timestamp, uuid, jsonb, integer, boolean, primaryKey, index, customType } from 'drizzle-orm/pg-core';

const bytea = customType<{ data: Buffer }>({
  dataType() { return 'bytea'; },
});

export const users = pgTable('users', {
  id: text('id').primaryKey(),
  email: text('email').notNull().unique(),
  name: text('name').notNull(),
  avatarUrl: text('avatar_url'),
  emailVerified: boolean('email_verified').notNull().default(false),
  role: text('role').notNull().default('user'), // user | admin | moderator
  createdAt: timestamp('created_at', { withTimezone: true }).notNull().defaultNow(),
  updatedAt: timestamp('updated_at', { withTimezone: true }).notNull().defaultNow(),
});

export const sessions = pgTable('sessions', {
  id: text('id').primaryKey(),
  userId: text('user_id')
    .notNull()
    .references(() => users.id, { onDelete: 'cascade' }),
  expiresAt: timestamp('expires_at', { withTimezone: true }).notNull(),
  token: text('token').notNull().unique(),
  ipAddress: text('ip_address'),
  userAgent: text('user_agent'),
  createdAt: timestamp('created_at', { withTimezone: true }).notNull().defaultNow(),
  updatedAt: timestamp('updated_at', { withTimezone: true }).notNull().defaultNow(),
});

export const accounts = pgTable('accounts', {
  id: text('id').primaryKey(),
  userId: text('user_id')
    .notNull()
    .references(() => users.id, { onDelete: 'cascade' }),
  accountId: text('account_id').notNull(),
  providerId: text('provider_id').notNull(),
  accessToken: text('access_token'),
  refreshToken: text('refresh_token'),
  expiresAt: timestamp('expires_at', { withTimezone: true }),
  password: text('password'),
  createdAt: timestamp('created_at', { withTimezone: true }).notNull().defaultNow(),
  updatedAt: timestamp('updated_at', { withTimezone: true }).notNull().defaultNow(),
});

export const verifications = pgTable('verifications', {
  id: text('id').primaryKey(),
  identifier: text('identifier').notNull(),
  value: text('value').notNull(),
  expiresAt: timestamp('expires_at', { withTimezone: true }).notNull(),
  createdAt: timestamp('created_at', { withTimezone: true }).notNull().defaultNow(),
  updatedAt: timestamp('updated_at', { withTimezone: true }).notNull().defaultNow(),
});

export const projects = pgTable('projects', {
  id: uuid('id').primaryKey().defaultRandom(),
  userId: text('user_id')
    .notNull()
    .references(() => users.id, { onDelete: 'cascade' }),
  name: text('name').notNull(),
  description: text('description'),
  thumbnailUrl: text('thumbnail_url'),
  createdAt: timestamp('created_at', { withTimezone: true }).notNull().defaultNow(),
  updatedAt: timestamp('updated_at', { withTimezone: true }).notNull().defaultNow(),
});

export const scenes = pgTable('scenes', {
  id: uuid('id').primaryKey().defaultRandom(),
  projectId: uuid('project_id')
    .notNull()
    .references(() => projects.id, { onDelete: 'cascade' }),
  name: text('name').notNull().default('Untitled Scene'),
  version: integer('version').notNull().default(1),
  data: jsonb('data').notNull().$type<Record<string, unknown>>(),
  createdAt: timestamp('created_at', { withTimezone: true }).notNull().defaultNow(),
  updatedAt: timestamp('updated_at', { withTimezone: true }).notNull().defaultNow(),
});

export const assets = pgTable('assets', {
  id: uuid('id').primaryKey().defaultRandom(),
  userId: text('user_id')
    .notNull()
    .references(() => users.id, { onDelete: 'cascade' }),
  name: text('name').notNull(),
  fileType: text('file_type').notNull(), // glTF, GLB, OBJ, FBX
  fileSize: integer('file_size').notNull(),
  storageKey: text('storage_key').notNull(),
  thumbnailUrl: text('thumbnail_url'),
  isPublic: boolean('is_public').notNull().default(false),
  createdAt: timestamp('created_at', { withTimezone: true }).notNull().defaultNow(),
  updatedAt: timestamp('updated_at', { withTimezone: true }).notNull().defaultNow(),
});

/** One run of the Studio's AI panel, from prompt or photo to final model (services/ai/studio.ts). */
export const aiGenerations = pgTable('ai_generations', {
  id: uuid('id').primaryKey().defaultRandom(),
  userId: text('user_id')
    .notNull()
    .references(() => users.id, { onDelete: 'cascade' }),
  sceneId: uuid('scene_id').references(() => scenes.id, { onDelete: 'set null' }),
  provider: text('provider').notNull(), // e.g. forge3d-trellis2 (self-hosted, see workers/)
  source: text('source').notNull().default('prompt'), // prompt | photo
  prompt: text('prompt'),
  /** The picture that goes to 3D: the uploaded photo, or the reference picture the user picked */
  imageUrl: text('image_url'),
  // drawing -> picking -> [viewing ->] previewing -> reviewing -> finishing -> done; or failed
  status: text('status').notNull().default('drawing'),
  jobId: text('job_id'),
  /** references | views | preview | final: the job running now, or the one that failed */
  jobKind: text('job_kind'),
  /**
   * The picture drawn from other sides (AI_MULTIVIEW=1), which the preview and final are built
   * from: azimuth in degrees around the object (0 is the picture's front) and elevation in
   * degrees above its middle. Null when the step didn't run or gave nothing.
   */
  views: jsonb('views').$type<{ url: string; azimuth: number; elevation: number }[]>(),
  /** Why the model is made from the picture alone: the multiview job failed or took too long */
  viewsError: text('views_error'),
  /**
   * The pictures drawn for a prompt. When the worker rates them: score (0..1, how good a start
   * for 3D) and issues (what makes it a worse one, e.g. "cut off at the bottom").
   */
  referenceImages: jsonb('reference_images').$type<
    { url: string; seed: number; score?: number; issues?: string[] }[]
  >(),
  seed: integer('seed'),
  previewUrl: text('preview_url'),
  previewTriangles: integer('preview_triangles'),
  finalUrl: text('final_url'),
  finalTriangles: integer('final_triangles'),
  /**
   * The pipeline that made the final, when the worker said: "1024_cascade", or "512" when it ran
   * out of GPU memory. Texture options made by another pipeline fit another shape.
   */
  finalPipeline: text('final_pipeline'),
  /**
   * Texture options (AI_TEXTURE_OPTIONS, on by default): more textures for the final's shape, made
   * after it, for the user to choose from. running | done | failed; null when none were asked for.
   * They never fail the generation.
   */
  texturesStatus: text('textures_status'),
  /** The textures job while it runs */
  texturesJobId: text('textures_job_id'),
  /** The textures made, besides the final's own: model file, triangles and the noise's seed */
  textures: jsonb('textures').$type<{ url: string; triangles: number; textureSeed: number }[]>(),
  /** Why there are no textures to choose from, or why some of them couldn't be made */
  texturesError: text('textures_error'),
  /**
   * The judge's pick (AI_TEXTURE_JUDGE=1): the texture a creator would rather use, numbered as the
   * panel numbers them (1 is the final's own texture, then `textures` in order). Null without one.
   */
  texturesPick: integer('textures_pick'),
  /** Why the judge picked it, in its one sentence */
  texturesPickWhy: text('textures_pick_why'),
  /** Why the judge gave no pick, when it was asked (the textures are offered all the same) */
  texturesJudgeError: text('textures_judge_error'),
  /** Whether the panel put the judge's pick in the creator's scene (it does so once) */
  texturesPickApplied: boolean('textures_pick_applied').notNull().default(false),
  /** The texture the creator last chose in the panel (1 is the final's own); null until they do */
  texturesChosen: integer('textures_chosen'),
  /** Attribution shown with the model, such as "Built with DINOv3" (a license requirement) */
  credits: jsonb('credits').$type<string[]>(),
  resultAssetId: uuid('result_asset_id').references(() => assets.id),
  creditsUsed: integer('credits_used').notNull().default(0),
  durationMs: integer('duration_ms'),
  errorMessage: text('error_message'),
  createdAt: timestamp('created_at', { withTimezone: true }).notNull().defaultNow(),
  updatedAt: timestamp('updated_at', { withTimezone: true }).notNull().defaultNow(),
}, (table) => [
  index('ai_generations_user_created_idx').on(table.userId, table.createdAt),
]);

// ==================== Phase 2: Community & Collaboration ====================

export const profiles = pgTable('profiles', {
  id: text('id')
    .primaryKey()
    .references(() => users.id, { onDelete: 'cascade' }),
  username: text('username').notNull().unique(),
  displayName: text('display_name'),
  bio: text('bio'),
  websiteUrl: text('website_url'),
  avatarUrl: text('avatar_url'),
  headerUrl: text('header_url'),
  followerCount: integer('follower_count').notNull().default(0),
  followingCount: integer('following_count').notNull().default(0),
  createdAt: timestamp('created_at', { withTimezone: true }).notNull().defaultNow(),
  updatedAt: timestamp('updated_at', { withTimezone: true }).notNull().defaultNow(),
});

export const categories = pgTable('categories', {
  id: uuid('id').primaryKey().defaultRandom(),
  name: text('name').notNull().unique(),
  slug: text('slug').notNull().unique(),
  sortOrder: integer('sort_order').notNull().default(0),
});

export const experiences = pgTable('experiences', (t) => ({
  id: t.uuid('id').primaryKey().defaultRandom(),
  creatorId: t.text('creator_id').notNull().references(() => users.id, { onDelete: 'cascade' }),
  projectId: t.uuid('project_id').references(() => projects.id, { onDelete: 'set null' }),
  sceneId: t.uuid('scene_id').references(() => scenes.id, { onDelete: 'set null' }),
  title: t.text('title').notNull(),
  description: t.text('description'),
  slug: t.text('slug').notNull().unique(),
  status: t.text('status').notNull().default('draft'), // draft|processing|published|unpublished|removed
  formatType: t.text('format_type').notNull().default('turntable'), // turntable|animated|video|interactive
  sceneUrl: t.text('scene_url'),
  thumbnailUrl: t.text('thumbnail_url'),
  previewUrl: t.text('preview_url'),
  experienceData: t.jsonb('experience_data').$type<Record<string, unknown>>(),
  viewCount: t.integer('view_count').notNull().default(0),
  likeCount: t.integer('like_count').notNull().default(0),
  commentCount: t.integer('comment_count').notNull().default(0),
  polyCount: t.integer('poly_count'),
  fileSize: t.integer('file_size'),
  categoryId: t.uuid('category_id').references(() => categories.id),
  publishedAt: t.timestamp('published_at', { withTimezone: true }),
  createdAt: t.timestamp('created_at', { withTimezone: true }).notNull().defaultNow(),
  updatedAt: t.timestamp('updated_at', { withTimezone: true }).notNull().defaultNow(),
}), (table) => [
  index('experiences_creator_idx').on(table.creatorId),
  index('experiences_status_published_idx').on(table.status, table.publishedAt),
  index('experiences_category_idx').on(table.categoryId),
]);

export const experienceTags = pgTable('experience_tags', {
  experienceId: uuid('experience_id')
    .notNull()
    .references(() => experiences.id, { onDelete: 'cascade' }),
  tag: text('tag').notNull(),
}, (table) => [
  primaryKey({ columns: [table.experienceId, table.tag] }),
  index('experience_tags_tag_idx').on(table.tag),
]);

export const follows = pgTable('follows', {
  followerId: text('follower_id')
    .notNull()
    .references(() => users.id, { onDelete: 'cascade' }),
  followingId: text('following_id')
    .notNull()
    .references(() => users.id, { onDelete: 'cascade' }),
  createdAt: timestamp('created_at', { withTimezone: true }).notNull().defaultNow(),
}, (table) => [
  primaryKey({ columns: [table.followerId, table.followingId] }),
  index('follows_following_idx').on(table.followingId),
]);

export const likes = pgTable('likes', {
  userId: text('user_id')
    .notNull()
    .references(() => users.id, { onDelete: 'cascade' }),
  experienceId: uuid('experience_id')
    .notNull()
    .references(() => experiences.id, { onDelete: 'cascade' }),
  createdAt: timestamp('created_at', { withTimezone: true }).notNull().defaultNow(),
}, (table) => [
  primaryKey({ columns: [table.userId, table.experienceId] }),
]);

export const comments = pgTable('comments', (t) => ({
  id: t.uuid('id').primaryKey().defaultRandom(),
  experienceId: t.uuid('experience_id').notNull().references(() => experiences.id, { onDelete: 'cascade' }),
  userId: t.text('user_id').notNull().references(() => users.id, { onDelete: 'cascade' }),
  parentId: t.uuid('parent_id'),
  body: t.text('body').notNull(),
  likeCount: t.integer('like_count').notNull().default(0),
  status: t.text('status').notNull().default('visible'), // visible|hidden|removed
  createdAt: t.timestamp('created_at', { withTimezone: true }).notNull().defaultNow(),
  updatedAt: t.timestamp('updated_at', { withTimezone: true }).notNull().defaultNow(),
}), (table) => [
  index('comments_experience_idx').on(table.experienceId, table.createdAt),
  index('comments_parent_idx').on(table.parentId),
]);

export const reports = pgTable('reports', {
  id: uuid('id').primaryKey().defaultRandom(),
  reporterId: text('reporter_id')
    .notNull()
    .references(() => users.id, { onDelete: 'cascade' }),
  targetType: text('target_type').notNull(), // experience|comment|user
  targetId: text('target_id').notNull(),
  reason: text('reason').notNull(),
  status: text('status').notNull().default('pending'), // pending|reviewed|resolved|dismissed
  reviewedBy: text('reviewed_by').references(() => users.id),
  reviewedAt: timestamp('reviewed_at', { withTimezone: true }),
  createdAt: timestamp('created_at', { withTimezone: true }).notNull().defaultNow(),
});

export const feedEntries = pgTable('feed_entries', (t) => ({
  id: t.uuid('id').primaryKey().defaultRandom(),
  userId: t.text('user_id').notNull().references(() => users.id, { onDelete: 'cascade' }),
  experienceId: t.uuid('experience_id').notNull().references(() => experiences.id, { onDelete: 'cascade' }),
  createdAt: t.timestamp('created_at', { withTimezone: true }).notNull().defaultNow(),
}), (table) => [
  index('feed_entries_user_idx').on(table.userId, table.createdAt),
]);

export const collabDocuments = pgTable('collab_documents', {
  sceneId: uuid('scene_id')
    .primaryKey()
    .references(() => scenes.id, { onDelete: 'cascade' }),
  state: bytea('state').notNull(),
  updatedAt: timestamp('updated_at', { withTimezone: true }).notNull().defaultNow(),
});
