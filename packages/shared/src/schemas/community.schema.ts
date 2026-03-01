import { z } from 'zod';

// ---------- Profile Schemas ----------

export const profileSetupSchema = z.object({
  username: z
    .string()
    .min(3)
    .max(30)
    .regex(/^[a-z0-9][a-z0-9-]*[a-z0-9]$/, 'Username must be lowercase alphanumeric with hyphens, cannot start/end with hyphen')
    .transform((s) => s.toLowerCase()),
});

export type ProfileSetupInput = z.infer<typeof profileSetupSchema>;

export const profileUpdateSchema = z.object({
  displayName: z.string().min(1).max(100).optional(),
  bio: z.string().max(500).optional(),
  websiteUrl: z.string().url().max(500).optional().or(z.literal('')),
  avatarUrl: z.string().url().max(500).optional().or(z.literal('')),
});

export type ProfileUpdateInput = z.infer<typeof profileUpdateSchema>;

// ---------- Social Schemas ----------

export const followInputSchema = z.object({
  userId: z.string().min(1),
});

export type FollowInput = z.infer<typeof followInputSchema>;

export const followerListSchema = z.object({
  userId: z.string().min(1),
  cursor: z.string().optional(),
  limit: z.number().int().min(1).max(50).default(20),
});

export type FollowerListInput = z.infer<typeof followerListSchema>;

// ---------- Interaction Schemas ----------

export const likeInputSchema = z.object({
  experienceId: z.string().uuid(),
});

export type LikeInput = z.infer<typeof likeInputSchema>;

export const commentInputSchema = z.object({
  experienceId: z.string().uuid(),
  body: z.string().min(1).max(2000),
  parentId: z.string().uuid().optional(),
});

export type CommentInput = z.infer<typeof commentInputSchema>;

export const commentListSchema = z.object({
  experienceId: z.string().uuid(),
  sort: z.enum(['newest', 'popular']).default('newest'),
  cursor: z.string().optional(),
  limit: z.number().int().min(1).max(50).default(20),
});

export type CommentListInput = z.infer<typeof commentListSchema>;

export const deleteCommentSchema = z.object({
  commentId: z.string().uuid(),
});

export type DeleteCommentInput = z.infer<typeof deleteCommentSchema>;
