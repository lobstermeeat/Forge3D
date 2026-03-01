import { z } from 'zod';

export const searchExperiencesSchema = z.object({
  query: z.string().min(1).max(200),
  categoryId: z.string().uuid().optional(),
  formatType: z.enum(['turntable', 'animated', 'video', 'interactive']).optional(),
  sort: z.enum(['relevance', 'newest', 'popular']).default('relevance'),
  limit: z.number().int().min(1).max(50).default(20),
  offset: z.number().int().min(0).default(0),
});

export const reportInputSchema = z.object({
  targetType: z.enum(['experience', 'comment', 'user']),
  targetId: z.string().min(1),
  reason: z.string().min(10).max(1000),
});

export const reportListSchema = z.object({
  status: z.enum(['pending', 'reviewed', 'resolved', 'dismissed']).optional(),
  cursor: z.string().optional(),
  limit: z.number().int().min(1).max(50).default(20),
});

export const reviewReportSchema = z.object({
  reportId: z.string().uuid(),
  action: z.enum(['resolve', 'dismiss']),
});
