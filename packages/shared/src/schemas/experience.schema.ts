import { z } from 'zod';
import { sceneDataSchema } from './scene.schema';

// ---------- Experience Format Schemas ----------

const cameraKeyframeSchema = z.object({
  time: z.number().min(0),
  position: z.tuple([z.number(), z.number(), z.number()]),
  target: z.tuple([z.number(), z.number(), z.number()]),
  easing: z.enum(['linear', 'ease-in', 'ease-out', 'ease-in-out']),
});

const interactionDefSchema = z.object({
  entityId: z.string(),
  action: z.enum(['showInfo', 'navigate', 'playAnimation', 'openUrl']),
  payload: z.record(z.unknown()),
});

const experienceFormatSchema = z.discriminatedUnion('type', [
  z.object({ type: z.literal('turntable'), speed: z.number().min(0).max(10), axis: z.enum(['y', 'x']) }),
  z.object({ type: z.literal('animated'), duration: z.number().positive() }),
  z.object({ type: z.literal('video'), keyframes: z.array(cameraKeyframeSchema).min(2) }),
  z.object({ type: z.literal('interactive'), interactions: z.array(interactionDefSchema) }),
]);

// ---------- Experience Data Schema ----------

export const experienceDataSchema = z.object({
  version: z.number().int().positive(),
  scene: sceneDataSchema,
  camera: z.object({
    position: z.tuple([z.number(), z.number(), z.number()]),
    target: z.tuple([z.number(), z.number(), z.number()]),
    fov: z.number().min(1).max(179),
  }),
  environment: z.object({
    backgroundColor: z.tuple([z.number(), z.number(), z.number()]),
    ambientIntensity: z.number().min(0).max(5),
  }),
  format: experienceFormatSchema,
});

export type ExperienceDataInput = z.infer<typeof experienceDataSchema>;

// ---------- Publish Input Schema ----------

export const publishExperienceSchema = z.object({
  sceneId: z.string().uuid(),
  title: z.string().min(1).max(200),
  description: z.string().max(2000).optional(),
  categoryId: z.string().uuid().optional(),
  tags: z.array(z.string().min(1).max(50).transform((s) => s.toLowerCase())).max(10).default([]),
  formatType: z.enum(['turntable', 'animated', 'video', 'interactive']).default('turntable'),
  formatConfig: experienceFormatSchema.optional(),
  thumbnail: z.string().optional(), // base64 data URL from client
});

export type PublishExperienceInput = z.infer<typeof publishExperienceSchema>;

// ---------- Query Schemas ----------

export const experienceSlugSchema = z.string().min(1).max(100);

export const experienceListSchema = z.object({
  cursor: z.string().optional(),
  limit: z.number().int().min(1).max(50).default(20),
  categoryId: z.string().uuid().optional(),
});

export type ExperienceListInput = z.infer<typeof experienceListSchema>;
