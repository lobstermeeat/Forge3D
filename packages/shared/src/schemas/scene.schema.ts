import { z } from 'zod';

const transformDataSchema = z.object({
  position: z.tuple([z.number(), z.number(), z.number()]),
  rotation: z.tuple([z.number(), z.number(), z.number(), z.number()]),
  scale: z.tuple([z.number(), z.number(), z.number()]),
});

const meshRendererDataSchema = z.object({
  geometryType: z.enum(['box', 'sphere', 'plane', 'cylinder', 'torus', 'imported']),
  geometryParams: z.record(z.number()).optional(),
  materialIndex: z.number().int().min(0),
});

const cameraDataSchema = z.object({
  fov: z.number().min(1).max(179),
  near: z.number().positive(),
  far: z.number().positive(),
});

const lightDataSchema = z.object({
  type: z.enum(['directional', 'point', 'ambient', 'spot']),
  color: z.tuple([z.number(), z.number(), z.number()]),
  intensity: z.number().min(0),
});

const entityDataSchema = z.object({
  id: z.string(),
  name: z.string(),
  parentId: z.string().optional(),
  components: z.object({
    transform: transformDataSchema.optional(),
    meshRenderer: meshRendererDataSchema.optional(),
    camera: cameraDataSchema.optional(),
    light: lightDataSchema.optional(),
  }),
});

const materialDescriptorSchema = z.object({
  name: z.string(),
  type: z.enum(['standard', 'physical']),
  color: z.tuple([z.number(), z.number(), z.number()]),
  metalness: z.number().min(0).max(1),
  roughness: z.number().min(0).max(1),
  emissive: z.tuple([z.number(), z.number(), z.number()]).optional(),
  emissiveIntensity: z.number().min(0).optional(),
  opacity: z.number().min(0).max(1).optional(),
  transparent: z.boolean().optional(),
});

export const sceneDataSchema = z.object({
  version: z.number().int().positive(),
  entities: z.array(entityDataSchema),
  materials: z.array(materialDescriptorSchema),
});

export const saveSceneSchema = z.object({
  projectId: z.string().uuid(),
  sceneId: z.string().uuid().optional(),
  name: z.string().min(1).max(200),
  data: sceneDataSchema,
});

export type SaveSceneInput = z.infer<typeof saveSceneSchema>;
