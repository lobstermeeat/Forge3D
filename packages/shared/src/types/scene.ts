export interface SceneData {
  version: number;
  entities: EntityData[];
  materials: MaterialDescriptor[];
}

export interface EntityData {
  id: string;
  name: string;
  parentId?: string;
  components: {
    transform?: TransformData;
    meshRenderer?: MeshRendererData;
    camera?: CameraData;
    light?: LightData;
    model?: ModelData;
  };
}

/**
 * A 3D model file the entity shows, such as one made with Orainge's AI. It loads from `url` and
 * rests on the entity's origin by its bottom centre, so the transform places and sizes it.
 */
export interface ModelData {
  url: string;
  /** An AI generation, or a file someone imported. */
  source: 'ai' | 'import';
  /** The AI generation it came from, so the AI panel can find it again (to make its final). */
  generationId?: string;
  /** For AI models: the quick preview or the final. */
  quality?: 'preview' | 'final';
  /** Attribution shown with the model, such as "Built with DINOv3". */
  credits?: string[];
}

export interface TransformData {
  position: [number, number, number];
  rotation: [number, number, number, number]; // quaternion xyzw
  scale: [number, number, number];
}

export interface MeshRendererData {
  geometryType: 'box' | 'sphere' | 'plane' | 'cylinder' | 'torus' | 'imported';
  geometryParams?: Record<string, number>;
  materialIndex: number;
}

export interface CameraData {
  fov: number;
  near: number;
  far: number;
}

export interface LightData {
  type: 'directional' | 'point' | 'ambient' | 'spot';
  color: [number, number, number];
  intensity: number;
}

export interface MaterialDescriptor {
  name: string;
  type: 'standard' | 'physical';
  color: [number, number, number];
  metalness: number;
  roughness: number;
  emissive?: [number, number, number];
  emissiveIntensity?: number;
  opacity?: number;
  transparent?: boolean;
}
