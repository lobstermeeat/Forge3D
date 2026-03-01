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
  };
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
