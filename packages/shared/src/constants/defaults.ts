import type { MaterialDescriptor, TransformData, CameraData } from '../types/scene';

export const DEFAULT_TRANSFORM: TransformData = {
  position: [0, 0, 0],
  rotation: [0, 0, 0, 1], // identity quaternion
  scale: [1, 1, 1],
};

export const DEFAULT_MATERIAL: MaterialDescriptor = {
  name: 'Default',
  type: 'standard',
  color: [0.8, 0.8, 0.8],
  metalness: 0.0,
  roughness: 0.5,
};

export const DEFAULT_CAMERA: CameraData = {
  fov: 60,
  near: 0.1,
  far: 1000,
};

export const SCENE_VERSION = 1;
