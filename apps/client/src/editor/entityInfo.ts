import type { Entity } from '@forge3d/engine';
import type { LightData, MeshRendererData, ModelData } from '@forge3d/shared';

export interface EntityInfo {
  icon: string;
  /** What the object is, e.g. "Part · Sphere". */
  type: string;
  color: string;
  kind: 'part' | 'light' | 'model' | 'mesh' | 'group';
}

const LIGHT_INFO: Record<LightData['type'], { icon: string; type: string }> = {
  directional: { icon: 'sun', type: 'Sun light' },
  point: { icon: 'bulb', type: 'Point light' },
  ambient: { icon: 'ambient', type: 'Ambient light' },
  spot: { icon: 'spot', type: 'Spot light' },
};

const PART_NAME: Record<string, string> = {
  box: 'Box',
  sphere: 'Sphere',
  cylinder: 'Cylinder',
  plane: 'Plane',
  torus: 'Torus',
};

export function describeEntity(entity: Entity): EntityInfo {
  const light = entity.getComponent<LightData>('light');
  if (light) {
    const info = LIGHT_INFO[light.type] ?? LIGHT_INFO.point;
    return { ...info, color: '#f5b84b', kind: 'light' };
  }
  const model = entity.getComponent<ModelData>('model');
  if (model) {
    const quality = model.quality === 'preview' ? ' · preview' : '';
    const type = model.source === 'ai' ? `AI model${quality}` : 'Model';
    return { icon: 'model', type, color: '#8fb3ff', kind: 'model' };
  }
  const mesh = entity.getComponent<MeshRendererData>('meshRenderer');
  if (mesh?.geometryType === 'imported') {
    return { icon: 'box', type: 'Imported mesh', color: '#8fb3ff', kind: 'mesh' };
  }
  if (mesh) {
    return {
      icon: mesh.geometryType,
      type: `Part · ${PART_NAME[mesh.geometryType] ?? mesh.geometryType}`,
      color: '#c9cdd4',
      kind: 'part',
    };
  }
  if (entity.children.length > 0) {
    return { icon: 'model', type: 'Model', color: '#8fb3ff', kind: 'model' };
  }
  return { icon: 'group', type: 'Group', color: '#878d97', kind: 'group' };
}
