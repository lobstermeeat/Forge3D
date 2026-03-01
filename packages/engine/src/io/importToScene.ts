import * as THREE from 'three';
import { Entity } from '../ecs/Entity';
import type { SceneManager } from '../scene/SceneManager';
import type { TransformData, MeshRendererData } from '@forge3d/shared';

/**
 * Converts a loaded THREE.Group (from GLTF import) into SceneManager entities.
 * Returns array of created entity IDs.
 *
 * Note: Imported geometry is NOT serialized into entity data — the entity stores
 * geometryType: 'imported' as a stub. The actual Three.js objects must be
 * registered separately via SceneBridge.addExternalObject().
 */
export function importGroupToScene(
  group: THREE.Group,
  sceneManager: SceneManager,
  fileName: string,
): { rootId: string; meshEntries: Array<{ entityId: string; object: THREE.Object3D }> } {
  const meshEntries: Array<{ entityId: string; object: THREE.Object3D }> = [];

  // Create a root entity for the imported group
  const rootEntity = new Entity(fileName);
  rootEntity.transform = readTransform(group);
  sceneManager.addEntity(rootEntity);

  // Traverse children and create entities for meshes
  group.traverse((child) => {
    if (child === group) return;
    if (child instanceof THREE.Mesh) {
      const entity = new Entity(child.name || 'ImportedMesh');
      entity.parentId = rootEntity.id;
      entity.transform = readTransform(child);
      entity.setComponent<MeshRendererData>('meshRenderer', {
        geometryType: 'imported',
        materialIndex: 0,
      });
      sceneManager.addEntity(entity);
      meshEntries.push({ entityId: entity.id, object: child });
    }
  });

  return { rootId: rootEntity.id, meshEntries };
}

function readTransform(obj: THREE.Object3D): TransformData {
  return {
    position: [obj.position.x, obj.position.y, obj.position.z],
    rotation: [obj.quaternion.x, obj.quaternion.y, obj.quaternion.z, obj.quaternion.w],
    scale: [obj.scale.x, obj.scale.y, obj.scale.z],
  };
}
