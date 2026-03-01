import * as THREE from 'three';
import type {
  ExperienceData,
  EntityData,
  MaterialDescriptor,
  LightData,
  MeshRendererData,
} from '@forge3d/shared';

/**
 * Lightweight scene builder for the viewer.
 * Builds a Three.js scene directly from ExperienceData without the ECS/SceneManager overhead.
 */
export class SceneBuilder {
  private materialCache = new Map<string, THREE.Material>();

  build(data: ExperienceData, scene: THREE.Scene): void {
    // Set background color
    const bg = data.environment.backgroundColor;
    scene.background = new THREE.Color(bg[0], bg[1], bg[2]);

    // Add ambient light from environment
    const ambient = new THREE.AmbientLight(0xffffff, data.environment.ambientIntensity);
    scene.add(ambient);

    // Build entities
    const entityMap = new Map<string, THREE.Object3D>();

    for (const entity of data.scene.entities) {
      const obj = this.buildEntity(entity, data.scene.materials);
      entityMap.set(entity.id, obj);
    }

    // Establish hierarchy
    for (const entity of data.scene.entities) {
      const obj = entityMap.get(entity.id);
      if (!obj) continue;

      if (entity.parentId) {
        const parent = entityMap.get(entity.parentId);
        if (parent) {
          parent.add(obj);
          continue;
        }
      }
      scene.add(obj);
    }
  }

  private buildEntity(entity: EntityData, materials: MaterialDescriptor[]): THREE.Object3D {
    const { components } = entity;
    let obj: THREE.Object3D;

    if (components.light) {
      obj = this.buildLight(components.light);
    } else if (components.meshRenderer) {
      obj = this.buildMesh(components.meshRenderer, materials);
    } else {
      obj = new THREE.Group();
    }

    obj.name = entity.name;
    obj.userData.entityId = entity.id;

    // Apply transform
    if (components.transform) {
      const t = components.transform;
      obj.position.set(t.position[0], t.position[1], t.position[2]);
      obj.quaternion.set(t.rotation[0], t.rotation[1], t.rotation[2], t.rotation[3]);
      obj.scale.set(t.scale[0], t.scale[1], t.scale[2]);
    }

    return obj;
  }

  private buildMesh(data: MeshRendererData, materials: MaterialDescriptor[]): THREE.Mesh {
    const geometry = this.createGeometry(data);
    const descriptor = materials[data.materialIndex];
    const material = descriptor ? this.createMaterial(descriptor) : new THREE.MeshStandardMaterial();
    return new THREE.Mesh(geometry, material);
  }

  private buildLight(data: LightData): THREE.Light {
    const color = new THREE.Color(data.color[0], data.color[1], data.color[2]);

    switch (data.type) {
      case 'directional': {
        const light = new THREE.DirectionalLight(color, data.intensity);
        light.castShadow = true;
        return light;
      }
      case 'point':
        return new THREE.PointLight(color, data.intensity);
      case 'ambient':
        return new THREE.AmbientLight(color, data.intensity);
      case 'spot':
        return new THREE.SpotLight(color, data.intensity);
      default:
        return new THREE.PointLight(color, data.intensity);
    }
  }

  private createGeometry(data: MeshRendererData): THREE.BufferGeometry {
    const p = data.geometryParams ?? {};
    switch (data.geometryType) {
      case 'box':
        return new THREE.BoxGeometry(p['width'] ?? 1, p['height'] ?? 1, p['depth'] ?? 1);
      case 'sphere':
        return new THREE.SphereGeometry(p['radius'] ?? 0.5, p['widthSegments'] ?? 32, p['heightSegments'] ?? 16);
      case 'plane':
        return new THREE.PlaneGeometry(p['width'] ?? 1, p['height'] ?? 1);
      case 'cylinder':
        return new THREE.CylinderGeometry(p['radiusTop'] ?? 0.5, p['radiusBottom'] ?? 0.5, p['height'] ?? 1, p['radialSegments'] ?? 32);
      case 'torus':
        return new THREE.TorusGeometry(p['radius'] ?? 0.5, p['tube'] ?? 0.2, p['radialSegments'] ?? 16, p['tubularSegments'] ?? 48);
      default:
        return new THREE.BoxGeometry(1, 1, 1);
    }
  }

  private createMaterial(desc: MaterialDescriptor): THREE.Material {
    const key = JSON.stringify(desc);
    const cached = this.materialCache.get(key);
    if (cached) return cached;

    const color = new THREE.Color(desc.color[0], desc.color[1], desc.color[2]);
    const opts = {
      color,
      metalness: desc.metalness,
      roughness: desc.roughness,
      emissive: desc.emissive ? new THREE.Color(...desc.emissive) : undefined,
      emissiveIntensity: desc.emissiveIntensity ?? 0,
      opacity: desc.opacity ?? 1,
      transparent: desc.transparent ?? false,
    };

    const material = desc.type === 'physical'
      ? new THREE.MeshPhysicalMaterial(opts)
      : new THREE.MeshStandardMaterial(opts);

    this.materialCache.set(key, material);
    return material;
  }

  dispose(): void {
    for (const mat of this.materialCache.values()) {
      mat.dispose();
    }
    this.materialCache.clear();
  }
}
