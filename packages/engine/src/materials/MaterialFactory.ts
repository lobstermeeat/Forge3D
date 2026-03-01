import * as THREE from 'three';
import type { MaterialDescriptor } from '@forge3d/shared';

export class MaterialFactory {
  private cache: Map<string, THREE.Material> = new Map();

  create(descriptor: MaterialDescriptor): THREE.Material {
    const key = JSON.stringify(descriptor);
    const cached = this.cache.get(key);
    if (cached) return cached;

    const color = new THREE.Color(descriptor.color[0], descriptor.color[1], descriptor.color[2]);

    let material: THREE.Material;

    if (descriptor.type === 'physical') {
      material = new THREE.MeshPhysicalMaterial({
        color,
        metalness: descriptor.metalness,
        roughness: descriptor.roughness,
        emissive: descriptor.emissive
          ? new THREE.Color(...descriptor.emissive)
          : undefined,
        emissiveIntensity: descriptor.emissiveIntensity ?? 0,
        opacity: descriptor.opacity ?? 1,
        transparent: descriptor.transparent ?? false,
      });
    } else {
      material = new THREE.MeshStandardMaterial({
        color,
        metalness: descriptor.metalness,
        roughness: descriptor.roughness,
        emissive: descriptor.emissive
          ? new THREE.Color(...descriptor.emissive)
          : undefined,
        emissiveIntensity: descriptor.emissiveIntensity ?? 0,
        opacity: descriptor.opacity ?? 1,
        transparent: descriptor.transparent ?? false,
      });
    }

    this.cache.set(key, material);
    return material;
  }

  clearCache(): void {
    for (const material of this.cache.values()) {
      material.dispose();
    }
    this.cache.clear();
  }
}
