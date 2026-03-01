import * as THREE from 'three';
import type { MeshRendererData } from '@forge3d/shared';

export class MeshFactory {
  createGeometry(data: MeshRendererData): THREE.BufferGeometry {
    const params = data.geometryParams ?? {};

    switch (data.geometryType) {
      case 'box':
        return new THREE.BoxGeometry(
          params['width'] ?? 1,
          params['height'] ?? 1,
          params['depth'] ?? 1,
        );
      case 'sphere':
        return new THREE.SphereGeometry(
          params['radius'] ?? 0.5,
          params['widthSegments'] ?? 32,
          params['heightSegments'] ?? 16,
        );
      case 'plane':
        return new THREE.PlaneGeometry(
          params['width'] ?? 1,
          params['height'] ?? 1,
        );
      case 'cylinder':
        return new THREE.CylinderGeometry(
          params['radiusTop'] ?? 0.5,
          params['radiusBottom'] ?? 0.5,
          params['height'] ?? 1,
          params['radialSegments'] ?? 32,
        );
      case 'torus':
        return new THREE.TorusGeometry(
          params['radius'] ?? 0.5,
          params['tube'] ?? 0.2,
          params['radialSegments'] ?? 16,
          params['tubularSegments'] ?? 48,
        );
      case 'imported':
        // Return placeholder; actual mesh is loaded from file
        return new THREE.BoxGeometry(1, 1, 1);
      default:
        return new THREE.BoxGeometry(1, 1, 1);
    }
  }
}
