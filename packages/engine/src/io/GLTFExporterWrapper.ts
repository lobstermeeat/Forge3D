import * as THREE from 'three';
import { GLTFExporter } from 'three/examples/jsm/exporters/GLTFExporter.js';

/**
 * Exports an array of Three.js objects as a GLB binary buffer.
 * Wraps objects in a temporary scene for export.
 */
export async function exportSceneAsGLB(objects: THREE.Object3D[]): Promise<ArrayBuffer> {
  const exporter = new GLTFExporter();
  const tempScene = new THREE.Scene();

  for (const obj of objects) {
    tempScene.add(obj.clone());
  }

  return new Promise<ArrayBuffer>((resolve, reject) => {
    exporter.parse(
      tempScene,
      (result) => resolve(result as ArrayBuffer),
      (error) => reject(error),
      { binary: true },
    );
  });
}
