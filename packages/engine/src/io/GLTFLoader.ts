import * as THREE from 'three';
import { GLTFLoader as ThreeGLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { DRACOLoader } from 'three/examples/jsm/loaders/DRACOLoader.js';

export interface LoadResult {
  scene: THREE.Group;
  animations: THREE.AnimationClip[];
}

export class AssetLoader {
  private gltfLoader: ThreeGLTFLoader;
  private dracoLoader: DRACOLoader;

  constructor(dracoPath = '/draco/') {
    this.dracoLoader = new DRACOLoader();
    this.dracoLoader.setDecoderPath(dracoPath);

    this.gltfLoader = new ThreeGLTFLoader();
    this.gltfLoader.setDRACOLoader(this.dracoLoader);
  }

  async loadGLTF(url: string): Promise<LoadResult> {
    return new Promise((resolve, reject) => {
      this.gltfLoader.load(
        url,
        (gltf) => {
          resolve({
            scene: gltf.scene,
            animations: gltf.animations,
          });
        },
        undefined,
        (error) => reject(new Error(`Failed to load GLTF: ${error}`)),
      );
    });
  }

  async loadFromFile(file: File): Promise<LoadResult> {
    const url = URL.createObjectURL(file);
    try {
      return await this.loadGLTF(url);
    } finally {
      URL.revokeObjectURL(url);
    }
  }

  dispose(): void {
    this.dracoLoader.dispose();
  }
}
