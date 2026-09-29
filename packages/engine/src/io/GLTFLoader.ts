import * as THREE from 'three';
import { GLTFLoader as ThreeGLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { DRACOLoader } from 'three/examples/jsm/loaders/DRACOLoader.js';
import { KTX2Loader } from 'three/examples/jsm/loaders/KTX2Loader.js';
import { MeshoptDecoder } from 'three/examples/jsm/libs/meshopt_decoder.module.js';

export interface LoadResult {
  scene: THREE.Group;
  animations: THREE.AnimationClip[];
}

export interface AssetLoaderOptions {
  /** Where the Draco and Basis decoders are served (`draco/` and `basis/` below it). */
  decoderPath?: string;
  /**
   * The renderer that will draw the model. KTX2 textures (what the AI pipeline produces)
   * are transcoded to a GPU format this renderer supports; without it they can't load.
   */
  renderer?: unknown;
}

export class AssetLoader {
  private gltfLoader: ThreeGLTFLoader;
  private dracoLoader: DRACOLoader;
  private ktx2Loader: KTX2Loader | null = null;

  constructor({ decoderPath = '/decoders/', renderer }: AssetLoaderOptions = {}) {
    this.dracoLoader = new DRACOLoader();
    this.dracoLoader.setDecoderPath(`${decoderPath}draco/`);

    this.gltfLoader = new ThreeGLTFLoader();
    this.gltfLoader.setDRACOLoader(this.dracoLoader);
    // meshopt geometry (EXT_meshopt_compression), as written by gltfpack
    this.gltfLoader.setMeshoptDecoder(MeshoptDecoder);

    if (renderer) {
      this.ktx2Loader = new KTX2Loader();
      this.ktx2Loader.setTranscoderPath(`${decoderPath}basis/`);
      this.ktx2Loader.detectSupport(renderer as THREE.WebGLRenderer);
      this.gltfLoader.setKTX2Loader(this.ktx2Loader);
    }
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
    this.ktx2Loader?.dispose();
  }
}
