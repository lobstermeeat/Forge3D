import * as THREE from 'three';
import { TransformControls as ThreeTransformControls } from 'three/examples/jsm/controls/TransformControls.js';

export type TransformMode = 'translate' | 'rotate' | 'scale';

export interface TransformControlsOptions {
  camera: THREE.Camera;
  domElement: HTMLElement;
  scene: THREE.Scene;
}

export class GizmoControls {
  private controls: ThreeTransformControls;
  private scene: THREE.Scene;
  private onChange: ((object: THREE.Object3D) => void) | null = null;
  private onDragStart: (() => void) | null = null;
  private onDragEnd: (() => void) | null = null;

  constructor(options: TransformControlsOptions) {
    this.scene = options.scene;
    this.controls = new ThreeTransformControls(options.camera, options.domElement);
    this.controls.setSize(0.75);
    this.scene.add(this.controls.getHelper());

    this.controls.addEventListener('change', () => {
      if (this.controls.object && this.onChange) {
        this.onChange(this.controls.object);
      }
    });

    this.controls.addEventListener('dragging-changed', (event) => {
      if ((event as unknown as { value: boolean }).value) {
        this.onDragStart?.();
      } else {
        this.onDragEnd?.();
      }
    });

    // Keyboard shortcuts for mode switching
    const onKeyDown = (e: KeyboardEvent) => {
      switch (e.key.toLowerCase()) {
        case 'g':
          this.setMode('translate');
          break;
        case 'r':
          this.setMode('rotate');
          break;
        case 's':
          if (!e.ctrlKey && !e.metaKey) this.setMode('scale');
          break;
      }

      // Snap toggle with Shift
      if (e.shiftKey) {
        this.controls.setTranslationSnap(1);
        this.controls.setRotationSnap(THREE.MathUtils.degToRad(15));
        this.controls.setScaleSnap(0.25);
      }
    };

    const onKeyUp = (e: KeyboardEvent) => {
      if (!e.shiftKey) {
        this.controls.setTranslationSnap(null);
        this.controls.setRotationSnap(null);
        this.controls.setScaleSnap(null);
      }
    };

    options.domElement.addEventListener('keydown', onKeyDown);
    options.domElement.addEventListener('keyup', onKeyUp);
  }

  attach(object: THREE.Object3D): void {
    this.controls.attach(object);
  }

  detach(): void {
    this.controls.detach();
  }

  setMode(mode: TransformMode): void {
    this.controls.setMode(mode);
  }

  getMode(): TransformMode {
    return this.controls.mode as TransformMode;
  }

  isDragging(): boolean {
    return this.controls.dragging;
  }

  onTransformChange(callback: (object: THREE.Object3D) => void): void {
    this.onChange = callback;
  }

  onDragStarted(callback: () => void): void {
    this.onDragStart = callback;
  }

  onDragEnded(callback: () => void): void {
    this.onDragEnd = callback;
  }

  dispose(): void {
    this.scene.remove(this.controls.getHelper());
    this.controls.dispose();
  }
}
