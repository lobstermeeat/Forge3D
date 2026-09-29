import * as THREE from 'three';
import { TransformControls as ThreeTransformControls } from 'three/examples/jsm/controls/TransformControls.js';

export type TransformMode = 'translate' | 'rotate' | 'scale';

export interface TransformControlsOptions {
  camera: THREE.Camera;
  domElement: HTMLElement;
  scene: THREE.Scene;
}

/** Snap increments. `null` turns snapping off for that operation. */
export interface SnapSettings {
  translate: number | null;
  rotateDegrees: number | null;
  scale: number | null;
}

/** Increments used while Shift is held and persistent snapping is off. */
const SHIFT_SNAP: SnapSettings = { translate: 1, rotateDegrees: 15, scale: 0.25 };

export class GizmoControls {
  private controls: ThreeTransformControls;
  private scene: THREE.Scene;
  private domElement: HTMLElement;
  private snap: SnapSettings = { translate: null, rotateDegrees: null, scale: null };
  private shiftHeld = false;
  private onChange: ((object: THREE.Object3D) => void) | null = null;
  private onDragStart: (() => void) | null = null;
  private onDragEnd: (() => void) | null = null;

  constructor(options: TransformControlsOptions) {
    this.scene = options.scene;
    this.domElement = options.domElement;
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

    this.domElement.addEventListener('keydown', this.onKeyDown);
    this.domElement.addEventListener('keyup', this.onKeyUp);
  }

  // Shift gives temporary snapping when persistent snapping is off
  private onKeyDown = (e: KeyboardEvent): void => {
    if (e.key === 'Shift' && !this.shiftHeld) {
      this.shiftHeld = true;
      this.applySnap();
    }
  };

  private onKeyUp = (e: KeyboardEvent): void => {
    if (e.key === 'Shift') {
      this.shiftHeld = false;
      this.applySnap();
    }
  };

  private applySnap(): void {
    const s = this.shiftHeld
      ? {
          translate: this.snap.translate ?? SHIFT_SNAP.translate,
          rotateDegrees: this.snap.rotateDegrees ?? SHIFT_SNAP.rotateDegrees,
          scale: this.snap.scale ?? SHIFT_SNAP.scale,
        }
      : this.snap;
    this.controls.setTranslationSnap(s.translate);
    this.controls.setRotationSnap(
      s.rotateDegrees === null ? null : THREE.MathUtils.degToRad(s.rotateDegrees),
    );
    this.controls.setScaleSnap(s.scale);
  }

  /** Persistent snapping used for every drag. */
  setSnap(snap: SnapSettings): void {
    this.snap = { ...snap };
    this.applySnap();
  }

  setSpace(space: 'world' | 'local'): void {
    this.controls.setSpace(space);
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

  /** True while the pointer is over a gizmo handle. */
  isHovered(): boolean {
    return this.controls.axis !== null;
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
    this.domElement.removeEventListener('keydown', this.onKeyDown);
    this.domElement.removeEventListener('keyup', this.onKeyUp);
    this.scene.remove(this.controls.getHelper());
    this.controls.dispose();
  }
}
