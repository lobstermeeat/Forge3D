import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

export interface ViewerControlsOptions {
  camera: THREE.PerspectiveCamera;
  domElement: HTMLElement;
  initialPosition?: [number, number, number];
  initialTarget?: [number, number, number];
}

/**
 * Lightweight orbit controls for the experience viewer.
 * Supports mouse + touch input, damping, and reset.
 */
export class ViewerControls {
  readonly controls: OrbitControls;
  private initialPosition: THREE.Vector3;
  private initialTarget: THREE.Vector3;

  constructor(opts: ViewerControlsOptions) {
    this.controls = new OrbitControls(opts.camera, opts.domElement);

    // Damping for smooth motion
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.1;

    // Distance constraints
    this.controls.minDistance = 0.5;
    this.controls.maxDistance = 200;

    // Touch support is built-in with OrbitControls:
    // ONE finger = rotate, TWO fingers = zoom + pan

    // Store initial state for reset
    const pos = opts.initialPosition ?? [5, 5, 5];
    const tgt = opts.initialTarget ?? [0, 0, 0];
    this.initialPosition = new THREE.Vector3(pos[0], pos[1], pos[2]);
    this.initialTarget = new THREE.Vector3(tgt[0], tgt[1], tgt[2]);

    opts.camera.position.copy(this.initialPosition);
    this.controls.target.copy(this.initialTarget);
    this.controls.update();
  }

  update(): void {
    this.controls.update();
  }

  reset(): void {
    this.controls.object.position.copy(this.initialPosition);
    this.controls.target.copy(this.initialTarget);
    this.controls.update();
  }

  /** Enable/disable auto-rotate (for turntable format) */
  setAutoRotate(enabled: boolean, speed = 2): void {
    this.controls.autoRotate = enabled;
    this.controls.autoRotateSpeed = speed;
  }

  setEnabled(enabled: boolean): void {
    this.controls.enabled = enabled;
  }

  dispose(): void {
    this.controls.dispose();
  }
}
