import * as THREE from 'three';
import { OrbitControls as ThreeOrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

/**
 * Mouse layout for the viewport camera.
 * - `studio`: left button is free for selecting and dragging gizmos, right-drag orbits,
 *   middle-drag pans (Roblox Studio / game-engine editor convention).
 * - `orbit`: left-drag orbits, middle/right-drag pans (model-viewer convention).
 */
export type MouseScheme = 'studio' | 'orbit';

export interface OrbitControlsOptions {
  camera: THREE.PerspectiveCamera;
  domElement: HTMLElement;
  enableDamping?: boolean;
  dampingFactor?: number;
  minDistance?: number;
  maxDistance?: number;
  mouseScheme?: MouseScheme;
}

export class ViewportControls {
  private controls: ThreeOrbitControls;
  private defaultTarget = new THREE.Vector3(0, 0, 0);
  private defaultPosition = new THREE.Vector3(5, 5, 5);
  private tween: {
    fromPos: THREE.Vector3;
    toPos: THREE.Vector3;
    fromTarget: THREE.Vector3;
    toTarget: THREE.Vector3;
    start: number;
    duration: number;
  } | null = null;

  constructor(options: OrbitControlsOptions) {
    this.controls = new ThreeOrbitControls(options.camera, options.domElement);
    this.controls.enableDamping = options.enableDamping ?? true;
    this.controls.dampingFactor = options.dampingFactor ?? 0.1;
    this.controls.minDistance = options.minDistance ?? 0.5;
    this.controls.maxDistance = options.maxDistance ?? 200;
    this.setMouseScheme(options.mouseScheme ?? 'studio');
    this.controls.touches = {
      ONE: THREE.TOUCH.ROTATE,
      TWO: THREE.TOUCH.DOLLY_PAN,
    };

    this.defaultPosition.copy(options.camera.position);

    // Double-click empty space to reset camera
    options.domElement.addEventListener('dblclick', () => this.reset());
  }

  setMouseScheme(scheme: MouseScheme): void {
    this.controls.mouseButtons =
      scheme === 'studio'
        ? {
            // Left button stays unbound so clicks select and drag gizmos
            LEFT: null,
            MIDDLE: THREE.MOUSE.PAN,
            RIGHT: THREE.MOUSE.ROTATE,
          }
        : {
            LEFT: THREE.MOUSE.ROTATE,
            MIDDLE: THREE.MOUSE.PAN,
            RIGHT: THREE.MOUSE.PAN,
          };
  }

  update(): void {
    if (this.tween) {
      const t = Math.min(1, (performance.now() - this.tween.start) / this.tween.duration);
      const e = t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;
      this.controls.object.position.lerpVectors(this.tween.fromPos, this.tween.toPos, e);
      this.controls.target.lerpVectors(this.tween.fromTarget, this.tween.toTarget, e);
      if (t >= 1) this.tween = null;
    }
    // While disabled (e.g. during Play), another controller owns the camera
    if (!this.controls.enabled) return;
    this.controls.update();
  }

  reset(): void {
    if (!this.controls.enabled) return;
    this.tween = null;
    this.controls.object.position.copy(this.defaultPosition);
    this.controls.target.copy(this.defaultTarget);
    this.controls.update();
  }

  focusOn(target: THREE.Vector3, distance = 5): void {
    const direction = this.controls.object.position.clone().sub(this.controls.target).normalize();
    if (direction.lengthSq() === 0) direction.set(1, 1, 1).normalize();
    this.setView(target.clone().add(direction.multiplyScalar(distance)), target);
  }

  /** Smoothly move the camera to a new position looking at `target`. */
  setView(position: THREE.Vector3, target: THREE.Vector3, durationMs = 450): void {
    if (durationMs <= 0) {
      this.tween = null;
      this.controls.object.position.copy(position);
      this.controls.target.copy(target);
      this.controls.update();
      return;
    }
    this.tween = {
      fromPos: this.controls.object.position.clone(),
      toPos: position.clone(),
      fromTarget: this.controls.target.clone(),
      toTarget: target.clone(),
      start: performance.now(),
      duration: durationMs,
    };
  }

  /** Get the orbit target (look-at point). */
  get target(): THREE.Vector3 {
    return this.controls.target;
  }

  setEnabled(enabled: boolean): void {
    this.controls.enabled = enabled;
    if (!enabled) this.tween = null;
  }

  isEnabled(): boolean {
    return this.controls.enabled;
  }

  dispose(): void {
    this.controls.dispose();
  }
}
