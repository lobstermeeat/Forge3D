import * as THREE from 'three';
import { OrbitControls as ThreeOrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

export interface OrbitControlsOptions {
  camera: THREE.PerspectiveCamera;
  domElement: HTMLElement;
  enableDamping?: boolean;
  dampingFactor?: number;
  minDistance?: number;
  maxDistance?: number;
}

export class ViewportControls {
  private controls: ThreeOrbitControls;
  private defaultTarget = new THREE.Vector3(0, 0, 0);
  private defaultPosition = new THREE.Vector3(5, 5, 5);

  constructor(options: OrbitControlsOptions) {
    this.controls = new ThreeOrbitControls(options.camera, options.domElement);
    this.controls.enableDamping = options.enableDamping ?? true;
    this.controls.dampingFactor = options.dampingFactor ?? 0.1;
    this.controls.minDistance = options.minDistance ?? 0.5;
    this.controls.maxDistance = options.maxDistance ?? 200;
    this.controls.mouseButtons = {
      LEFT: THREE.MOUSE.ROTATE,
      MIDDLE: THREE.MOUSE.PAN,
      RIGHT: THREE.MOUSE.PAN,
    };
    this.controls.touches = {
      ONE: THREE.TOUCH.ROTATE,
      TWO: THREE.TOUCH.DOLLY_PAN,
    };

    this.defaultPosition.copy(options.camera.position);

    // Double-click to reset camera
    options.domElement.addEventListener('dblclick', () => this.reset());
  }

  update(): void {
    this.controls.update();
  }

  reset(): void {
    this.controls.object.position.copy(this.defaultPosition);
    this.controls.target.copy(this.defaultTarget);
    this.controls.update();
  }

  focusOn(target: THREE.Vector3, distance = 5): void {
    this.controls.target.copy(target);
    const direction = this.controls.object.position.clone().sub(target).normalize();
    this.controls.object.position.copy(target).add(direction.multiplyScalar(distance));
    this.controls.update();
  }

  /** Get the orbit target (look-at point). */
  get target(): THREE.Vector3 {
    return this.controls.target;
  }

  setEnabled(enabled: boolean): void {
    this.controls.enabled = enabled;
  }

  dispose(): void {
    this.controls.dispose();
  }
}
