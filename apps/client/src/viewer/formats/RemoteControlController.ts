import * as THREE from 'three';
import type { RemoteControlCameraSync } from '@forge3d/shared';
import type { FormatController } from './types';

/**
 * Format controller for the "viewer" side of a remote control session.
 * Smoothly interpolates camera position/target toward the latest state
 * received from the controller.
 */
export class RemoteControlController implements FormatController {
  private targetPosition = new THREE.Vector3();
  private targetLookAt = new THREE.Vector3();
  private targetFov: number;
  private hasTarget = false;
  private active = true;

  /** Interpolation speed (higher = snappier). */
  private lerpSpeed = 8;

  constructor(
    private camera: THREE.PerspectiveCamera,
    private orbitTarget: THREE.Vector3,
  ) {
    this.targetPosition.copy(camera.position);
    this.targetLookAt.copy(orbitTarget);
    this.targetFov = camera.fov;
  }

  /** Called when a camera-sync message arrives from the controller. */
  applyCameraSync(msg: RemoteControlCameraSync): void {
    this.targetPosition.set(msg.position[0], msg.position[1], msg.position[2]);
    this.targetLookAt.set(msg.target[0], msg.target[1], msg.target[2]);
    this.targetFov = msg.fov;
    this.hasTarget = true;
  }

  update(dt: number): void {
    if (!this.active || !this.hasTarget) return;

    const t = 1 - Math.exp(-this.lerpSpeed * dt);

    this.camera.position.lerp(this.targetPosition, t);
    this.orbitTarget.lerp(this.targetLookAt, t);

    if (Math.abs(this.camera.fov - this.targetFov) > 0.01) {
      this.camera.fov += (this.targetFov - this.camera.fov) * t;
      this.camera.updateProjectionMatrix();
    }
  }

  togglePlayPause(): boolean {
    this.active = !this.active;
    return this.active;
  }

  isPlaying(): boolean {
    return this.active;
  }

  dispose(): void {
    // No resources to clean up
  }
}
