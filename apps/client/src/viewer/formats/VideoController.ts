import * as THREE from 'three';
import type { CameraKeyframe } from '@forge3d/shared';
import { ease } from '@forge3d/engine';
import type { ViewerControls } from '../ViewerControls';
import type { FormatController } from './types';

/**
 * Video format: plays a camera flythrough along a keyframe path.
 * Interpolates camera position and target between keyframes with easing.
 */
export class VideoController implements FormatController {
  private playing = true;
  private currentTime = 0;
  private duration: number;
  private keyframes: CameraKeyframe[];
  private camera: THREE.PerspectiveCamera;

  // Reusable vectors to avoid per-frame allocations
  private posA = new THREE.Vector3();
  private posB = new THREE.Vector3();
  private tgtA = new THREE.Vector3();
  private tgtB = new THREE.Vector3();

  constructor(
    private controls: ViewerControls,
    camera: THREE.PerspectiveCamera,
    keyframes: CameraKeyframe[],
  ) {
    this.camera = camera;
    this.keyframes = [...keyframes].sort((a, b) => a.time - b.time);
    this.duration = this.keyframes[this.keyframes.length - 1]?.time ?? 0;

    // Start playing — disable orbit controls so camera is driven programmatically
    this.controls.setEnabled(false);
    this.applyCamera();
  }

  togglePlayPause(): boolean {
    this.playing = !this.playing;
    this.controls.setEnabled(!this.playing);
    return this.playing;
  }

  isPlaying(): boolean {
    return this.playing;
  }

  getDuration(): number {
    return this.duration;
  }

  getCurrentTime(): number {
    return this.currentTime;
  }

  seek(time: number): void {
    this.currentTime = Math.max(0, Math.min(time, this.duration));
    this.applyCamera();
  }

  update(dt: number): void {
    if (!this.playing || this.duration === 0) return;

    this.currentTime += dt;
    if (this.currentTime >= this.duration) {
      this.currentTime = 0; // loop
    }

    this.applyCamera();
  }

  private applyCamera(): void {
    const kfs = this.keyframes;
    if (kfs.length === 0) return;

    // Find the two keyframes that bracket currentTime
    let idxB = kfs.findIndex((kf) => kf.time >= this.currentTime);
    if (idxB === -1) idxB = kfs.length - 1;
    const idxA = Math.max(0, idxB - 1);

    const kfA = kfs[idxA]!;
    const kfB = kfs[idxB]!;

    if (idxA === idxB || kfA.time === kfB.time) {
      // At or before first keyframe, or same keyframe
      this.camera.position.set(kfA.position[0], kfA.position[1], kfA.position[2]);
      this.controls.controls.target.set(kfA.target[0], kfA.target[1], kfA.target[2]);
      return;
    }

    // Compute local t and apply easing from the target keyframe
    const localT = (this.currentTime - kfA.time) / (kfB.time - kfA.time);
    const easedT = ease(kfB.easing, localT);

    this.posA.set(kfA.position[0], kfA.position[1], kfA.position[2]);
    this.posB.set(kfB.position[0], kfB.position[1], kfB.position[2]);
    this.tgtA.set(kfA.target[0], kfA.target[1], kfA.target[2]);
    this.tgtB.set(kfB.target[0], kfB.target[1], kfB.target[2]);

    this.camera.position.lerpVectors(this.posA, this.posB, easedT);
    this.controls.controls.target.lerpVectors(this.tgtA, this.tgtB, easedT);
  }

  dispose(): void {
    this.controls.setEnabled(true);
  }
}
