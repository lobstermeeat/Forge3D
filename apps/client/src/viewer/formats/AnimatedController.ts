import * as THREE from 'three';
import type { FormatController } from './types';

export interface AnimatedConfig {
  duration: number;
}

/**
 * Animated format: plays GLTF skeletal/morph animations found in the scene.
 * Uses THREE.AnimationMixer to drive all AnimationClip objects.
 */
export class AnimatedController implements FormatController {
  private mixer: THREE.AnimationMixer | null = null;
  private actions: THREE.AnimationAction[] = [];
  private playing = true;
  private totalDuration: number;

  constructor(scene: THREE.Scene, config: AnimatedConfig) {
    this.totalDuration = config.duration;

    // Find animation clips attached to objects in the scene
    const clips: THREE.AnimationClip[] = [];
    scene.traverse((obj) => {
      const anims = (obj as THREE.Object3D & { animations?: THREE.AnimationClip[] }).animations;
      if (anims && anims.length > 0) {
        clips.push(...anims);
      }
    });

    if (clips.length > 0) {
      this.mixer = new THREE.AnimationMixer(scene);
      for (const clip of clips) {
        const action = this.mixer.clipAction(clip);
        action.play();
        this.actions.push(action);
      }
      // Use actual clip duration if longer than configured
      const maxClipDuration = Math.max(...clips.map((c) => c.duration));
      if (maxClipDuration > this.totalDuration) {
        this.totalDuration = maxClipDuration;
      }
    }
  }

  togglePlayPause(): boolean {
    this.playing = !this.playing;
    if (this.mixer) {
      this.mixer.timeScale = this.playing ? 1 : 0;
    }
    return this.playing;
  }

  isPlaying(): boolean {
    return this.playing;
  }

  getDuration(): number {
    return this.totalDuration;
  }

  getCurrentTime(): number {
    return this.mixer?.time ?? 0;
  }

  seek(time: number): void {
    if (this.mixer) {
      this.mixer.setTime(time);
    }
  }

  update(dt: number): void {
    if (!this.playing) return;
    this.mixer?.update(dt);
  }

  dispose(): void {
    for (const action of this.actions) {
      action.stop();
    }
    this.actions = [];
    this.mixer = null;
  }
}
