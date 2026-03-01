import type { ViewerControls } from '../ViewerControls';
import type { FormatController } from './types';

export interface TurntableConfig {
  speed: number; // rotations per second factor
  axis: 'y' | 'x';
}

/**
 * Turntable format: auto-rotates the camera around the scene center.
 * User can pause/resume and override with manual interaction.
 */
export class TurntableController implements FormatController {
  private playing = true;
  private idleTimeout: ReturnType<typeof setTimeout> | null = null;
  private userInteracting = false;

  constructor(
    private controls: ViewerControls,
    private config: TurntableConfig,
  ) {
    this.applyAutoRotate();

    // Detect user interaction to pause auto-rotate
    const el = this.controls.controls.domElement;
    if (el) {
      el.addEventListener('pointerdown', this.onInteractionStart);
      el.addEventListener('pointerup', this.onInteractionEnd);
      el.addEventListener('wheel', this.onInteractionStart);
    }
  }

  private applyAutoRotate(): void {
    this.controls.setAutoRotate(this.playing && !this.userInteracting, this.config.speed * 2);
  }

  private onInteractionStart = (): void => {
    this.userInteracting = true;
    this.applyAutoRotate();

    if (this.idleTimeout) {
      clearTimeout(this.idleTimeout);
      this.idleTimeout = null;
    }
  };

  private onInteractionEnd = (): void => {
    // Resume auto-rotate after 3 seconds idle
    this.idleTimeout = setTimeout(() => {
      this.userInteracting = false;
      this.applyAutoRotate();
    }, 3000);
  };

  togglePlayPause(): boolean {
    this.playing = !this.playing;
    this.applyAutoRotate();
    return this.playing;
  }

  isPlaying(): boolean {
    return this.playing;
  }

  update(_dt: number): void {
    // Auto-rotate is handled internally by OrbitControls
  }

  dispose(): void {
    if (this.idleTimeout) clearTimeout(this.idleTimeout);

    const el = this.controls.controls.domElement;
    if (el) {
      el.removeEventListener('pointerdown', this.onInteractionStart);
      el.removeEventListener('pointerup', this.onInteractionEnd);
      el.removeEventListener('wheel', this.onInteractionStart);
    }

    this.controls.setAutoRotate(false);
  }
}
