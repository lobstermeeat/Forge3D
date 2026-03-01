import * as THREE from 'three';
import type { InteractionDef } from '@forge3d/shared';
import type { FormatController } from './types';

export type ShowInfoCallback = (entityId: string, payload: Record<string, unknown>) => void;
export type NavigateCallback = (url: string) => void;

export interface InteractiveCallbacks {
  onShowInfo?: ShowInfoCallback;
  onNavigate?: NavigateCallback;
}

/**
 * Interactive format: clickable hotspots on 3D objects that trigger actions.
 * Raycasts on hover (highlight) and click (dispatch action).
 */
export class InteractiveController implements FormatController {
  private raycaster = new THREE.Raycaster();
  private mouse = new THREE.Vector2();
  private interactiveObjects = new Map<THREE.Object3D, InteractionDef>();
  private hoveredObject: THREE.Object3D | null = null;
  private originalEmissive = new Map<THREE.Mesh, { color: THREE.Color; intensity: number }>();
  private active = true;

  constructor(
    private camera: THREE.PerspectiveCamera,
    private scene: THREE.Scene,
    private canvas: HTMLCanvasElement,
    interactions: InteractionDef[],
    private callbacks: InteractiveCallbacks = {},
  ) {
    // Map entityIds to scene objects
    for (const def of interactions) {
      scene.traverse((obj) => {
        if (obj.userData.entityId === def.entityId || obj.name === def.entityId) {
          this.interactiveObjects.set(obj, def);
        }
      });
    }

    canvas.addEventListener('pointermove', this.onPointerMove);
    canvas.addEventListener('click', this.onClick);
  }

  private onPointerMove = (e: PointerEvent): void => {
    if (!this.active) return;

    const rect = this.canvas.getBoundingClientRect();
    this.mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
    this.mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;

    this.raycaster.setFromCamera(this.mouse, this.camera);
    const targets = Array.from(this.interactiveObjects.keys());
    const hits = this.raycaster.intersectObjects(targets, true);

    // Walk up parent chain to find the interactive root object
    let hitObj: THREE.Object3D | null = null;
    for (const hit of hits) {
      let current: THREE.Object3D | null = hit.object;
      while (current) {
        if (this.interactiveObjects.has(current)) {
          hitObj = current;
          break;
        }
        current = current.parent;
      }
      if (hitObj) break;
    }

    if (hitObj !== this.hoveredObject) {
      this.clearHighlight();
      this.hoveredObject = hitObj;
      if (hitObj) this.applyHighlight(hitObj);
    }

    this.canvas.style.cursor = hitObj ? 'pointer' : 'default';
  };

  private onClick = (): void => {
    if (!this.active || !this.hoveredObject) return;

    const def = this.interactiveObjects.get(this.hoveredObject);
    if (!def) return;

    switch (def.action) {
      case 'showInfo':
        this.callbacks.onShowInfo?.(def.entityId, def.payload);
        break;
      case 'openUrl': {
        const url = def.payload['url'];
        if (typeof url === 'string') window.open(url, '_blank', 'noopener');
        break;
      }
      case 'navigate': {
        const navUrl = def.payload['url'];
        if (typeof navUrl === 'string') this.callbacks.onNavigate?.(navUrl);
        break;
      }
      case 'playAnimation':
        // Find AnimationMixer on the object and play clip
        this.playObjectAnimation(this.hoveredObject, def.payload);
        break;
    }
  };

  private playObjectAnimation(obj: THREE.Object3D, payload: Record<string, unknown>): void {
    const clipName = payload['clip'] as string | undefined;
    const clips = (obj as THREE.Object3D & { animations?: THREE.AnimationClip[] }).animations;
    if (!clips || clips.length === 0) return;

    const clip = clipName ? clips.find((c) => c.name === clipName) : clips[0];
    if (!clip) return;

    const mixer = new THREE.AnimationMixer(obj);
    const action = mixer.clipAction(clip);
    action.setLoop(THREE.LoopOnce, 1);
    action.clampWhenFinished = true;
    action.play();

    // Drive the mixer for the clip duration
    const start = performance.now();
    const tick = () => {
      const elapsed = (performance.now() - start) / 1000;
      if (elapsed < clip.duration + 0.1) {
        mixer.update(1 / 60);
        requestAnimationFrame(tick);
      } else {
        action.stop();
      }
    };
    requestAnimationFrame(tick);
  }

  private applyHighlight(obj: THREE.Object3D): void {
    obj.traverse((child) => {
      if (child instanceof THREE.Mesh && child.material instanceof THREE.MeshStandardMaterial) {
        this.originalEmissive.set(child, {
          color: child.material.emissive.clone(),
          intensity: child.material.emissiveIntensity,
        });
        child.material.emissive.set(0x89b4fa); // Catppuccin blue
        child.material.emissiveIntensity = 0.3;
      }
    });
  }

  private clearHighlight(): void {
    for (const [mesh, original] of this.originalEmissive) {
      if (mesh.material instanceof THREE.MeshStandardMaterial) {
        mesh.material.emissive.copy(original.color);
        mesh.material.emissiveIntensity = original.intensity;
      }
    }
    this.originalEmissive.clear();
  }

  togglePlayPause(): boolean {
    this.active = !this.active;
    if (!this.active) {
      this.clearHighlight();
      this.canvas.style.cursor = 'default';
    }
    return this.active;
  }

  isPlaying(): boolean {
    return this.active;
  }

  update(_dt: number): void {
    // Interactive format doesn't need per-frame updates
  }

  dispose(): void {
    this.clearHighlight();
    this.canvas.removeEventListener('pointermove', this.onPointerMove);
    this.canvas.removeEventListener('click', this.onClick);
    this.canvas.style.cursor = 'default';
  }
}
