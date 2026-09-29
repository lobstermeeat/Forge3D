import * as THREE from 'three';
import { RoomEnvironment } from 'three/examples/jsm/environments/RoomEnvironment.js';

interface PMREMGeneratorLike {
  fromScene(scene: THREE.Scene, sigma?: number): { texture: THREE.Texture };
}

/** A PMREM generator class: `THREE.PMREMGenerator` for WebGL, the one from `three/webgpu` for WebGPU. */
export type PMREMGeneratorClass = new (renderer: never) => PMREMGeneratorLike;

/**
 * Neutral studio lighting for PBR materials, shared by the editor and the viewer.
 *
 * Metallic and glossy surfaces reflect their surroundings; with only ambient and
 * direct lights they render nearly black. This bakes a procedural room into a
 * prefiltered environment map, so there is no HDRI file to download or license.
 */
export function createStudioEnvironment(
  renderer: unknown,
  PMREMGenerator: PMREMGeneratorClass = THREE.PMREMGenerator as unknown as PMREMGeneratorClass,
): THREE.Texture {
  const pmrem = new PMREMGenerator(renderer as never);
  const room = new RoomEnvironment();
  const texture = pmrem.fromScene(room, 0.04).texture;
  room.dispose();
  return texture;
}

/**
 * Tone mapping shared by the editor and the viewer, so Play shows what viewers see.
 * Khronos PBR Neutral keeps the colours creators pick (ACES shifts and desaturates them).
 */
export const VIEWER_TONE_MAPPING = THREE.NeutralToneMapping;

/** How strongly the studio environment lights the scene (on top of the 0.4 ambient). */
export const STUDIO_ENVIRONMENT_INTENSITY = 0.4;

/**
 * The colour to render so that, after Khronos PBR Neutral tone mapping, it shows as `target`.
 * Keeps editor chrome (sky, grid, backgrounds) at its designed colours. Neutral only
 * subtracts a small offset below its highlight compression (linear 0.76), which covers
 * every colour this is used for.
 */
export function preToneMapped(target: THREE.Color): THREE.Color {
  const min = Math.min(target.r, target.g, target.b);
  // Neutral subtracts offset(x) = x < 0.08 ? x - 6.25x² : 0.04, where x is the smallest channel
  const offset = min < 0.04 ? Math.sqrt(min / 6.25) - min : 0.04;
  return new THREE.Color(target.r + offset, target.g + offset, target.b + offset);
}
