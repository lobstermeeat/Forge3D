import * as THREE from 'three';
import type { MeshRendererData, TransformData } from '@forge3d/shared';
import type { SceneManager } from '@forge3d/engine';

export type Primitive = Exclude<MeshRendererData['geometryType'], 'imported'>;

export const PRIMITIVES: { type: Primitive; label: string }[] = [
  { type: 'box', label: 'Box' },
  { type: 'sphere', label: 'Sphere' },
  { type: 'cylinder', label: 'Cylinder' },
  { type: 'plane', label: 'Plane' },
  { type: 'torus', label: 'Torus' },
];

export function primitiveLabel(type: Primitive): string {
  return PRIMITIVES.find((p) => p.type === type)?.label ?? type;
}

const GROUND = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);

/** Transform that rests a default-sized primitive on the ground at `point`. */
export function restingTransform(type: Primitive, point: THREE.Vector3): TransformData {
  const q = new THREE.Quaternion();
  let y = 0.5; // unit box, 0.5-radius sphere and 1-high cylinder
  if (type === 'plane') {
    q.setFromEuler(new THREE.Euler(-Math.PI / 2, 0, 0)); // lie flat instead of standing up
    y = 0.001;
  } else if (type === 'torus') {
    y = 0.7; // ring radius 0.5 + tube 0.2, standing upright
  }
  return { position: [point.x, y, point.z], rotation: [q.x, q.y, q.z, q.w], scale: [1, 1, 1] };
}

/** Point on the ground under normalised device coordinates, or `fallback` when looking at the sky. */
export function groundPointAt(
  camera: THREE.Camera,
  ndc: THREE.Vector2,
  fallback: THREE.Vector3,
): THREE.Vector3 {
  const ray = new THREE.Raycaster();
  ray.setFromCamera(ndc, camera);
  const p = new THREE.Vector3();
  const hit = ray.ray.intersectPlane(GROUND, p);
  if (!hit || p.distanceTo(camera.position) > 60) return fallback.clone().setY(0);
  return p;
}

export function snapTo(value: number, step: number): number {
  return Math.round(value / step) * step;
}

/** Nudge `p` sideways (relative to the camera) until it doesn't land on another top-level part. */
export function freeSpot(
  p: THREE.Vector3,
  sceneManager: SceneManager,
  camera: THREE.Camera,
  step = 1.5,
): THREE.Vector3 {
  const right = new THREE.Vector3();
  camera.getWorldDirection(right);
  right.cross(camera.up).setY(0);
  if (right.lengthSq() < 1e-6) right.set(1, 0, 0);
  right.normalize();

  const occupied = sceneManager
    .getAllEntities()
    .filter((e) => e.hasComponent('meshRenderer') && !e.parentId)
    .map((e) => e.transform.position);

  for (let i = 0; i < 12; i++) {
    const clash = occupied.some((q) => Math.hypot(q[0] - p.x, q[2] - p.z) < 1.1);
    if (!clash) break;
    p.addScaledVector(right, step);
  }
  return p;
}
