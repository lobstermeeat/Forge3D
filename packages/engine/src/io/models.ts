import * as THREE from 'three';

/**
 * Wraps a loaded model so that it rests on its holder's origin by the centre of its base, with
 * shadows on. Every model is placed the same way wherever it loads (editor or published viewer),
 * so an entity's transform alone decides where it stands.
 */
export function prepareModel(content: THREE.Object3D): THREE.Object3D {
  const pivot = new THREE.Group();
  pivot.name = 'Model';
  pivot.userData['modelContent'] = true;
  pivot.add(content);
  pivot.updateMatrixWorld(true);
  const box = new THREE.Box3().setFromObject(content, true);
  if (!box.isEmpty()) {
    const centre = box.getCenter(new THREE.Vector3());
    content.position.x -= centre.x;
    content.position.z -= centre.z;
    content.position.y -= box.min.y;
  }
  content.traverse((object) => {
    if ((object as THREE.Mesh).isMesh) {
      object.castShadow = true;
      object.receiveShadow = true;
    }
  });
  return pivot;
}

/** Frees the GPU memory of an object's meshes, materials and textures. */
export function disposeObject(root: THREE.Object3D): void {
  root.traverse((object) => {
    const mesh = object as THREE.Mesh;
    if (!mesh.isMesh) return;
    mesh.geometry.dispose();
    for (const material of [mesh.material].flat()) {
      for (const value of Object.values(material)) {
        if (value instanceof THREE.Texture) value.dispose();
      }
      material.dispose();
    }
  });
}
