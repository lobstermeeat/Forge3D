import * as THREE from 'three';
import type { TransformData, MeshRendererData, MaterialDescriptor, LightData } from '@forge3d/shared';
import { SceneManager } from './SceneManager';
import { Entity } from '../ecs/Entity';
import { MeshFactory } from '../core/MeshFactory';
import { MaterialFactory } from '../materials/MaterialFactory';

export class SceneBridge {
  private entityToObject = new Map<string, THREE.Object3D>();
  private objectToEntity = new Map<THREE.Object3D, string>();
  private meshFactory = new MeshFactory();
  private materialFactory = new MaterialFactory();
  private unsubscribe: (() => void) | null = null;

  constructor(
    private sceneManager: SceneManager,
    private threeScene: THREE.Scene,
  ) {}

  connect(): void {
    this.unsubscribe = this.sceneManager.subscribe(() => this.sync());
    this.sync();
  }

  disconnect(): void {
    this.unsubscribe?.();
    this.unsubscribe = null;
    for (const obj of this.entityToObject.values()) {
      this.threeScene.remove(obj);
    }
    this.entityToObject.clear();
    this.objectToEntity.clear();
  }

  private sync(): void {
    const entities = this.sceneManager.getAllEntities();
    const currentIds = new Set(entities.map((e) => e.id));

    // Remove stale objects
    for (const [id, obj] of this.entityToObject) {
      if (!currentIds.has(id)) {
        this.threeScene.remove(obj);
        this.objectToEntity.delete(obj);
        this.entityToObject.delete(id);
      }
    }

    // Add or update
    for (const entity of entities) {
      let obj = this.entityToObject.get(entity.id);

      if (!obj) {
        obj = this.createObject(entity) ?? undefined;
        if (obj) {
          this.entityToObject.set(entity.id, obj);
          this.objectToEntity.set(obj, entity.id);
          this.threeScene.add(obj);
        }
      }

      if (obj) {
        this.applyTransform(obj, entity.transform);
        obj.name = entity.name;
      }
    }
  }

  private createObject(entity: Entity): THREE.Object3D | null {
    // Check for light component
    const lightData = entity.getComponent<LightData>('light');
    if (lightData) {
      return this.createLight(lightData, entity.name);
    }

    const meshData = entity.getComponent<MeshRendererData>('meshRenderer');
    if (!meshData) {
      const group = new THREE.Group();
      group.name = entity.name;
      return group;
    }

    const geometry = this.meshFactory.createGeometry(meshData);
    const materials = this.sceneManager.getMaterials();
    const matDesc = materials[meshData.materialIndex] ?? materials[0]!;
    const material = this.materialFactory.create(matDesc!);

    const mesh = new THREE.Mesh(geometry, material);
    mesh.name = entity.name;
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    return mesh;
  }

  private createLight(data: LightData, name: string): THREE.Object3D {
    const color = new THREE.Color(data.color[0], data.color[1], data.color[2]);
    let light: THREE.Light;

    switch (data.type) {
      case 'directional':
        light = new THREE.DirectionalLight(color, data.intensity);
        break;
      case 'point':
        light = new THREE.PointLight(color, data.intensity);
        break;
      case 'ambient':
        light = new THREE.AmbientLight(color, data.intensity);
        break;
      case 'spot':
        light = new THREE.SpotLight(color, data.intensity);
        break;
      default:
        light = new THREE.PointLight(color, data.intensity);
    }

    light.name = name;
    return light;
  }

  private applyTransform(obj: THREE.Object3D, t: TransformData): void {
    obj.position.set(t.position[0], t.position[1], t.position[2]);
    obj.quaternion.set(t.rotation[0], t.rotation[1], t.rotation[2], t.rotation[3]);
    obj.scale.set(t.scale[0], t.scale[1], t.scale[2]);
  }

  getObject(entityId: string): THREE.Object3D | undefined {
    return this.entityToObject.get(entityId);
  }

  getEntityId(object: THREE.Object3D): string | undefined {
    let current: THREE.Object3D | null = object;
    while (current) {
      const id = this.objectToEntity.get(current);
      if (id) return id;
      current = current.parent;
    }
    return undefined;
  }

  getManagedObjects(): THREE.Object3D[] {
    return Array.from(this.entityToObject.values());
  }

  readTransform(entityId: string): TransformData | null {
    const obj = this.entityToObject.get(entityId);
    if (!obj) return null;
    return {
      position: [obj.position.x, obj.position.y, obj.position.z],
      rotation: [obj.quaternion.x, obj.quaternion.y, obj.quaternion.z, obj.quaternion.w],
      scale: [obj.scale.x, obj.scale.y, obj.scale.z],
    };
  }

  updateMaterial(entityId: string, descriptor: MaterialDescriptor): void {
    const obj = this.entityToObject.get(entityId);
    if (obj instanceof THREE.Mesh) {
      obj.material = this.materialFactory.create(descriptor);
    }
  }

  /** Register an already-created Three.js object (e.g. from GLTF import) */
  addExternalObject(entityId: string, obj: THREE.Object3D): void {
    this.entityToObject.set(entityId, obj);
    this.objectToEntity.set(obj, entityId);
    this.threeScene.add(obj);
  }

  /** Replace a light object after property changes */
  updateLight(entityId: string, data: LightData): void {
    const oldObj = this.entityToObject.get(entityId);
    if (oldObj) {
      this.threeScene.remove(oldObj);
      this.objectToEntity.delete(oldObj);
    }
    const entity = this.sceneManager.getEntity(entityId);
    const newLight = this.createLight(data, entity?.name ?? 'Light');
    if (entity) {
      this.applyTransform(newLight, entity.transform);
    }
    this.entityToObject.set(entityId, newLight);
    this.objectToEntity.set(newLight, entityId);
    this.threeScene.add(newLight);
  }

  dispose(): void {
    this.disconnect();
    this.materialFactory.clearCache();
  }
}
