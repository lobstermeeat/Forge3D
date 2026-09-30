import * as THREE from 'three';
import type {
  TransformData,
  MeshRendererData,
  MaterialDescriptor,
  LightData,
  ModelData,
} from '@forge3d/shared';
import { SceneManager } from './SceneManager';
import { Entity } from '../ecs/Entity';
import { MeshFactory } from '../core/MeshFactory';
import { MaterialFactory } from '../materials/MaterialFactory';
import { disposeObject, prepareModel } from '../io/models';

/** Loads a model file (a GLB) into three.js objects. The app supplies it: KTX2 needs the renderer. */
export type ModelLoader = (url: string) => Promise<THREE.Object3D>;

export interface ModelLoadEvent {
  entityId: string;
  url: string;
  status: 'loaded' | 'failed';
  error?: unknown;
}

export class SceneBridge {
  private entityToObject = new Map<string, THREE.Object3D>();
  private objectToEntity = new Map<THREE.Object3D, string>();
  private meshFactory = new MeshFactory();
  private materialFactory = new MaterialFactory();
  private unsubscribe: (() => void) | null = null;
  private modelLoader: ModelLoader | null = null;
  /** The file each model entity shows, or is loading: a changed url loads again */
  private modelUrls = new Map<string, string>();
  private modelListeners = new Set<(event: ModelLoadEvent) => void>();

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
      obj.removeFromParent();
    }
    this.entityToObject.clear();
    this.objectToEntity.clear();
    this.modelUrls.clear();
  }

  /**
   * Sets how model files are loaded. Model entities (a `model` component) show nothing until one
   * is set; setting it loads every model in the scene.
   */
  setModelLoader(loader: ModelLoader | null): void {
    this.modelLoader = loader;
    this.modelUrls.clear();
    if (this.unsubscribe) this.sync();
  }

  /** Called whenever a model finishes loading or fails. Returns a function that stops it. */
  onModelLoad(listener: (event: ModelLoadEvent) => void): () => void {
    this.modelListeners.add(listener);
    return () => this.modelListeners.delete(listener);
  }

  private sync(): void {
    const entities = this.sceneManager.getAllEntities();
    const currentIds = new Set(entities.map((e) => e.id));

    // Remove stale objects
    for (const [id, obj] of this.entityToObject) {
      if (!currentIds.has(id)) {
        obj.removeFromParent();
        this.objectToEntity.delete(obj);
        this.entityToObject.delete(id);
        this.modelUrls.delete(id);
      }
    }

    // Create objects for new entities first, so parents exist before children attach
    for (const entity of entities) {
      if (this.entityToObject.has(entity.id)) continue;
      const obj = this.createObject(entity);
      if (obj) {
        this.entityToObject.set(entity.id, obj);
        this.objectToEntity.set(obj, entity.id);
      }
    }

    // Entity transforms are local to the parent entity, so mirror the hierarchy
    for (const entity of entities) {
      const obj = this.entityToObject.get(entity.id);
      if (!obj) continue;
      const parent = this.parentObjectFor(entity);
      if (obj.parent !== parent) parent.add(obj);
      this.applyTransform(obj, entity.transform);
      obj.name = entity.name;
      const model = entity.getComponent<ModelData>('model');
      if (model && this.modelUrls.get(entity.id) !== model.url) {
        this.loadModel(entity.id, obj, model.url);
      }
    }
  }

  /** Loads a model entity's file into its object, replacing what it showed before. */
  private loadModel(entityId: string, holder: THREE.Object3D, url: string): void {
    if (!this.modelLoader) return;
    // Recorded before loading, so later syncs don't start the same load again (or retry a failure)
    this.modelUrls.set(entityId, url);
    this.modelLoader(url).then(
      (content) => {
        // Superseded while it loaded: another file, or the entity was removed or rebuilt
        if (this.modelUrls.get(entityId) !== url || this.entityToObject.get(entityId) !== holder) {
          disposeObject(content);
          return;
        }
        for (const old of holder.children.filter((child) => child.userData['modelContent'])) {
          holder.remove(old);
          disposeObject(old);
        }
        holder.add(prepareModel(content));
        this.emitModel({ entityId, url, status: 'loaded' });
      },
      (error: unknown) => {
        if (this.modelUrls.get(entityId) === url) {
          this.emitModel({ entityId, url, status: 'failed', error });
        }
      },
    );
  }

  private emitModel(event: ModelLoadEvent): void {
    for (const listener of this.modelListeners) listener(event);
  }

  private createObject(entity: Entity): THREE.Object3D | null {
    // A model file loads into this group (see loadModel), so transforms and picking work at once
    if (entity.hasComponent('model')) {
      const group = new THREE.Group();
      group.name = entity.name;
      return group;
    }

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

  /** What a click can hit: every entity's object, plus the meshes inside model entities. */
  getPickableObjects(): THREE.Object3D[] {
    const objects: THREE.Object3D[] = [];
    for (const [id, obj] of this.entityToObject) {
      objects.push(obj);
      if (!this.modelUrls.has(id)) continue;
      obj.traverse((child) => {
        if (child !== obj && (child as THREE.Mesh).isMesh) objects.push(child);
      });
    }
    return objects;
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
    // sync() already made a placeholder for this entity when it was added; drop it
    const placeholder = this.entityToObject.get(entityId);
    if (placeholder && placeholder !== obj) {
      placeholder.removeFromParent();
      this.objectToEntity.delete(placeholder);
    }
    this.entityToObject.set(entityId, obj);
    this.objectToEntity.set(obj, entityId);
    const entity = this.sceneManager.getEntity(entityId);
    (entity ? this.parentObjectFor(entity) : this.threeScene).add(obj);
  }

  /** The object an entity's object should live under: its parent entity's object, or the scene. */
  private parentObjectFor(entity: Entity): THREE.Object3D {
    const parent = entity.parentId ? this.entityToObject.get(entity.parentId) : undefined;
    return parent ?? this.threeScene;
  }

  /** Replace a light object after property changes */
  updateLight(entityId: string, data: LightData): void {
    const oldObj = this.entityToObject.get(entityId);
    if (oldObj) {
      oldObj.removeFromParent();
      this.objectToEntity.delete(oldObj);
    }
    const entity = this.sceneManager.getEntity(entityId);
    const newLight = this.createLight(data, entity?.name ?? 'Light');
    if (entity) {
      this.applyTransform(newLight, entity.transform);
    }
    this.entityToObject.set(entityId, newLight);
    this.objectToEntity.set(newLight, entityId);
    (entity ? this.parentObjectFor(entity) : this.threeScene).add(newLight);
  }

  dispose(): void {
    this.disconnect();
    this.materialFactory.clearCache();
  }
}
