import type { EntityData, TransformData } from '@forge3d/shared';
import { DEFAULT_TRANSFORM, generateId } from '@forge3d/shared';

export class Entity {
  readonly id: string;
  name: string;
  parentId: string | undefined;
  children: string[] = [];
  components: Map<string, unknown> = new Map();

  constructor(name: string, id?: string) {
    this.id = id ?? generateId();
    this.name = name;
    this.setComponent('transform', { ...DEFAULT_TRANSFORM });
  }

  setComponent<T>(key: string, data: T): void {
    this.components.set(key, data);
  }

  getComponent<T>(key: string): T | undefined {
    return this.components.get(key) as T | undefined;
  }

  removeComponent(key: string): boolean {
    return this.components.delete(key);
  }

  hasComponent(key: string): boolean {
    return this.components.has(key);
  }

  get transform(): TransformData {
    return this.getComponent<TransformData>('transform')!;
  }

  set transform(data: TransformData) {
    this.setComponent('transform', data);
  }

  serialize(): EntityData {
    const data: EntityData = {
      id: this.id,
      name: this.name,
      parentId: this.parentId,
      components: {
        transform: this.getComponent('transform'),
      },
    };

    if (this.hasComponent('meshRenderer')) {
      data.components.meshRenderer = this.getComponent('meshRenderer');
    }
    if (this.hasComponent('camera')) {
      data.components.camera = this.getComponent('camera');
    }
    if (this.hasComponent('light')) {
      data.components.light = this.getComponent('light');
    }

    return data;
  }

  static deserialize(data: EntityData): Entity {
    const entity = new Entity(data.name, data.id);
    entity.parentId = data.parentId;

    if (data.components.transform) {
      entity.setComponent('transform', data.components.transform);
    }
    if (data.components.meshRenderer) {
      entity.setComponent('meshRenderer', data.components.meshRenderer);
    }
    if (data.components.camera) {
      entity.setComponent('camera', data.components.camera);
    }
    if (data.components.light) {
      entity.setComponent('light', data.components.light);
    }

    return entity;
  }
}
