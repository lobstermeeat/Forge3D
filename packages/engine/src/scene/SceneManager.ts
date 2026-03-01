import type { SceneData, MaterialDescriptor } from '@forge3d/shared';
import { DEFAULT_MATERIAL, SCENE_VERSION, generateId } from '@forge3d/shared';
import { Entity } from '../ecs/Entity';

export class SceneManager {
  private entities: Map<string, Entity> = new Map();
  private materials: MaterialDescriptor[] = [{ ...DEFAULT_MATERIAL }];
  private selectedEntityId: string | null = null;
  private listeners: Set<() => void> = new Set();

  getEntity(id: string): Entity | undefined {
    return this.entities.get(id);
  }

  getAllEntities(): Entity[] {
    return Array.from(this.entities.values());
  }

  getRootEntities(): Entity[] {
    return this.getAllEntities().filter((e) => !e.parentId);
  }

  getChildren(parentId: string): Entity[] {
    return this.getAllEntities().filter((e) => e.parentId === parentId);
  }

  addEntity(entity: Entity): void {
    this.entities.set(entity.id, entity);
    if (entity.parentId) {
      const parent = this.entities.get(entity.parentId);
      if (parent && !parent.children.includes(entity.id)) {
        parent.children.push(entity.id);
      }
    }
    this.notify();
  }

  removeEntity(id: string): void {
    const entity = this.entities.get(id);
    if (!entity) return;

    // Remove children recursively
    for (const childId of [...entity.children]) {
      this.removeEntity(childId);
    }

    // Remove from parent's children list
    if (entity.parentId) {
      const parent = this.entities.get(entity.parentId);
      if (parent) {
        parent.children = parent.children.filter((cid) => cid !== id);
      }
    }

    this.entities.delete(id);
    if (this.selectedEntityId === id) {
      this.selectedEntityId = null;
    }
    this.notify();
  }

  selectEntity(id: string | null): void {
    this.selectedEntityId = id;
    this.notify();
  }

  getSelectedEntity(): Entity | null {
    if (!this.selectedEntityId) return null;
    return this.entities.get(this.selectedEntityId) ?? null;
  }

  getMaterials(): MaterialDescriptor[] {
    return this.materials;
  }

  addMaterial(material: MaterialDescriptor): number {
    this.materials.push(material);
    this.notify();
    return this.materials.length - 1;
  }

  updateMaterial(index: number, material: Partial<MaterialDescriptor>): void {
    const existing = this.materials[index];
    if (!existing) return;
    this.materials[index] = { ...existing, ...material };
    this.notify();
  }

  serialize(): SceneData {
    return {
      version: SCENE_VERSION,
      entities: this.getAllEntities().map((e) => e.serialize()),
      materials: [...this.materials],
    };
  }

  deserialize(data: SceneData): void {
    this.entities.clear();
    this.materials = data.materials.length > 0 ? [...data.materials] : [{ ...DEFAULT_MATERIAL }];
    this.selectedEntityId = null;

    for (const entityData of data.entities) {
      const entity = Entity.deserialize(entityData);
      this.entities.set(entity.id, entity);
    }

    // Rebuild children arrays
    for (const entity of this.entities.values()) {
      if (entity.parentId) {
        const parent = this.entities.get(entity.parentId);
        if (parent && !parent.children.includes(entity.id)) {
          parent.children.push(entity.id);
        }
      }
    }

    this.notify();
  }

  clear(): void {
    this.entities.clear();
    this.materials = [{ ...DEFAULT_MATERIAL }];
    this.selectedEntityId = null;
    this.notify();
  }

  subscribe(listener: () => void): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  notifyChange(): void {
    this.notify();
  }

  private notify(): void {
    for (const listener of this.listeners) {
      listener();
    }
  }
}
