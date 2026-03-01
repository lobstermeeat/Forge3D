import type { Command } from './CommandHistory';
import type { TransformData, MeshRendererData } from '@forge3d/shared';
import { SceneManager } from '../scene/SceneManager';
import { Entity } from '../ecs/Entity';

export class AddEntityCommand implements Command {
  description: string;
  private entityId: string;
  private entityName: string;
  private meshRenderer: MeshRendererData | undefined;
  private transform: TransformData | undefined;

  constructor(
    private sceneManager: SceneManager,
    name: string,
    meshRenderer?: MeshRendererData,
    transform?: TransformData,
  ) {
    const entity = new Entity(name);
    this.entityId = entity.id;
    this.entityName = name;
    this.meshRenderer = meshRenderer;
    this.transform = transform;
    this.description = `Add ${name}`;
  }

  execute(): void {
    if (this.sceneManager.getEntity(this.entityId)) return;
    const entity = new Entity(this.entityName, this.entityId);
    if (this.meshRenderer) {
      entity.setComponent('meshRenderer', this.meshRenderer);
    }
    if (this.transform) {
      entity.transform = this.transform;
    }
    this.sceneManager.addEntity(entity);
  }

  undo(): void {
    this.sceneManager.removeEntity(this.entityId);
  }

  getEntityId(): string {
    return this.entityId;
  }
}

export class RemoveEntityCommand implements Command {
  description: string;
  private entitySnapshot: ReturnType<Entity['serialize']> | null = null;

  constructor(
    private sceneManager: SceneManager,
    private entityId: string,
  ) {
    const entity = sceneManager.getEntity(entityId);
    this.description = `Remove ${entity?.name ?? entityId}`;
  }

  execute(): void {
    const entity = this.sceneManager.getEntity(this.entityId);
    if (entity) {
      this.entitySnapshot = entity.serialize();
    }
    this.sceneManager.removeEntity(this.entityId);
  }

  undo(): void {
    if (this.entitySnapshot) {
      const entity = Entity.deserialize(this.entitySnapshot);
      this.sceneManager.addEntity(entity);
    }
  }
}

export class TransformCommand implements Command {
  description = 'Transform';

  constructor(
    private sceneManager: SceneManager,
    private entityId: string,
    private before: TransformData,
    private after: TransformData,
  ) {}

  execute(): void {
    const entity = this.sceneManager.getEntity(this.entityId);
    if (entity) {
      entity.transform = { ...this.after };
      this.sceneManager.notifyChange();
    }
  }

  undo(): void {
    const entity = this.sceneManager.getEntity(this.entityId);
    if (entity) {
      entity.transform = { ...this.before };
      this.sceneManager.notifyChange();
    }
  }
}

export class RenameEntityCommand implements Command {
  description: string;
  private oldName: string;

  constructor(
    private sceneManager: SceneManager,
    private entityId: string,
    private newName: string,
  ) {
    const entity = sceneManager.getEntity(entityId);
    this.oldName = entity?.name ?? '';
    this.description = `Rename to ${newName}`;
  }

  execute(): void {
    const entity = this.sceneManager.getEntity(this.entityId);
    if (entity) {
      entity.name = this.newName;
      this.sceneManager.notifyChange();
    }
  }

  undo(): void {
    const entity = this.sceneManager.getEntity(this.entityId);
    if (entity) {
      entity.name = this.oldName;
      this.sceneManager.notifyChange();
    }
  }
}
