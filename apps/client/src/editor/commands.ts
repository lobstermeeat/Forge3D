import { Entity } from '@forge3d/engine';
import type { Command, SceneManager } from '@forge3d/engine';
import type { EntityData, ModelData } from '@forge3d/shared';

/** Adds an entity from a full snapshot (used for lights and duplicates, which need more than a mesh). */
export class AddSnapshotCommand implements Command {
  description: string;

  constructor(
    private sceneManager: SceneManager,
    private data: EntityData,
  ) {
    this.description = `Add ${data.name}`;
  }

  execute(): void {
    if (this.sceneManager.getEntity(this.data.id)) return;
    this.sceneManager.addEntity(Entity.deserialize(structuredClone(this.data)));
  }

  undo(): void {
    this.sceneManager.removeEntity(this.data.id);
  }

  getEntityId(): string {
    return this.data.id;
  }
}

/** Points a model entity at another file, such as the final that replaces its preview. */
export class SetModelCommand implements Command {
  description: string;

  constructor(
    private sceneManager: SceneManager,
    private entityId: string,
    private before: ModelData,
    private after: ModelData,
  ) {
    this.description = after.quality === 'final' ? 'Use final model' : 'Change model';
  }

  execute(): void {
    this.apply(this.after);
  }

  undo(): void {
    this.apply(this.before);
  }

  private apply(model: ModelData): void {
    const entity = this.sceneManager.getEntity(this.entityId);
    if (!entity) return;
    entity.setComponent('model', structuredClone(model));
    this.sceneManager.notifyChange();
  }
}
