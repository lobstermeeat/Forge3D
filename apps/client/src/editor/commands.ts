import { Entity } from '@forge3d/engine';
import type { Command, SceneManager } from '@forge3d/engine';
import type { EntityData } from '@forge3d/shared';

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
