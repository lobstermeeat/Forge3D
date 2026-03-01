import * as Y from 'yjs';
import type { TransformData, MeshRendererData, LightData, CameraData, MaterialDescriptor } from '@forge3d/shared';
import { Entity } from '../ecs/Entity';
import type { SceneManager } from '../scene/SceneManager';

/**
 * Two-way binding between SceneManager and a Yjs shared document.
 *
 * Y.Doc structure:
 *   doc.getMap('entities')   → Y.Map<entityId, Y.Map<componentKey, JSON>>
 *   doc.getArray('materials') → Y.Array<MaterialDescriptor>
 *   doc.getMap('meta')        → Y.Map<'name'|'version', value>
 *
 * Local SceneManager changes  → pushed into Y.Doc  → synced to peers
 * Remote Y.Doc changes        → applied to SceneManager → UI re-renders
 */
export class YjsSceneBinding {
  private entitiesMap: Y.Map<Y.Map<unknown>>;
  private materialsArray: Y.Array<MaterialDescriptor>;
  private metaMap: Y.Map<unknown>;

  /** Prevents echo: when we write to Yjs from a local change, ignore the Yjs observe callback */
  private suppressRemote = false;
  /** Prevents echo: when we apply remote Yjs changes, ignore SceneManager's notify */
  private suppressLocal = false;

  private unsubscribeScene: (() => void) | null = null;
  private observers: (() => void)[] = [];

  constructor(
    private doc: Y.Doc,
    private sceneManager: SceneManager,
  ) {
    this.entitiesMap = doc.getMap('entities') as Y.Map<Y.Map<unknown>>;
    this.materialsArray = doc.getArray('materials');
    this.metaMap = doc.getMap('meta');
  }

  /**
   * Start two-way sync.
   * Call once after the Yjs provider is connected and initial state is loaded.
   */
  bind(): void {
    // 1. Seed Yjs from current SceneManager state (only if Yjs is empty — new doc)
    if (this.entitiesMap.size === 0) {
      this.pushFullStateToYjs();
    } else {
      // Yjs already has state (reconnect / peer loaded) — apply to SceneManager
      this.applyFullStateFromYjs();
    }

    // 2. Listen for local SceneManager changes → push to Yjs
    this.unsubscribeScene = this.sceneManager.subscribe(() => {
      if (this.suppressLocal) return;
      this.pushFullStateToYjs();
    });

    // 3. Listen for remote Yjs changes → apply to SceneManager
    const onEntitiesChange = () => {
      if (this.suppressRemote) return;
      this.applyFullStateFromYjs();
    };

    const onMaterialsChange = () => {
      if (this.suppressRemote) return;
      this.applyMaterialsFromYjs();
    };

    this.entitiesMap.observeDeep(onEntitiesChange);
    this.materialsArray.observe(onMaterialsChange);

    this.observers.push(
      () => this.entitiesMap.unobserveDeep(onEntitiesChange),
      () => this.materialsArray.unobserve(onMaterialsChange),
    );
  }

  unbind(): void {
    this.unsubscribeScene?.();
    this.unsubscribeScene = null;
    for (const unsub of this.observers) unsub();
    this.observers = [];
  }

  // ─── Local → Yjs ───────────────────────────────────────────

  private pushFullStateToYjs(): void {
    this.suppressRemote = true;

    this.doc.transact(() => {
      const currentEntities = this.sceneManager.getAllEntities();
      const currentIds = new Set(currentEntities.map((e) => e.id));

      // Remove entities from Yjs that no longer exist locally
      for (const key of this.entitiesMap.keys()) {
        if (!currentIds.has(key)) {
          this.entitiesMap.delete(key);
        }
      }

      // Upsert entities
      for (const entity of currentEntities) {
        this.pushEntityToYjs(entity);
      }

      // Sync materials
      const materials = this.sceneManager.getMaterials();
      // Replace materials array
      if (this.materialsArray.length > 0) {
        this.materialsArray.delete(0, this.materialsArray.length);
      }
      this.materialsArray.push(materials);
    });

    this.suppressRemote = false;
  }

  private pushEntityToYjs(entity: Entity): void {
    let yEntity = this.entitiesMap.get(entity.id);
    if (!yEntity) {
      yEntity = new Y.Map<unknown>();
      this.entitiesMap.set(entity.id, yEntity);
    }

    yEntity.set('name', entity.name);
    yEntity.set('parentId', entity.parentId ?? null);

    // Components
    const transform = entity.getComponent<TransformData>('transform');
    if (transform) yEntity.set('transform', transform);

    const meshRenderer = entity.getComponent<MeshRendererData>('meshRenderer');
    if (meshRenderer) {
      yEntity.set('meshRenderer', meshRenderer);
    } else {
      yEntity.delete('meshRenderer');
    }

    const light = entity.getComponent<LightData>('light');
    if (light) {
      yEntity.set('light', light);
    } else {
      yEntity.delete('light');
    }

    const camera = entity.getComponent<CameraData>('camera');
    if (camera) {
      yEntity.set('camera', camera);
    } else {
      yEntity.delete('camera');
    }
  }

  // ─── Yjs → Local ───────────────────────────────────────────

  private applyFullStateFromYjs(): void {
    this.suppressLocal = true;

    // Clear and rebuild
    this.sceneManager.clear();

    // Restore materials
    this.applyMaterialsFromYjs();

    // Restore entities
    for (const [id, yEntity] of this.entitiesMap) {
      const entity = this.yMapToEntity(id, yEntity);
      this.sceneManager.addEntity(entity);
    }

    this.suppressLocal = false;
  }

  private applyMaterialsFromYjs(): void {
    this.suppressLocal = true;

    const materials = this.materialsArray.toArray() as MaterialDescriptor[];
    // Re-set materials on SceneManager — use the internal approach
    // SceneManager doesn't have a `setMaterials` method, so we update one by one
    const currentMats = this.sceneManager.getMaterials();

    // Extend or update
    for (let i = 0; i < materials.length; i++) {
      if (i < currentMats.length) {
        this.sceneManager.updateMaterial(i, materials[i]!);
      } else {
        this.sceneManager.addMaterial(materials[i]!);
      }
    }

    this.suppressLocal = false;
  }

  private yMapToEntity(id: string, yEntity: Y.Map<unknown>): Entity {
    const name = (yEntity.get('name') as string) ?? 'Unnamed';
    const entity = new Entity(name, id);
    entity.parentId = (yEntity.get('parentId') as string | null) ?? undefined;

    const transform = yEntity.get('transform') as TransformData | undefined;
    if (transform) entity.setComponent('transform', transform);

    const meshRenderer = yEntity.get('meshRenderer') as MeshRendererData | undefined;
    if (meshRenderer) entity.setComponent('meshRenderer', meshRenderer);

    const light = yEntity.get('light') as LightData | undefined;
    if (light) entity.setComponent('light', light);

    const camera = yEntity.get('camera') as CameraData | undefined;
    if (camera) entity.setComponent('camera', camera);

    return entity;
  }
}
