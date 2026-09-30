import type { MaterialDescriptor, MeshRendererData } from '@forge3d/shared';
import type { CommandHistory, SceneBridge, SceneManager } from '@forge3d/engine';

export const SWATCHES: { hex: string; name: string }[] = [
  { hex: '#f4efe6', name: 'Rice paper' },
  { hex: '#b9bec7', name: 'Ash' },
  { hex: '#6b717c', name: 'Slate' },
  { hex: '#2b2e34', name: 'Ink' },
  { hex: '#ffe2b6', name: 'Lantern' },
  { hex: '#ff7a2f', name: 'Ember' },
  { hex: '#d2452f', name: 'Vermilion' },
  { hex: '#8c2f24', name: 'Lacquer' },
  { hex: '#e0a83a', name: 'Ginkgo' },
  { hex: '#6e8b3d', name: 'Moss' },
  { hex: '#2f7f7a', name: 'Celadon' },
  { hex: '#3f6fb5', name: 'Indigo' },
];

export interface MaterialPreset {
  key: string;
  label: string;
  patch: Partial<MaterialDescriptor>;
}

export const MATERIAL_PRESETS: MaterialPreset[] = [
  {
    key: 'plastic',
    label: 'Plastic',
    patch: { metalness: 0, roughness: 0.5, opacity: 1, transparent: false, emissiveIntensity: 0 },
  },
  {
    key: 'matte',
    label: 'Matte',
    patch: { metalness: 0, roughness: 0.95, opacity: 1, transparent: false, emissiveIntensity: 0 },
  },
  {
    key: 'metal',
    label: 'Metal',
    patch: {
      metalness: 1,
      roughness: 0.3,
      opacity: 1,
      transparent: false,
      emissiveIntensity: 0,
    },
  },
  {
    key: 'glass',
    label: 'Glass',
    patch: {
      metalness: 0,
      roughness: 0.05,
      opacity: 0.35,
      transparent: true,
      emissiveIntensity: 0,
    },
  },
  {
    key: 'glow',
    label: 'Glow',
    patch: { metalness: 0, roughness: 0.6, opacity: 1, transparent: false, emissiveIntensity: 1.5 },
  },
];

export function hexToRgb(hex: string): [number, number, number] {
  const n = parseInt(hex.replace('#', '').slice(0, 6), 16) || 0;
  return [((n >> 16) & 255) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255];
}

export function rgbToHex(c: [number, number, number]): string {
  const h = (v: number) =>
    Math.round(Math.min(1, Math.max(0, v)) * 255)
      .toString(16)
      .padStart(2, '0');
  return `#${h(c[0])}${h(c[1])}${h(c[2])}`;
}

export function materialOf(sm: SceneManager, entityId: string): MaterialDescriptor | null {
  const mr = sm.getEntity(entityId)?.getComponent<MeshRendererData>('meshRenderer');
  if (!mr || mr.geometryType === 'imported') return null;
  return sm.getMaterials()[mr.materialIndex] ?? null;
}

function setMaterialIndex(sm: SceneManager, entityId: string, index: number): void {
  const entity = sm.getEntity(entityId);
  const mr = entity?.getComponent<MeshRendererData>('meshRenderer');
  if (entity && mr)
    entity.setComponent<MeshRendererData>('meshRenderer', { ...mr, materialIndex: index });
}

/**
 * One material edit gesture (a colour pick, a slider drag, a preset click).
 *
 * New parts all share material 0, so the first change forks a private copy for this
 * object instead of recolouring every other part. `apply` previews live; `commit`
 * records a single undo step for the whole gesture.
 */
export class MaterialEdit {
  private readonly beforeIndex: number;
  private readonly beforeDesc: MaterialDescriptor;
  private index: number;
  private forked = false;
  private changed = false;

  private constructor(
    private sm: SceneManager,
    private bridge: React.RefObject<SceneBridge | null>,
    private entityId: string,
    index: number,
  ) {
    this.beforeIndex = index;
    this.index = index;
    this.beforeDesc = { ...this.sm.getMaterials()[index]! };
  }

  static start(
    sm: SceneManager,
    bridge: React.RefObject<SceneBridge | null>,
    entityId: string,
  ): MaterialEdit | null {
    const mr = sm.getEntity(entityId)?.getComponent<MeshRendererData>('meshRenderer');
    if (!mr || mr.geometryType === 'imported' || !sm.getMaterials()[mr.materialIndex]) return null;
    return new MaterialEdit(sm, bridge, entityId, mr.materialIndex);
  }

  private isShared(index: number): boolean {
    return this.sm.getAllEntities().some((e) => {
      if (e.id === this.entityId) return false;
      return e.getComponent<MeshRendererData>('meshRenderer')?.materialIndex === index;
    });
  }

  apply(patch: Partial<MaterialDescriptor>): void {
    if (!this.forked && this.isShared(this.index)) {
      const entity = this.sm.getEntity(this.entityId);
      const copy: MaterialDescriptor = {
        ...this.sm.getMaterials()[this.index]!,
        name: entity?.name ?? 'Material',
      };
      this.index = this.sm.addMaterial(copy);
      setMaterialIndex(this.sm, this.entityId, this.index);
      this.forked = true;
    }
    this.changed = true;
    this.sm.updateMaterial(this.index, patch);
    this.bridge.current?.updateMaterial(this.entityId, this.sm.getMaterials()[this.index]!);
  }

  commit(history: CommandHistory, description = 'Edit material'): void {
    if (!this.changed) return;
    const { sm, bridge, entityId, beforeIndex, beforeDesc, forked } = this;
    const afterIndex = this.index;
    const afterDesc = { ...sm.getMaterials()[afterIndex]! };
    const show = (index: number) => {
      const desc = sm.getMaterials()[index];
      if (desc) bridge.current?.updateMaterial(entityId, desc);
      sm.notifyChange();
    };
    history.pushExecuted({
      description,
      execute() {
        setMaterialIndex(sm, entityId, afterIndex);
        sm.updateMaterial(afterIndex, afterDesc);
        show(afterIndex);
      },
      undo() {
        setMaterialIndex(sm, entityId, beforeIndex);
        // A forked edit never touched the shared material, so only restore in-place edits
        if (!forked) sm.updateMaterial(beforeIndex, beforeDesc);
        show(beforeIndex);
      },
    });
    this.changed = false;
  }
}
