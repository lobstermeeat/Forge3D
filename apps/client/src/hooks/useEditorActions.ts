import { useMemo } from 'react';
import * as THREE from 'three';
import {
  AddEntityCommand,
  RemoveEntityCommand,
  RenameEntityCommand,
  TransformCommand,
  AssetLoader,
  importGroupToScene,
  exportSceneAsGLB,
} from '@forge3d/engine';
import type { CommandHistory, SceneBridge, SceneManager, ViewportControls } from '@forge3d/engine';
import type { LightData, MeshRendererData, TransformData } from '@forge3d/shared';
import { DEFAULT_TRANSFORM, generateId } from '@forge3d/shared';
import { useEditorStore } from '@/stores/editorStore';
import type { CameraView } from '@/stores/editorStore';
import { logOutput } from '@/stores/outputStore';
import { useAssetStore } from '@/stores/assetStore';
import { AddSnapshotCommand } from '@/editor/commands';
import {
  freeSpot,
  groundPointAt,
  primitiveLabel,
  restingTransform,
  snapTo,
  type Primitive,
} from '@/editor/placement';

interface ActionDeps {
  sceneManager: SceneManager;
  history: CommandHistory;
  bridge: React.RefObject<SceneBridge | null>;
  controls: React.RefObject<ViewportControls | null>;
  getCamera: () => THREE.Camera | null;
  onSelectionChange?: (id: string | null) => void;
}

const LIGHT_LABEL: Record<LightData['type'], string> = {
  directional: 'Sun Light',
  point: 'Point Light',
  ambient: 'Ambient Light',
  spot: 'Spot Light',
};

function download(blob: Blob, name: string) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = name;
  a.click();
  URL.revokeObjectURL(url);
}

export type EditorActions = ReturnType<typeof createActions>;

function createActions(d: ActionDeps) {
  const { sceneManager, history, bridge, controls, getCamera } = d;
  const store = () => useEditorStore.getState();

  const select = (id: string | null) => {
    store().selectEntity(id);
    sceneManager.selectEntity(id);
    d.onSelectionChange?.(id);
  };

  const selectedId = (id?: string | null) => (id === undefined ? store().selectedEntityId : id);

  /** Where a new object should go: under the view centre, on the ground, not on top of another part. */
  const spawnPoint = (at?: THREE.Vector3): THREE.Vector3 => {
    const camera = getCamera();
    const target = controls.current?.target ?? new THREE.Vector3();
    let p =
      at?.clone() ??
      (camera ? groundPointAt(camera, new THREE.Vector2(0, -0.1), target) : target.clone());
    if (!at && camera) p = freeSpot(p, sceneManager, camera);
    const { snapEnabled, moveStep } = store();
    if (snapEnabled) {
      p.x = snapTo(p.x, moveStep);
      p.z = snapTo(p.z, moveStep);
    }
    return p;
  };

  const objectBounds = (id: string): THREE.Box3 | null => {
    const obj = bridge.current?.getObject(id);
    if (!obj) return null;
    const box = new THREE.Box3().setFromObject(obj, true);
    return box.isEmpty() ? null : box;
  };

  const actions = {
    select,

    addPrimitive(type: Primitive, at?: THREE.Vector3) {
      const name = primitiveLabel(type);
      const meshRenderer: MeshRendererData = { geometryType: type, materialIndex: 0 };
      const cmd = new AddEntityCommand(
        sceneManager,
        name,
        meshRenderer,
        restingTransform(type, spawnPoint(at)),
      );
      history.execute(cmd);
      select(cmd.getEntityId());
      logOutput('ok', `Inserted ${name}`);
    },

    addLight(type: LightData['type']) {
      const p = spawnPoint();
      const transform: TransformData = {
        ...DEFAULT_TRANSFORM,
        position: type === 'ambient' ? [0, 0, 0] : [p.x, 3, p.z],
      };
      const cmd = new AddSnapshotCommand(sceneManager, {
        id: generateId(),
        name: LIGHT_LABEL[type],
        components: { transform, light: { type, color: [1, 1, 1], intensity: 1 } },
      });
      history.execute(cmd);
      select(cmd.getEntityId());
      logOutput('ok', `Inserted ${LIGHT_LABEL[type]}`);
    },

    async importModel(file: File, opts: { remember?: boolean } = {}) {
      if (!/\.(gltf|glb)$/i.test(file.name)) {
        logOutput('warn', `${file.name} isn't a .glb or .gltf file`);
        return;
      }
      logOutput('info', `Importing ${file.name}…`);
      const loader = new AssetLoader();
      try {
        const { scene: group } = await loader.loadFromFile(file);
        const base = file.name.replace(/\.(gltf|glb)$/i, '');

        group.updateMatrixWorld(true);
        const bounds = new THREE.Box3().setFromObject(group, true);

        // Meshes become direct children of one root entity, so bake any intermediate
        // node transforms into each mesh, relative to the file's root
        const meshes: THREE.Mesh[] = [];
        group.traverse((o) => {
          if ((o as THREE.Mesh).isMesh) meshes.push(o as THREE.Mesh);
        });
        const toRoot = group.matrixWorld.clone().invert();
        const rel = new THREE.Matrix4();
        for (const mesh of meshes) {
          rel.multiplyMatrices(toRoot, mesh.matrixWorld);
          rel.decompose(mesh.position, mesh.quaternion, mesh.scale);
        }

        // Rest the model on the ground, centred where the view is looking
        const p = spawnPoint();
        if (bounds.isEmpty()) {
          group.position.set(p.x, 0, p.z);
        } else {
          const c = bounds.getCenter(new THREE.Vector3());
          group.position.x += p.x - c.x;
          group.position.z += p.z - c.z;
          group.position.y -= bounds.min.y;
        }
        const { rootId, meshEntries } = importGroupToScene(group, sceneManager, base);
        for (const entry of meshEntries) {
          bridge.current?.addExternalObject(entry.entityId, entry.object);
        }
        select(rootId);
        if (opts.remember !== false) {
          useAssetStore.getState().addImport({ id: generateId(), name: base, file });
        }
        logOutput(
          'ok',
          `Imported ${file.name} — ${meshEntries.length} mesh${meshEntries.length === 1 ? '' : 'es'}`,
        );
      } catch (err) {
        logOutput(
          'error',
          `Couldn't import ${file.name}: ${err instanceof Error ? err.message : String(err)}`,
        );
      } finally {
        loader.dispose();
      }
    },

    duplicate(id?: string | null) {
      const eid = selectedId(id);
      const entity = eid ? sceneManager.getEntity(eid) : undefined;
      if (!entity) return;
      const mr = entity.getComponent<MeshRendererData>('meshRenderer');
      if (entity.children.length > 0 || mr?.geometryType === 'imported') {
        logOutput(
          'warn',
          `${entity.name} has imported parts, so it can't be duplicated yet — import the file again instead`,
        );
        return;
      }
      const data = structuredClone(entity.serialize());
      data.id = generateId();
      const t = data.components.transform ?? { ...DEFAULT_TRANSFORM };
      const step = Math.max(store().moveStep, 1);
      data.components.transform = {
        ...t,
        position: [t.position[0] + step, t.position[1], t.position[2]],
      };
      const cmd = new AddSnapshotCommand(sceneManager, data);
      history.execute(cmd);
      select(cmd.getEntityId());
      logOutput('info', `Duplicated ${entity.name}`);
    },

    remove(id?: string | null) {
      const eid = selectedId(id);
      const entity = eid ? sceneManager.getEntity(eid) : undefined;
      if (!eid || !entity) return;
      history.execute(new RemoveEntityCommand(sceneManager, eid));
      if (store().selectedEntityId === eid) select(null);
      logOutput('info', `Deleted ${entity.name}`);
    },

    rename(id: string, name: string) {
      const entity = sceneManager.getEntity(id);
      const next = name.trim();
      if (!entity || !next || next === entity.name) return;
      const before = entity.name;
      history.execute(new RenameEntityCommand(sceneManager, id, next));
      logOutput('info', `Renamed ${before} → ${next}`);
    },

    setTransform(id: string, patch: Partial<TransformData>, label = 'Transform') {
      const entity = sceneManager.getEntity(id);
      if (!entity) return;
      const before = { ...entity.transform };
      const after = { ...before, ...patch };
      if (JSON.stringify(before) === JSON.stringify(after)) return;
      const cmd = new TransformCommand(sceneManager, id, before, after);
      cmd.description = label;
      history.execute(cmd);
    },

    focus(id?: string | null) {
      const eid = selectedId(id);
      if (!eid || !controls.current) return;
      const box = objectBounds(eid);
      if (box) {
        const sphere = box.getBoundingSphere(new THREE.Sphere());
        controls.current.focusOn(sphere.center, Math.max(2.5, sphere.radius * 3));
      } else {
        const t = sceneManager.getEntity(eid)?.transform.position;
        if (t) controls.current.focusOn(new THREE.Vector3(t[0], t[1], t[2]), 5);
      }
    },

    setView(view: CameraView) {
      const c = controls.current;
      const camera = getCamera();
      if (!c || !camera) return;
      const target = c.target.clone();
      const dist = Math.max(4, camera.position.distanceTo(target));
      const offset =
        view === 'top'
          ? new THREE.Vector3(0, dist, 0.001)
          : view === 'front'
            ? new THREE.Vector3(0, 0, dist)
            : view === 'side'
              ? new THREE.Vector3(dist, 0, 0)
              : new THREE.Vector3(6, 5, 8).normalize().multiplyScalar(dist);
      c.setView(target.clone().add(offset), target);
      store().setCameraView(view);
    },

    resetCamera() {
      controls.current?.reset();
      store().setCameraView('perspective');
    },

    dropToGround(id?: string | null) {
      const eid = selectedId(id);
      const entity = eid ? sceneManager.getEntity(eid) : undefined;
      const box = eid ? objectBounds(eid) : null;
      if (!eid || !entity || !box) return;
      const p = entity.transform.position;
      actions.setTransform(eid, { position: [p[0], p[1] - box.min.y, p[2]] }, 'Drop to ground');
      logOutput('info', `Dropped ${entity.name} to the ground`);
    },

    snapToGrid(id?: string | null) {
      const eid = selectedId(id);
      const entity = eid ? sceneManager.getEntity(eid) : undefined;
      if (!eid || !entity) return;
      const step = store().moveStep;
      const p = entity.transform.position;
      actions.setTransform(
        eid,
        { position: [snapTo(p[0], step), p[1], snapTo(p[2], step)] },
        'Snap to grid',
      );
    },

    undo() {
      if (!history.canUndo()) return;
      history.undo();
      sceneManager.notifyChange();
      const id = store().selectedEntityId;
      if (id && !sceneManager.getEntity(id)) select(null);
    },

    redo() {
      if (!history.canRedo()) return;
      history.redo();
      sceneManager.notifyChange();
    },

    saveToFile() {
      const json = JSON.stringify(sceneManager.serialize(), null, 2);
      download(new Blob([json], { type: 'application/json' }), 'scene.forge3d.json');
      logOutput('ok', 'Downloaded scene.forge3d.json');
    },

    openFile(file: File) {
      const reader = new FileReader();
      reader.onload = () => {
        try {
          sceneManager.deserialize(JSON.parse(reader.result as string));
          history.clear();
          select(null);
          logOutput('ok', `Opened ${file.name}`);
        } catch (err) {
          logOutput(
            'error',
            `Couldn't open ${file.name}: ${err instanceof Error ? err.message : String(err)}`,
          );
        }
      };
      reader.readAsText(file);
    },

    async exportGLB() {
      const objects = bridge.current?.getRootObjects();
      if (!objects || objects.length === 0) {
        logOutput('warn', 'Nothing to export yet');
        return;
      }
      try {
        const buffer = await exportSceneAsGLB(objects);
        download(new Blob([buffer], { type: 'model/gltf-binary' }), 'scene.glb');
        logOutput('ok', 'Exported scene.glb');
      } catch (err) {
        logOutput(
          'error',
          `GLB export failed: ${err instanceof Error ? err.message : String(err)}`,
        );
      }
    },
  };

  return actions;
}

export function useEditorActions(deps: ActionDeps): EditorActions {
  const { sceneManager, history, bridge, controls, getCamera, onSelectionChange } = deps;
  return useMemo(
    () => createActions({ sceneManager, history, bridge, controls, getCamera, onSelectionChange }),
    [sceneManager, history, bridge, controls, getCamera, onSelectionChange],
  );
}
