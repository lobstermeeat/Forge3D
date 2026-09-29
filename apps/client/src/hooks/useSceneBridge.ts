import { useEffect, useRef, useState, useCallback } from 'react';
import * as THREE from 'three';
import {
  SceneBridge,
  GizmoControls,
  type SceneManager,
  type CommandHistory,
} from '@forge3d/engine';
import type { TransformData } from '@forge3d/shared';
import { useEditorStore } from '@/stores/editorStore';
import type { FrameCallback } from '@/hooks/useEngine';

interface UseSceneBridgeArgs {
  threeScene: THREE.Scene | null;
  camera: THREE.PerspectiveCamera | null;
  canvas: HTMLCanvasElement | null;
  sceneManager: SceneManager;
  history: CommandHistory;
  ready: boolean;
  addFrameCallback?: (cb: FrameCallback) => () => void;
  /** Called when a click in the viewport picks (or clears) an entity. */
  onPick?: (entityId: string | null) => void;
}

export interface HierarchyNode {
  id: string;
  name: string;
  children: HierarchyNode[];
}

/** Pointer travel (px) above which a press counts as a drag, not a click. */
const CLICK_SLOP = 4;
const SELECT_COLOR = 0xff7a2f;
const HOVER_COLOR = 0xffffff;

const UNIT_EDGES = new THREE.EdgesGeometry(new THREE.BoxGeometry(1, 1, 1));
const NO_ROTATION = new THREE.Quaternion();
const _box = new THREE.Box3();
const _center = new THREE.Vector3();
const _size = new THREE.Vector3();
const _fit = new THREE.Matrix4();

/** Wireframe box drawn on top of everything, positioned by hand each frame. */
function makeBoxLines(color: number, opacity: number): THREE.LineSegments {
  const lines = new THREE.LineSegments(
    UNIT_EDGES,
    new THREE.LineBasicMaterial({ color, transparent: true, opacity, depthTest: false }),
  );
  lines.matrixAutoUpdate = false;
  lines.renderOrder = 999;
  lines.visible = false;
  lines.raycast = () => {};
  return lines;
}

/**
 * Fit `lines` around `obj`. Parts get a box that turns with them (their geometry bounds);
 * groups and imported models get a world-aligned box. Returns false if there is nothing to box.
 */
function fitBoxLines(lines: THREE.LineSegments, obj: THREE.Object3D): boolean {
  obj.updateWorldMatrix(true, false);
  const mesh = obj as THREE.Mesh;
  if (mesh.isMesh && mesh.geometry) {
    if (!mesh.geometry.boundingBox) mesh.geometry.computeBoundingBox();
    _box.copy(mesh.geometry.boundingBox!);
    _box.getCenter(_center);
    _box.getSize(_size).addScalar(0.02);
    _fit.compose(_center, NO_ROTATION, _size);
    lines.matrix.multiplyMatrices(mesh.matrixWorld, _fit);
  } else {
    _box.setFromObject(obj);
    if (_box.isEmpty()) return false;
    _box.getCenter(_center);
    _box.getSize(_size).addScalar(0.02);
    lines.matrix.compose(_center, NO_ROTATION, _size);
  }
  lines.matrixWorldNeedsUpdate = true;
  return true;
}

export function useSceneBridge(args: UseSceneBridgeArgs) {
  const { threeScene, camera, canvas, sceneManager, history, ready, addFrameCallback, onPick } = args;
  const bridgeRef = useRef<SceneBridge | null>(null);
  const gizmoRef = useRef<GizmoControls | null>(null);
  const [hierarchyNodes, setHierarchyNodes] = useState<HierarchyNode[]>([]);
  const [sceneVersion, setSceneVersion] = useState(0);
  const pressRef = useRef<{ x: number; y: number; gizmo: boolean } | null>(null);
  const suppressClickRef = useRef(false);
  const onPickRef = useRef(onPick);
  onPickRef.current = onPick;

  const selectedEntityId = useEditorStore((s) => s.selectedEntityId);
  const transformMode = useEditorStore((s) => s.transformMode);
  const selectOnly = useEditorStore((s) => s.selectOnly);
  const isPlaying = useEditorStore((s) => s.isPlaying);
  const snapEnabled = useEditorStore((s) => s.snapEnabled);
  const moveStep = useEditorStore((s) => s.moveStep);
  const rotateStep = useEditorStore((s) => s.rotateStep);
  const space = useEditorStore((s) => s.space);

  /** Clicking any part of a model selects the whole model, as in Studio. */
  const topLevelId = useCallback(
    (id: string): string => {
      let entity = sceneManager.getEntity(id);
      while (entity?.parentId) {
        const parent = sceneManager.getEntity(entity.parentId);
        if (!parent) break;
        entity = parent;
      }
      return entity?.id ?? id;
    },
    [sceneManager],
  );

  const pickAt = useCallback(
    (clientX: number, clientY: number): string | null => {
      const bridge = bridgeRef.current;
      if (!bridge || !camera || !canvas) return null;
      const rect = canvas.getBoundingClientRect();
      const mouse = new THREE.Vector2(
        ((clientX - rect.left) / rect.width) * 2 - 1,
        -((clientY - rect.top) / rect.height) * 2 + 1,
      );
      const raycaster = new THREE.Raycaster();
      raycaster.setFromCamera(mouse, camera);
      const hits = raycaster.intersectObjects(bridge.getManagedObjects(), false);
      for (const hit of hits) {
        const entityId = bridge.getEntityId(hit.object);
        if (entityId) return topLevelId(entityId);
      }
      return null;
    },
    [camera, canvas, topLevelId],
  );

  // Create bridge, gizmo and selection helpers once the renderer is ready
  useEffect(() => {
    if (!ready || !threeScene || !camera || !canvas) return;

    const bridge = new SceneBridge(sceneManager, threeScene);
    bridge.connect();
    bridgeRef.current = bridge;

    const gizmo = new GizmoControls({ camera, domElement: canvas, scene: threeScene });
    gizmoRef.current = gizmo;

    const select = makeBoxLines(SELECT_COLOR, 0.95);
    const hover = makeBoxLines(HOVER_COLOR, 0.35);
    threeScene.add(select, hover);

    // Gizmo drag → capture before/after for undo
    let transformBefore: TransformData | null = null;
    let draggingEntityId: string | null = null;

    gizmo.onDragStarted(() => {
      suppressClickRef.current = true;
      const selId = useEditorStore.getState().selectedEntityId;
      if (selId) {
        draggingEntityId = selId;
        transformBefore = bridge.readTransform(selId);
      }
    });

    gizmo.onDragEnded(() => {
      if (draggingEntityId && transformBefore) {
        const after = bridge.readTransform(draggingEntityId);
        if (after) {
          const entity = sceneManager.getEntity(draggingEntityId);
          if (entity) {
            entity.transform = after;
          }
          const beforeSnap = { ...transformBefore };
          const afterSnap = { ...after };
          const eid = draggingEntityId;
          history.pushExecuted({
            execute() {
              const e = sceneManager.getEntity(eid);
              if (e) {
                e.transform = { ...afterSnap };
                sceneManager.notifyChange();
              }
            },
            undo() {
              const e = sceneManager.getEntity(eid);
              if (e) {
                e.transform = { ...beforeSnap };
                sceneManager.notifyChange();
              }
            },
            description: 'Transform',
          });
          sceneManager.notifyChange();
        }
      }
      transformBefore = null;
      draggingEntityId = null;
    });

    // Live preview during drag
    gizmo.onTransformChange(() => {
      const selId = useEditorStore.getState().selectedEntityId;
      if (selId) {
        const t = bridge.readTransform(selId);
        if (t) {
          const entity = sceneManager.getEntity(selId);
          if (entity) entity.transform = t;
          setSceneVersion((v) => v + 1);
        }
      }
    });

    // Rebuild hierarchy on scene changes
    const rebuildHierarchy = () => {
      const buildNode = (entityId: string): HierarchyNode | null => {
        const entity = sceneManager.getEntity(entityId);
        if (!entity) return null;
        const children = sceneManager
          .getChildren(entityId)
          .map((c) => buildNode(c.id))
          .filter((n): n is HierarchyNode => n !== null);
        return { id: entity.id, name: entity.name, children };
      };

      const roots = sceneManager.getRootEntities();
      const nodes = roots
        .map((e) => buildNode(e.id))
        .filter((n): n is HierarchyNode => n !== null);
      setHierarchyNodes(nodes);
      setSceneVersion((v) => v + 1);
    };

    const unsub = sceneManager.subscribe(rebuildHierarchy);
    rebuildHierarchy();

    // Distinguish clicks from camera/gizmo drags
    const onPointerDown = (e: PointerEvent) => {
      suppressClickRef.current = false;
      pressRef.current = { x: e.clientX, y: e.clientY, gizmo: gizmo.isHovered() };
    };

    // Hover highlight (one raycast per frame at most)
    let hoverPoint: { x: number; y: number } | null = null;
    const onPointerMove = (e: PointerEvent) => {
      hoverPoint = e.buttons === 0 ? { x: e.clientX, y: e.clientY } : null;
    };
    const onPointerLeave = () => {
      hoverPoint = null;
      if (useEditorStore.getState().hoveredEntityId) useEditorStore.getState().setHovered(null);
    };
    canvas.addEventListener('pointerdown', onPointerDown);
    canvas.addEventListener('pointermove', onPointerMove);
    canvas.addEventListener('pointerleave', onPointerLeave);

    // Keep the boxes glued to their objects and resolve hover once per frame
    const removeFrame = addFrameCallback?.(() => {
      const state = useEditorStore.getState();
      if (hoverPoint && !state.isPlaying && !gizmo.isDragging()) {
        const id = gizmo.isHovered() ? null : pickAt(hoverPoint.x, hoverPoint.y);
        if (id !== state.hoveredEntityId) state.setHovered(id);
        hoverPoint = null;
      }
      for (const [helper, id] of [
        [select, state.isPlaying ? null : state.selectedEntityId],
        [hover, state.isPlaying || state.hoveredEntityId === state.selectedEntityId ? null : state.hoveredEntityId],
      ] as const) {
        const obj = id ? bridge.getObject(id) : undefined;
        helper.visible = obj ? fitBoxLines(helper, obj) : false;
      }
    });

    return () => {
      removeFrame?.();
      canvas.removeEventListener('pointerdown', onPointerDown);
      canvas.removeEventListener('pointermove', onPointerMove);
      canvas.removeEventListener('pointerleave', onPointerLeave);
      threeScene.remove(select, hover);
      (select.material as THREE.Material).dispose();
      (hover.material as THREE.Material).dispose();
      unsub();
      gizmo.dispose();
      bridge.dispose();
      bridgeRef.current = null;
      gizmoRef.current = null;
    };
  }, [ready, threeScene, camera, canvas, sceneManager, history, addFrameCallback, pickAt]);

  // Sync gizmo mode, snapping and space
  useEffect(() => {
    gizmoRef.current?.setMode(transformMode);
  }, [transformMode, ready]);

  useEffect(() => {
    gizmoRef.current?.setSnap(
      snapEnabled
        ? { translate: moveStep, rotateDegrees: rotateStep, scale: 0.1 }
        : { translate: null, rotateDegrees: null, scale: null },
    );
  }, [snapEnabled, moveStep, rotateStep, ready]);

  useEffect(() => {
    gizmoRef.current?.setSpace(space);
  }, [space, ready]);

  // Attach the gizmo to the selection (never while playing or with the Select tool)
  useEffect(() => {
    const gizmo = gizmoRef.current;
    const bridge = bridgeRef.current;
    if (!gizmo || !bridge) return;

    const obj = selectedEntityId && !isPlaying && !selectOnly ? bridge.getObject(selectedEntityId) : undefined;
    if (obj) gizmo.attach(obj);
    else gizmo.detach();
  }, [selectedEntityId, sceneVersion, isPlaying, selectOnly]);

  // Click-to-select (ignores the release of a camera orbit or gizmo drag)
  const handleCanvasClick = useCallback(
    (event: React.MouseEvent<HTMLCanvasElement>) => {
      const press = pressRef.current;
      pressRef.current = null;
      if (useEditorStore.getState().isPlaying) return;
      if (suppressClickRef.current || gizmoRef.current?.isDragging()) return;
      if (press && (press.gizmo || Math.hypot(event.clientX - press.x, event.clientY - press.y) > CLICK_SLOP)) {
        return;
      }
      onPickRef.current?.(pickAt(event.clientX, event.clientY));
    },
    [pickAt],
  );

  return {
    bridge: bridgeRef,
    gizmo: gizmoRef,
    hierarchyNodes,
    handleCanvasClick,
    pickAt,
    sceneVersion,
  };
}
