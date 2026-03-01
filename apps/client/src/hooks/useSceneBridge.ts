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

interface UseSceneBridgeArgs {
  threeScene: THREE.Scene | null;
  camera: THREE.PerspectiveCamera | null;
  canvas: HTMLCanvasElement | null;
  sceneManager: SceneManager;
  history: CommandHistory;
  ready: boolean;
}

export interface HierarchyNode {
  id: string;
  name: string;
  children: HierarchyNode[];
}

export function useSceneBridge(args: UseSceneBridgeArgs) {
  const { threeScene, camera, canvas, sceneManager, history, ready } = args;
  const bridgeRef = useRef<SceneBridge | null>(null);
  const gizmoRef = useRef<GizmoControls | null>(null);
  const [hierarchyNodes, setHierarchyNodes] = useState<HierarchyNode[]>([]);
  const [sceneVersion, setSceneVersion] = useState(0);

  const selectedEntityId = useEditorStore((s) => s.selectedEntityId);
  const transformMode = useEditorStore((s) => s.transformMode);
  const selectEntity = useEditorStore((s) => s.selectEntity);

  // Create bridge and gizmo once renderer is ready
  useEffect(() => {
    if (!ready || !threeScene || !camera || !canvas) return;

    const bridge = new SceneBridge(sceneManager, threeScene);
    bridge.connect();
    bridgeRef.current = bridge;

    const gizmo = new GizmoControls({ camera, domElement: canvas, scene: threeScene });
    gizmoRef.current = gizmo;

    // Gizmo drag → capture before/after for undo
    let transformBefore: TransformData | null = null;
    let draggingEntityId: string | null = null;

    gizmo.onDragStarted(() => {
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

    return () => {
      unsub();
      gizmo.dispose();
      bridge.dispose();
      bridgeRef.current = null;
      gizmoRef.current = null;
    };
  }, [ready, threeScene, camera, canvas, sceneManager, history]);

  // Sync gizmo mode
  useEffect(() => {
    gizmoRef.current?.setMode(transformMode);
  }, [transformMode]);

  // Sync gizmo attachment to selection
  useEffect(() => {
    const gizmo = gizmoRef.current;
    const bridge = bridgeRef.current;
    if (!gizmo || !bridge) return;

    if (selectedEntityId) {
      const obj = bridge.getObject(selectedEntityId);
      if (obj) gizmo.attach(obj);
      else gizmo.detach();
    } else {
      gizmo.detach();
    }
  }, [selectedEntityId, sceneVersion]);

  // Raycasting for click-to-select
  const handleCanvasClick = useCallback(
    (event: React.MouseEvent<HTMLCanvasElement>) => {
      const bridge = bridgeRef.current;
      if (!bridge || !camera || !canvas) return;
      if (gizmoRef.current?.isDragging()) return;

      const rect = canvas.getBoundingClientRect();
      const mouse = new THREE.Vector2(
        ((event.clientX - rect.left) / rect.width) * 2 - 1,
        -((event.clientY - rect.top) / rect.height) * 2 + 1,
      );

      const raycaster = new THREE.Raycaster();
      raycaster.setFromCamera(mouse, camera);

      const objects = bridge.getManagedObjects();
      const hits = raycaster.intersectObjects(objects, false);

      if (hits.length > 0) {
        const entityId = bridge.getEntityId(hits[0]!.object);
        if (entityId) {
          selectEntity(entityId);
          sceneManager.selectEntity(entityId);
          return;
        }
      }
      selectEntity(null);
      sceneManager.selectEntity(null);
    },
    [camera, canvas, selectEntity, sceneManager],
  );

  return {
    bridge: bridgeRef,
    gizmo: gizmoRef,
    hierarchyNodes,
    handleCanvasClick,
    sceneVersion,
  };
}
