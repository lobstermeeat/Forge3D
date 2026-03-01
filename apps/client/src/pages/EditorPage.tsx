import { useRef, useCallback, useEffect } from 'react';
import { useParams } from 'react-router-dom';
import Viewport from '@/components/Viewport';
import { Toolbar } from '@/components/Toolbar';
import { HierarchyPanel } from '@/components/HierarchyPanel';
import { InspectorPanel } from '@/components/InspectorPanel';
import { useEngine } from '@/hooks/useEngine';
import { useSceneBridge } from '@/hooks/useSceneBridge';
import { useKeyboardShortcuts } from '@/hooks/useKeyboardShortcuts';
import { useAutoSave } from '@/hooks/useAutoSave';
import { useEditorStore } from '@/stores/editorStore';
import {
  AddEntityCommand,
  RemoveEntityCommand,
  RenameEntityCommand,
  Entity,
  AssetLoader,
  importGroupToScene,
} from '@forge3d/engine';
import { trpc } from '@/api/trpc';
import type { MeshRendererData, LightData, SceneData } from '@forge3d/shared';
import type * as THREE from 'three';

export function EditorPage() {
  const { projectId, sceneId } = useParams<{ projectId?: string; sceneId?: string }>();
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const sceneLoadedRef = useRef(false);

  const { controls, sceneManager, history, rendererType, ready, getScene, getCamera } =
    useEngine(canvasRef);

  const selectEntity = useEditorStore((s) => s.selectEntity);

  // Load scene from server when route params present
  const sceneQuery = trpc.scene.getById.useQuery(
    { id: sceneId ?? '' },
    { enabled: !!sceneId && ready },
  );

  useEffect(() => {
    if (sceneQuery.data?.data && ready && !sceneLoadedRef.current) {
      sceneManager.deserialize(sceneQuery.data.data as unknown as SceneData);
      history.clear();
      sceneLoadedRef.current = true;
    }
  }, [sceneQuery.data, ready, sceneManager, history]);

  // Reset load flag when sceneId changes
  useEffect(() => {
    sceneLoadedRef.current = false;
  }, [sceneId]);

  // Auto-save to server when scene changes
  const saveStatus = useAutoSave(sceneManager, sceneId, ready && sceneLoadedRef.current);

  const { bridge, hierarchyNodes, handleCanvasClick } = useSceneBridge({
    threeScene: ready ? getScene() : null,
    camera: ready ? (getCamera() as THREE.PerspectiveCamera | null) : null,
    canvas: canvasRef.current,
    sceneManager,
    history,
    ready,
  });

  const addPrimitive = useCallback(
    (type: MeshRendererData['geometryType']) => {
      const meshRenderer: MeshRendererData = { geometryType: type, materialIndex: 0 };
      const cmd = new AddEntityCommand(
        sceneManager,
        type.charAt(0).toUpperCase() + type.slice(1),
        meshRenderer,
      );
      history.execute(cmd);
      selectEntity(cmd.getEntityId());
      sceneManager.selectEntity(cmd.getEntityId());
    },
    [sceneManager, history, selectEntity],
  );

  const addLight = useCallback(
    (type: LightData['type']) => {
      const lightData: LightData = { type, color: [1, 1, 1], intensity: 1 };
      const entity = new Entity(type.charAt(0).toUpperCase() + type.slice(1) + ' Light');
      entity.setComponent<LightData>('light', lightData);
      // Position directional/point lights above origin
      if (type !== 'ambient') {
        entity.transform = {
          ...entity.transform,
          position: [0, 3, 2],
        };
      }
      sceneManager.addEntity(entity);
      selectEntity(entity.id);
      sceneManager.selectEntity(entity.id);
    },
    [sceneManager, selectEntity],
  );

  const handleImportGLTF = useCallback(
    async (file: File) => {
      const loader = new AssetLoader();
      try {
        const { scene: group } = await loader.loadFromFile(file);
        const { rootId, meshEntries } = importGroupToScene(
          group,
          sceneManager,
          file.name.replace(/\.(gltf|glb)$/i, ''),
        );

        // Register imported Three.js objects with the bridge
        for (const entry of meshEntries) {
          bridge.current?.addExternalObject(entry.entityId, entry.object);
        }

        selectEntity(rootId);
        sceneManager.selectEntity(rootId);
      } catch (err) {
        console.error('GLTF import failed:', err);
      } finally {
        loader.dispose();
      }
    },
    [sceneManager, bridge, selectEntity],
  );

  const handleSelect = useCallback(
    (id: string) => {
      selectEntity(id);
      sceneManager.selectEntity(id);
    },
    [selectEntity, sceneManager],
  );

  const handleDelete = useCallback(
    (id: string) => {
      const cmd = new RemoveEntityCommand(sceneManager, id);
      history.execute(cmd);
      selectEntity(null);
    },
    [sceneManager, history, selectEntity],
  );

  const handleRename = useCallback(
    (id: string, name: string) => {
      const cmd = new RenameEntityCommand(sceneManager, id, name);
      history.execute(cmd);
    },
    [sceneManager, history],
  );

  useKeyboardShortcuts({ history, sceneManager, controls, canvas: canvasRef.current });

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100vh', background: '#11111b' }}>
      <Toolbar
        onAddPrimitive={addPrimitive}
        onAddLight={addLight}
        onImportGLTF={handleImportGLTF}
        sceneManager={sceneManager}
        history={history}
        bridge={bridge}
        threeScene={ready ? getScene() : null}
        saveStatus={sceneId ? saveStatus : undefined}
      />
      <div style={{ display: 'flex', flex: 1, overflow: 'hidden' }}>
        <div style={{ width: 240, flexShrink: 0, borderRight: '1px solid #313244' }}>
          <HierarchyPanel
            nodes={hierarchyNodes}
            onSelect={handleSelect}
            onDelete={handleDelete}
            onRename={handleRename}
          />
        </div>
        <div style={{ flex: 1 }}>
          <Viewport ref={canvasRef} rendererType={rendererType} ready={ready} onClick={handleCanvasClick} />
        </div>
        <div style={{ width: 280, flexShrink: 0, borderLeft: '1px solid #313244' }}>
          <InspectorPanel sceneManager={sceneManager} history={history} bridge={bridge} />
        </div>
      </div>
    </div>
  );
}
