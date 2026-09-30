import { useRef, useState, useCallback, useEffect } from 'react';
import { useParams } from 'react-router-dom';
import * as THREE from 'three';
import Viewport from '@/components/Viewport';
import { TopBar } from '@/components/TopBar';
import { Ribbon } from '@/components/Ribbon';
import { LibraryPanel, LIBRARY_DRAG_TYPE, type LibraryDragPayload } from '@/components/LibraryPanel';
import { ExplorerPanel } from '@/components/ExplorerPanel';
import { PropertiesPanel } from '@/components/PropertiesPanel';
import { OutputPanel } from '@/components/OutputPanel';
import { StatusBar } from '@/components/StatusBar';
import { ViewportOverlay } from '@/components/ViewportOverlay';
import { PublishDialog } from '@/components/PublishDialog';
import { CursorOverlay } from '@/components/CursorOverlay';
import { IconSprite } from '@/editor/Icon';
import { describeEntity } from '@/editor/entityInfo';
import { MATERIAL_PRESETS, MaterialEdit, hexToRgb, materialOf, rgbToHex } from '@/editor/materials';
import { groundPointAt } from '@/editor/placement';
import { useEngine } from '@/hooks/useEngine';
import { useSceneBridge } from '@/hooks/useSceneBridge';
import { useKeyboardShortcuts } from '@/hooks/useKeyboardShortcuts';
import { useAutoSave } from '@/hooks/useAutoSave';
import { useCollaboration } from '@/hooks/useCollaboration';
import { useEditorActions, type EditorActions } from '@/hooks/useEditorActions';
import { usePlayMode, FORMAT_LABEL } from '@/hooks/usePlayMode';
import { useEditorStore } from '@/stores/editorStore';
import { useFormatStore } from '@/stores/formatStore';
import { useAssetStore } from '@/stores/assetStore';
import { logOutput } from '@/stores/outputStore';
import { trpc } from '@/api/trpc';
import type { SceneData } from '@forge3d/shared';
import '@/editor/editor.css';

export function EditorPage() {
  const { sceneId } = useParams<{ projectId?: string; sceneId?: string }>();
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const importInputRef = useRef<HTMLInputElement>(null);
  const sceneLoadedRef = useRef(false);
  const [publishOpen, setPublishOpen] = useState(false);
  const [dropActive, setDropActive] = useState(false);

  const engine = useEngine(canvasRef);
  const {
    controls,
    sceneManager,
    history,
    backendLabel,
    ready,
    getScene,
    getCamera,
    addFrameCallback,
    setGridVisible,
    setPlayPreview,
  } = engine;

  const selectedId = useEditorStore((s) => s.selectedEntityId);
  const panels = useEditorStore((s) => s.panels);
  const showGrid = useEditorStore((s) => s.showGrid);
  const formatType = useFormatStore((s) => s.formatType);

  // Load scene from server when route params present
  const sceneQuery = trpc.scene.getById.useQuery({ id: sceneId ?? '' }, { enabled: !!sceneId && ready });
  const sceneName = (sceneQuery.data as { name?: string } | null | undefined)?.name ?? (sceneId ? 'Loading…' : 'Untitled scene');

  useEffect(() => {
    if (sceneQuery.data?.data && ready && !sceneLoadedRef.current) {
      sceneManager.deserialize(sceneQuery.data.data as unknown as SceneData);
      history.clear();
      sceneLoadedRef.current = true;
      logOutput('ok', `Opened “${(sceneQuery.data as { name?: string }).name ?? 'scene'}”`);
    }
  }, [sceneQuery.data, ready, sceneManager, history]);

  // Reset load flag when sceneId changes
  useEffect(() => {
    sceneLoadedRef.current = false;
  }, [sceneId]);

  useEffect(() => {
    if (sceneQuery.error) logOutput('error', `Couldn't load the scene: ${sceneQuery.error.message}`);
  }, [sceneQuery.error]);

  // Auto-save to server when scene changes
  const { saveStatus, saveNow } = useAutoSave(sceneManager, sceneId, ready && sceneLoadedRef.current);

  // Real-time collaboration via Yjs
  const collab = useCollaboration(sceneId, sceneManager, ready && sceneLoadedRef.current);
  const updateSelection = collab?.updateSelection;

  // The bridge needs a pick handler and the actions need the bridge; a ref breaks the cycle
  const actionsRef = useRef<EditorActions | null>(null);

  const { bridge, hierarchyNodes, handleCanvasClick, sceneVersion } = useSceneBridge({
    threeScene: ready ? getScene() : null,
    camera: ready ? (getCamera() as THREE.PerspectiveCamera | null) : null,
    canvas: canvasRef.current,
    sceneManager,
    history,
    ready,
    addFrameCallback,
    onPick: (id) => actionsRef.current?.select(id),
  });

  const editorActions = useEditorActions({
    sceneManager,
    history,
    bridge,
    controls,
    getCamera,
    onSelectionChange: updateSelection,
  });
  actionsRef.current = editorActions;

  const { play, stop, info, closeInfo } = usePlayMode({
    ready,
    canvas: canvasRef.current,
    getCamera: getCamera as () => THREE.PerspectiveCamera | null,
    getScene,
    controls,
    bridge,
    sceneManager,
    addFrameCallback,
    setGridVisible,
    setPlayPreview,
  });

  // Ctrl+S: save now on a dashboard scene; a scratch scene has nowhere to save to
  const save = useCallback(() => {
    if (saveNow()) return;
    if (sceneId)
      logOutput(
        'warn',
        sceneQuery.error ? 'The scene didn’t load, so there’s nothing to save.' : 'The scene is still loading, so nothing was saved yet.',
      );
    else
      logOutput(
        'info',
        'This scratch scene isn’t saved online. Use File › Download scene file to keep a copy, or open a scene from your dashboard.',
      );
  }, [saveNow, sceneId, sceneQuery.error]);

  useKeyboardShortcuts({ actions: editorActions, play, stop, save });

  useEffect(() => {
    if (ready) setGridVisible(showGrid && !useEditorStore.getState().isPlaying);
  }, [ready, showGrid, setGridVisible]);

  useEffect(() => {
    if (ready && backendLabel) logOutput('info', `Renderer ready — ${backendLabel}`);
  }, [ready, backendLabel]);

  // Drop any selection that no longer exists (undo, collaborator delete)
  useEffect(() => {
    if (selectedId && !sceneManager.getEntity(selectedId)) editorActions.select(null);
  }, [sceneVersion, selectedId, sceneManager, editorActions]);

  const requestImport = useCallback(() => importInputRef.current?.click(), []);

  // Library drag-and-drop into the viewport
  const onDragOver = (e: React.DragEvent) => {
    if (!e.dataTransfer.types.includes(LIBRARY_DRAG_TYPE) && !e.dataTransfer.types.includes('Files')) return;
    e.preventDefault();
    e.dataTransfer.dropEffect = 'copy';
    if (!dropActive) setDropActive(true);
  };
  const onDrop = (e: React.DragEvent) => {
    setDropActive(false);
    const raw = e.dataTransfer.getData(LIBRARY_DRAG_TYPE);
    if (!raw) {
      const file = e.dataTransfer.files?.[0];
      if (file) {
        e.preventDefault();
        void editorActions.importModel(file);
      }
      return;
    }
    e.preventDefault();
    if (useEditorStore.getState().isPlaying) return;
    const payload = JSON.parse(raw) as LibraryDragPayload;
    const camera = getCamera();
    const canvas = canvasRef.current;
    let at: THREE.Vector3 | undefined;
    if (camera && canvas) {
      const r = canvas.getBoundingClientRect();
      const ndc = new THREE.Vector2(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
      at = groundPointAt(camera, ndc, controls.current?.target ?? new THREE.Vector3());
    }
    if (payload.kind === 'part') editorActions.addPrimitive(payload.type, at);
    else if (payload.kind === 'light') editorActions.addLight(payload.type);
    else {
      const asset = useAssetStore.getState().imports.find((a) => a.id === payload.id);
      if (asset) void editorActions.importModel(asset.file, { remember: false });
    }
  };

  // Ribbon Color / Material menus paint the selection in one undo step
  const selected = selectedId ? sceneManager.getEntity(selectedId) : undefined;
  const selectedMaterial = selected ? materialOf(sceneManager, selected.id) : null;
  const paint = selectedMaterial
    ? {
        color: rgbToHex(selectedMaterial.color),
        apply: ({ color, preset }: { color?: string; preset?: string }) => {
          if (!selected) return;
          const edit = MaterialEdit.start(sceneManager, bridge, selected.id);
          const mat = materialOf(sceneManager, selected.id);
          if (!edit || !mat) return;
          if (color) {
            const rgb = hexToRgb(color);
            edit.apply((mat.emissiveIntensity ?? 0) > 0 ? { color: rgb, emissive: rgb } : { color: rgb });
            edit.commit(history, 'Paint');
            logOutput('info', `Painted ${selected.name}`);
          } else if (preset) {
            const p = MATERIAL_PRESETS.find((m) => m.key === preset);
            if (!p) return;
            edit.apply({ ...p.patch, emissive: mat.color });
            edit.commit(history, `${p.label} material`);
            logOutput('info', `${selected.name} is now ${p.label.toLowerCase()}`);
          }
        },
      }
    : null;
  const selectionInfo = selected ? describeEntity(selected) : null;

  const peek = (cmd: { description: string } | undefined) => cmd?.description;
  const objectCount = sceneManager.getAllEntities().length;
  const showRight = panels.explorer || panels.properties;
  const formatLabel = FORMAT_LABEL[formatType];

  return (
    <div className="f3">
      <IconSprite />
      <TopBar
        title={sceneName}
        formatLabel={formatLabel}
        saveStatus={sceneId ? saveStatus : undefined}
        canUndo={history.canUndo()}
        canRedo={history.canRedo()}
        undoLabel={peek(history.peekUndo())}
        redoLabel={peek(history.peekRedo())}
        canPublish={!!sceneId}
        onPublish={() => setPublishOpen(true)}
        onRequestImport={requestImport}
        actions={editorActions}
        collab={collab ? { connected: collab.connected, peers: collab.peers } : null}
      />
      <Ribbon
        actions={editorActions}
        play={play}
        stop={stop}
        paint={paint}
        selection={selected ? { id: selected.id, editable: !!selectedMaterial, isModel: selectionInfo?.kind === 'model' || selectionInfo?.kind === 'mesh' } : null}
        onRequestImport={requestImport}
      />

      <div className="f3-main">
        {panels.library && <LibraryPanel actions={editorActions} onRequestImport={requestImport} />}

        <div className="f3-center">
          <div
            className="f3-viewport"
            onDragOver={onDragOver}
            onDragLeave={(e) => {
              if (!e.currentTarget.contains(e.relatedTarget as Node)) setDropActive(false);
            }}
            onDrop={onDrop}
          >
            <Viewport ref={canvasRef} onClick={handleCanvasClick} />
            <ViewportOverlay
              actions={editorActions}
              getCamera={getCamera}
              addFrameCallback={addFrameCallback}
              formatLabel={formatLabel}
              stop={stop}
              info={info}
              closeInfo={closeInfo}
              empty={ready && objectCount === 0 && (!sceneId || sceneLoadedRef.current)}
              dropActive={dropActive}
            />
            {collab && collab.peers.length > 0 && (
              <CursorOverlay
                peers={collab.peers}
                camera={ready ? (getCamera() as THREE.PerspectiveCamera | null) : null}
                canvasElement={canvasRef.current}
              />
            )}
          </div>
          {panels.output && (
            <OutputPanel
              actions={editorActions}
              sceneManager={sceneManager}
              play={play}
              stop={stop}
              camera={ready ? (getCamera() as THREE.PerspectiveCamera | null) : null}
              controls={controls}
            />
          )}
        </div>

        {showRight && (
          <aside className="f3-right" aria-label="Explorer and properties">
            {panels.explorer && (
              <ExplorerPanel
                sceneName={sceneName}
                nodes={hierarchyNodes}
                sceneManager={sceneManager}
                actions={editorActions}
                peers={collab?.peers ?? []}
                fill={!panels.properties}
              />
            )}
            {panels.properties && (
              <PropertiesPanel
                sceneName={sceneName}
                sceneManager={sceneManager}
                history={history}
                bridge={bridge}
                actions={editorActions}
                version={sceneVersion}
              />
            )}
          </aside>
        )}
      </div>

      <StatusBar backendLabel={backendLabel} objectCount={objectCount} selectionName={selected?.name ?? null} formatLabel={formatLabel} />

      <input
        ref={importInputRef}
        type="file"
        accept=".gltf,.glb"
        aria-label="Import model"
        style={{ display: 'none' }}
        onChange={(e) => {
          const file = e.target.files?.[0];
          e.target.value = '';
          if (file) void editorActions.importModel(file);
        }}
      />

      {sceneId && (
        <PublishDialog
          open={publishOpen}
          onClose={() => setPublishOpen(false)}
          sceneId={sceneId}
          sceneManager={sceneManager}
          canvasElement={canvasRef.current}
        />
      )}
    </div>
  );
}
