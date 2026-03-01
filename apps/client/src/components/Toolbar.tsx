import { useRef } from 'react';
import * as THREE from 'three';
import { useEditorStore } from '@/stores/editorStore';
import type { TransformMode, SceneManager, CommandHistory } from '@forge3d/engine';
import type { SceneBridge } from '@forge3d/engine';
import { exportSceneAsGLB } from '@forge3d/engine';
import type { MeshRendererData, LightData } from '@forge3d/shared';
import type { SaveStatus } from '@/hooks/useAutoSave';

const modes: { key: TransformMode; label: string; shortcut: string }[] = [
  { key: 'translate', label: 'Move', shortcut: 'G' },
  { key: 'rotate', label: 'Rotate', shortcut: 'R' },
  { key: 'scale', label: 'Scale', shortcut: 'S' },
];

const primitives: { type: MeshRendererData['geometryType']; label: string }[] = [
  { type: 'box', label: 'Box' },
  { type: 'sphere', label: 'Sphere' },
  { type: 'plane', label: 'Plane' },
  { type: 'cylinder', label: 'Cylinder' },
  { type: 'torus', label: 'Torus' },
];

const lights: { type: LightData['type']; label: string }[] = [
  { type: 'directional', label: 'Dir Light' },
  { type: 'point', label: 'Point Light' },
  { type: 'ambient', label: 'Ambient' },
];

interface ToolbarProps {
  onAddPrimitive: (type: MeshRendererData['geometryType']) => void;
  onAddLight?: (type: LightData['type']) => void;
  onImportGLTF?: (file: File) => void;
  onPublish?: () => void;
  sceneManager: SceneManager;
  history: CommandHistory;
  bridge: React.RefObject<SceneBridge | null>;
  threeScene?: THREE.Scene | null;
  saveStatus?: SaveStatus;
  canPublish?: boolean;
}

export function Toolbar({
  onAddPrimitive,
  onAddLight,
  onImportGLTF,
  onPublish,
  sceneManager,
  history,
  bridge,
  threeScene,
  saveStatus,
  canPublish,
}: ToolbarProps) {
  const transformMode = useEditorStore((s) => s.transformMode);
  const setTransformMode = useEditorStore((s) => s.setTransformMode);
  const showGrid = useEditorStore((s) => s.showGrid);
  const toggleGrid = useEditorStore((s) => s.toggleGrid);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const gltfInputRef = useRef<HTMLInputElement>(null);

  const handleSave = () => {
    const data = sceneManager.serialize();
    const json = JSON.stringify(data, null, 2);
    const blob = new Blob([json], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'scene.forge3d.json';
    a.click();
    URL.revokeObjectURL(url);
  };

  const handleLoad = () => {
    fileInputRef.current?.click();
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      try {
        const data = JSON.parse(reader.result as string);
        sceneManager.deserialize(data);
        history.clear();
      } catch (err) {
        console.error('Failed to load scene:', err);
      }
    };
    reader.readAsText(file);
    e.target.value = '';
  };

  const handleExportGLB = async () => {
    const objects = bridge.current?.getManagedObjects();
    if (!objects || objects.length === 0) return;
    try {
      const buffer = await exportSceneAsGLB(objects);
      const blob = new Blob([buffer], { type: 'model/gltf-binary' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'scene.glb';
      a.click();
      URL.revokeObjectURL(url);
    } catch (err) {
      console.error('GLB export failed:', err);
    }
  };

  const handleImportGLTF = () => {
    gltfInputRef.current?.click();
  };

  const handleGLTFFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    onImportGLTF?.(file);
    e.target.value = '';
  };

  const handleUndo = () => {
    history.undo();
    sceneManager.notifyChange();
  };

  const handleRedo = () => {
    history.redo();
    sceneManager.notifyChange();
  };

  const statusColor =
    saveStatus === 'saving'
      ? '#f9e2af'
      : saveStatus === 'saved'
        ? '#a6e3a1'
        : saveStatus === 'error'
          ? '#f38ba8'
          : 'transparent';

  return (
    <div
      role="toolbar"
      aria-label="Editor toolbar"
      style={{
        display: 'flex',
        gap: 4,
        padding: '6px 12px',
        background: '#181825',
        borderBottom: '1px solid #313244',
        alignItems: 'center',
        flexWrap: 'wrap',
      }}
    >
      {/* Transform modes */}
      {modes.map((mode) => (
        <button
          key={mode.key}
          title={`${mode.label} (${mode.shortcut})`}
          aria-pressed={transformMode === mode.key}
          onClick={() => setTransformMode(mode.key)}
          style={btnStyle(transformMode === mode.key)}
        >
          {mode.label}
        </button>
      ))}

      <Separator />

      {/* Add primitives */}
      {primitives.map((p) => (
        <button key={p.type} title={`Add ${p.label}`} onClick={() => onAddPrimitive(p.type)} style={btnStyle(false)}>
          + {p.label}
        </button>
      ))}

      <Separator />

      {/* Add lights */}
      {lights.map((l) => (
        <button
          key={l.type}
          title={`Add ${l.label}`}
          onClick={() => onAddLight?.(l.type)}
          style={btnStyle(false)}
        >
          + {l.label}
        </button>
      ))}

      <Separator />

      {/* Undo / Redo */}
      <button title="Undo (Ctrl+Z)" onClick={handleUndo} style={btnStyle(false)}>
        Undo
      </button>
      <button title="Redo (Ctrl+Shift+Z)" onClick={handleRedo} style={btnStyle(false)}>
        Redo
      </button>

      <Separator />

      {/* File I/O */}
      <button title="Save scene (JSON)" onClick={handleSave} style={btnStyle(false)}>
        Save
      </button>
      <button title="Load scene (JSON)" onClick={handleLoad} style={btnStyle(false)}>
        Load
      </button>
      <button title="Export as GLB" onClick={handleExportGLB} style={btnStyle(false)}>
        Export GLB
      </button>
      <button title="Import GLTF/GLB" onClick={handleImportGLTF} style={btnStyle(false)}>
        Import
      </button>
      <input
        ref={fileInputRef}
        type="file"
        accept=".json"
        onChange={handleFileChange}
        style={{ display: 'none' }}
      />
      <input
        ref={gltfInputRef}
        type="file"
        accept=".gltf,.glb"
        onChange={handleGLTFFileChange}
        style={{ display: 'none' }}
      />

      {saveStatus && (
        <span style={{ fontSize: 11, color: statusColor, marginLeft: 4 }}>
          {saveStatus === 'saving'
            ? 'Saving...'
            : saveStatus === 'saved'
              ? 'Saved'
              : saveStatus === 'error'
                ? 'Save failed'
                : ''}
        </span>
      )}

      <div style={{ flex: 1 }} />

      {/* Publish */}
      {canPublish && (
        <button
          title="Publish experience"
          onClick={onPublish}
          style={{
            padding: '4px 14px',
            border: 'none',
            borderRadius: 4,
            background: '#2563eb',
            color: '#fff',
            cursor: 'pointer',
            fontSize: 12,
            fontWeight: 600,
          }}
        >
          Publish
        </button>
      )}

      {/* Grid toggle */}
      <button
        title="Toggle grid"
        aria-pressed={showGrid}
        onClick={toggleGrid}
        style={btnStyle(showGrid)}
      >
        Grid
      </button>
    </div>
  );
}

function Separator() {
  return <div style={{ width: 1, height: 20, background: '#313244', margin: '0 4px' }} />;
}

function btnStyle(active: boolean): React.CSSProperties {
  return {
    padding: '4px 10px',
    border: 'none',
    borderRadius: 4,
    background: active ? '#2563eb' : '#313244',
    color: active ? '#fff' : '#ccc',
    cursor: 'pointer',
    fontSize: 12,
  };
}
