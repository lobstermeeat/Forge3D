import { useState, useEffect, useCallback } from 'react';
import * as THREE from 'three';
import { useEditorStore } from '@/stores/editorStore';
import { useFormatStore } from '@/stores/formatStore';
import type { SceneManager, CommandHistory, SceneBridge } from '@forge3d/engine';
import { TransformCommand, RenameEntityCommand } from '@forge3d/engine';
import type { TransformData, MeshRendererData, MaterialDescriptor, LightData } from '@forge3d/shared';
import { InteractionEditor } from './InteractionEditor';

interface InspectorPanelProps {
  sceneManager: SceneManager;
  history: CommandHistory;
  bridge: React.RefObject<SceneBridge | null>;
}

export function InspectorPanel({ sceneManager, history, bridge }: InspectorPanelProps) {
  const selectedId = useEditorStore((s) => s.selectedEntityId);
  const formatType = useFormatStore((s) => s.formatType);
  const [, forceUpdate] = useState(0);

  useEffect(() => {
    const unsub = sceneManager.subscribe(() => forceUpdate((v) => v + 1));
    return unsub;
  }, [sceneManager]);

  const entity = selectedId ? sceneManager.getEntity(selectedId) : null;

  if (!entity) {
    return (
      <div style={{ padding: 16, color: '#666', fontSize: 13 }}>
        No entity selected
      </div>
    );
  }

  const transform = entity.transform;
  const meshRenderer = entity.getComponent<MeshRendererData>('meshRenderer');
  const lightData = entity.getComponent<LightData>('light');
  const materials = sceneManager.getMaterials();
  const material = meshRenderer ? materials[meshRenderer.materialIndex] ?? null : null;

  return (
    <div style={{ padding: 12, color: '#ccc', fontSize: 13, overflowY: 'auto', height: '100%', background: '#1e1e2e' }}>
      <SectionHeader label="Entity" />
      <NameInput
        value={entity.name}
        onChange={(name) => {
          const cmd = new RenameEntityCommand(sceneManager, entity.id, name);
          history.execute(cmd);
        }}
      />

      <SectionHeader label="Transform" />
      <Vec3Row
        label="Position"
        value={transform.position}
        onChange={(pos) => {
          const before = { ...transform };
          const after: TransformData = { ...transform, position: pos };
          history.execute(new TransformCommand(sceneManager, entity.id, before, after));
        }}
      />
      <EulerRow
        label="Rotation"
        quaternion={transform.rotation}
        onChange={(quat) => {
          const before = { ...transform };
          const after: TransformData = { ...transform, rotation: quat };
          history.execute(new TransformCommand(sceneManager, entity.id, before, after));
        }}
      />
      <Vec3Row
        label="Scale"
        value={transform.scale}
        step={0.1}
        onChange={(scale) => {
          const before = { ...transform };
          const after: TransformData = { ...transform, scale };
          history.execute(new TransformCommand(sceneManager, entity.id, before, after));
        }}
      />

      {material && meshRenderer && (
        <>
          <SectionHeader label="Material" />
          <ColorRow
            label="Color"
            value={material.color}
            onChange={(color) => {
              sceneManager.updateMaterial(meshRenderer.materialIndex, { color });
              bridge.current?.updateMaterial(entity.id, { ...material, color });
            }}
          />
          <SliderRow
            label="Metalness"
            value={material.metalness}
            onChange={(metalness) => {
              sceneManager.updateMaterial(meshRenderer.materialIndex, { metalness });
              bridge.current?.updateMaterial(entity.id, { ...material, metalness });
            }}
          />
          <SliderRow
            label="Roughness"
            value={material.roughness}
            onChange={(roughness) => {
              sceneManager.updateMaterial(meshRenderer.materialIndex, { roughness });
              bridge.current?.updateMaterial(entity.id, { ...material, roughness });
            }}
          />
        </>
      )}

      {lightData && (
        <>
          <SectionHeader label="Light" />
          <div style={{ fontSize: 12, color: '#a6adc8', marginBottom: 6 }}>
            Type: {lightData.type}
          </div>
          <ColorRow
            label="Color"
            value={lightData.color}
            onChange={(color) => {
              const updated: LightData = { ...lightData, color };
              entity.setComponent('light', updated);
              bridge.current?.updateLight(entity.id, updated);
              sceneManager.notifyChange();
            }}
          />
          <SliderRow
            label="Intensity"
            value={lightData.intensity}
            min={0}
            max={10}
            step={0.1}
            onChange={(intensity) => {
              const updated: LightData = { ...lightData, intensity };
              entity.setComponent('light', updated);
              bridge.current?.updateLight(entity.id, updated);
              sceneManager.notifyChange();
            }}
          />
        </>
      )}

      {formatType === 'interactive' && <InteractionEditor />}
    </div>
  );
}

// --- Sub-components ---

function SectionHeader({ label }: { label: string }) {
  return (
    <div style={{ fontSize: 11, fontWeight: 600, color: '#888', textTransform: 'uppercase', letterSpacing: 1, margin: '12px 0 6px' }}>
      {label}
    </div>
  );
}

function NameInput({ value, onChange }: { value: string; onChange: (v: string) => void }) {
  const [local, setLocal] = useState(value);
  useEffect(() => setLocal(value), [value]);

  return (
    <input
      value={local}
      onChange={(e) => setLocal(e.target.value)}
      onBlur={() => { if (local !== value) onChange(local); }}
      onKeyDown={(e) => { if (e.key === 'Enter') { (e.target as HTMLInputElement).blur(); } }}
      style={inputStyle}
    />
  );
}

function Vec3Row({
  label,
  value,
  step = 0.1,
  onChange,
}: {
  label: string;
  value: [number, number, number];
  step?: number;
  onChange: (v: [number, number, number]) => void;
}) {
  const axes = ['X', 'Y', 'Z'] as const;
  return (
    <div style={{ marginBottom: 6 }}>
      <div style={{ fontSize: 11, color: '#888', marginBottom: 2 }}>{label}</div>
      <div style={{ display: 'flex', gap: 4 }}>
        {axes.map((axis, i) => (
          <div key={axis} style={{ flex: 1 }}>
            <label style={{ fontSize: 10, color: '#666' }}>{axis}</label>
            <input
              type="number"
              step={step}
              value={Number((value[i] ?? 0).toFixed(3))}
              onChange={(e) => {
                const next = [...value] as [number, number, number];
                next[i] = parseFloat(e.target.value) || 0;
                onChange(next);
              }}
              style={{ ...inputStyle, width: '100%' }}
            />
          </div>
        ))}
      </div>
    </div>
  );
}

function EulerRow({
  label,
  quaternion,
  onChange,
}: {
  label: string;
  quaternion: [number, number, number, number];
  onChange: (quat: [number, number, number, number]) => void;
}) {
  const euler = new THREE.Euler().setFromQuaternion(
    new THREE.Quaternion(quaternion[0], quaternion[1], quaternion[2], quaternion[3]),
  );
  const degrees: [number, number, number] = [
    THREE.MathUtils.radToDeg(euler.x),
    THREE.MathUtils.radToDeg(euler.y),
    THREE.MathUtils.radToDeg(euler.z),
  ];

  return (
    <Vec3Row
      label={label}
      value={degrees}
      step={1}
      onChange={(deg) => {
        const e = new THREE.Euler(
          THREE.MathUtils.degToRad(deg[0]),
          THREE.MathUtils.degToRad(deg[1]),
          THREE.MathUtils.degToRad(deg[2]),
        );
        const q = new THREE.Quaternion().setFromEuler(e);
        onChange([q.x, q.y, q.z, q.w]);
      }}
    />
  );
}

function ColorRow({
  label,
  value,
  onChange,
}: {
  label: string;
  value: [number, number, number];
  onChange: (v: [number, number, number]) => void;
}) {
  const hex =
    '#' +
    new THREE.Color(value[0], value[1], value[2])
      .getHexString();

  return (
    <div style={{ marginBottom: 6 }}>
      <div style={{ fontSize: 11, color: '#888', marginBottom: 2 }}>{label}</div>
      <input
        type="color"
        value={hex}
        onChange={(e) => {
          const c = new THREE.Color(e.target.value);
          onChange([c.r, c.g, c.b]);
        }}
        style={{ width: '100%', height: 28, border: 'none', borderRadius: 4, cursor: 'pointer', background: '#313244' }}
      />
    </div>
  );
}

function SliderRow({
  label,
  value,
  min = 0,
  max = 1,
  step = 0.01,
  onChange,
}: {
  label: string;
  value: number;
  min?: number;
  max?: number;
  step?: number;
  onChange: (v: number) => void;
}) {
  return (
    <div style={{ marginBottom: 6 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, color: '#888', marginBottom: 2 }}>
        <span>{label}</span>
        <span>{value.toFixed(2)}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        style={{ width: '100%' }}
      />
    </div>
  );
}

const inputStyle: React.CSSProperties = {
  padding: '4px 8px',
  background: '#313244',
  border: '1px solid #45475a',
  borderRadius: 4,
  color: '#cdd6f4',
  fontSize: 12,
  outline: 'none',
};
