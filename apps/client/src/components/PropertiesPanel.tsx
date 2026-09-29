import { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import type { CommandHistory, SceneBridge, SceneManager } from '@forge3d/engine';
import type {
  ExperienceFormatType,
  LightData,
  MaterialDescriptor,
  MeshRendererData,
  TransformData,
} from '@forge3d/shared';
import { Icon } from '@/editor/Icon';
import { PropRow, Section, Select } from '@/editor/ui';
import { describeEntity } from '@/editor/entityInfo';
import { MATERIAL_PRESETS, MaterialEdit, hexToRgb, materialOf, rgbToHex } from '@/editor/materials';
import { useEditorStore } from '@/stores/editorStore';
import { useFormatStore } from '@/stores/formatStore';
import { FORMAT_LABEL } from '@/hooks/usePlayMode';
import type { EditorActions } from '@/hooks/useEditorActions';
import { InteractionEditor } from './InteractionEditor';

interface PropertiesPanelProps {
  sceneName: string;
  sceneManager: SceneManager;
  history: CommandHistory;
  bridge: React.RefObject<SceneBridge | null>;
  actions: EditorActions;
  /** Bumps whenever the scene changes, so values stay live during gizmo drags. */
  version: number;
}

const AXES = ['x', 'y', 'z'] as const;
const AXIS_COLOR = { x: 'var(--ax-x)', y: 'var(--ax-y)', z: 'var(--ax-z)' };

function fmt(v: number, digits: number): string {
  const s = v.toFixed(digits);
  return /^-0(\.0+)?$/.test(s) ? s.slice(1) : s;
}

/** Number field that edits on Enter/blur, cancels on Esc and nudges with ↑/↓ (Shift ×10). */
function NumField({
  axis,
  value,
  digits,
  step,
  label,
  onCommit,
  disabled,
}: {
  axis: (typeof AXES)[number];
  value: number;
  digits: number;
  step: number;
  label: string;
  onCommit: (v: number) => void;
  disabled?: boolean;
}) {
  const [draft, setDraft] = useState<string | null>(null);
  const commit = () => {
    if (draft === null) return;
    const v = parseFloat(draft.replace(',', '.'));
    setDraft(null);
    if (Number.isFinite(v) && v !== value) onCommit(v);
  };
  return (
    <label className="f3-num">
      <b style={{ color: AXIS_COLOR[axis] }}>{axis.toUpperCase()}</b>
      <input
        inputMode="decimal"
        aria-label={`${label} ${axis.toUpperCase()}`}
        disabled={disabled}
        value={draft ?? fmt(value, digits)}
        onChange={(e) => setDraft(e.target.value)}
        onBlur={commit}
        onKeyDown={(e) => {
          if (e.key === 'Enter') e.currentTarget.blur();
          else if (e.key === 'Escape') {
            setDraft(null);
            e.currentTarget.blur();
          } else if (e.key === 'ArrowUp' || e.key === 'ArrowDown') {
            e.preventDefault();
            setDraft(null);
            onCommit(value + (e.key === 'ArrowUp' ? 1 : -1) * step * (e.shiftKey ? 10 : 1));
          }
        }}
      />
    </label>
  );
}

function VecRow({
  label,
  values,
  digits,
  step,
  onCommit,
  disabled,
}: {
  label: string;
  values: [number, number, number];
  digits: number;
  step: number;
  onCommit: (axis: 0 | 1 | 2, v: number) => void;
  disabled?: boolean;
}) {
  return (
    <PropRow label={label} variant="v">
      <div className="f3-vec">
        {AXES.map((axis, i) => (
          <NumField
            key={axis}
            axis={axis}
            label={label}
            value={values[i]!}
            digits={digits}
            step={step}
            disabled={disabled}
            onCommit={(v) => onCommit(i as 0 | 1 | 2, v)}
          />
        ))}
      </div>
    </PropRow>
  );
}

function Slider({
  label,
  value,
  min,
  max,
  step,
  display,
  onChange,
  disabled,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  display: string;
  onChange: (v: number) => void;
  disabled?: boolean;
}) {
  return (
    <PropRow label={label}>
      <div className="f3-rngrow">
        <input
          type="range"
          className="f3-rng"
          aria-label={label}
          min={min}
          max={max}
          step={step}
          value={value}
          disabled={disabled}
          onChange={(e) => onChange(parseFloat(e.target.value))}
        />
        <span className="f3-rv">{display}</span>
      </div>
    </PropRow>
  );
}

/**
 * Material edits preview live and land in history as one undo step once the
 * user pauses, so a slider drag or a colour-picker session is a single undo.
 */
function useMaterialGesture(
  sceneManager: SceneManager,
  bridge: React.RefObject<SceneBridge | null>,
  history: CommandHistory,
  entityId: string | null,
) {
  const editRef = useRef<{ id: string; edit: MaterialEdit } | null>(null);
  const timerRef = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);

  const flush = () => {
    clearTimeout(timerRef.current);
    editRef.current?.edit.commit(history);
    editRef.current = null;
  };

  // Commit any open gesture when the selection changes or the panel unmounts
  useEffect(() => flush, [entityId]); // eslint-disable-line react-hooks/exhaustive-deps

  return (patch: Partial<MaterialDescriptor>) => {
    if (!entityId) return;
    if (!editRef.current || editRef.current.id !== entityId) {
      flush();
      const edit = MaterialEdit.start(sceneManager, bridge, entityId);
      if (!edit) return;
      editRef.current = { id: entityId, edit };
    }
    editRef.current.edit.apply(patch);
    clearTimeout(timerRef.current);
    timerRef.current = setTimeout(flush, 500);
  };
}

function presetKey(m: MaterialDescriptor): string {
  const found = MATERIAL_PRESETS.find(
    (p) =>
      Math.abs((p.patch.metalness ?? 0) - m.metalness) < 0.02 &&
      Math.abs((p.patch.roughness ?? 0) - m.roughness) < 0.02 &&
      Math.abs((p.patch.opacity ?? 1) - (m.opacity ?? 1)) < 0.02 &&
      Math.abs((p.patch.emissiveIntensity ?? 0) - (m.emissiveIntensity ?? 0)) < 0.02,
  );
  return found?.key ?? 'custom';
}

function ExperienceSettings({ sceneManager }: { sceneManager: SceneManager }) {
  const f = useFormatStore();
  const setDockTab = useEditorStore((s) => s.setDockTab);
  const entities = sceneManager.getAllEntities();
  const parts = entities.filter((e) => e.hasComponent('meshRenderer')).length;
  const lights = entities.filter((e) => e.hasComponent('light')).length;

  return (
    <>
      <Section title="Experience">
        <PropRow label="Format">
          <Select<ExperienceFormatType>
            label="Format"
            value={f.formatType}
            onChange={f.setFormatType}
            options={(Object.keys(FORMAT_LABEL) as ExperienceFormatType[]).map((k) => ({
              value: k,
              label: FORMAT_LABEL[k],
            }))}
          />
        </PropRow>
        {f.formatType === 'turntable' && (
          <>
            <Slider
              label="Spin speed"
              value={f.turntableSpeed}
              min={0.1}
              max={5}
              step={0.1}
              display={`${f.turntableSpeed.toFixed(1)}×`}
              onChange={f.setTurntableSpeed}
            />
            <PropRow label="Spin axis">
              <Select<'y' | 'x'>
                label="Spin axis"
                value={f.turntableAxis}
                onChange={f.setTurntableAxis}
                options={[
                  { value: 'y', label: 'Vertical (Y)' },
                  { value: 'x', label: 'Horizontal (X)' },
                ]}
              />
            </PropRow>
          </>
        )}
        {f.formatType === 'animated' && (
          <PropRow label="Length">
            <div className="f3-rngrow">
              <input
                className="f3-input f3-mono"
                aria-label="Animation length in seconds"
                type="number"
                min={1}
                step={1}
                value={f.animationDuration}
                onChange={(e) => f.setAnimationDuration(parseFloat(e.target.value) || 5)}
                style={{ width: 70 }}
              />
              <span style={{ color: 'var(--tx3)', fontSize: 12 }}>seconds</span>
            </div>
          </PropRow>
        )}
        {f.formatType === 'video' && (
          <>
            <PropRow label="Camera keys">
              <span className="f3-mono" style={{ fontSize: 12 }}>
                {f.keyframes.length}
              </span>
            </PropRow>
            {f.keyframes.length < 2 && (
              <div className="f3-note">Add at least two camera keyframes to play a 3D video.</div>
            )}
            <div style={{ padding: '4px 12px 0' }}>
              <button
                type="button"
                className="f3-btn ghost"
                style={{ height: 26, fontSize: 12 }}
                onClick={() => setDockTab('timeline')}
              >
                <Icon name="timeline" size={13} />
                Open timeline
              </button>
            </div>
          </>
        )}
        {f.formatType === 'interactive' && (
          <>
            <PropRow label="Interactions">
              <span className="f3-mono" style={{ fontSize: 12 }}>
                {f.interactions.length}
              </span>
            </PropRow>
            <div className="f3-note">
              Select an object to decide what happens when a viewer clicks it.
            </div>
          </>
        )}
      </Section>
      <Section title="Scene">
        <PropRow label="Parts">
          <span className="f3-mono" style={{ fontSize: 12 }}>
            {parts}
          </span>
        </PropRow>
        <PropRow label="Lights">
          <span className="f3-mono" style={{ fontSize: 12 }}>
            {lights}
          </span>
        </PropRow>
        <div className="f3-note">Press Play to see it exactly the way viewers will.</div>
      </Section>
    </>
  );
}

export function PropertiesPanel({
  sceneName,
  sceneManager,
  history,
  bridge,
  actions,
  version,
}: PropertiesPanelProps) {
  const selectedId = useEditorStore((s) => s.selectedEntityId);
  const playing = useEditorStore((s) => s.isPlaying);
  const formatType = useFormatStore((s) => s.formatType);
  const paint = useMaterialGesture(sceneManager, bridge, history, selectedId);
  const [nameDraft, setNameDraft] = useState<string | null>(null);
  void version;

  useEffect(() => setNameDraft(null), [selectedId]);

  const entity = selectedId ? sceneManager.getEntity(selectedId) : undefined;

  if (!entity) {
    return (
      <section
        aria-label="Properties"
        style={{ flexGrow: 1, display: 'flex', flexDirection: 'column', minHeight: 0 }}
      >
        <div className="f3-ph">
          <span className="f3-pt">Properties</span>
        </div>
        <div className="f3-obj">
          <div className="f3-obj-ic">
            <Icon name="scene" size={17} />
          </div>
          <div style={{ minWidth: 0, display: 'flex', flexDirection: 'column', gap: 1 }}>
            <span style={{ fontSize: 13, fontWeight: 600 }}>{sceneName}</span>
            <span style={{ fontSize: 11, color: 'var(--tx3)' }}>Scene</span>
          </div>
        </div>
        <div className="f3-props">
          <ExperienceSettings sceneManager={sceneManager} />
        </div>
      </section>
    );
  }

  const info = describeEntity(entity);
  const t = entity.transform;
  const euler = new THREE.Euler().setFromQuaternion(new THREE.Quaternion(...t.rotation));
  const degrees: [number, number, number] = [
    THREE.MathUtils.radToDeg(euler.x),
    THREE.MathUtils.radToDeg(euler.y),
    THREE.MathUtils.radToDeg(euler.z),
  ];
  const material = materialOf(sceneManager, entity.id);
  const mesh = entity.getComponent<MeshRendererData>('meshRenderer');
  const light = entity.getComponent<LightData>('light');

  const setPos = (axis: 0 | 1 | 2, v: number) => {
    const position = [...t.position] as TransformData['position'];
    position[axis] = v;
    actions.setTransform(entity.id, { position }, 'Move');
  };
  const setRot = (axis: 0 | 1 | 2, v: number) => {
    const deg = [...degrees] as [number, number, number];
    deg[axis] = v;
    const q = new THREE.Quaternion().setFromEuler(
      new THREE.Euler(
        THREE.MathUtils.degToRad(deg[0]),
        THREE.MathUtils.degToRad(deg[1]),
        THREE.MathUtils.degToRad(deg[2]),
      ),
    );
    actions.setTransform(entity.id, { rotation: [q.x, q.y, q.z, q.w] }, 'Rotate');
  };
  const setScale = (axis: 0 | 1 | 2, v: number) => {
    const scale = [...t.scale] as TransformData['scale'];
    scale[axis] = Math.abs(v) < 0.01 ? (v < 0 ? -0.01 : 0.01) : v;
    actions.setTransform(entity.id, { scale }, 'Scale');
  };

  const updateLight = (patch: Partial<LightData>) => {
    if (!light) return;
    const updated: LightData = { ...light, ...patch };
    entity.setComponent('light', updated);
    bridge.current?.updateLight(entity.id, updated);
    sceneManager.notifyChange();
  };

  const colorHex = material ? rgbToHex(material.color) : '#ffffff';
  const glow = material?.emissiveIntensity ?? 0;

  return (
    <section
      aria-label="Properties"
      style={{ flexGrow: 1, display: 'flex', flexDirection: 'column', minHeight: 0 }}
    >
      <div className="f3-ph">
        <span className="f3-pt">Properties</span>
      </div>
      <div className="f3-obj">
        <div className="f3-obj-ic" style={{ color: info.color }}>
          <Icon name={info.icon} size={17} />
        </div>
        <div style={{ minWidth: 0, flexGrow: 1, display: 'flex', flexDirection: 'column', gap: 1 }}>
          <input
            className="f3-obj-name"
            aria-label="Name"
            value={nameDraft ?? entity.name}
            disabled={playing}
            onChange={(e) => setNameDraft(e.target.value)}
            onBlur={() => {
              if (nameDraft !== null) actions.rename(entity.id, nameDraft);
              setNameDraft(null);
            }}
            onKeyDown={(e) => {
              if (e.key === 'Enter') e.currentTarget.blur();
              if (e.key === 'Escape') {
                setNameDraft(null);
                e.currentTarget.blur();
              }
            }}
          />
          <span style={{ fontSize: 11, color: 'var(--tx3)' }}>{info.type}</span>
        </div>
      </div>

      <div className="f3-props">
        <Section title="Transform">
          <VecRow
            label="Position"
            values={t.position}
            digits={2}
            step={0.1}
            onCommit={setPos}
            disabled={playing}
          />
          <VecRow
            label="Rotation"
            values={degrees}
            digits={1}
            step={5}
            onCommit={setRot}
            disabled={playing}
          />
          <VecRow
            label="Scale"
            values={t.scale}
            digits={2}
            step={0.1}
            onCommit={setScale}
            disabled={playing}
          />
        </Section>

        {material && (
          <Section title="Appearance">
            <PropRow label="Color">
              <div className="f3-rngrow">
                <label className="f3-sw" style={{ background: colorHex }}>
                  <input
                    type="color"
                    aria-label="Color"
                    value={colorHex}
                    disabled={playing}
                    onChange={(e) => {
                      const color = hexToRgb(e.target.value);
                      paint(glow > 0 ? { color, emissive: color } : { color });
                    }}
                  />
                </label>
                <span className="f3-mono" style={{ fontSize: 11.5, color: 'var(--tx2)' }}>
                  {colorHex.toUpperCase()}
                </span>
              </div>
            </PropRow>
            <PropRow label="Material">
              <Select
                label="Material"
                value={presetKey(material)}
                onChange={(key) => {
                  const preset = MATERIAL_PRESETS.find((p) => p.key === key);
                  if (preset) paint({ ...preset.patch, emissive: material.color });
                }}
                options={[
                  ...MATERIAL_PRESETS.map((p) => ({ value: p.key, label: p.label })),
                  ...(presetKey(material) === 'custom'
                    ? [{ value: 'custom', label: 'Custom' }]
                    : []),
                ]}
              />
            </PropRow>
            <Slider
              label="Metalness"
              value={material.metalness}
              min={0}
              max={1}
              step={0.01}
              display={fmt(material.metalness, 2)}
              disabled={playing}
              onChange={(metalness) => paint({ metalness })}
            />
            <Slider
              label="Roughness"
              value={material.roughness}
              min={0}
              max={1}
              step={0.01}
              display={fmt(material.roughness, 2)}
              disabled={playing}
              onChange={(roughness) => paint({ roughness })}
            />
            <Slider
              label="Glow"
              value={glow}
              min={0}
              max={4}
              step={0.05}
              display={fmt(glow, 1)}
              disabled={playing}
              onChange={(emissiveIntensity) =>
                paint({ emissiveIntensity, emissive: material.color })
              }
            />
            <Slider
              label="Opacity"
              value={material.opacity ?? 1}
              min={0.05}
              max={1}
              step={0.01}
              display={`${Math.round((material.opacity ?? 1) * 100)}%`}
              disabled={playing}
              onChange={(opacity) => paint({ opacity, transparent: opacity < 0.999 })}
            />
          </Section>
        )}

        {mesh?.geometryType === 'imported' && (
          <Section title="Appearance">
            <div className="f3-note" style={{ marginTop: 0 }}>
              Uses the materials from the imported file.
            </div>
          </Section>
        )}

        {light && (
          <Section title="Light">
            <PropRow label="Color">
              <div className="f3-rngrow">
                <label className="f3-sw" style={{ background: rgbToHex(light.color) }}>
                  <input
                    type="color"
                    aria-label="Light color"
                    value={rgbToHex(light.color)}
                    disabled={playing}
                    onChange={(e) => updateLight({ color: hexToRgb(e.target.value) })}
                  />
                </label>
                <span className="f3-mono" style={{ fontSize: 11.5, color: 'var(--tx2)' }}>
                  {rgbToHex(light.color).toUpperCase()}
                </span>
              </div>
            </PropRow>
            <Slider
              label="Intensity"
              value={light.intensity}
              min={0}
              max={10}
              step={0.1}
              display={fmt(light.intensity, 1)}
              disabled={playing}
              onChange={(intensity) => updateLight({ intensity })}
            />
          </Section>
        )}

        {formatType === 'interactive' && (
          <Section title="Interaction">
            <InteractionEditor />
          </Section>
        )}
      </div>
    </section>
  );
}
