import type { ReactNode } from 'react';
import type { LightData } from '@forge3d/shared';
import type { ExperienceFormatType } from '@forge3d/shared';
import { Icon } from '@/editor/Icon';
import { MenuAnchor, MenuItem } from '@/editor/ui';
import { PRIMITIVES } from '@/editor/placement';
import { SWATCHES, MATERIAL_PRESETS } from '@/editor/materials';
import { useEditorStore, type CameraView, type PanelKey } from '@/stores/editorStore';
import { useFormatStore } from '@/stores/formatStore';
import { FORMAT_LABEL } from '@/hooks/usePlayMode';
import type { EditorActions } from '@/hooks/useEditorActions';

const LIGHTS: { type: LightData['type']; label: string; icon: string }[] = [
  { type: 'directional', label: 'Sun', icon: 'sun' },
  { type: 'point', label: 'Point', icon: 'bulb' },
  { type: 'ambient', label: 'Ambient', icon: 'ambient' },
];

const FORMATS: ExperienceFormatType[] = ['turntable', 'interactive', 'animated', 'video'];

interface RibbonProps {
  actions: EditorActions;
  play: () => void;
  stop: () => void;
  /** Material editing for the selection (null when it has no editable material). */
  paint: {
    color: string | null;
    apply: (patch: { color?: string; preset?: string }) => void;
  } | null;
  selection: { id: string; editable: boolean; isModel: boolean } | null;
  onRequestImport: () => void;
}

function Group({
  label,
  children,
  small,
}: {
  label: string;
  children?: ReactNode;
  small?: ReactNode;
}) {
  return (
    <div className="f3-rg" role="group" aria-label={label}>
      <div className="f3-rg-b">
        {children}
        {small && <div className="f3-rsm">{small}</div>}
      </div>
      <div className="f3-rg-l">{label}</div>
    </div>
  );
}

interface BtnProps {
  icon: string;
  label: string;
  onClick?: () => void;
  pressed?: boolean;
  disabled?: boolean;
  title?: string;
  className?: string;
  caret?: boolean;
  expanded?: boolean;
}

function Big({
  icon,
  label,
  onClick,
  pressed,
  disabled,
  title,
  className,
  caret,
  expanded,
}: BtnProps) {
  return (
    <button
      type="button"
      className={className ? `f3-rbl ${className}` : 'f3-rbl'}
      onClick={onClick}
      aria-pressed={pressed}
      aria-expanded={expanded}
      disabled={disabled}
      title={title ?? label}
    >
      <Icon name={icon} size={21} />
      <span>
        {label}
        {caret && <Icon name="chevdown" size={10} />}
      </span>
    </button>
  );
}

function Small({
  icon,
  label,
  onClick,
  pressed,
  disabled,
  title,
  caret,
  expanded,
  check,
}: BtnProps & { check?: boolean }) {
  return (
    <button
      type="button"
      className="f3-rbs"
      onClick={onClick}
      aria-pressed={pressed}
      aria-expanded={expanded}
      disabled={disabled}
      title={title ?? label}
    >
      <Icon name={icon} size={14} />
      <span>{label}</span>
      {caret && <Icon name="chevdown" size={10} />}
      {check && <span className="f3-check" />}
    </button>
  );
}

export function Ribbon({ actions, play, stop, paint, selection, onRequestImport }: RibbonProps) {
  const s = useEditorStore();
  const formatType = useFormatStore((f) => f.formatType);
  const setFormatType = useFormatStore((f) => f.setFormatType);
  const playing = s.isPlaying;
  const noSel = !selection || playing;
  const tool = s.selectOnly ? 'select' : s.transformMode;

  const toolButtons = (
    <>
      <Big
        icon="select"
        label="Select"
        pressed={tool === 'select'}
        disabled={playing}
        onClick={s.setSelectOnly}
        title="Select — click objects without moving them"
      />
      <Big
        icon="move"
        label="Move"
        pressed={tool === 'translate'}
        disabled={playing}
        onClick={() => s.setTransformMode('translate')}
        title="Move (G)"
      />
      <Big
        icon="rotate"
        label="Rotate"
        pressed={tool === 'rotate'}
        disabled={playing}
        onClick={() => s.setTransformMode('rotate')}
        title="Rotate (R)"
      />
      <Big
        icon="scale"
        label="Scale"
        pressed={tool === 'scale'}
        disabled={playing}
        onClick={() => s.setTransformMode('scale')}
        title="Scale (S)"
      />
    </>
  );

  const partMenu = (
    <MenuAnchor
      popStyle={{ minWidth: 170 }}
      trigger={(open, toggle) => (
        <Big
          icon="box"
          label="Part"
          caret
          expanded={open}
          disabled={playing}
          onClick={toggle}
          title="Insert a part"
        />
      )}
    >
      {(close) =>
        PRIMITIVES.map((p) => (
          <MenuItem
            key={p.type}
            icon={p.type}
            label={p.label}
            onSelect={() => {
              close();
              actions.addPrimitive(p.type);
            }}
          />
        ))
      }
    </MenuAnchor>
  );

  const lightMenu = (
    <MenuAnchor
      popStyle={{ minWidth: 170 }}
      trigger={(open, toggle) => (
        <Big
          icon="bulb"
          label="Light"
          caret
          expanded={open}
          disabled={playing}
          onClick={toggle}
          title="Insert a light"
        />
      )}
    >
      {(close) =>
        LIGHTS.map((l) => (
          <MenuItem
            key={l.type}
            icon={l.icon}
            label={`${l.label} light`}
            onSelect={() => {
              close();
              actions.addLight(l.type);
            }}
          />
        ))
      }
    </MenuAnchor>
  );

  const snapButtons = (
    <>
      <Small
        icon="snap"
        label="Snap"
        pressed={s.snapEnabled}
        check
        onClick={s.toggleSnap}
        title="Snap moves and rotations to steps (hold Shift to snap while this is off)"
      />
      <Small
        icon="move"
        label={`Move ${s.moveStep} m`}
        onClick={s.cycleMoveStep}
        title="Change the move step"
      />
      <Small
        icon="rotate"
        label={`Rotate ${s.rotateStep}°`}
        onClick={s.cycleRotateStep}
        title="Change the rotation step"
      />
    </>
  );

  const colorMenu = (
    <MenuAnchor
      popStyle={{ minWidth: 0 }}
      trigger={(open, toggle) => (
        <Small
          icon="palette"
          label="Color"
          caret
          expanded={open}
          disabled={noSel || !paint}
          onClick={toggle}
        />
      )}
    >
      {(close) => (
        <div className="f3-swatches">
          {SWATCHES.map((sw) => (
            <button
              key={sw.hex}
              type="button"
              className="f3-swatch"
              aria-label={sw.name}
              title={sw.name}
              aria-pressed={paint?.color?.toLowerCase() === sw.hex}
              style={{ background: sw.hex }}
              onClick={() => {
                paint?.apply({ color: sw.hex });
                close();
              }}
            />
          ))}
        </div>
      )}
    </MenuAnchor>
  );

  const materialMenu = (
    <MenuAnchor
      popStyle={{ minWidth: 160 }}
      trigger={(open, toggle) => (
        <Small
          icon="material"
          label="Material"
          caret
          expanded={open}
          disabled={noSel || !paint}
          onClick={toggle}
        />
      )}
    >
      {(close) =>
        MATERIAL_PRESETS.map((m) => (
          <MenuItem
            key={m.key}
            icon="material"
            label={m.label}
            onSelect={() => {
              paint?.apply({ preset: m.key });
              close();
            }}
          />
        ))
      }
    </MenuAnchor>
  );

  const testButtons = (
    <>
      <Big
        icon="play"
        label="Play"
        className="play"
        disabled={playing}
        onClick={play}
        title="Play-test the experience (F5)"
      />
      <Big
        icon="stop"
        label="Stop"
        className="stop"
        disabled={!playing}
        onClick={stop}
        title="Stop and return to editing (Shift+F5 or Esc)"
      />
    </>
  );

  let groups: ReactNode;
  switch (s.ribbonTab) {
    case 'model':
      groups = (
        <>
          <Group label="Transform">
            <Big
              icon="move"
              label="Move"
              pressed={tool === 'translate'}
              disabled={playing}
              onClick={() => s.setTransformMode('translate')}
              title="Move (G)"
            />
            <Big
              icon="rotate"
              label="Rotate"
              pressed={tool === 'rotate'}
              disabled={playing}
              onClick={() => s.setTransformMode('rotate')}
              title="Rotate (R)"
            />
            <Big
              icon="scale"
              label="Scale"
              pressed={tool === 'scale'}
              disabled={playing}
              onClick={() => s.setTransformMode('scale')}
              title="Scale (S)"
            />
          </Group>
          <Group
            label="Arrange"
            small={
              <>
                <Small
                  icon="drop"
                  label="Drop to ground"
                  disabled={noSel}
                  onClick={() => actions.dropToGround()}
                />
                <Small
                  icon="grid"
                  label="Snap to grid"
                  disabled={noSel}
                  onClick={() => actions.snapToGrid()}
                />
                <Small
                  icon="focus"
                  label="Focus"
                  disabled={noSel}
                  onClick={() => actions.focus()}
                  title="Focus the camera on the selection (F)"
                />
              </>
            }
          />
          <Group
            label="Space"
            small={
              <>
                <Small
                  icon="globe"
                  label="World"
                  pressed={s.space === 'world'}
                  check
                  onClick={() => s.setSpace('world')}
                  title="Gizmo follows world axes"
                />
                <Small
                  icon="local"
                  label="Local"
                  pressed={s.space === 'local'}
                  check
                  onClick={() => s.setSpace('local')}
                  title="Gizmo follows the object's own axes"
                />
              </>
            }
          />
          <Group label="Snap" small={snapButtons} />
          <Group
            label="Parts"
            small={PRIMITIVES.map((p) => (
              <Small
                key={p.type}
                icon={p.type}
                label={p.label}
                disabled={playing}
                onClick={() => actions.addPrimitive(p.type)}
              />
            ))}
          />
          <Group
            label="Lights"
            small={LIGHTS.map((l) => (
              <Small
                key={l.type}
                icon={l.icon}
                label={l.label}
                disabled={playing}
                onClick={() => actions.addLight(l.type)}
              />
            ))}
          />
        </>
      );
      break;

    case 'test':
      groups = (
        <>
          <Group label="Run">{testButtons}</Group>
          <Group label="Experience">
            <MenuAnchor
              popStyle={{ minWidth: 220 }}
              trigger={(open, toggle) => (
                <Big
                  icon="sliders"
                  label={FORMAT_LABEL[formatType]}
                  caret
                  expanded={open}
                  disabled={playing}
                  onClick={toggle}
                  title="What viewers get when they open this scene"
                />
              )}
            >
              {(close) =>
                FORMATS.map((f) => (
                  <MenuItem
                    key={f}
                    icon="sliders"
                    label={FORMAT_LABEL[f]}
                    checked={formatType === f}
                    onSelect={() => {
                      setFormatType(f);
                      close();
                    }}
                  />
                ))
              }
            </MenuAnchor>
            <Big
              icon="timeline"
              label="Timeline"
              pressed={s.dockTab === 'timeline' && s.panels.output}
              onClick={() => s.setDockTab('timeline')}
              title="Camera keyframes for 3D Video"
            />
          </Group>
        </>
      );
      break;

    case 'view': {
      const panel = (key: PanelKey, label: string, icon: string) => (
        <Big
          key={key}
          icon={icon}
          label={label}
          pressed={s.panels[key]}
          onClick={() => s.togglePanel(key)}
          title={`Show or hide ${label}`}
        />
      );
      const view = (v: CameraView, label: string) => (
        <Small
          key={v}
          icon="camera"
          label={label}
          pressed={s.cameraView === v}
          disabled={playing}
          onClick={() => actions.setView(v)}
        />
      );
      groups = (
        <>
          <Group label="Panels">
            {panel('library', 'Library', 'library')}
            {panel('explorer', 'Explorer', 'scene')}
            {panel('properties', 'Properties', 'sliders')}
            {panel('output', 'Output', 'terminal')}
          </Group>
          <Group
            label="Viewport"
            small={
              <Small icon="grid" label="Grid" pressed={s.showGrid} check onClick={s.toggleGrid} />
            }
          />
          <Group
            label="Camera"
            small={
              <>
                {view('perspective', 'Perspective')}
                {view('top', 'Top')}
                {view('front', 'Front')}
                {view('side', 'Side')}
                <Small
                  icon="reset"
                  label="Reset"
                  disabled={playing}
                  onClick={actions.resetCamera}
                  title="Reset the camera (double-click empty space)"
                />
                <Small
                  icon="focus"
                  label="Focus"
                  disabled={noSel}
                  onClick={() => actions.focus()}
                  title="Focus the selection (F)"
                />
              </>
            }
          />
        </>
      );
      break;
    }

    case 'home':
    default:
      groups = (
        <>
          <Group label="Tools">{toolButtons}</Group>
          <Group label="Insert">
            {partMenu}
            {lightMenu}
            <Big
              icon="import"
              label="Import"
              disabled={playing}
              onClick={onRequestImport}
              title="Import a .glb or .gltf model"
            />
            <Big
              icon="sparkle"
              label="AI model"
              disabled={playing}
              onClick={s.openAIPanel}
              title="Make a 3D model from a description or a photo"
            />
          </Group>
          <Group
            label="Edit"
            small={
              <>
                {colorMenu}
                {materialMenu}
                <Small
                  icon="focus"
                  label="Focus"
                  disabled={noSel}
                  onClick={() => actions.focus()}
                  title="Focus the camera on the selection (F)"
                />
                <Small
                  icon="duplicate"
                  label="Duplicate"
                  disabled={noSel || selection?.isModel}
                  onClick={() => actions.duplicate()}
                  title={
                    selection?.isModel
                      ? 'Imported models can’t be duplicated yet'
                      : 'Duplicate (Ctrl+D)'
                  }
                />
                <Small
                  icon="drop"
                  label="Drop to ground"
                  disabled={noSel}
                  onClick={() => actions.dropToGround()}
                />
                <Small
                  icon="trash"
                  label="Delete"
                  disabled={noSel}
                  onClick={() => actions.remove()}
                  title="Delete (Del)"
                />
              </>
            }
          />
          <Group label="Snap" small={snapButtons} />
          <Group label="Test">{testButtons}</Group>
        </>
      );
  }

  return (
    <div className="f3-ribbon" role="toolbar" aria-label={`${s.ribbonTab} tools`}>
      {groups}
    </div>
  );
}
