import { useEffect, useRef } from 'react';
import * as THREE from 'three';
import { Icon } from '@/editor/Icon';
import { MenuAnchor, MenuItem } from '@/editor/ui';
import { useEditorStore, type CameraView } from '@/stores/editorStore';
import type { EditorActions } from '@/hooks/useEditorActions';
import type { FrameCallback } from '@/hooks/useEngine';
import type { InfoCard } from '@/hooks/usePlayMode';

const VIEWS: { key: CameraView; label: string }[] = [
  { key: 'perspective', label: 'Perspective' },
  { key: 'top', label: 'Top' },
  { key: 'front', label: 'Front' },
  { key: 'side', label: 'Side' },
];

const AXES = [
  { key: 'x', dir: new THREE.Vector3(1, 0, 0), color: 'var(--ax-x)', view: 'side' as CameraView },
  { key: 'y', dir: new THREE.Vector3(0, 1, 0), color: 'var(--ax-y)', view: 'top' as CameraView },
  { key: 'z', dir: new THREE.Vector3(0, 0, 1), color: 'var(--ax-z)', view: 'front' as CameraView },
];

/** Small orientation widget; its DOM is updated directly each frame so React doesn't re-render. */
function AxisIndicator({
  getCamera,
  addFrameCallback,
  onPick,
}: {
  getCamera: () => THREE.Camera | null;
  addFrameCallback: (cb: FrameCallback) => () => void;
  onPick: (view: CameraView) => void;
}) {
  const groupRefs = useRef<(SVGGElement | null)[]>([]);
  const lineRefs = useRef<(SVGLineElement | null)[]>([]);

  useEffect(() => {
    const inv = new THREE.Quaternion();
    const v = new THREE.Vector3();
    return addFrameCallback(() => {
      const camera = getCamera();
      if (!camera) return;
      inv.copy(camera.quaternion).invert();
      AXES.forEach((axis, i) => {
        v.copy(axis.dir).applyQuaternion(inv);
        const x = 32 + v.x * 21;
        const y = 32 - v.y * 21;
        lineRefs.current[i]?.setAttribute('x2', x.toFixed(1));
        lineRefs.current[i]?.setAttribute('y2', y.toFixed(1));
        const g = groupRefs.current[i];
        if (g) {
          g.setAttribute('transform', `translate(${x.toFixed(1)} ${y.toFixed(1)})`);
          g.style.opacity = v.z < -0.2 ? '0.55' : '1';
        }
      });
    });
  }, [addFrameCallback, getCamera]);

  return (
    <svg
      width="64"
      height="64"
      viewBox="0 0 64 64"
      role="group"
      aria-label="View axes — click one to look along it"
    >
      <circle cx="32" cy="32" r="30" fill="rgba(14,15,18,0.5)" />
      {AXES.map((axis, i) => (
        <line
          key={`l${axis.key}`}
          ref={(el) => {
            lineRefs.current[i] = el;
          }}
          x1="32"
          y1="32"
          x2="32"
          y2="32"
          stroke={axis.color}
          strokeWidth="2.2"
          strokeLinecap="round"
        />
      ))}
      {AXES.map((axis, i) => (
        <g
          key={axis.key}
          ref={(el) => {
            groupRefs.current[i] = el;
          }}
          role="button"
          tabIndex={0}
          aria-label={`Look along ${axis.key.toUpperCase()} (${axis.view} view)`}
          style={{ cursor: 'pointer' }}
          onClick={() => onPick(axis.view)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' || e.key === ' ') onPick(axis.view);
          }}
        >
          <circle r="8" fill={axis.color} />
          <text
            textAnchor="middle"
            dy="3.6"
            fontSize="10"
            fontWeight="700"
            fill="#101114"
            style={{ pointerEvents: 'none' }}
          >
            {axis.key.toUpperCase()}
          </text>
        </g>
      ))}
    </svg>
  );
}

interface ViewportOverlayProps {
  actions: EditorActions;
  getCamera: () => THREE.Camera | null;
  addFrameCallback: (cb: FrameCallback) => () => void;
  formatLabel: string;
  stop: () => void;
  info: InfoCard | null;
  closeInfo: () => void;
  empty: boolean;
  dropActive: boolean;
}

export function ViewportOverlay({
  actions,
  getCamera,
  addFrameCallback,
  formatLabel,
  stop,
  info,
  closeInfo,
  empty,
  dropActive,
}: ViewportOverlayProps) {
  const playing = useEditorStore((s) => s.isPlaying);
  const cameraView = useEditorStore((s) => s.cameraView);
  const showGrid = useEditorStore((s) => s.showGrid);
  const toggleGrid = useEditorStore((s) => s.toggleGrid);

  return (
    <>
      {!playing && (
        <div
          style={{
            position: 'absolute',
            left: 12,
            top: 12,
            zIndex: 6,
            display: 'flex',
            flexDirection: 'column',
            gap: 8,
            alignItems: 'flex-start',
          }}
        >
          <div style={{ display: 'flex', gap: 6 }}>
            <MenuAnchor
              popStyle={{ top: 30, minWidth: 190 }}
              trigger={(open, toggle) => (
                <button
                  type="button"
                  className="f3-vchip"
                  aria-haspopup="menu"
                  aria-expanded={open}
                  onClick={toggle}
                >
                  <Icon name="camera" size={13} />
                  {VIEWS.find((v) => v.key === cameraView)?.label ?? 'Perspective'}
                  <Icon name="chevdown" size={11} />
                </button>
              )}
            >
              {(close) => (
                <>
                  {VIEWS.map((v) => (
                    <MenuItem
                      key={v.key}
                      icon="camera"
                      label={v.label}
                      checked={cameraView === v.key}
                      onSelect={() => {
                        actions.setView(v.key);
                        close();
                      }}
                    />
                  ))}
                  <div className="f3-pop-sep" />
                  <MenuItem
                    icon="focus"
                    label="Focus selection"
                    shortcut="F"
                    onSelect={() => {
                      actions.focus();
                      close();
                    }}
                  />
                  <MenuItem
                    icon="reset"
                    label="Reset camera"
                    onSelect={() => {
                      actions.resetCamera();
                      close();
                    }}
                  />
                </>
              )}
            </MenuAnchor>
            <button
              type="button"
              className="f3-vchip"
              aria-pressed={showGrid}
              onClick={toggleGrid}
              title="Show or hide the grid"
            >
              <Icon name="grid" size={13} />
              Grid {showGrid ? 'on' : 'off'}
            </button>
          </div>
          <span className="f3-hint">
            Right-drag orbit · Middle-drag pan · Scroll zoom · F focus
          </span>
        </div>
      )}

      {!playing && (
        <div style={{ position: 'absolute', right: 10, top: 10, zIndex: 6 }}>
          <AxisIndicator
            getCamera={getCamera}
            addFrameCallback={addFrameCallback}
            onPick={(v) => actions.setView(v)}
          />
        </div>
      )}

      {empty && !playing && (
        <div
          style={{
            position: 'absolute',
            inset: 0,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            pointerEvents: 'none',
            zIndex: 4,
          }}
        >
          <div
            className="f3-glass"
            style={{
              padding: '14px 18px',
              borderRadius: 12,
              display: 'flex',
              alignItems: 'center',
              gap: 12,
              fontSize: 12.5,
              color: 'var(--tx2)',
              maxWidth: 380,
            }}
          >
            <Icon name="box" size={20} style={{ color: 'var(--acc)' }} />
            <span>
              Your scene is empty. Drag a part in from the Library, or use Part on the Home tab.
            </span>
          </div>
        </div>
      )}

      {dropActive && <div className="f3-dropzone" />}

      {playing && (
        <>
          <div className="f3-playing-frame" />
          <div
            className="f3-glass"
            role="status"
            style={{
              position: 'absolute',
              top: 12,
              left: '50%',
              transform: 'translateX(-50%)',
              zIndex: 8,
              display: 'flex',
              alignItems: 'center',
              gap: 10,
              height: 36,
              padding: '0 5px 0 12px',
              borderRadius: 10,
              fontSize: 12.5,
              whiteSpace: 'nowrap',
            }}
          >
            <span
              className="f3-dot f3-pulse"
              style={{ width: 8, height: 8, background: 'var(--ok)' }}
            />
            <span style={{ fontWeight: 600 }}>Playing</span>
            <span style={{ color: 'var(--tx2)' }}>{formatLabel} · what viewers see</span>
            <button
              type="button"
              className="f3-btn ghost"
              style={{ height: 26, padding: '0 10px' }}
              onClick={stop}
            >
              <Icon name="stop" size={11} style={{ color: 'var(--err)' }} />
              Stop
            </button>
          </div>
        </>
      )}

      {playing && info && (
        <div
          className="f3-glass"
          role="dialog"
          aria-label={info.title}
          style={{
            position: 'absolute',
            top: 60,
            left: '50%',
            transform: 'translateX(-50%)',
            zIndex: 9,
            width: 320,
            padding: '14px 16px',
            borderRadius: 12,
          }}
        >
          <div style={{ display: 'flex', alignItems: 'flex-start', gap: 8 }}>
            <strong style={{ fontSize: 13.5, flexGrow: 1 }}>{info.title}</strong>
            <button
              type="button"
              className="f3-ibtn"
              style={{ width: 24, height: 24 }}
              aria-label="Close"
              onClick={closeInfo}
            >
              <Icon name="close" size={14} />
            </button>
          </div>
          {info.description && (
            <p
              style={{ margin: '6px 0 0', fontSize: 12.5, lineHeight: '18px', color: 'var(--tx2)' }}
            >
              {info.description}
            </p>
          )}
        </div>
      )}
    </>
  );
}
