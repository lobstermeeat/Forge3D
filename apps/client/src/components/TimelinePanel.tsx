import { useState, useRef, useCallback, useEffect } from 'react';
import * as THREE from 'three';
import { ease } from '@forge3d/engine';
import type { ViewportControls } from '@forge3d/engine';
import type { CameraKeyframe } from '@forge3d/shared';
import { useFormatStore } from '@/stores/formatStore';

interface TimelinePanelProps {
  camera: THREE.PerspectiveCamera | null;
  orbitControls: React.RefObject<ViewportControls | null> | null;
}

const EASING_OPTIONS = ['linear', 'ease-in', 'ease-out', 'ease-in-out'] as const;

function formatTime(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = (seconds % 60).toFixed(1);
  return `${m}:${s.padStart(4, '0')}`;
}

export function TimelinePanel({ camera, orbitControls }: TimelinePanelProps) {
  const {
    keyframes,
    selectedKeyframeIndex,
    previewPlaying,
    addKeyframe,
    updateKeyframe,
    removeKeyframe,
    selectKeyframe,
    setPreviewPlaying,
  } = useFormatStore();

  const [collapsed, setCollapsed] = useState(false);
  const previewRef = useRef<{ animId: number; savedPos: THREE.Vector3; savedTarget: THREE.Vector3 } | null>(null);

  const totalDuration = keyframes.length > 0 ? keyframes[keyframes.length - 1]!.time : 0;

  const handleRecordKeyframe = useCallback(() => {
    if (!camera) return;

    const target = orbitControls?.current?.target ?? new THREE.Vector3();

    const time = totalDuration + 1; // Default: 1 second after last keyframe
    const kf: CameraKeyframe = {
      time,
      position: [camera.position.x, camera.position.y, camera.position.z],
      target: [target.x, target.y, target.z],
      easing: 'ease-in-out',
    };
    addKeyframe(kf);
  }, [camera, orbitControls, totalDuration, addKeyframe]);

  const handlePreviewToggle = useCallback(() => {
    if (previewPlaying) {
      // Stop preview, restore camera
      if (previewRef.current) {
        cancelAnimationFrame(previewRef.current.animId);
        if (camera) {
          camera.position.copy(previewRef.current.savedPos);
          const target = orbitControls?.current?.target;
          if (target) target.copy(previewRef.current.savedTarget);
        }
        previewRef.current = null;
      }
      setPreviewPlaying(false);
      return;
    }

    if (!camera || keyframes.length < 2) return;

    // Save camera state
    const target = orbitControls?.current?.target;
    previewRef.current = {
      animId: 0,
      savedPos: camera.position.clone(),
      savedTarget: target ? target.clone() : new THREE.Vector3(),
    };

    setPreviewPlaying(true);

    let currentTime = 0;
    const duration = keyframes[keyframes.length - 1]!.time;
    let lastTimestamp = performance.now();

    const posA = new THREE.Vector3();
    const posB = new THREE.Vector3();
    const tgtA = new THREE.Vector3();
    const tgtB = new THREE.Vector3();

    function tick() {
      const now = performance.now();
      const dt = (now - lastTimestamp) / 1000;
      lastTimestamp = now;
      currentTime += dt;

      if (currentTime >= duration) {
        // Done — restore camera
        if (previewRef.current && camera) {
          camera.position.copy(previewRef.current.savedPos);
          if (target) target.copy(previewRef.current.savedTarget);
          previewRef.current = null;
        }
        setPreviewPlaying(false);
        return;
      }

      // Interpolate camera
      let idxB = keyframes.findIndex((kf) => kf.time >= currentTime);
      if (idxB === -1) idxB = keyframes.length - 1;
      const idxA = Math.max(0, idxB - 1);

      const kfA = keyframes[idxA]!;
      const kfB = keyframes[idxB]!;

      if (kfA.time !== kfB.time) {
        const localT = (currentTime - kfA.time) / (kfB.time - kfA.time);
        const easedT = ease(kfB.easing, localT);

        posA.set(kfA.position[0], kfA.position[1], kfA.position[2]);
        posB.set(kfB.position[0], kfB.position[1], kfB.position[2]);
        tgtA.set(kfA.target[0], kfA.target[1], kfA.target[2]);
        tgtB.set(kfB.target[0], kfB.target[1], kfB.target[2]);

        camera!.position.lerpVectors(posA, posB, easedT);
        if (target) target.lerpVectors(tgtA, tgtB, easedT);
      }

      previewRef.current!.animId = requestAnimationFrame(tick);
    }

    previewRef.current.animId = requestAnimationFrame(tick);
  }, [camera, orbitControls, keyframes, previewPlaying, setPreviewPlaying]);

  // Cleanup preview on unmount
  useEffect(() => {
    return () => {
      if (previewRef.current) {
        cancelAnimationFrame(previewRef.current.animId);
      }
    };
  }, []);

  const selectedKf = selectedKeyframeIndex !== null ? keyframes[selectedKeyframeIndex] : null;

  return (
    <div style={{ background: '#181825', borderTop: '1px solid #313244' }}>
      {/* Header */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          padding: '6px 12px',
          cursor: 'pointer',
        }}
        onClick={() => setCollapsed(!collapsed)}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <span style={{ fontSize: 11, fontWeight: 600, color: '#cdd6f4' }}>Timeline</span>
          <span style={{ fontSize: 10, color: '#6c7086' }}>
            {keyframes.length} keyframe{keyframes.length !== 1 ? 's' : ''} · {formatTime(totalDuration)}
          </span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          {!collapsed && (
            <>
              <button
                onClick={(e) => { e.stopPropagation(); handleRecordKeyframe(); }}
                style={btnStyle}
                title="Record camera position as keyframe"
              >
                + Keyframe
              </button>
              <button
                onClick={(e) => { e.stopPropagation(); handlePreviewToggle(); }}
                disabled={keyframes.length < 2}
                style={{
                  ...btnStyle,
                  opacity: keyframes.length < 2 ? 0.4 : 1,
                  cursor: keyframes.length < 2 ? 'not-allowed' : 'pointer',
                }}
              >
                {previewPlaying ? 'Stop' : 'Preview'}
              </button>
            </>
          )}
          <span style={{ fontSize: 10, color: '#6c7086' }}>{collapsed ? '▲' : '▼'}</span>
        </div>
      </div>

      {/* Body */}
      {!collapsed && (
        <div style={{ padding: '0 12px 8px' }}>
          {/* Track */}
          <div
            style={{
              position: 'relative',
              height: 32,
              background: '#313244',
              borderRadius: 4,
              marginBottom: 8,
            }}
          >
            {totalDuration > 0 &&
              keyframes.map((kf, i) => {
                const left = (kf.time / totalDuration) * 100;
                const isSelected = selectedKeyframeIndex === i;
                return (
                  <div
                    key={i}
                    onClick={() => selectKeyframe(isSelected ? null : i)}
                    style={{
                      position: 'absolute',
                      left: `${left}%`,
                      top: '50%',
                      transform: 'translate(-50%, -50%) rotate(45deg)',
                      width: 10,
                      height: 10,
                      background: isSelected ? '#f38ba8' : '#89b4fa',
                      cursor: 'pointer',
                      borderRadius: 2,
                    }}
                    title={`${kf.time.toFixed(1)}s`}
                  />
                );
              })}
          </div>

          {/* Selected keyframe details */}
          {selectedKf && selectedKeyframeIndex !== null && (
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
              <label style={detailLabelStyle}>Time:</label>
              <input
                type="number"
                value={selectedKf.time}
                step={0.1}
                min={0}
                onChange={(e) =>
                  updateKeyframe(selectedKeyframeIndex, { time: parseFloat(e.target.value) || 0 })
                }
                style={detailInputStyle}
              />
              <label style={detailLabelStyle}>Easing:</label>
              <select
                value={selectedKf.easing}
                onChange={(e) =>
                  updateKeyframe(selectedKeyframeIndex, {
                    easing: e.target.value as CameraKeyframe['easing'],
                  })
                }
                style={{ ...detailInputStyle, width: 100 }}
              >
                {EASING_OPTIONS.map((e) => (
                  <option key={e} value={e}>
                    {e}
                  </option>
                ))}
              </select>
              <button
                onClick={() => {
                  if (camera && selectedKf) {
                    camera.position.set(
                      selectedKf.position[0],
                      selectedKf.position[1],
                      selectedKf.position[2],
                    );
                    const target = orbitControls?.current?.target;
                    if (target) {
                      target.set(selectedKf.target[0], selectedKf.target[1], selectedKf.target[2]);
                    }
                  }
                }}
                style={btnStyle}
                title="Move camera to this keyframe"
              >
                Go To
              </button>
              <button
                onClick={() => removeKeyframe(selectedKeyframeIndex)}
                style={{ ...btnStyle, color: '#f38ba8' }}
              >
                Delete
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

const btnStyle: React.CSSProperties = {
  padding: '3px 8px',
  borderRadius: 4,
  border: '1px solid #313244',
  background: '#1e1e2e',
  color: '#cdd6f4',
  cursor: 'pointer',
  fontSize: 10,
  fontWeight: 500,
};

const detailLabelStyle: React.CSSProperties = {
  fontSize: 10,
  color: '#a6adc8',
};

const detailInputStyle: React.CSSProperties = {
  width: 60,
  padding: '2px 6px',
  borderRadius: 4,
  border: '1px solid #313244',
  background: '#11111b',
  color: '#cdd6f4',
  fontSize: 10,
};
