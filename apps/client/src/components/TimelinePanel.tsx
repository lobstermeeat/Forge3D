import { useRef, useCallback, useEffect } from 'react';
import * as THREE from 'three';
import { ease } from '@forge3d/engine';
import type { ViewportControls } from '@forge3d/engine';
import type { CameraKeyframe } from '@forge3d/shared';
import { useFormatStore } from '@/stores/formatStore';
import { Icon } from '@/editor/Icon';

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
    <div style={{ padding: '8px 12px 10px', display: 'flex', flexDirection: 'column', gap: 8 }}>
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
        <span style={{ fontSize: 12, fontWeight: 600, color: 'var(--tx1)' }}>Camera path</span>
        <span className="f3-mono" style={{ fontSize: 11, color: 'var(--tx3)' }}>
          {keyframes.length} keyframe{keyframes.length !== 1 ? 's' : ''} · {formatTime(totalDuration)}
        </span>
        <div style={{ marginLeft: 'auto', display: 'flex', gap: 6 }}>
          <button type="button" className="f3-btn ghost" style={{ height: 26, fontSize: 12 }} onClick={handleRecordKeyframe} title="Record the current camera as a keyframe">
            <Icon name="keyadd" size={13} />
            Key camera here
          </button>
          <button
            type="button"
            className="f3-btn ghost"
            style={{ height: 26, fontSize: 12 }}
            onClick={handlePreviewToggle}
            disabled={keyframes.length < 2}
            title={keyframes.length < 2 ? 'Add at least two keyframes to preview' : 'Fly the camera along the path'}
          >
            <Icon name={previewPlaying ? 'stop' : 'play'} size={11} />
            {previewPlaying ? 'Stop' : 'Preview'}
          </button>
        </div>
      </div>

      {/* Track */}
      <div
        style={{
          position: 'relative',
          height: 30,
          background: 'var(--bg3)',
          borderRadius: 6,
          margin: '0 6px',
        }}
      >
        {keyframes.length === 0 && (
          <span style={{ position: 'absolute', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 11.5, color: 'var(--tx3)' }}>
            Frame a shot in the viewport, then press “Key camera here”.
          </span>
        )}
        {keyframes.map((kf, i) => {
          const left = totalDuration > 0 ? (kf.time / totalDuration) * 100 : 0;
          const isSelected = selectedKeyframeIndex === i;
          return (
            <button
              type="button"
              key={i}
              onClick={() => selectKeyframe(isSelected ? null : i)}
              aria-label={`Keyframe at ${kf.time.toFixed(1)} seconds`}
              aria-pressed={isSelected}
              style={{
                position: 'absolute',
                left: `${left}%`,
                top: '50%',
                transform: 'translate(-50%, -50%) rotate(45deg)',
                width: 11,
                height: 11,
                background: isSelected ? 'var(--acc)' : 'var(--bg1)',
                boxShadow: 'inset 0 0 0 1.5px var(--acc)',
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
          <label style={detailLabelStyle}>
            Time
            <input
              type="number"
              className="f3-input f3-mono"
              value={selectedKf.time}
              step={0.1}
              min={0}
              onChange={(e) => updateKeyframe(selectedKeyframeIndex, { time: parseFloat(e.target.value) || 0 })}
              style={{ width: 70 }}
            />
          </label>
          <label style={detailLabelStyle}>
            Easing
            <select
              className="f3-input"
              value={selectedKf.easing}
              onChange={(e) =>
                updateKeyframe(selectedKeyframeIndex, {
                  easing: e.target.value as CameraKeyframe['easing'],
                })
              }
              style={{ width: 120 }}
            >
              {EASING_OPTIONS.map((e) => (
                <option key={e} value={e}>
                  {e}
                </option>
              ))}
            </select>
          </label>
          <button
            type="button"
            className="f3-btn ghost"
            style={{ height: 26, fontSize: 12 }}
            onClick={() => {
              if (camera && selectedKf) {
                camera.position.set(selectedKf.position[0], selectedKf.position[1], selectedKf.position[2]);
                const target = orbitControls?.current?.target;
                if (target) {
                  target.set(selectedKf.target[0], selectedKf.target[1], selectedKf.target[2]);
                }
              }
            }}
            title="Move the camera to this keyframe"
          >
            Go to
          </button>
          <button type="button" className="f3-btn danger" style={{ height: 26, fontSize: 12 }} onClick={() => removeKeyframe(selectedKeyframeIndex)}>
            Delete
          </button>
        </div>
      )}
    </div>
  );
}

const detailLabelStyle: React.CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  gap: 6,
  fontSize: 12,
  color: 'var(--tx2)',
};
