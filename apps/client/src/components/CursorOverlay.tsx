import { useMemo } from 'react';
import * as THREE from 'three';
import type { AwarenessUser } from '@/hooks/useCollaboration';

interface CursorOverlayProps {
  peers: AwarenessUser[];
  camera: THREE.Camera | null;
  canvasElement: HTMLCanvasElement | null;
}

/**
 * Renders peer cursor labels projected from 3D world → 2D screen coordinates.
 * Overlay sits on top of the 3D viewport using absolute positioning.
 */
export function CursorOverlay({ peers, camera, canvasElement }: CursorOverlayProps) {
  const cursorPositions = useMemo(() => {
    if (!camera || !canvasElement) return [];

    const rect = canvasElement.getBoundingClientRect();
    const result: { clientId: number; name: string; color: string; x: number; y: number }[] = [];

    for (const peer of peers) {
      if (!peer.cursor) continue;

      const worldPos = new THREE.Vector3(peer.cursor[0], peer.cursor[1], peer.cursor[2]);
      const screenPos = worldPos.project(camera);

      // Convert from NDC (-1..1) to pixel coords
      const x = (screenPos.x * 0.5 + 0.5) * rect.width;
      const y = (-screenPos.y * 0.5 + 0.5) * rect.height;

      // Skip if behind camera
      if (screenPos.z > 1) continue;

      // Skip if offscreen
      if (x < -20 || x > rect.width + 20 || y < -20 || y > rect.height + 20) continue;

      result.push({
        clientId: peer.clientId,
        name: peer.name,
        color: peer.color,
        x,
        y,
      });
    }

    return result;
  }, [peers, camera, canvasElement]);

  if (cursorPositions.length === 0) return null;

  return (
    <div
      style={{
        position: 'absolute',
        inset: 0,
        pointerEvents: 'none',
        overflow: 'hidden',
      }}
    >
      {cursorPositions.map((cursor) => (
        <div
          key={cursor.clientId}
          style={{
            position: 'absolute',
            left: cursor.x,
            top: cursor.y,
            transform: 'translate(-50%, -100%)',
          }}
        >
          {/* Cursor dot */}
          <div
            style={{
              width: 8,
              height: 8,
              borderRadius: '50%',
              background: cursor.color,
              border: '1.5px solid white',
              margin: '0 auto 2px',
            }}
          />
          {/* Name label */}
          <div
            style={{
              background: cursor.color,
              color: '#1e1e2e',
              fontSize: 10,
              fontWeight: 600,
              padding: '1px 6px',
              borderRadius: 4,
              whiteSpace: 'nowrap',
            }}
          >
            {cursor.name}
          </div>
        </div>
      ))}
    </div>
  );
}
