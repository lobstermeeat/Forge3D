import { useEffect, useRef, useState, useCallback } from 'react';
import type { RemoteControlMessage, RemoteControlRole, RemoteControlCameraSync } from '@forge3d/shared';

export interface RemoteControlState {
  connected: boolean;
  viewerCount: number;
  error: string | null;
  /** Send camera state to all viewers (controller only). */
  sendCameraSync: (position: [number, number, number], target: [number, number, number], fov: number) => void;
}

/**
 * React hook for remote control WebSocket connection.
 *
 * - Controller: sends camera-sync messages to broadcast view to all viewers.
 * - Viewer: receives camera-sync messages and invokes the onCameraSync callback.
 */
export function useRemoteControl(
  sessionId: string | undefined,
  role: RemoteControlRole,
  onCameraSync?: (msg: RemoteControlCameraSync) => void,
): RemoteControlState {
  const [connected, setConnected] = useState(false);
  const [viewerCount, setViewerCount] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const onCameraSyncRef = useRef(onCameraSync);
  onCameraSyncRef.current = onCameraSync;

  useEffect(() => {
    if (!sessionId) return;

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const url = `${protocol}//${window.location.host}/ws/remote/${sessionId}?role=${role}`;
    const ws = new WebSocket(url);
    wsRef.current = ws;

    ws.onopen = () => {
      setConnected(true);
      setError(null);
    };

    ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data as string) as RemoteControlMessage;
        switch (msg.type) {
          case 'session-info':
            setViewerCount(msg.viewerCount);
            break;
          case 'camera-sync':
            onCameraSyncRef.current?.(msg);
            break;
          case 'error':
            setError(msg.message);
            break;
        }
      } catch {
        // Ignore malformed messages
      }
    };

    ws.onclose = () => {
      setConnected(false);
    };

    ws.onerror = () => {
      setError('Connection failed');
      setConnected(false);
    };

    return () => {
      ws.close();
      wsRef.current = null;
      setConnected(false);
      setViewerCount(0);
      setError(null);
    };
  }, [sessionId, role]);

  const sendCameraSync = useCallback(
    (position: [number, number, number], target: [number, number, number], fov: number) => {
      const ws = wsRef.current;
      if (ws && ws.readyState === WebSocket.OPEN) {
        const msg: RemoteControlCameraSync = {
          type: 'camera-sync',
          position,
          target,
          fov,
        };
        ws.send(JSON.stringify(msg));
      }
    },
    [],
  );

  return { connected, viewerCount, error, sendCameraSync };
}
