import { useEffect, useRef, useState, useCallback } from 'react';
import * as Y from 'yjs';
import { WebsocketProvider } from 'y-websocket';
import { YjsSceneBinding } from '@forge3d/engine';
import type { SceneManager } from '@forge3d/engine';

export interface AwarenessUser {
  clientId: number;
  name: string;
  color: string;
  avatarUrl: string | null;
  cursor: [number, number, number] | null;
  selection: string | null;
}

export interface CollaborationState {
  connected: boolean;
  peerCount: number;
  peers: AwarenessUser[];
  doc: Y.Doc | null;
  awareness: WebsocketProvider['awareness'] | null;
  updateCursor: (position: [number, number, number] | null) => void;
  updateSelection: (entityId: string | null) => void;
}

const COLLAB_COLORS = [
  '#f38ba8', '#fab387', '#f9e2af', '#a6e3a1',
  '#94e2d5', '#89b4fa', '#b4befe', '#cba6f7',
];

function pickColor(clientId: number): string {
  return COLLAB_COLORS[clientId % COLLAB_COLORS.length]!;
}

/**
 * React hook for real-time scene collaboration via Yjs.
 *
 * Returns null when sceneId is undefined (offline mode).
 * When sceneId is provided, connects to the Hocuspocus WebSocket server
 * and binds the Yjs doc to the SceneManager.
 */
export function useCollaboration(
  sceneId: string | undefined,
  sceneManager: SceneManager,
  ready: boolean,
): CollaborationState | null {
  const [connected, setConnected] = useState(false);
  const [peerCount, setPeerCount] = useState(0);
  const [peers, setPeers] = useState<AwarenessUser[]>([]);

  const docRef = useRef<Y.Doc | null>(null);
  const providerRef = useRef<WebsocketProvider | null>(null);
  const bindingRef = useRef<YjsSceneBinding | null>(null);

  // If no sceneId, return null (offline/unsaved mode)
  const isEnabled = !!sceneId && ready;

  useEffect(() => {
    if (!isEnabled || !sceneId) return;

    const doc = new Y.Doc();
    docRef.current = doc;

    // Connect to Hocuspocus via y-websocket provider
    const wsUrl = `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}/ws/collab`;
    const provider = new WebsocketProvider(wsUrl, sceneId, doc, {
      connect: true,
    });
    providerRef.current = provider;

    // Set local awareness state
    const userName = 'User'; // TODO: get from auth context
    provider.awareness.setLocalStateField('user', {
      name: userName,
      color: pickColor(doc.clientID),
      avatarUrl: null,
    });

    // Track connection status
    provider.on('status', ({ status }: { status: string }) => {
      setConnected(status === 'connected');
    });

    // Track peers via awareness
    const updatePeers = () => {
      const states = provider.awareness.getStates();
      const peerList: AwarenessUser[] = [];
      states.forEach((state, clientId) => {
        if (clientId === doc.clientID) return; // skip self
        const user = state['user'] as { name?: string; color?: string; avatarUrl?: string | null } | undefined;
        peerList.push({
          clientId,
          name: user?.name ?? 'Anonymous',
          color: user?.color ?? pickColor(clientId),
          avatarUrl: user?.avatarUrl ?? null,
          cursor: (state['cursor'] as [number, number, number] | null) ?? null,
          selection: (state['selection'] as string | null) ?? null,
        });
      });
      setPeers(peerList);
      setPeerCount(peerList.length);
    };

    provider.awareness.on('change', updatePeers);
    updatePeers();

    // Bind Yjs doc to SceneManager once synced
    provider.on('sync', (isSynced: boolean) => {
      if (isSynced && !bindingRef.current) {
        const binding = new YjsSceneBinding(doc, sceneManager);
        binding.bind();
        bindingRef.current = binding;
      }
    });

    return () => {
      bindingRef.current?.unbind();
      bindingRef.current = null;
      provider.awareness.off('change', updatePeers);
      provider.disconnect();
      provider.destroy();
      doc.destroy();
      docRef.current = null;
      providerRef.current = null;
      setConnected(false);
      setPeerCount(0);
      setPeers([]);
    };
  }, [isEnabled, sceneId, sceneManager]);

  const updateCursor = useCallback((position: [number, number, number] | null) => {
    providerRef.current?.awareness.setLocalStateField('cursor', position);
  }, []);

  const updateSelection = useCallback((entityId: string | null) => {
    providerRef.current?.awareness.setLocalStateField('selection', entityId);
  }, []);

  if (!isEnabled) return null;

  return {
    connected,
    peerCount,
    peers,
    doc: docRef.current,
    awareness: providerRef.current?.awareness ?? null,
    updateCursor,
    updateSelection,
  };
}
