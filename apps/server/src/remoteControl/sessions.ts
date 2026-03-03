import { randomUUID } from 'node:crypto';
import type { WebSocket } from 'ws';
import type { RemoteControlMessage, RemoteControlRole } from '@forge3d/shared';

interface RemoteClient {
  ws: WebSocket;
  role: RemoteControlRole;
}

interface Session {
  sessionId: string;
  experienceSlug: string;
  controller: WebSocket | null;
  viewers: Set<WebSocket>;
  createdAt: Date;
}

const sessions = new Map<string, Session>();

/** Create a new remote control session for an experience. */
export function createSession(experienceSlug: string): string {
  const sessionId = randomUUID().slice(0, 8);
  sessions.set(sessionId, {
    sessionId,
    experienceSlug,
    controller: null,
    viewers: new Set(),
    createdAt: new Date(),
  });
  return sessionId;
}

/** Get session info (or null if not found). */
export function getSession(sessionId: string) {
  const session = sessions.get(sessionId);
  if (!session) return null;
  return {
    sessionId: session.sessionId,
    experienceSlug: session.experienceSlug,
    viewerCount: session.viewers.size,
    createdAt: session.createdAt.toISOString(),
  };
}

/** Handle a new WebSocket connection to a remote control session. */
export function handleConnection(ws: WebSocket, sessionId: string, role: RemoteControlRole): void {
  const session = sessions.get(sessionId);
  if (!session) {
    sendMessage(ws, { type: 'error', message: 'Session not found' });
    ws.close();
    return;
  }

  if (role === 'controller') {
    if (session.controller !== null) {
      sendMessage(ws, { type: 'error', message: 'Session already has a controller' });
      ws.close();
      return;
    }
    session.controller = ws;
  } else {
    session.viewers.add(ws);
  }

  // Confirm join
  sendMessage(ws, {
    type: 'session-info',
    sessionId,
    viewerCount: session.viewers.size,
  });

  // Notify controller of updated viewer count
  if (role === 'viewer' && session.controller) {
    sendMessage(session.controller, {
      type: 'session-info',
      sessionId,
      viewerCount: session.viewers.size,
    });
  }

  ws.on('message', (data) => {
    try {
      const msg = JSON.parse(data.toString()) as RemoteControlMessage;
      handleMessage(session, ws, role, msg);
    } catch {
      // Ignore malformed messages
    }
  });

  ws.on('close', () => {
    handleDisconnect(session, ws, role);
  });
}

function handleMessage(
  session: Session,
  ws: WebSocket,
  role: RemoteControlRole,
  msg: RemoteControlMessage,
): void {
  // Only controllers can broadcast camera sync
  if (msg.type === 'camera-sync' && role === 'controller') {
    // Broadcast to all viewers
    const payload = JSON.stringify(msg);
    for (const viewer of session.viewers) {
      if (viewer.readyState === ws.OPEN) {
        viewer.send(payload);
      }
    }
  }
}

function handleDisconnect(session: Session, ws: WebSocket, role: RemoteControlRole): void {
  if (role === 'controller') {
    session.controller = null;
    // Notify all viewers that controller disconnected
    const payload = JSON.stringify({ type: 'error', message: 'Presenter disconnected' } satisfies RemoteControlMessage);
    for (const viewer of session.viewers) {
      if (viewer.readyState === ws.OPEN) {
        viewer.send(payload);
      }
    }
    // Clean up the session when controller leaves
    for (const viewer of session.viewers) {
      viewer.close();
    }
    sessions.delete(session.sessionId);
  } else {
    session.viewers.delete(ws);
    // Notify controller of updated viewer count
    if (session.controller && session.controller.readyState === ws.OPEN) {
      sendMessage(session.controller, {
        type: 'session-info',
        sessionId: session.sessionId,
        viewerCount: session.viewers.size,
      });
    }
  }
}

function sendMessage(ws: WebSocket, msg: RemoteControlMessage): void {
  if (ws.readyState === ws.OPEN) {
    ws.send(JSON.stringify(msg));
  }
}
