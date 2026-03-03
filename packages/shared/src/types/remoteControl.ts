// ---------- Remote Control Types ----------

/**
 * Remote control allows a presenter to broadcast their camera view
 * to all connected viewers in real-time via WebSocket.
 */

export type RemoteControlRole = 'controller' | 'viewer';

export interface RemoteControlSession {
  sessionId: string;
  experienceSlug: string;
  createdAt: string;
}

// ---------- WebSocket Message Types ----------

export type RemoteControlMessage =
  | RemoteControlCameraSync
  | RemoteControlJoin
  | RemoteControlLeave
  | RemoteControlSessionInfo
  | RemoteControlError;

export interface RemoteControlCameraSync {
  type: 'camera-sync';
  position: [number, number, number];
  target: [number, number, number];
  fov: number;
}

export interface RemoteControlJoin {
  type: 'join';
  role: RemoteControlRole;
  sessionId: string;
}

export interface RemoteControlLeave {
  type: 'leave';
}

export interface RemoteControlSessionInfo {
  type: 'session-info';
  sessionId: string;
  viewerCount: number;
}

export interface RemoteControlError {
  type: 'error';
  message: string;
}
