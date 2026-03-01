import type { AwarenessUser } from '@/hooks/useCollaboration';

interface PresencePanelProps {
  connected: boolean;
  peers: AwarenessUser[];
}

/**
 * Shows connected collaborators in the left sidebar.
 * Displays a colored dot + name for each peer.
 */
export function PresencePanel({ connected, peers }: PresencePanelProps) {
  return (
    <div
      style={{
        padding: '8px 12px',
        borderBottom: '1px solid #313244',
        background: '#181825',
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 4 }}>
        <div
          style={{
            width: 8,
            height: 8,
            borderRadius: '50%',
            background: connected ? '#a6e3a1' : '#f38ba8',
          }}
        />
        <span style={{ fontSize: 11, color: '#a6adc8', fontWeight: 600 }}>
          {connected ? `${peers.length + 1} online` : 'Offline'}
        </span>
      </div>

      {peers.length > 0 && (
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>
          {peers.map((peer) => (
            <div
              key={peer.clientId}
              title={peer.name}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: 4,
                padding: '2px 8px',
                borderRadius: 12,
                background: '#313244',
                fontSize: 11,
                color: '#cdd6f4',
              }}
            >
              <div
                style={{
                  width: 6,
                  height: 6,
                  borderRadius: '50%',
                  background: peer.color,
                  flexShrink: 0,
                }}
              />
              <span style={{ maxWidth: 80, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                {peer.name}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
