import { useRef } from 'react';
import { Link } from 'react-router-dom';
import { Icon } from '@/editor/Icon';
import { MenuAnchor, MenuItem } from '@/editor/ui';
import { useEditorStore, type RibbonTab } from '@/stores/editorStore';
import type { EditorActions } from '@/hooks/useEditorActions';
import type { SaveStatus } from '@/hooks/useAutoSave';
import type { AwarenessUser } from '@/hooks/useCollaboration';

const TABS: { key: RibbonTab; label: string }[] = [
  { key: 'home', label: 'Home' },
  { key: 'model', label: 'Model' },
  { key: 'test', label: 'Test' },
  { key: 'view', label: 'View' },
];

const SAVE_TEXT: Record<SaveStatus, string> = {
  idle: 'Edited',
  saving: 'Saving…',
  saved: 'Saved',
  error: 'Save failed',
};

const SAVE_DOT: Record<SaveStatus, string> = {
  idle: 'var(--tx3)',
  saving: 'var(--warn)',
  saved: 'var(--ok)',
  error: 'var(--err)',
};

function initials(name: string): string {
  const parts = name.trim().split(/\s+/);
  return ((parts[0]?.[0] ?? '?') + (parts[1]?.[0] ?? '')).toUpperCase();
}

interface TopBarProps {
  title: string;
  formatLabel: string;
  saveStatus?: SaveStatus;
  canUndo: boolean;
  canRedo: boolean;
  undoLabel?: string;
  redoLabel?: string;
  canPublish: boolean;
  onPublish: () => void;
  onRequestImport: () => void;
  onAbout: () => void;
  actions: EditorActions;
  collab: { connected: boolean; peers: AwarenessUser[] } | null;
}

export function TopBar({
  title,
  formatLabel,
  saveStatus,
  canUndo,
  canRedo,
  undoLabel,
  redoLabel,
  canPublish,
  onPublish,
  onRequestImport,
  onAbout,
  actions,
  collab,
}: TopBarProps) {
  const ribbonTab = useEditorStore((s) => s.ribbonTab);
  const setRibbonTab = useEditorStore((s) => s.setRibbonTab);
  const isPlaying = useEditorStore((s) => s.isPlaying);
  const sceneInput = useRef<HTMLInputElement>(null);

  return (
    <header className="f3-topbar">
      <Link to="/dashboard" className="f3-brand" title="Back to your projects">
        <svg width="22" height="22" viewBox="0 0 24 24" aria-hidden="true">
          <path d="M12 2.5l8.5 4.8L12 12 3.5 7.3z" style={{ fill: 'var(--acc)' }} />
          <path d="M3.5 7.3L12 12v9.5l-8.5-4.8z" style={{ fill: '#d9dce1' }} />
          <path d="M20.5 7.3L12 12v9.5l8.5-4.8z" style={{ fill: '#767d8a' }} />
        </svg>
        <b>Orainge</b>
        <span>Studio</span>
      </Link>
      <div className="f3-vsep" />

      <MenuAnchor
        popStyle={{ top: 42, minWidth: 250 }}
        trigger={(open, toggle) => (
          <button
            type="button"
            className="f3-tab"
            aria-haspopup="menu"
            aria-expanded={open}
            onClick={toggle}
            style={{ marginLeft: 4 }}
          >
            File
          </button>
        )}
      >
        {(close) => (
          <>
            <MenuItem
              icon="import"
              label="Import model (.glb, .gltf)…"
              onSelect={() => {
                close();
                onRequestImport();
              }}
            />
            <MenuItem
              icon="open"
              label="Open scene file…"
              onSelect={() => {
                close();
                sceneInput.current?.click();
              }}
            />
            <MenuItem
              icon="save"
              label="Download scene file"
              onSelect={() => {
                close();
                actions.saveToFile();
              }}
            />
            <div className="f3-pop-sep" />
            <MenuItem
              icon="publish"
              label="Publish…"
              disabled={!canPublish}
              onSelect={() => {
                close();
                onPublish();
              }}
            />
            {!canPublish && (
              <div className="f3-pop-note">Open a scene from your dashboard to publish it.</div>
            )}
            <div className="f3-pop-sep" />
            <MenuItem
              icon="info"
              label="About Orainge Studio"
              onSelect={() => {
                close();
                onAbout();
              }}
            />
          </>
        )}
      </MenuAnchor>

      <nav aria-label="Ribbon tabs" style={{ display: 'flex', alignItems: 'center' }}>
        {TABS.map((t) => (
          <button
            key={t.key}
            type="button"
            className="f3-tab"
            aria-pressed={ribbonTab === t.key}
            onClick={() => setRibbonTab(t.key)}
          >
            {t.label}
          </button>
        ))}
      </nav>

      <div className="f3-title">
        <strong>{title}</strong>
        <span className="f3-badge">{formatLabel}</span>
        {saveStatus && saveStatus !== 'idle' && (
          <span className="f3-save" role="status">
            <span className="f3-dot" style={{ background: SAVE_DOT[saveStatus] }} />
            {SAVE_TEXT[saveStatus]}
          </span>
        )}
      </div>

      <div
        style={{
          marginLeft: 'auto',
          display: 'flex',
          alignItems: 'center',
          gap: 4,
          paddingRight: 12,
        }}
      >
        <button
          type="button"
          className="f3-ibtn"
          onClick={actions.undo}
          disabled={!canUndo || isPlaying}
          aria-label="Undo"
          title={undoLabel ? `Undo ${undoLabel} (Ctrl+Z)` : 'Undo (Ctrl+Z)'}
        >
          <Icon name="undo" size={17} />
        </button>
        <button
          type="button"
          className="f3-ibtn"
          onClick={actions.redo}
          disabled={!canRedo || isPlaying}
          aria-label="Redo"
          title={redoLabel ? `Redo ${redoLabel} (Ctrl+Shift+Z)` : 'Redo (Ctrl+Shift+Z)'}
        >
          <Icon name="redo" size={17} />
        </button>

        {collab && (
          <>
            <div className="f3-vsep" style={{ margin: '0 8px' }} />
            <div
              title={
                collab.connected
                  ? `${collab.peers.length + 1} editing`
                  : 'Offline — changes sync when you reconnect'
              }
              style={{ display: 'flex', alignItems: 'center', gap: 6, marginRight: 4 }}
            >
              <span
                className="f3-dot"
                style={{ background: collab.connected ? 'var(--ok)' : 'var(--err)' }}
              />
              <div style={{ display: 'flex' }}>
                {collab.peers.slice(0, 4).map((peer, i) => (
                  <span
                    key={peer.clientId}
                    className="f3-avatar"
                    title={peer.name}
                    style={{ background: peer.color, marginLeft: i === 0 ? 0 : -6 }}
                  >
                    {initials(peer.name)}
                  </span>
                ))}
                {collab.peers.length > 4 && (
                  <span
                    className="f3-avatar"
                    style={{ background: 'var(--bg4)', color: 'var(--tx1)', marginLeft: -6 }}
                  >
                    +{collab.peers.length - 4}
                  </span>
                )}
              </div>
            </div>
          </>
        )}

        <button
          type="button"
          className="f3-btn pri"
          onClick={onPublish}
          disabled={!canPublish || isPlaying}
          title={
            canPublish
              ? 'Publish this scene to Orainge'
              : 'Open a scene from your dashboard to publish it'
          }
          style={{ marginLeft: 8 }}
        >
          Publish
        </button>
      </div>

      <input
        ref={sceneInput}
        type="file"
        accept=".json"
        aria-label="Open scene file"
        style={{ display: 'none' }}
        onChange={(e) => {
          const file = e.target.files?.[0];
          e.target.value = '';
          if (file) actions.openFile(file);
        }}
      />
    </header>
  );
}
