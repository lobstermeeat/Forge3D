import { useEffect, useRef, useState } from 'react';
import type * as THREE from 'three';
import type { SceneManager, ViewportControls } from '@forge3d/engine';
import { Icon } from '@/editor/Icon';
import { runCommand } from '@/editor/commandBar';
import { useEditorStore } from '@/stores/editorStore';
import { useFormatStore } from '@/stores/formatStore';
import { useOutputStore, type LogLevel } from '@/stores/outputStore';
import type { EditorActions } from '@/hooks/useEditorActions';
import { TimelinePanel } from './TimelinePanel';

type Filter = 'all' | 'warn';

const LEVEL_ICON: Record<LogLevel, { icon: string; color: string }> = {
  info: { icon: 'info', color: 'var(--tx3)' },
  ok: { icon: 'ok', color: 'var(--ok)' },
  warn: { icon: 'warn', color: 'var(--warn)' },
  error: { icon: 'error', color: 'var(--err)' },
  cmd: { icon: 'cmd', color: 'var(--tx2)' },
};

interface OutputPanelProps {
  actions: EditorActions;
  sceneManager: SceneManager;
  play: () => void;
  stop: () => void;
  camera: THREE.PerspectiveCamera | null;
  controls: React.RefObject<ViewportControls | null>;
}

export function OutputPanel({
  actions,
  sceneManager,
  play,
  stop,
  camera,
  controls,
}: OutputPanelProps) {
  const dockTab = useEditorStore((s) => s.dockTab);
  const setDockTab = useEditorStore((s) => s.setDockTab);
  const formatType = useFormatStore((s) => s.formatType);
  const setFormatType = useFormatStore((s) => s.setFormatType);
  const lines = useOutputStore((s) => s.lines);
  const clear = useOutputStore((s) => s.clear);
  const [filter, setFilter] = useState<Filter>('all');
  const [cmd, setCmd] = useState('');
  const [lastCmd, setLastCmd] = useState('');
  const listRef = useRef<HTMLDivElement>(null);

  const shown =
    filter === 'all' ? lines : lines.filter((l) => l.level === 'warn' || l.level === 'error');

  // Follow new output
  useEffect(() => {
    const el = listRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [shown.length, dockTab]);

  return (
    <section className="f3-dock" aria-label="Output and timeline">
      <div className="f3-subtabs" style={{ paddingRight: 8 }} role="tablist">
        <button
          type="button"
          role="tab"
          className="f3-st"
          aria-selected={dockTab === 'output'}
          onClick={() => setDockTab('output')}
        >
          <Icon name="terminal" size={14} />
          Output
        </button>
        <button
          type="button"
          role="tab"
          className="f3-st"
          aria-selected={dockTab === 'timeline'}
          onClick={() => setDockTab('timeline')}
        >
          <Icon name="timeline" size={14} />
          Timeline
        </button>
        {dockTab === 'output' && (
          <div
            style={{
              marginLeft: 'auto',
              height: 31,
              display: 'flex',
              gap: 4,
              alignItems: 'center',
            }}
          >
            <button
              type="button"
              className="f3-chip"
              style={{ height: 22 }}
              aria-pressed={filter === 'all'}
              onClick={() => setFilter('all')}
            >
              All
            </button>
            <button
              type="button"
              className="f3-chip"
              style={{ height: 22 }}
              aria-pressed={filter === 'warn'}
              onClick={() => setFilter('warn')}
            >
              Problems
            </button>
            <button
              type="button"
              className="f3-ibtn"
              style={{ width: 26, height: 26 }}
              aria-label="Clear output"
              title="Clear output"
              onClick={clear}
            >
              <Icon name="trash" size={15} />
            </button>
          </div>
        )}
      </div>

      {dockTab === 'output' ? (
        <>
          <div ref={listRef} className="f3-log" role="log" aria-live="polite">
            {shown.length === 0 && (
              <div className="f3-lg">
                <span className="t" />
                <span />
                <span className="m" style={{ color: 'var(--tx3)' }}>
                  {filter === 'warn'
                    ? 'No problems.'
                    : 'Actions, warnings and command results show up here.'}
                </span>
              </div>
            )}
            {shown.map((l) => (
              <div key={l.id} className={`f3-lg ${l.level}`}>
                <span className="t">{l.time}</span>
                <Icon
                  name={LEVEL_ICON[l.level].icon}
                  size={13}
                  style={{ marginTop: 2.5, color: LEVEL_ICON[l.level].color }}
                />
                <span className="m">{l.text}</span>
              </div>
            ))}
          </div>
          <label className="f3-cmdbar">
            <Icon name="cmd" size={14} style={{ color: 'var(--acc)' }} />
            <input
              className="f3-mono"
              aria-label="Command"
              value={cmd}
              placeholder="Run a command — try “help” or “insert sphere”"
              onChange={(e) => setCmd(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter') {
                  e.preventDefault();
                  runCommand(cmd, { actions, sceneManager, play, stop });
                  if (cmd.trim()) setLastCmd(cmd);
                  setCmd('');
                } else if (e.key === 'ArrowUp' && lastCmd) {
                  e.preventDefault();
                  setCmd(lastCmd);
                }
              }}
            />
          </label>
        </>
      ) : (
        <div style={{ flexGrow: 1, overflowY: 'auto', minHeight: 0 }}>
          {formatType !== 'video' && (
            <div
              className="f3-note"
              style={{ margin: '8px 12px 0', display: 'flex', alignItems: 'center', gap: 8 }}
            >
              <Icon name="info" size={14} />
              <span>Camera keyframes drive the 3D Video format.</span>
              <button
                type="button"
                className="f3-btn ghost"
                style={{ height: 24, fontSize: 11.5 }}
                onClick={() => setFormatType('video')}
              >
                Switch to 3D Video
              </button>
            </div>
          )}
          <TimelinePanel camera={camera} orbitControls={controls} />
        </div>
      )}
    </section>
  );
}
