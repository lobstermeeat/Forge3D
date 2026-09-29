import { useEffect, useRef, useState } from 'react';
import type { ReactElement } from 'react';
import type { SceneManager } from '@forge3d/engine';
import { Icon } from '@/editor/Icon';
import { describeEntity } from '@/editor/entityInfo';
import { useEditorStore } from '@/stores/editorStore';
import type { HierarchyNode } from '@/hooks/useSceneBridge';
import type { EditorActions } from '@/hooks/useEditorActions';
import type { AwarenessUser } from '@/hooks/useCollaboration';

interface ExplorerPanelProps {
  sceneName: string;
  nodes: HierarchyNode[];
  sceneManager: SceneManager;
  actions: EditorActions;
  peers: AwarenessUser[];
  /** Grow to fill the column when Properties is hidden. */
  fill: boolean;
}

function matches(node: HierarchyNode, q: string): boolean {
  return node.name.toLowerCase().includes(q) || node.children.some((c) => matches(c, q));
}

function countNodes(nodes: HierarchyNode[]): number {
  return nodes.reduce((n, node) => n + 1 + countNodes(node.children), 0);
}

export function ExplorerPanel({
  sceneName,
  nodes,
  sceneManager,
  actions,
  peers,
  fill,
}: ExplorerPanelProps) {
  const selectedId = useEditorStore((s) => s.selectedEntityId);
  const setHovered = useEditorStore((s) => s.setHovered);
  const playing = useEditorStore((s) => s.isPlaying);
  const [collapsed, setCollapsed] = useState<Record<string, boolean>>({});
  const [filter, setFilter] = useState('');
  const [renaming, setRenaming] = useState<string | null>(null);
  const treeRef = useRef<HTMLDivElement>(null);
  const q = filter.trim().toLowerCase();

  // Reveal the selection: expand its ancestors and scroll it into view
  useEffect(() => {
    if (!selectedId) return;
    let parentId = sceneManager.getEntity(selectedId)?.parentId;
    const open: Record<string, boolean> = {};
    while (parentId) {
      open[parentId] = false;
      parentId = sceneManager.getEntity(parentId)?.parentId;
    }
    if (Object.keys(open).length) setCollapsed((c) => ({ ...c, ...open }));
    requestAnimationFrame(() => {
      treeRef.current
        ?.querySelector(`[data-row="${selectedId}"]`)
        ?.scrollIntoView({ block: 'nearest' });
    });
  }, [selectedId, sceneManager]);

  const peerColor = (id: string) => peers.find((p) => p.selection === id)?.color;

  const rows: ReactElement[] = [];
  const walk = (list: HierarchyNode[], depth: number) => {
    for (const node of list) {
      if (q && !matches(node, q)) continue;
      const entity = sceneManager.getEntity(node.id);
      if (!entity) continue;
      const info = describeEntity(entity);
      const hasKids = node.children.length > 0;
      const open = q ? true : !collapsed[node.id];
      const selected = selectedId === node.id;
      const peer = peerColor(node.id);
      rows.push(
        <div
          key={node.id}
          role="treeitem"
          aria-selected={selected}
          aria-expanded={hasKids ? open : undefined}
          aria-level={depth + 2}
          data-row={node.id}
          className="f3-er"
          style={{ paddingLeft: 4 + (depth + 1) * 14 }}
          onPointerEnter={() => setHovered(node.id)}
          onPointerLeave={() => setHovered(null)}
        >
          <button
            type="button"
            className="f3-car"
            tabIndex={-1}
            aria-label={open ? `Collapse ${node.name}` : `Expand ${node.name}`}
            style={{ visibility: hasKids ? 'visible' : 'hidden' }}
            onClick={() => setCollapsed((c) => ({ ...c, [node.id]: open }))}
          >
            <Icon name="caret" size={10} style={{ transform: `rotate(${open ? 90 : 0}deg)` }} />
          </button>
          {renaming === node.id ? (
            <div className="f3-er-main">
              <Icon name={info.icon} size={14} className="f3-er-ic" style={{ color: info.color }} />
              <input
                className="f3-er-rename"
                aria-label={`Rename ${node.name}`}
                defaultValue={node.name}
                autoFocus
                onFocus={(e) => e.currentTarget.select()}
                onBlur={(e) => {
                  actions.rename(node.id, e.currentTarget.value);
                  setRenaming(null);
                }}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') e.currentTarget.blur();
                  if (e.key === 'Escape') {
                    e.currentTarget.value = node.name;
                    e.currentTarget.blur();
                  }
                }}
              />
            </div>
          ) : (
            <button
              type="button"
              className="f3-er-main"
              disabled={playing}
              onClick={() => actions.select(node.id)}
              onDoubleClick={() => setRenaming(node.id)}
              onKeyDown={(e) => {
                if (e.key === 'F2') setRenaming(node.id);
              }}
              title={`${info.type} — double-click to rename`}
            >
              <Icon
                name={info.icon}
                size={14}
                className="f3-er-ic"
                style={{ color: selected ? undefined : info.color }}
              />
              <span>{node.name}</span>
            </button>
          )}
          {peer && (
            <span
              className="f3-peerdot"
              style={{ background: peer }}
              title="A collaborator has this selected"
            />
          )}
          {!playing && renaming !== node.id && (
            <div className="f3-er-acts">
              <button
                type="button"
                className="f3-ab"
                aria-label={`Focus ${node.name}`}
                title="Focus (F)"
                onClick={() => actions.focus(node.id)}
              >
                <Icon name="focus" size={13} />
              </button>
              <button
                type="button"
                className="f3-ab"
                aria-label={`Delete ${node.name}`}
                title="Delete (Del)"
                onClick={() => actions.remove(node.id)}
              >
                <Icon name="trash" size={13} />
              </button>
            </div>
          )}
        </div>,
      );
      if (hasKids && open) walk(node.children, depth + 1);
    }
  };
  walk(nodes, 0);

  const total = countNodes(nodes);

  return (
    <section
      aria-label="Explorer"
      style={{
        display: 'flex',
        flexDirection: 'column',
        minHeight: 0,
        flex: fill ? '1 1 auto' : '0 0 300px',
        borderBottom: fill ? 'none' : '1px solid var(--line)',
      }}
    >
      <div className="f3-ph">
        <span className="f3-pt">Explorer</span>
        <span style={{ fontSize: 11, color: 'var(--tx3)' }}>
          {total} object{total === 1 ? '' : 's'}
        </span>
        <div style={{ marginLeft: 'auto', display: 'flex', gap: 2 }}>
          <button
            type="button"
            className="f3-ibtn"
            style={{ width: 24, height: 24 }}
            aria-label="Collapse all"
            title="Collapse all"
            onClick={() =>
              setCollapsed(
                Object.fromEntries(sceneManager.getAllEntities().map((e) => [e.id, true])),
              )
            }
          >
            <Icon name="collapse" size={14} />
          </button>
        </div>
      </div>
      <div style={{ padding: '8px 8px 6px', flexShrink: 0 }}>
        <label className="f3-fld" style={{ height: 26 }}>
          <Icon name="search" size={13} />
          <input
            aria-label="Filter objects"
            placeholder="Filter objects"
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
          />
        </label>
      </div>
      <div ref={treeRef} className="f3-tree" role="tree" aria-label="Scene objects">
        <div
          role="treeitem"
          aria-selected={selectedId === null}
          aria-level={1}
          className="f3-er"
          style={{ paddingLeft: 4 }}
          onPointerEnter={() => setHovered(null)}
        >
          <span className="f3-car" />
          <button
            type="button"
            className="f3-er-main"
            disabled={playing}
            onClick={() => actions.select(null)}
            title="Scene — experience settings"
          >
            <Icon
              name="scene"
              size={14}
              className="f3-er-ic"
              style={{ color: selectedId === null ? undefined : 'var(--tx1)' }}
            />
            <span style={{ fontWeight: 500 }}>{sceneName}</span>
          </button>
        </div>
        {rows}
        {total === 0 && (
          <div className="f3-empty" style={{ margin: '10px 4px 0' }}>
            <Icon name="box" size={20} />
            <span>
              Empty scene. Insert a part from the ribbon, or drag one in from the Library.
            </span>
          </div>
        )}
        {total > 0 && rows.length === 0 && q && (
          <div className="f3-empty" style={{ margin: '10px 4px 0' }}>
            <span>No objects match “{filter}”.</span>
          </div>
        )}
      </div>
    </section>
  );
}
