import { useEditorStore } from '@/stores/editorStore';

interface HierarchyNode {
  id: string;
  name: string;
  children: HierarchyNode[];
}

interface HierarchyPanelProps {
  nodes: HierarchyNode[];
  onSelect: (id: string) => void;
  onDelete: (id: string) => void;
  onRename: (id: string, name: string) => void;
}

function TreeNode({
  node,
  depth,
  selectedId,
  onSelect,
  onDelete,
  onRename,
}: {
  node: HierarchyNode;
  depth: number;
  selectedId: string | null;
  onSelect: (id: string) => void;
  onDelete: (id: string) => void;
  onRename: (id: string, name: string) => void;
}) {
  const isSelected = node.id === selectedId;

  return (
    <div>
      <div
        role="treeitem"
        tabIndex={0}
        aria-selected={isSelected}
        style={{
          paddingLeft: depth * 16 + 8,
          paddingTop: 4,
          paddingBottom: 4,
          background: isSelected ? '#2563eb' : 'transparent',
          color: isSelected ? '#fff' : '#ccc',
          cursor: 'pointer',
          fontSize: 13,
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
        }}
        onClick={() => onSelect(node.id)}
        onDoubleClick={() => {
          const name = prompt('Rename entity:', node.name);
          if (name) onRename(node.id, name);
        }}
        onKeyDown={(e) => {
          if (e.key === 'Enter') onSelect(node.id);
          if (e.key === 'Delete' || e.key === 'Backspace') onDelete(node.id);
        }}
      >
        <span>{node.name}</span>
      </div>
      {node.children.map((child) => (
        <TreeNode
          key={child.id}
          node={child}
          depth={depth + 1}
          selectedId={selectedId}
          onSelect={onSelect}
          onDelete={onDelete}
          onRename={onRename}
        />
      ))}
    </div>
  );
}

export function HierarchyPanel({ nodes, onSelect, onDelete, onRename }: HierarchyPanelProps) {
  const selectedId = useEditorStore((s) => s.selectedEntityId);

  return (
    <div
      role="tree"
      aria-label="Scene hierarchy"
      style={{
        background: '#1e1e2e',
        height: '100%',
        overflow: 'auto',
        padding: 4,
      }}
    >
      <div
        style={{
          padding: '8px',
          fontSize: 12,
          fontWeight: 600,
          color: '#888',
          textTransform: 'uppercase',
          letterSpacing: 1,
        }}
      >
        Scene
      </div>
      {nodes.map((node) => (
        <TreeNode
          key={node.id}
          node={node}
          depth={0}
          selectedId={selectedId}
          onSelect={onSelect}
          onDelete={onDelete}
          onRename={onRename}
        />
      ))}
    </div>
  );
}
