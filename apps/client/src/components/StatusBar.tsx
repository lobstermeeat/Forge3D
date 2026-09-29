import { useEditorStore } from '@/stores/editorStore';

interface StatusBarProps {
  backendLabel: string | null;
  objectCount: number;
  selectionName: string | null;
  formatLabel: string;
}

export function StatusBar({
  backendLabel,
  objectCount,
  selectionName,
  formatLabel,
}: StatusBarProps) {
  const snapEnabled = useEditorStore((s) => s.snapEnabled);
  const moveStep = useEditorStore((s) => s.moveStep);
  const rotateStep = useEditorStore((s) => s.rotateStep);
  const space = useEditorStore((s) => s.space);
  const playing = useEditorStore((s) => s.isPlaying);

  return (
    <footer className="f3-statusbar f3-mono">
      <span style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
        <span
          className="f3-dot"
          style={{ background: backendLabel ? 'var(--ok)' : 'var(--warn)' }}
        />
        {backendLabel ?? 'Starting renderer…'}
      </span>
      <span>
        {objectCount} object{objectCount === 1 ? '' : 's'}
      </span>
      {playing && (
        <span style={{ color: 'var(--ok)', display: 'flex', alignItems: 'center', gap: 6 }}>
          <span className="f3-dot f3-pulse" style={{ background: 'var(--ok)' }} />
          Playing · {formatLabel}
        </span>
      )}
      <span style={{ marginLeft: 'auto' }}>
        {snapEnabled ? `Snap ${moveStep} m · ${rotateStep}°` : 'Snap off'}
      </span>
      <span>{space === 'local' ? 'Local' : 'World'}</span>
      <span
        style={{ color: 'var(--tx2)', maxWidth: 260, overflow: 'hidden', textOverflow: 'ellipsis' }}
      >
        {selectionName ?? 'Nothing selected'}
      </span>
    </footer>
  );
}
