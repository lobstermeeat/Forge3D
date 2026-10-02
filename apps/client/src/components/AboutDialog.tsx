import { useEffect } from 'react';

interface Credit {
  name: string;
  detail: string;
}

/**
 * The models behind Orainge's AI generation. The DINOv3 License requires "Built with DINOv3"
 * to be shown in the product; keep it here and next to generated assets.
 */
export const AI_CREDITS: Credit[] = [
  { name: 'Built with DINOv3', detail: 'Image understanding · Meta · DINOv3 License' },
  { name: 'TRELLIS.2', detail: 'Image to 3D · Microsoft · MIT' },
  { name: 'FLUX.1 [schnell]', detail: 'Reference images · Black Forest Labs · Apache-2.0' },
  { name: 'BiRefNet', detail: 'Background removal · MIT' },
];

const OPEN_SOURCE: Credit[] = [
  { name: 'three.js', detail: 'Rendering · MIT' },
  { name: 'Yjs', detail: 'Real-time collaboration · MIT' },
  { name: 'Geist', detail: 'Typeface · SIL Open Font License' },
];

function CreditList({ title, items }: { title: string; items: Credit[] }) {
  return (
    <section className="f3-credits">
      <h3>{title}</h3>
      <ul>
        {items.map((c) => (
          <li key={c.name}>
            <span>{c.name}</span>
            <span>{c.detail}</span>
          </li>
        ))}
      </ul>
    </section>
  );
}

export function AboutDialog({ open, onClose }: { open: boolean; onClose: () => void }) {
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [open, onClose]);

  if (!open) return null;
  return (
    <div
      className="f3-modal-backdrop"
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <div
        className="f3-modal f3-about"
        role="dialog"
        aria-modal="true"
        aria-labelledby="f3-about-title"
      >
        <div className="f3-modal-head">
          <div>
            <h2 id="f3-about-title">Orainge Studio</h2>
            <p>Build interactive 3D experiences in the browser.</p>
          </div>
        </div>
        <div className="f3-modal-body">
          <CreditList title="AI generation" items={AI_CREDITS} />
          <CreditList title="Open source" items={OPEN_SOURCE} />
        </div>
        <div className="f3-modal-foot">
          <button type="button" className="f3-btn" onClick={onClose}>
            Close
          </button>
        </div>
      </div>
    </div>
  );
}
