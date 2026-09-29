import { forwardRef } from 'react';

interface ViewportProps {
  onClick?: React.MouseEventHandler<HTMLCanvasElement>;
}

/** The WebGPU/WebGL canvas. Overlays and the renderer label live in the editor chrome. */
const Viewport = forwardRef<HTMLCanvasElement, ViewportProps>(({ onClick }, ref) => {
  return (
    <div style={{ position: 'relative', width: '100%', height: '100%' }}>
      <canvas
        ref={ref}
        aria-label="3D viewport"
        style={{ width: '100%', height: '100%', display: 'block', touchAction: 'none' }}
        tabIndex={0}
        onClick={onClick}
        onContextMenu={(e) => e.preventDefault()}
      />
    </div>
  );
});

Viewport.displayName = 'Viewport';
export default Viewport;
