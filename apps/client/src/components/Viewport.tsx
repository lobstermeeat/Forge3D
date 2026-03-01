import { forwardRef } from 'react';

interface ViewportProps {
  rendererType: string | null;
  ready: boolean;
  onClick?: React.MouseEventHandler<HTMLCanvasElement>;
}

const Viewport = forwardRef<HTMLCanvasElement, ViewportProps>(
  ({ rendererType, ready, onClick }, ref) => {
    return (
      <div style={{ position: 'relative', width: '100%', height: '100%' }}>
        <canvas
          ref={ref}
          style={{ width: '100%', height: '100%', display: 'block' }}
          tabIndex={0}
          onClick={onClick}
        />
        {ready && rendererType && (
          <div
            style={{
              position: 'absolute',
              bottom: 8,
              right: 8,
              padding: '4px 8px',
              background: 'rgba(0,0,0,0.6)',
              color: '#aaa',
              fontSize: 12,
              borderRadius: 4,
              fontFamily: 'monospace',
            }}
          >
            {rendererType.toUpperCase()}
          </div>
        )}
      </div>
    );
  },
);

Viewport.displayName = 'Viewport';
export default Viewport;
