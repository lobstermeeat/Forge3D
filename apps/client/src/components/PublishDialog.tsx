import { useState, useCallback, useRef, useEffect } from 'react';
import { trpc } from '@/api/trpc';
import type { SceneManager } from '@forge3d/engine';
import type { ExperienceFormatType } from '@forge3d/shared';
import { useFormatStore } from '@/stores/formatStore';

interface PublishDialogProps {
  open: boolean;
  onClose: () => void;
  sceneId: string;
  sceneManager: SceneManager;
  canvasElement: HTMLCanvasElement | null;
}

const FORMAT_OPTIONS: { value: ExperienceFormatType; label: string; description: string }[] = [
  { value: 'turntable', label: 'Turntable', description: 'Auto-rotating 3D view' },
  { value: 'interactive', label: 'Interactive', description: 'Click and explore freely' },
  { value: 'animated', label: 'Animated', description: 'Looping animation playback' },
  { value: 'video', label: '3D Video', description: 'Camera path flythrough' },
];

export function PublishDialog({
  open,
  onClose,
  sceneId,
  sceneManager: _sceneManager,
  canvasElement,
}: PublishDialogProps) {
  const [title, setTitle] = useState('');
  const [description, setDescription] = useState('');
  const [tags, setTags] = useState('');
  const formatStore = useFormatStore();
  const [formatType, setFormatType] = useState<ExperienceFormatType>(formatStore.formatType);
  const [thumbnailDataUrl, setThumbnailDataUrl] = useState<string | null>(null);

  // Sync format type to store
  useEffect(() => {
    formatStore.setFormatType(formatType);
  }, [formatType]); // eslint-disable-line react-hooks/exhaustive-deps
  const [step, setStep] = useState<'form' | 'publishing' | 'success' | 'error'>('form');
  const [publishedSlug, setPublishedSlug] = useState<string | null>(null);
  const [errorMessage, setErrorMessage] = useState('');
  const backdropRef = useRef<HTMLDivElement>(null);

  const publishMutation = trpc.experience.publish.useMutation();

  const captureThumbnail = useCallback(() => {
    if (!canvasElement) return;
    const dataUrl = canvasElement.toDataURL('image/jpeg', 0.85);
    setThumbnailDataUrl(dataUrl);
  }, [canvasElement]);

  const handlePublish = async () => {
    if (!title.trim()) return;

    setStep('publishing');

    const tagList = tags
      .split(',')
      .map((t) => t.trim())
      .filter((t) => t.length > 0)
      .slice(0, 10);

    try {
      const formatConfig = useFormatStore.getState().getFormatConfig();
      const result = await publishMutation.mutateAsync({
        sceneId,
        title: title.trim(),
        description: description.trim() || undefined,
        formatType,
        formatConfig,
        tags: tagList,
        thumbnail: thumbnailDataUrl ?? undefined,
      });

      setPublishedSlug(result.slug);
      setStep('success');
    } catch (err) {
      setErrorMessage(err instanceof Error ? err.message : 'Publishing failed');
      setStep('error');
    }
  };

  const handleBackdropClick = (e: React.MouseEvent) => {
    if (e.target === backdropRef.current) {
      onClose();
    }
  };

  const handleClose = () => {
    setStep('form');
    setTitle('');
    setDescription('');
    setTags('');
    setFormatType('turntable');
    setThumbnailDataUrl(null);
    setPublishedSlug(null);
    setErrorMessage('');
    onClose();
  };

  if (!open) return null;

  return (
    <div
      ref={backdropRef}
      onClick={handleBackdropClick}
      style={{
        position: 'fixed',
        inset: 0,
        background: 'rgba(6,7,9,0.62)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        zIndex: 1000,
      }}
    >
      <div
        style={{
          background: 'var(--bg2)',
          borderRadius: 16,
          width: 540,
          maxHeight: '86vh',
          overflow: 'auto',
          padding: 24,
          boxShadow: 'inset 0 0 0 1px var(--line2), 0 30px 80px rgba(0,0,0,0.55)',
          color: 'var(--tx1)',
          userSelect: 'none',
        }}
      >
        {step === 'form' && (
          <>
            <h2 style={{ margin: '0 0 4px', fontSize: 17, fontWeight: 600, color: 'var(--tx1)' }}>
              Publish to Orainge
            </h2>
            <p style={{ margin: '0 0 18px', fontSize: 12.5, color: 'var(--tx2)' }}>
              Viewers can explore your scene but can’t download its models.
            </p>

            {/* Thumbnail */}
            <div style={{ marginBottom: 16 }}>
              <label style={labelStyle}>Thumbnail</label>
              {thumbnailDataUrl ? (
                <div style={{ position: 'relative' }}>
                  <img
                    src={thumbnailDataUrl}
                    alt="Thumbnail preview"
                    style={{ width: '100%', borderRadius: 10, border: '1px solid var(--line2)' }}
                  />
                  <button onClick={captureThumbnail} style={{ ...smallBtnStyle, position: 'absolute', bottom: 8, right: 8 }}>
                    Recapture
                  </button>
                </div>
              ) : (
                <button onClick={captureThumbnail} style={actionBtnStyle}>
                  Capture from Viewport
                </button>
              )}
            </div>

            {/* Title */}
            <div style={{ marginBottom: 12 }}>
              <label style={labelStyle}>Title *</label>
              <input
                type="text"
                value={title}
                onChange={(e) => setTitle(e.target.value)}
                placeholder="My 3D Experience"
                maxLength={200}
                style={inputStyle}
              />
            </div>

            {/* Description */}
            <div style={{ marginBottom: 12 }}>
              <label style={labelStyle}>Description</label>
              <textarea
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                placeholder="Describe your 3D experience..."
                maxLength={2000}
                rows={3}
                style={{ ...inputStyle, resize: 'vertical' }}
              />
            </div>

            {/* Tags */}
            <div style={{ marginBottom: 12 }}>
              <label style={labelStyle}>Tags (comma-separated)</label>
              <input
                type="text"
                value={tags}
                onChange={(e) => setTags(e.target.value)}
                placeholder="3d, art, lowpoly, character"
                style={inputStyle}
              />
            </div>

            {/* Format */}
            <div style={{ marginBottom: 20 }}>
              <label style={labelStyle}>Format</label>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
                {FORMAT_OPTIONS.map((opt) => (
                  <button
                    key={opt.value}
                    onClick={() => setFormatType(opt.value)}
                    style={{
                      padding: '10px 12px',
                      borderRadius: 8,
                      border: formatType === opt.value ? '1.5px solid var(--acc)' : '1px solid var(--line2)',
                      background: formatType === opt.value ? 'var(--acc-soft)' : 'var(--bg1)',
                      color: 'var(--tx1)',
                      cursor: 'pointer',
                      textAlign: 'left',
                    }}
                  >
                    <div style={{ fontSize: 13, fontWeight: 600 }}>{opt.label}</div>
                    <div style={{ fontSize: 11.5, color: 'var(--tx2)', marginTop: 3 }}>{opt.description}</div>
                  </button>
                ))}
              </div>
            </div>

            {/* Format-specific config */}
            <div style={{ marginBottom: 16 }}>
              {formatType === 'turntable' && (
                <div style={{ display: 'flex', gap: 12, alignItems: 'center' }}>
                  <label style={{ fontSize: 12, color: 'var(--tx2)' }}>Speed</label>
                  <input
                    type="range"
                    min={0.1}
                    max={5}
                    step={0.1}
                    value={formatStore.turntableSpeed}
                    onChange={(e) => formatStore.setTurntableSpeed(parseFloat(e.target.value))}
                    style={{ flex: 1 }}
                  />
                  <span style={{ fontSize: 11, color: 'var(--tx3)', minWidth: 24 }}>{formatStore.turntableSpeed.toFixed(1)}</span>
                  <label style={{ fontSize: 12, color: 'var(--tx2)', marginLeft: 8 }}>Axis</label>
                  <select
                    value={formatStore.turntableAxis}
                    onChange={(e) => formatStore.setTurntableAxis(e.target.value as 'y' | 'x')}
                    style={{ ...inputStyle, width: 50, padding: '4px 6px' }}
                  >
                    <option value="y">Y</option>
                    <option value="x">X</option>
                  </select>
                </div>
              )}
              {formatType === 'video' && (
                <div style={{ fontSize: 12, color: 'var(--tx2)' }}>
                  {formatStore.keyframes.length} keyframe{formatStore.keyframes.length !== 1 ? 's' : ''} configured
                  {formatStore.keyframes.length < 2 && (
                    <span style={{ color: 'var(--err)', marginLeft: 8 }}>
                      (min 2 required — use Timeline in editor)
                    </span>
                  )}
                </div>
              )}
              {formatType === 'interactive' && (
                <div style={{ fontSize: 12, color: 'var(--tx2)' }}>
                  {formatStore.interactions.length} interaction{formatStore.interactions.length !== 1 ? 's' : ''} configured
                  {formatStore.interactions.length === 0 && (
                    <span style={{ color: 'var(--tx3)', marginLeft: 8 }}>
                      (assign interactions via Inspector panel)
                    </span>
                  )}
                </div>
              )}
              {formatType === 'animated' && (
                <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                  <label style={{ fontSize: 12, color: 'var(--tx2)' }}>Duration (seconds)</label>
                  <input
                    type="number"
                    min={1}
                    step={1}
                    value={formatStore.animationDuration}
                    onChange={(e) => formatStore.setAnimationDuration(parseFloat(e.target.value) || 5)}
                    style={{ ...inputStyle, width: 80, padding: '4px 8px' }}
                  />
                </div>
              )}
            </div>

            {/* Actions */}
            <div style={{ display: 'flex', gap: 8, justifyContent: 'flex-end' }}>
              <button onClick={handleClose} style={cancelBtnStyle}>
                Cancel
              </button>
              <button
                onClick={handlePublish}
                disabled={!title.trim()}
                style={{
                  ...publishBtnStyle,
                  opacity: title.trim() ? 1 : 0.5,
                  cursor: title.trim() ? 'pointer' : 'not-allowed',
                }}
              >
                Publish
              </button>
            </div>
          </>
        )}

        {step === 'publishing' && (
          <div style={{ textAlign: 'center', padding: '40px 0' }}>
            <div style={{ fontSize: 18, marginBottom: 8 }}>Publishing...</div>
            <div style={{ color: 'var(--tx2)', fontSize: 13 }}>Preparing your experience for the world</div>
          </div>
        )}

        {step === 'success' && (
          <div style={{ textAlign: 'center', padding: '24px 0' }}>
            <div style={{ fontSize: 40, marginBottom: 12 }}>&#10003;</div>
            <div style={{ fontSize: 18, fontWeight: 600, marginBottom: 8 }}>Published!</div>
            <div style={{ color: 'var(--tx2)', fontSize: 13, marginBottom: 16 }}>
              Your experience is live and ready to share.
            </div>
            {publishedSlug && (
              <div style={{ marginBottom: 16 }}>
                <input
                  type="text"
                  readOnly
                  value={`${window.location.origin}/view/${publishedSlug}`}
                  style={{ ...inputStyle, textAlign: 'center', cursor: 'text' }}
                  onClick={(e) => (e.target as HTMLInputElement).select()}
                />
              </div>
            )}
            <button onClick={handleClose} style={publishBtnStyle}>
              Done
            </button>
          </div>
        )}

        {step === 'error' && (
          <div style={{ textAlign: 'center', padding: '24px 0' }}>
            <div style={{ fontSize: 18, fontWeight: 600, marginBottom: 8, color: 'var(--err)' }}>
              Publishing Failed
            </div>
            <div style={{ color: 'var(--tx2)', fontSize: 13, marginBottom: 16 }}>{errorMessage}</div>
            <div style={{ display: 'flex', gap: 8, justifyContent: 'center' }}>
              <button onClick={() => setStep('form')} style={cancelBtnStyle}>
                Back
              </button>
              <button onClick={handlePublish} style={publishBtnStyle}>
                Retry
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

const labelStyle: React.CSSProperties = {
  display: 'block',
  fontSize: 12,
  fontWeight: 600,
  color: 'var(--tx2)',
  marginBottom: 4,
};

const inputStyle: React.CSSProperties = {
  width: '100%',
  padding: '8px 12px',
  borderRadius: 7,
  border: '1px solid transparent',
  background: 'var(--bg3)',
  color: 'var(--tx1)',
  fontSize: 13,
  outline: 'none',
  boxSizing: 'border-box',
};

const actionBtnStyle: React.CSSProperties = {
  padding: '10px 16px',
  borderRadius: 10,
  border: '1px dashed var(--line2)',
  background: 'var(--bg1)',
  color: 'var(--tx2)',
  cursor: 'pointer',
  width: '100%',
  fontSize: 13,
};

const smallBtnStyle: React.CSSProperties = {
  padding: '4px 10px',
  borderRadius: 7,
  border: '1px solid var(--line2)',
  background: 'rgba(14,15,18,0.8)',
  color: 'var(--tx1)',
  cursor: 'pointer',
  fontSize: 11,
};

const cancelBtnStyle: React.CSSProperties = {
  padding: '8px 20px',
  borderRadius: 7,
  border: '1px solid var(--line2)',
  background: 'transparent',
  color: 'var(--tx1)',
  cursor: 'pointer',
  fontSize: 13,
};

const publishBtnStyle: React.CSSProperties = {
  padding: '8px 20px',
  borderRadius: 7,
  border: 'none',
  background: 'var(--acc)',
  color: 'var(--acc-ink)',
  cursor: 'pointer',
  fontSize: 13,
  fontWeight: 600,
};
