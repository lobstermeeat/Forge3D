import { useFormatStore } from '@/stores/formatStore';
import { useEditorStore } from '@/stores/editorStore';
import type { InteractionDef } from '@forge3d/shared';

const ACTION_OPTIONS: { value: InteractionDef['action']; label: string }[] = [
  { value: 'showInfo', label: 'Show Info' },
  { value: 'openUrl', label: 'Open URL' },
  { value: 'navigate', label: 'Navigate' },
  { value: 'playAnimation', label: 'Play Animation' },
];

export function InteractionEditor() {
  const selectedEntityId = useEditorStore((s) => s.selectedEntityId);
  const { interactions, addInteraction, updateInteraction, removeInteraction } =
    useFormatStore();

  if (!selectedEntityId) return null;

  const interaction = interactions.find((d) => d.entityId === selectedEntityId);

  const handleAdd = () => {
    addInteraction({
      entityId: selectedEntityId,
      action: 'showInfo',
      payload: { title: '', description: '' },
    });
  };

  const handleActionChange = (action: InteractionDef['action']) => {
    // Reset payload when action type changes
    let payload: Record<string, unknown> = {};
    switch (action) {
      case 'showInfo':
        payload = { title: '', description: '' };
        break;
      case 'openUrl':
      case 'navigate':
        payload = { url: '' };
        break;
      case 'playAnimation':
        payload = { clip: '' };
        break;
    }
    updateInteraction(selectedEntityId, { action, payload });
  };

  const handlePayloadChange = (key: string, value: string) => {
    if (!interaction) return;
    updateInteraction(selectedEntityId, {
      payload: { ...interaction.payload, [key]: value },
    });
  };

  return (
    <div style={{ padding: '8px 12px', borderTop: '1px solid #313244' }}>
      <div style={{ fontSize: 11, fontWeight: 600, color: '#a6adc8', marginBottom: 6 }}>
        Interaction
      </div>

      {!interaction ? (
        <button onClick={handleAdd} style={addBtnStyle}>
          + Add Interaction
        </button>
      ) : (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
          {/* Action type */}
          <select
            value={interaction.action}
            onChange={(e) => handleActionChange(e.target.value as InteractionDef['action'])}
            style={inputStyle}
          >
            {ACTION_OPTIONS.map((opt) => (
              <option key={opt.value} value={opt.value}>
                {opt.label}
              </option>
            ))}
          </select>

          {/* Payload fields */}
          {interaction.action === 'showInfo' && (
            <>
              <input
                type="text"
                placeholder="Title"
                value={(interaction.payload['title'] as string) ?? ''}
                onChange={(e) => handlePayloadChange('title', e.target.value)}
                style={inputStyle}
              />
              <textarea
                placeholder="Description"
                value={(interaction.payload['description'] as string) ?? ''}
                onChange={(e) => handlePayloadChange('description', e.target.value)}
                rows={2}
                style={{ ...inputStyle, resize: 'vertical' }}
              />
            </>
          )}

          {(interaction.action === 'openUrl' || interaction.action === 'navigate') && (
            <input
              type="text"
              placeholder="URL"
              value={(interaction.payload['url'] as string) ?? ''}
              onChange={(e) => handlePayloadChange('url', e.target.value)}
              style={inputStyle}
            />
          )}

          {interaction.action === 'playAnimation' && (
            <input
              type="text"
              placeholder="Animation clip name"
              value={(interaction.payload['clip'] as string) ?? ''}
              onChange={(e) => handlePayloadChange('clip', e.target.value)}
              style={inputStyle}
            />
          )}

          <button
            onClick={() => removeInteraction(selectedEntityId)}
            style={{ ...addBtnStyle, color: '#f38ba8', borderColor: '#f38ba8' }}
          >
            Remove Interaction
          </button>
        </div>
      )}
    </div>
  );
}

const inputStyle: React.CSSProperties = {
  width: '100%',
  padding: '4px 8px',
  borderRadius: 4,
  border: '1px solid #313244',
  background: '#11111b',
  color: '#cdd6f4',
  fontSize: 11,
  outline: 'none',
  boxSizing: 'border-box',
};

const addBtnStyle: React.CSSProperties = {
  width: '100%',
  padding: '6px 0',
  borderRadius: 4,
  border: '1px dashed #313244',
  background: 'transparent',
  color: '#a6adc8',
  cursor: 'pointer',
  fontSize: 11,
};
