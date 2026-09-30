import { useFormatStore } from '@/stores/formatStore';
import { useEditorStore } from '@/stores/editorStore';
import type { InteractionDef } from '@forge3d/shared';
import { Icon } from '@/editor/Icon';
import { PropRow, Select } from '@/editor/ui';

const ACTION_OPTIONS: { value: InteractionDef['action']; label: string }[] = [
  { value: 'showInfo', label: 'Show info card' },
  { value: 'openUrl', label: 'Open a link' },
  { value: 'navigate', label: 'Go to another page' },
  { value: 'playAnimation', label: 'Play an animation' },
];

export function InteractionEditor() {
  const selectedEntityId = useEditorStore((s) => s.selectedEntityId);
  const playing = useEditorStore((s) => s.isPlaying);
  const { interactions, addInteraction, updateInteraction, removeInteraction } = useFormatStore();

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

  if (!interaction) {
    return (
      <div style={{ padding: '0 12px' }}>
        <button
          type="button"
          className="f3-btn ghost"
          style={{ width: '100%', boxShadow: 'inset 0 0 0 1px var(--line2)', borderStyle: 'dashed' }}
          disabled={playing}
          onClick={handleAdd}
        >
          <Icon name="bolt" size={14} />
          Add a click action
        </button>
      </div>
    );
  }

  const text = (key: string) => (interaction.payload[key] as string) ?? '';

  return (
    <>
      <PropRow label="When clicked">
        <Select label="When clicked" value={interaction.action} options={ACTION_OPTIONS} onChange={handleActionChange} />
      </PropRow>

      {interaction.action === 'showInfo' && (
        <>
          <PropRow label="Title">
            <input className="f3-input" aria-label="Info card title" placeholder="Card title" value={text('title')} onChange={(e) => handlePayloadChange('title', e.target.value)} />
          </PropRow>
          <PropRow label="Text" variant="top">
            <textarea
              className="f3-input"
              aria-label="Info card text"
              placeholder="What viewers read"
              rows={3}
              value={text('description')}
              onChange={(e) => handlePayloadChange('description', e.target.value)}
              style={{ resize: 'vertical' }}
            />
          </PropRow>
        </>
      )}

      {(interaction.action === 'openUrl' || interaction.action === 'navigate') && (
        <PropRow label="URL">
          <input className="f3-input" aria-label="URL" placeholder="https://" value={text('url')} onChange={(e) => handlePayloadChange('url', e.target.value)} />
        </PropRow>
      )}

      {interaction.action === 'playAnimation' && (
        <PropRow label="Clip">
          <input className="f3-input" aria-label="Animation clip name" placeholder="Animation clip name" value={text('clip')} onChange={(e) => handlePayloadChange('clip', e.target.value)} />
        </PropRow>
      )}

      <div style={{ padding: '4px 12px 0' }}>
        <button type="button" className="f3-btn danger" style={{ height: 26, fontSize: 12 }} disabled={playing} onClick={() => removeInteraction(selectedEntityId)}>
          <Icon name="trash" size={13} />
          Remove click action
        </button>
      </div>
    </>
  );
}
