import { useEffect } from 'react';
import * as THREE from 'three';
import { RemoveEntityCommand } from '@forge3d/engine';
import type { CommandHistory, SceneManager, ViewportControls } from '@forge3d/engine';
import { useEditorStore } from '@/stores/editorStore';

interface ShortcutArgs {
  history: CommandHistory;
  sceneManager: SceneManager;
  controls: React.RefObject<ViewportControls | null>;
  canvas: HTMLCanvasElement | null;
}

export function useKeyboardShortcuts({ history, sceneManager, controls, canvas }: ShortcutArgs) {
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      const isInput =
        e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement;
      if (isInput) return;

      // Ctrl+Z = Undo
      if ((e.ctrlKey || e.metaKey) && e.key === 'z' && !e.shiftKey) {
        e.preventDefault();
        history.undo();
        sceneManager.notifyChange();
      }

      // Ctrl+Shift+Z = Redo
      if ((e.ctrlKey || e.metaKey) && (e.key === 'Z' || (e.key === 'z' && e.shiftKey))) {
        e.preventDefault();
        history.redo();
        sceneManager.notifyChange();
      }

      // Delete selected
      if ((e.key === 'Delete' || e.key === 'Backspace')) {
        const selectedEntityId = useEditorStore.getState().selectedEntityId;
        if (selectedEntityId) {
          e.preventDefault();
          const cmd = new RemoveEntityCommand(sceneManager, selectedEntityId);
          history.execute(cmd);
          useEditorStore.getState().selectEntity(null);
          sceneManager.selectEntity(null);
        }
      }

      // F = focus on selected
      if (e.key === 'f') {
        const selectedEntityId = useEditorStore.getState().selectedEntityId;
        if (selectedEntityId && controls.current) {
          const entity = sceneManager.getEntity(selectedEntityId);
          if (entity) {
            const t = entity.transform;
            const target = new THREE.Vector3(t.position[0], t.position[1], t.position[2]);
            controls.current.focusOn(target);
          }
        }
      }

      // G / R / S = transform mode
      const { setTransformMode } = useEditorStore.getState();
      if (e.key === 'g') setTransformMode('translate');
      if (e.key === 'r') setTransformMode('rotate');
      if (e.key === 's' && !e.ctrlKey && !e.metaKey) setTransformMode('scale');

      // Escape = deselect
      if (e.key === 'Escape') {
        useEditorStore.getState().selectEntity(null);
        sceneManager.selectEntity(null);
      }
    };

    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [history, sceneManager, controls, canvas]);
}
