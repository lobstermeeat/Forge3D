import { useEffect, useRef } from 'react';
import { useEditorStore } from '@/stores/editorStore';
import type { EditorActions } from '@/hooks/useEditorActions';

interface ShortcutArgs {
  actions: EditorActions;
  play: () => void;
  stop: () => void;
  save: () => void;
}

function isTyping(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false;
  return (
    target instanceof HTMLInputElement ||
    target instanceof HTMLTextAreaElement ||
    target instanceof HTMLSelectElement ||
    target.isContentEditable
  );
}

/** Blur the focused field so a half-typed value is committed before play or save. */
function commitFocusedField(target: EventTarget | null): void {
  if (isTyping(target)) (target as HTMLElement).blur();
}

/**
 * Editor keyboard shortcuts.
 * G / R / S — move / rotate / scale · F — focus selection · Del — delete
 * Ctrl+D — duplicate · Ctrl+Z / Ctrl+Shift+Z — undo / redo · Ctrl+S — save
 * F5 / Shift+F5 — play / stop · Esc — stop or deselect
 */
export function useKeyboardShortcuts({ actions, play, stop, save }: ShortcutArgs) {
  const argsRef = useRef({ actions, play, stop, save });
  argsRef.current = { actions, play, stop, save };

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      const { actions: a, play: startPlay, stop: stopPlay, save: saveScene } = argsRef.current;
      const mod = e.ctrlKey || e.metaKey;
      const key = e.key.toLowerCase();

      // These work even while a field has focus: F5 would otherwise reload the page,
      // and Ctrl+S would open the browser's "Save page" dialog.
      if (e.key === 'F5') {
        e.preventDefault();
        commitFocusedField(e.target);
        if (e.shiftKey) stopPlay();
        else startPlay();
        return;
      }
      if (mod && !e.shiftKey && !e.altKey && key === 's') {
        e.preventDefault();
        commitFocusedField(e.target);
        saveScene();
        return;
      }

      if (isTyping(e.target)) return;
      const state = useEditorStore.getState();

      if (e.key === 'Escape') {
        if (state.isPlaying) stopPlay();
        else a.select(null);
        return;
      }

      // Editing is paused while play-testing
      if (state.isPlaying) return;

      if (mod && key === 'z') {
        e.preventDefault();
        if (e.shiftKey) a.redo();
        else a.undo();
        return;
      }
      if (mod && key === 'y') {
        e.preventDefault();
        a.redo();
        return;
      }
      if (mod && key === 'd') {
        e.preventDefault();
        a.duplicate();
        return;
      }
      if (mod || e.altKey) return;

      if (e.key === 'Delete' || e.key === 'Backspace') {
        if (state.selectedEntityId) {
          e.preventDefault();
          a.remove();
        }
        return;
      }

      switch (key) {
        case 'f':
          a.focus();
          break;
        case 'g':
          state.setTransformMode('translate');
          break;
        case 'r':
          state.setTransformMode('rotate');
          break;
        case 's':
          state.setTransformMode('scale');
          break;
      }
    };

    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, []);
}
