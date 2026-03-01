import { create } from 'zustand';
import type { TransformMode } from '@forge3d/engine';

interface EditorState {
  selectedEntityId: string | null;
  transformMode: TransformMode;
  isPlaying: boolean;
  showGrid: boolean;
  sidebarPanel: 'hierarchy' | 'materials' | 'ai' | 'assets';

  selectEntity: (id: string | null) => void;
  setTransformMode: (mode: TransformMode) => void;
  setPlaying: (playing: boolean) => void;
  toggleGrid: () => void;
  setSidebarPanel: (panel: EditorState['sidebarPanel']) => void;
}

export const useEditorStore = create<EditorState>((set) => ({
  selectedEntityId: null,
  transformMode: 'translate',
  isPlaying: false,
  showGrid: true,
  sidebarPanel: 'hierarchy',

  selectEntity: (id) => set({ selectedEntityId: id }),
  setTransformMode: (mode) => set({ transformMode: mode }),
  setPlaying: (playing) => set({ isPlaying: playing }),
  toggleGrid: () => set((s) => ({ showGrid: !s.showGrid })),
  setSidebarPanel: (panel) => set({ sidebarPanel: panel }),
}));
