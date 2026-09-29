import { create } from 'zustand';
import type { TransformMode } from '@forge3d/engine';

export type RibbonTab = 'home' | 'model' | 'test' | 'view';
export type PanelKey = 'library' | 'explorer' | 'properties' | 'output';
export type DockTab = 'output' | 'timeline';
export type CameraView = 'perspective' | 'top' | 'front' | 'side';

const MOVE_STEPS = [0.25, 0.5, 1, 2] as const;
const ROTATE_STEPS = [5, 15, 45, 90] as const;

interface EditorState {
  selectedEntityId: string | null;
  hoveredEntityId: string | null;
  transformMode: TransformMode;
  /** Select tool: clicking selects, no transform gizmo. */
  selectOnly: boolean;
  isPlaying: boolean;
  showGrid: boolean;
  sidebarPanel: 'hierarchy' | 'materials' | 'ai' | 'assets';

  ribbonTab: RibbonTab;
  panels: Record<PanelKey, boolean>;
  dockTab: DockTab;
  snapEnabled: boolean;
  moveStep: number;
  rotateStep: number;
  space: 'world' | 'local';
  cameraView: CameraView;

  selectEntity: (id: string | null) => void;
  setHovered: (id: string | null) => void;
  setTransformMode: (mode: TransformMode) => void;
  setSelectOnly: () => void;
  setPlaying: (playing: boolean) => void;
  toggleGrid: () => void;
  setSidebarPanel: (panel: EditorState['sidebarPanel']) => void;
  setRibbonTab: (tab: RibbonTab) => void;
  togglePanel: (panel: PanelKey) => void;
  showPanel: (panel: PanelKey) => void;
  setDockTab: (tab: DockTab) => void;
  toggleSnap: () => void;
  cycleMoveStep: () => void;
  cycleRotateStep: () => void;
  setSpace: (space: 'world' | 'local') => void;
  setCameraView: (view: CameraView) => void;
}

export const useEditorStore = create<EditorState>((set) => ({
  selectedEntityId: null,
  hoveredEntityId: null,
  transformMode: 'translate',
  selectOnly: false,
  isPlaying: false,
  showGrid: true,
  sidebarPanel: 'hierarchy',

  ribbonTab: 'home',
  panels: { library: true, explorer: true, properties: true, output: true },
  dockTab: 'output',
  snapEnabled: true,
  moveStep: 1,
  rotateStep: 15,
  space: 'world',
  cameraView: 'perspective',

  selectEntity: (id) => set({ selectedEntityId: id }),
  setHovered: (id) => set({ hoveredEntityId: id }),
  setTransformMode: (mode) => set({ transformMode: mode, selectOnly: false }),
  setSelectOnly: () => set({ selectOnly: true }),
  setPlaying: (playing) => set({ isPlaying: playing }),
  toggleGrid: () => set((s) => ({ showGrid: !s.showGrid })),
  setSidebarPanel: (panel) => set({ sidebarPanel: panel }),
  setRibbonTab: (tab) => set({ ribbonTab: tab }),
  togglePanel: (panel) => set((s) => ({ panels: { ...s.panels, [panel]: !s.panels[panel] } })),
  showPanel: (panel) => set((s) => ({ panels: { ...s.panels, [panel]: true } })),
  setDockTab: (tab) => set((s) => ({ dockTab: tab, panels: { ...s.panels, output: true } })),
  toggleSnap: () => set((s) => ({ snapEnabled: !s.snapEnabled })),
  cycleMoveStep: () =>
    set((s) => {
      const i = MOVE_STEPS.indexOf(s.moveStep as (typeof MOVE_STEPS)[number]);
      return { moveStep: MOVE_STEPS[(i + 1) % MOVE_STEPS.length]!, snapEnabled: true };
    }),
  cycleRotateStep: () =>
    set((s) => {
      const i = ROTATE_STEPS.indexOf(s.rotateStep as (typeof ROTATE_STEPS)[number]);
      return { rotateStep: ROTATE_STEPS[(i + 1) % ROTATE_STEPS.length]!, snapEnabled: true };
    }),
  setSpace: (space) => set({ space }),
  setCameraView: (view) => set({ cameraView: view }),
}));
