import { create } from 'zustand';
import type {
  ExperienceFormatType,
  ExperienceFormat,
  CameraKeyframe,
  InteractionDef,
} from '@forge3d/shared';

interface FormatState {
  formatType: ExperienceFormatType;

  // Video format
  keyframes: CameraKeyframe[];
  selectedKeyframeIndex: number | null;
  previewPlaying: boolean;

  // Interactive format
  interactions: InteractionDef[];

  // Animated format
  animationDuration: number;

  // Turntable format
  turntableSpeed: number;
  turntableAxis: 'y' | 'x';

  // Actions
  setFormatType: (type: ExperienceFormatType) => void;

  // Keyframe actions
  addKeyframe: (kf: CameraKeyframe) => void;
  updateKeyframe: (index: number, partial: Partial<CameraKeyframe>) => void;
  removeKeyframe: (index: number) => void;
  selectKeyframe: (index: number | null) => void;
  setPreviewPlaying: (playing: boolean) => void;

  // Interaction actions
  addInteraction: (def: InteractionDef) => void;
  updateInteraction: (entityId: string, partial: Partial<InteractionDef>) => void;
  removeInteraction: (entityId: string) => void;

  // Config actions
  setAnimationDuration: (d: number) => void;
  setTurntableSpeed: (s: number) => void;
  setTurntableAxis: (a: 'y' | 'x') => void;

  /** Build the final ExperienceFormat object for publishing. */
  getFormatConfig: () => ExperienceFormat;
  reset: () => void;
}

const initialState = {
  formatType: 'turntable' as ExperienceFormatType,
  keyframes: [] as CameraKeyframe[],
  selectedKeyframeIndex: null as number | null,
  previewPlaying: false,
  interactions: [] as InteractionDef[],
  animationDuration: 5,
  turntableSpeed: 1,
  turntableAxis: 'y' as 'y' | 'x',
};

export const useFormatStore = create<FormatState>((set, get) => ({
  ...initialState,

  setFormatType: (type) => set({ formatType: type }),

  // Keyframes
  addKeyframe: (kf) =>
    set((s) => ({
      keyframes: [...s.keyframes, kf].sort((a, b) => a.time - b.time),
    })),
  updateKeyframe: (index, partial) =>
    set((s) => ({
      keyframes: s.keyframes.map((kf, i) =>
        i === index ? { ...kf, ...partial } : kf,
      ).sort((a, b) => a.time - b.time),
    })),
  removeKeyframe: (index) =>
    set((s) => ({
      keyframes: s.keyframes.filter((_, i) => i !== index),
      selectedKeyframeIndex:
        s.selectedKeyframeIndex === index ? null : s.selectedKeyframeIndex,
    })),
  selectKeyframe: (index) => set({ selectedKeyframeIndex: index }),
  setPreviewPlaying: (playing) => set({ previewPlaying: playing }),

  // Interactions
  addInteraction: (def) =>
    set((s) => ({ interactions: [...s.interactions, def] })),
  updateInteraction: (entityId, partial) =>
    set((s) => ({
      interactions: s.interactions.map((d) =>
        d.entityId === entityId ? { ...d, ...partial } : d,
      ),
    })),
  removeInteraction: (entityId) =>
    set((s) => ({
      interactions: s.interactions.filter((d) => d.entityId !== entityId),
    })),

  // Config
  setAnimationDuration: (d) => set({ animationDuration: d }),
  setTurntableSpeed: (s) => set({ turntableSpeed: s }),
  setTurntableAxis: (a) => set({ turntableAxis: a }),

  getFormatConfig: (): ExperienceFormat => {
    const s = get();
    switch (s.formatType) {
      case 'turntable':
        return { type: 'turntable', speed: s.turntableSpeed, axis: s.turntableAxis };
      case 'video':
        return { type: 'video', keyframes: s.keyframes };
      case 'animated':
        return { type: 'animated', duration: s.animationDuration };
      case 'interactive':
        return { type: 'interactive', interactions: s.interactions };
      default:
        return { type: 'turntable', speed: 1, axis: 'y' };
    }
  },

  reset: () => set(initialState),
}));
