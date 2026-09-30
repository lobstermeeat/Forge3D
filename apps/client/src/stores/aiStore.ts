import { create } from 'zustand';

/** The AI panel's state that outlives the panel (it unmounts when the Library shows another tab). */
interface AIState {
  /** The generation the panel follows */
  currentId: string | null;
  mode: 'describe' | 'photo';
  prompt: string;
  photo: { dataUrl: string; name: string } | null;
  /** Generations seen running this session: when they finish, the panel places them itself */
  watched: Record<string, true>;
  /** Whether the panel has looked for an unfinished generation since the page loaded */
  restored: boolean;
  setCurrentId: (id: string | null) => void;
  setMode: (mode: AIState['mode']) => void;
  setPrompt: (prompt: string) => void;
  setPhoto: (photo: AIState['photo']) => void;
  watch: (id: string) => void;
  markRestored: () => void;
}

export const useAIStore = create<AIState>((set) => ({
  currentId: null,
  mode: 'describe',
  prompt: '',
  photo: null,
  watched: {},
  restored: false,
  setCurrentId: (id) => set({ currentId: id }),
  setMode: (mode) => set({ mode }),
  setPrompt: (prompt) => set({ prompt }),
  setPhoto: (photo) => set({ photo }),
  watch: (id) => set((s) => (s.watched[id] ? s : { watched: { ...s.watched, [id]: true } })),
  markRestored: () => set({ restored: true }),
}));
