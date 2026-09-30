import { create } from 'zustand';

export type LogLevel = 'info' | 'ok' | 'warn' | 'error' | 'cmd';

export interface LogLine {
  id: number;
  time: string;
  level: LogLevel;
  text: string;
}

const MAX_LINES = 300;
let nextId = 1;

function clock(): string {
  const d = new Date();
  const p = (n: number) => String(n).padStart(2, '0');
  return `${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}`;
}

interface OutputState {
  lines: LogLine[];
  log: (level: LogLevel, text: string) => void;
  clear: () => void;
}

export const useOutputStore = create<OutputState>((set) => ({
  lines: [],
  log: (level, text) =>
    set((s) => {
      const lines = [...s.lines, { id: nextId++, time: clock(), level, text }];
      return { lines: lines.length > MAX_LINES ? lines.slice(-MAX_LINES) : lines };
    }),
  clear: () => set({ lines: [] }),
}));

/** Write a line to the editor's Output panel from anywhere. */
export function logOutput(level: LogLevel, text: string): void {
  useOutputStore.getState().log(level, text);
}
