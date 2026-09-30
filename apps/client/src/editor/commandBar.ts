import type { LightData } from '@forge3d/shared';
import type { EditorActions } from '@/hooks/useEditorActions';
import type { SceneManager } from '@forge3d/engine';
import { useEditorStore, type CameraView } from '@/stores/editorStore';
import { useOutputStore } from '@/stores/outputStore';
import { PRIMITIVES } from './placement';

interface CommandContext {
  actions: EditorActions;
  sceneManager: SceneManager;
  play: () => void;
  stop: () => void;
}

const LIGHT_WORDS: Record<string, LightData['type']> = {
  sun: 'directional',
  directional: 'directional',
  point: 'point',
  bulb: 'point',
  ambient: 'ambient',
};

const HELP =
  'Commands: insert <box | sphere | cylinder | plane | torus> · light <sun | point | ambient> · ' +
  'select <name> · delete · duplicate · focus · view <perspective | top | front | side> · ' +
  'grid · undo · redo · play · stop · clear';

/** Runs one line typed into the Output command bar. */
export function runCommand(raw: string, ctx: CommandContext): void {
  const out = useOutputStore.getState();
  const line = raw.trim();
  if (!line) return;
  out.log('cmd', `› ${line}`);

  const [head = '', ...rest] = line.split(/\s+/);
  const verb = head.toLowerCase();
  const arg = rest.join(' ').toLowerCase();
  const a = ctx.actions;
  const playing = useEditorStore.getState().isPlaying;

  if (playing && !['stop', 'help', 'clear'].includes(verb)) {
    out.log('warn', 'Stop play-testing first (Esc)');
    return;
  }

  switch (verb) {
    case 'help':
      out.log('info', HELP);
      return;
    case 'insert':
    case 'add':
    case 'part': {
      const p = PRIMITIVES.find(
        (x) =>
          x.type === arg || x.label.toLowerCase() === arg || (arg === 'block' && x.type === 'box'),
      );
      if (p) a.addPrimitive(p.type);
      else out.log('warn', `No part called “${arg}” — try box, sphere, cylinder, plane or torus`);
      return;
    }
    case 'light': {
      const type = LIGHT_WORDS[arg || 'point'];
      if (type) a.addLight(type);
      else out.log('warn', `No light called “${arg}” — try sun, point or ambient`);
      return;
    }
    case 'select': {
      const all = ctx.sceneManager.getAllEntities();
      const hit =
        all.find((e) => e.name.toLowerCase() === arg) ??
        all.find((e) => arg && e.name.toLowerCase().includes(arg));
      if (hit) {
        a.select(hit.id);
        out.log('info', `Selected ${hit.name}`);
      } else out.log('warn', `Nothing is named “${rest.join(' ')}”`);
      return;
    }
    case 'delete':
    case 'remove':
      a.remove();
      return;
    case 'duplicate':
      a.duplicate();
      return;
    case 'focus':
    case 'frame':
      a.focus();
      return;
    case 'view': {
      const views: CameraView[] = ['perspective', 'top', 'front', 'side'];
      const v = views.find((x) => x === arg);
      if (v) a.setView(v);
      else out.log('warn', 'Views: perspective, top, front, side');
      return;
    }
    case 'grid':
      useEditorStore.getState().toggleGrid();
      return;
    case 'undo':
      a.undo();
      return;
    case 'redo':
      a.redo();
      return;
    case 'play':
      ctx.play();
      return;
    case 'stop':
      ctx.stop();
      return;
    case 'clear':
      out.clear();
      return;
    default:
      out.log('warn', `Unknown command “${head}” — type help`);
  }
}
