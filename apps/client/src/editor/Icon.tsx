/**
 * Stroke icons for the editor, drawn on a 24px grid.
 * Render <IconSprite /> once per page, then use <Icon name="move" />.
 */
const PATHS: Record<string, string> = {
  select: '<path d="M6 3.5l12 6.8-5.4 1.4-2.4 5.3L6 3.5z"/>',
  move: '<path d="M12 3.5v17M3.5 12h17M12 3.5L9.8 5.7M12 3.5l2.2 2.2M12 20.5l-2.2-2.2M12 20.5l2.2-2.2M3.5 12l2.2-2.2M3.5 12l2.2 2.2M20.5 12l-2.2-2.2M20.5 12l-2.2 2.2"/>',
  rotate: '<path d="M19.5 12a7.5 7.5 0 1 1-2.2-5.3"/><path d="M19.8 4.2v3.6h-3.6"/>',
  scale: '<path d="M4 20l6.5-6.5M4 20v-5M4 20h5"/><path d="M20 4l-6.5 6.5M20 4v5M20 4h-5"/>',
  box: '<path d="M12 3.2l7.8 4.4v8.8L12 20.8l-7.8-4.4V7.6z"/><path d="M4.2 7.6L12 12l7.8-4.4M12 12v8.8"/>',
  sphere:
    '<circle cx="12" cy="12" r="8.5"/><path d="M3.5 12c0 1.9 3.8 3.4 8.5 3.4s8.5-1.5 8.5-3.4"/>',
  cylinder:
    '<ellipse cx="12" cy="6" rx="7" ry="2.5"/><path d="M5 6v12c0 1.4 3.1 2.5 7 2.5s7-1.1 7-2.5V6"/>',
  plane: '<path d="M3 15.5l6-6.5h12l-6 6.5z"/>',
  torus: '<ellipse cx="12" cy="12" rx="9" ry="5.8"/><ellipse cx="12" cy="11.5" rx="3.8" ry="1.9"/>',
  sun: '<circle cx="12" cy="12" r="3.8"/><path d="M12 2.8v2.4M12 18.8v2.4M2.8 12h2.4M18.8 12h2.4M5.5 5.5l1.7 1.7M16.8 16.8l1.7 1.7M5.5 18.5l1.7-1.7M16.8 7.2l1.7-1.7"/>',
  bulb: '<path d="M9.2 17.5h5.6M10.2 20.5h3.6"/><path d="M12 3.5a5.8 5.8 0 0 0-3.4 10.5c.5.4.6.9.6 1.5v2h5.6v-2c0-.6.1-1.1.6-1.5A5.8 5.8 0 0 0 12 3.5z"/>',
  ambient:
    '<circle cx="12" cy="12" r="3"/><circle cx="12" cy="12" r="7.5" stroke-dasharray="2 3"/>',
  spot: '<path d="M9 4h6l-1 5h-4z"/><path d="M10 9l-5 11h14L14 9"/>',
  import:
    '<path d="M12 3.5v11M7.8 10.6l4.2 4.2 4.2-4.2"/><path d="M4.5 14.5v4a1.5 1.5 0 0 0 1.5 1.5h12a1.5 1.5 0 0 0 1.5-1.5v-4"/>',
  export:
    '<path d="M12 14.5V3.5M8 7.5l4-4 4 4"/><path d="M4.5 14.5v4a1.5 1.5 0 0 0 1.5 1.5h12a1.5 1.5 0 0 0 1.5-1.5v-4"/>',
  open: '<path d="M3.5 7.2a1.4 1.4 0 0 1 1.4-1.4h4.3l1.9 2h7.1a1.4 1.4 0 0 1 1.4 1.4V10"/><path d="M3.5 18.2l2.2-7.3a1.4 1.4 0 0 1 1.3-.9h12.6a1 1 0 0 1 1 1.3l-2 6.9a1.4 1.4 0 0 1-1.3 1H4.9a1.4 1.4 0 0 1-1.4-1z"/>',
  save: '<path d="M5 3.5h11l3.5 3.5v12a1.5 1.5 0 0 1-1.5 1.5H6A1.5 1.5 0 0 1 4.5 19V5A1.5 1.5 0 0 1 6 3.5z"/><path d="M8 3.5v5h7v-5M8 20.5v-6h8v6"/>',
  undo: '<path d="M9 6.5L4.5 11 9 15.5"/><path d="M4.5 11h10a5 5 0 0 1 0 10H12"/>',
  redo: '<path d="M15 6.5l4.5 4.5-4.5 4.5"/><path d="M19.5 11h-10a5 5 0 0 0 0 10H12"/>',
  play: '<path d="M8 5.2v13.6L19 12z"/>',
  stop: '<rect x="6.5" y="6.5" width="11" height="11" rx="1.6"/>',
  pause:
    '<rect x="7" y="5.5" width="3.5" height="13" rx="1"/><rect x="13.5" y="5.5" width="3.5" height="13" rx="1"/>',
  grid: '<rect x="4" y="4" width="16" height="16" rx="1.5"/><path d="M4 9.3h16M4 14.7h16M9.3 4v16M14.7 4v16"/>',
  snap: '<path d="M6 4.5V12a6 6 0 0 0 12 0V4.5h-3.8V12a2.2 2.2 0 0 1-4.4 0V4.5z"/><path d="M6 8.3h3.8M14.2 8.3H18"/>',
  focus:
    '<path d="M4 9V5.5A1.5 1.5 0 0 1 5.5 4H9M15 4h3.5A1.5 1.5 0 0 1 20 5.5V9M20 15v3.5a1.5 1.5 0 0 1-1.5 1.5H15M9 20H5.5A1.5 1.5 0 0 1 4 18.5V15"/><circle cx="12" cy="12" r="2.6"/>',
  trash:
    '<path d="M4.5 7h15M9.5 7V4.8h5V7M6.5 7l.9 12.2a1.3 1.3 0 0 0 1.3 1.3h6.6a1.3 1.3 0 0 0 1.3-1.3L17.5 7"/>',
  duplicate:
    '<rect x="8.5" y="8.5" width="11" height="11" rx="1.6"/><path d="M4.5 15.5V5.8a1.3 1.3 0 0 1 1.3-1.3h9.7"/><path d="M14 11.5v5M11.5 14h5"/>',
  drop: '<path d="M12 3.5v11M8 10.5l4 4 4-4M4 20h16"/>',
  caret: '<path d="M9 6l6 6-6 6"/>',
  chevdown: '<path d="M6 9.5l6 6 6-6"/>',
  search: '<circle cx="11" cy="11" r="6.5"/><path d="M20 20l-4.2-4.2"/>',
  close: '<path d="M6 6l12 12M18 6L6 18"/>',
  check: '<path d="M5 12.5l4.5 4.5L19 7.5"/>',
  info: '<circle cx="12" cy="12" r="8.5"/><path d="M12 11v5.5M12 7.8v.2"/>',
  warn: '<path d="M12 4.2l8.8 15.3H3.2z"/><path d="M12 10v4.2M12 17v.2"/>',
  ok: '<circle cx="12" cy="12" r="8.5"/><path d="M8.3 12.3l2.5 2.5 5-5.2"/>',
  error: '<circle cx="12" cy="12" r="8.5"/><path d="M9 9l6 6M15 9l-6 6"/>',
  cmd: '<path d="M5 8l4.5 4L5 16M11.5 16.5H19"/>',
  camera:
    '<rect x="3" y="7" width="12.5" height="10" rx="1.8"/><path d="M15.5 10.5l5.5-3v9l-5.5-3"/>',
  scene:
    '<path d="M12 3.5l8.5 4.3L12 12 3.5 7.8z"/><path d="M3.5 12.2L12 16.4l8.5-4.2M3.5 16.4L12 20.6l8.5-4.2"/>',
  group:
    '<path d="M3.5 7.2a1.4 1.4 0 0 1 1.4-1.4h4.3l1.9 2h8.1a1.4 1.4 0 0 1 1.4 1.4v8.7a1.4 1.4 0 0 1-1.4 1.4H4.9a1.4 1.4 0 0 1-1.4-1.4z"/>',
  model:
    '<path d="M12 3l8 4.5v9L12 21l-8-4.5v-9z"/><path d="M12 12l8-4.5M12 12v9M12 12L4 7.5"/><path d="M8 5.2l8 4.6"/>',
  library:
    '<rect x="4" y="4" width="7" height="7" rx="1.4"/><rect x="13" y="4" width="7" height="7" rx="1.4"/><rect x="4" y="13" width="7" height="7" rx="1.4"/><rect x="13" y="13" width="7" height="7" rx="1.4"/>',
  eye: '<path d="M2.5 12S6 5.8 12 5.8 21.5 12 21.5 12 18 18.2 12 18.2 2.5 12 2.5 12z"/><circle cx="12" cy="12" r="2.8"/>',
  sliders:
    '<path d="M4 7h9M17 7h3M4 17h3M11 17h9"/><circle cx="15" cy="7" r="2"/><circle cx="9" cy="17" r="2"/>',
  timeline: '<path d="M4 7h16M4 12h16M4 17h16"/><path d="M8 5v4M15 10v4M11 15v4"/>',
  terminal:
    '<rect x="3.5" y="4.5" width="17" height="15" rx="2"/><path d="M7.5 9.5L10 12l-2.5 2.5M12.5 15H16"/>',
  publish:
    '<path d="M12 15V4M7.5 8.5L12 4l4.5 4.5"/><path d="M5 13.5v5A1.5 1.5 0 0 0 6.5 20h11a1.5 1.5 0 0 0 1.5-1.5v-5"/>',
  bolt: '<path d="M13 3L5.5 13.2h5.8L10.5 21 18 10.8h-5.8z"/>',
  globe:
    '<circle cx="12" cy="12" r="8.5"/><path d="M3.5 12h17M12 3.5c2.4 2.4 3.6 5.2 3.6 8.5S14.4 18.1 12 20.5C9.6 18.1 8.4 15.3 8.4 12S9.6 5.9 12 3.5z"/>',
  local: '<path d="M12 12V4.5M12 12l6.5 3.8M12 12l-6.5 3.8"/><circle cx="12" cy="12" r="1.6"/>',
  reset:
    '<path d="M4.5 12a7.5 7.5 0 1 0 2.2-5.3"/><path d="M4.2 4.2v3.6h3.6"/><circle cx="12" cy="12" r="1.8"/>',
  panels: '<rect x="3.5" y="4.5" width="17" height="15" rx="2"/><path d="M9 4.5v15M15 4.5v15"/>',
  collapse: '<rect x="4" y="4" width="16" height="16" rx="2.5"/><path d="M8.5 12h7"/>',
  users:
    '<circle cx="9" cy="8.5" r="3"/><path d="M3.5 19a5.5 5.5 0 0 1 11 0"/><path d="M15.5 5.8a3 3 0 0 1 0 5.4M17.5 14.2a5.5 5.5 0 0 1 3 4.8"/>',
  palette:
    '<path d="M12 3.5a8.5 8.5 0 0 0 0 17c1.2 0 1.8-.8 1.8-1.7 0-1.3-1.2-1.6-1.2-2.7 0-1 .8-1.6 1.8-1.6h2.1a4 4 0 0 0 4-4c0-3.9-3.8-7-8.5-7z"/><circle cx="7.8" cy="11" r="1"/><circle cx="10.5" cy="7.5" r="1"/><circle cx="15" cy="8" r="1"/>',
  material:
    '<circle cx="12" cy="12" r="8.5"/><path d="M7.5 9a5 5 0 0 1 4-3"/><path d="M3.8 13.8c2.8 1.6 13.6 1.6 16.4 0"/>',
  keyadd: '<path d="M10 5l6.5 6.5L10 18l-6.5-6.5z"/><path d="M18.5 3.5v5M16 6h5"/>',
  link: '<path d="M10 14a4 4 0 0 0 5.7 0l3-3a4 4 0 0 0-5.7-5.7l-1.2 1.2"/><path d="M14 10a4 4 0 0 0-5.7 0l-3 3a4 4 0 0 0 5.7 5.7l1.2-1.2"/>',
};

export type IconName = keyof typeof PATHS;

const FILLED = new Set(['play', 'stop', 'pause']);

export function IconSprite() {
  const symbols = Object.entries(PATHS)
    .map(([name, d]) => `<symbol id="f3i-${name}" viewBox="0 0 24 24">${d}</symbol>`)
    .join('');
  return (
    <svg
      width="0"
      height="0"
      aria-hidden="true"
      style={{ position: 'absolute', overflow: 'hidden' }}
      dangerouslySetInnerHTML={{ __html: symbols }}
    />
  );
}

export function Icon({
  name,
  size = 16,
  className,
  style,
}: {
  name: IconName | string;
  size?: number;
  className?: string;
  style?: React.CSSProperties;
}) {
  const cls = FILLED.has(name) ? 'f3-icf' : 'f3-ic';
  return (
    <svg
      className={className ? `${cls} ${className}` : cls}
      width={size}
      height={size}
      aria-hidden="true"
      focusable="false"
      style={style}
    >
      <use href={`#f3i-${name}`} />
    </svg>
  );
}
