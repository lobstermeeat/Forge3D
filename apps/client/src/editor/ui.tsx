import { useEffect, useRef, useState } from 'react';
import type { ReactNode } from 'react';
import { Icon } from './Icon';

/** Closes a popover on outside pointerdown or Escape. */
export function useDismiss(open: boolean, onClose: () => void) {
  const ref = useRef<HTMLDivElement>(null);
  const closeRef = useRef(onClose);
  closeRef.current = onClose;
  useEffect(() => {
    if (!open) return;
    const onDown = (e: PointerEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) closeRef.current();
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        e.stopPropagation();
        closeRef.current();
      }
    };
    document.addEventListener('pointerdown', onDown, true);
    document.addEventListener('keydown', onKey, true);
    return () => {
      document.removeEventListener('pointerdown', onDown, true);
      document.removeEventListener('keydown', onKey, true);
    };
  }, [open]);
  return ref;
}

/**
 * A trigger plus a popover menu. `trigger` receives the open state and a toggle;
 * `children` receives a close function for items that should dismiss the menu.
 */
export function MenuAnchor({
  trigger,
  children,
  popStyle,
  className,
}: {
  trigger: (open: boolean, toggle: () => void) => ReactNode;
  children: (close: () => void) => ReactNode;
  popStyle?: React.CSSProperties;
  className?: string;
}) {
  const [open, setOpen] = useState(false);
  const close = () => setOpen(false);
  const ref = useDismiss(open, close);
  return (
    <div ref={ref} className={className ?? 'f3-anchor'}>
      {trigger(open, () => setOpen((v) => !v))}
      {open && (
        <div className="f3-pop" role="menu" style={popStyle}>
          {children(close)}
        </div>
      )}
    </div>
  );
}

export function MenuItem({
  icon,
  label,
  shortcut,
  checked,
  disabled,
  onSelect,
}: {
  icon?: string;
  label: string;
  shortcut?: string;
  checked?: boolean;
  disabled?: boolean;
  onSelect: () => void;
}) {
  return (
    <button
      type="button"
      className="f3-pi"
      role={checked === undefined ? 'menuitem' : 'menuitemradio'}
      aria-checked={checked}
      disabled={disabled}
      onClick={onSelect}
    >
      {icon ? <Icon name={checked ? 'check' : icon} size={15} /> : null}
      <span>{label}</span>
      {shortcut ? <span className="k">{shortcut}</span> : null}
    </button>
  );
}

export function Section({
  title,
  children,
  defaultOpen = true,
}: {
  title: string;
  children: ReactNode;
  defaultOpen?: boolean;
}) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div className="f3-sec">
      <button type="button" className="f3-sech" aria-expanded={open} onClick={() => setOpen(!open)}>
        <Icon name="caret" size={10} style={{ transform: `rotate(${open ? 90 : 0}deg)` }} />
        {title}
      </button>
      {open && <div className="f3-sbody">{children}</div>}
    </div>
  );
}

export function PropRow({
  label,
  children,
  variant,
}: {
  label: string;
  children: ReactNode;
  variant?: 'v' | 'top';
}) {
  return (
    <div className={variant ? `f3-pr ${variant}` : 'f3-pr'}>
      <span className="f3-pl" title={label}>
        {label}
      </span>
      {children}
    </div>
  );
}

export function Select<T extends string>({
  value,
  options,
  onChange,
  label,
}: {
  value: T;
  options: { value: T; label: string }[];
  onChange: (value: T) => void;
  label: string;
}) {
  return (
    <div className="f3-dd">
      <select aria-label={label} value={value} onChange={(e) => onChange(e.target.value as T)}>
        {options.map((o) => (
          <option key={o.value} value={o.value}>
            {o.label}
          </option>
        ))}
      </select>
      <Icon name="chevdown" size={12} />
    </div>
  );
}

export function Toggle({
  on,
  onChange,
  label,
}: {
  on: boolean;
  onChange: (on: boolean) => void;
  label: string;
}) {
  return (
    <button
      type="button"
      className="f3-tg"
      aria-pressed={on}
      aria-label={label}
      onClick={() => onChange(!on)}
    />
  );
}
