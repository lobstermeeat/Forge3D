export type EasingName = 'linear' | 'ease-in' | 'ease-out' | 'ease-in-out';

const easings: Record<EasingName, (t: number) => number> = {
  'linear': (t) => t,
  'ease-in': (t) => t * t,
  'ease-out': (t) => t * (2 - t),
  'ease-in-out': (t) => (t < 0.5 ? 2 * t * t : -1 + (4 - 2 * t) * t),
};

/** Apply easing to a normalized t value (0-1). */
export function ease(name: EasingName, t: number): number {
  const clamped = Math.max(0, Math.min(1, t));
  return easings[name](clamped);
}
