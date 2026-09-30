const FILLER = new Set(['a', 'an', 'the', 'and', 'with', 'of', 'on', 'in', 'to', 'for']);
const MAX_LENGTH = 28;

/**
 * A scene name for a generated model: the prompt's first whole words, without a leading article
 * or a dangling "with". Photo runs (no prompt) are "Photo model".
 */
export function modelName(prompt: string | null | undefined): string {
  const words = (prompt ?? '')
    .trim()
    .replace(/^(an?|the)\s+/i, '')
    .split(/\s+/)
    .filter(Boolean);
  if (words.length === 0) return 'Photo model';
  const kept: string[] = [];
  for (const word of words) {
    if ([...kept, word].join(' ').length > MAX_LENGTH) break;
    kept.push(word);
  }
  while (kept.length > 1 && FILLER.has(kept[kept.length - 1]!.toLowerCase())) kept.pop();
  const name = kept.join(' ') || words[0]!.slice(0, MAX_LENGTH);
  return name.charAt(0).toUpperCase() + name.slice(1);
}
