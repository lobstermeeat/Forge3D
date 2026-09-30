import { describe, expect, it } from 'vitest';
import { modelName } from './modelName';

describe('modelName', () => {
  it('names a model after the first whole words of its prompt', () => {
    expect(modelName('a wooden treasure chest with iron bands')).toBe('Wooden treasure chest');
    expect(modelName('The red vintage scooter')).toBe('Red vintage scooter');
    expect(modelName('medieval longsword with a leather-wrapped grip')).toBe('Medieval longsword');
  });

  it('copes with photos, long words and odd spacing', () => {
    expect(modelName(null)).toBe('Photo model');
    expect(modelName('   ')).toBe('Photo model');
    expect(modelName('supercalifragilisticexpialidocious lamp')).toBe(
      'Supercalifragilisticexpialid',
    );
    expect(modelName('  a   brass   pocket watch ')).toBe('Brass pocket watch');
  });
});
