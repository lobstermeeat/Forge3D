// @vitest-environment jsdom
import { afterEach, describe, expect, it, vi } from 'vitest';
import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import type { TextureGeneration } from '@/editor/textureOptions';
import { TextureOptions } from './TextureOptions';

const final = { url: 'https://files.test/ai/g1/final-42.glb', triangles: 100_000 };
const optionUrl = (k: number) =>
  `https://files.test/ai/g1/final-42-texture-${1042 + 1000 * (k - 1)}.glb`;

function generation(textures: TextureGeneration['textures']): TextureGeneration {
  return { status: 'done', final, textures };
}

const done = generation({
  status: 'done',
  count: 3,
  options: [1, 2, 3].map((k) => ({
    url: optionUrl(k),
    triangles: 96_000,
    textureSeed: 42 + 1000 * k,
  })),
  error: null,
});

afterEach(cleanup);

describe('TextureOptions', () => {
  it('says the textures are being made, without anything to pick yet', () => {
    render(
      <TextureOptions
        generation={generation({ status: 'running', count: 3, options: [], error: null })}
        sceneUrl={final.url}
        onChoose={() => {}}
      />,
    );
    expect(screen.getByText('Making 3 more textures to choose from…')).toBeTruthy();
    expect(screen.queryByRole('radio')).toBeNull();
  });

  it('offers 4 textures, the one in the scene pressed, and picking one hands it over', () => {
    const onChoose = vi.fn();
    const { rerender } = render(
      <TextureOptions generation={done} sceneUrl={final.url} onChoose={onChoose} />,
    );
    const radios = screen.getAllByRole('radio');
    expect(radios.map((radio) => radio.textContent)).toEqual(['1', '2', '3', '4']);
    // The final's own texture is in the scene to begin with
    expect(radios.map((radio) => radio.getAttribute('aria-checked'))).toEqual([
      'true',
      'false',
      'false',
      'false',
    ]);
    expect(radios[0]!.getAttribute('aria-label')).toBe('Texture 1, the final’s own');

    fireEvent.click(radios[2]!);
    expect(onChoose).toHaveBeenCalledWith({ number: 3, url: optionUrl(2) });

    // The scene now shows texture 3
    rerender(<TextureOptions generation={done} sceneUrl={optionUrl(2)} onChoose={onChoose} />);
    expect(screen.getAllByRole('radio').map((radio) => radio.getAttribute('aria-pressed'))).toEqual(
      ['false', 'false', 'true', 'false'],
    );

    // Not in the scene: none is pressed, and picking one still hands it over (to place it)
    rerender(<TextureOptions generation={done} sceneUrl={null} onChoose={onChoose} />);
    expect(
      screen.getAllByRole('radio').some((radio) => radio.getAttribute('aria-checked') === 'true'),
    ).toBe(false);
    fireEvent.click(screen.getAllByRole('radio')[0]!);
    expect(onChoose).toHaveBeenLastCalledWith({ number: 1, url: final.url });
  });

  it("can't be used while the scene plays", () => {
    const onChoose = vi.fn();
    render(<TextureOptions generation={done} sceneUrl={final.url} disabled onChoose={onChoose} />);
    const radios = screen.getAllByRole('radio') as HTMLButtonElement[];
    expect(radios.every((radio) => radio.disabled)).toBe(true);
    fireEvent.click(radios[1]!);
    expect(onChoose).not.toHaveBeenCalled();
  });

  it('notes quietly when no textures could be made, and shows nothing when none were asked', () => {
    const { container, rerender } = render(
      <TextureOptions
        generation={generation({
          status: 'failed',
          count: 3,
          options: [],
          error: 'texture options need TRELLIS.2 finals',
        })}
        sceneUrl={final.url}
        onChoose={() => {}}
      />,
    );
    const note = screen.getByText('More textures couldn’t be made this time.');
    expect(note.getAttribute('title')).toBe('texture options need TRELLIS.2 finals');
    expect(screen.queryByRole('radio')).toBeNull();

    rerender(
      <TextureOptions generation={generation(null)} sceneUrl={final.url} onChoose={() => {}} />,
    );
    expect(container.textContent).toBe('');
  });
});
