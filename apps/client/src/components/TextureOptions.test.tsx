// @vitest-environment jsdom
import { afterEach, describe, expect, it, vi } from 'vitest';
import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import type { TextureGeneration } from '@/editor/textureOptions';
import { TextureOptions } from './TextureOptions';

type Textures = NonNullable<TextureGeneration['textures']>;

const final = { url: 'https://files.test/ai/g1/final-42.glb', triangles: 100_000 };
const optionUrl = (k: number) =>
  `https://files.test/ai/g1/final-42-texture-${1042 + 1000 * (k - 1)}.glb`;

function generation(textures: TextureGeneration['textures']): TextureGeneration {
  return { status: 'done', final, textures };
}

/** The texture options as the server shows them, without the judge unless `more` says */
function texturesOf(status: Textures['status'], more: Partial<Textures> = {}): Textures {
  return {
    status,
    count: 3,
    options: [],
    error: null,
    recommended: null,
    chosen: null,
    judgeError: null,
    ...more,
  };
}

const options = (count: number) =>
  Array.from({ length: count }, (_, i) => ({
    url: optionUrl(i + 1),
    triangles: 96_000,
    textureSeed: 42 + 1000 * (i + 1),
  }));

const done = generation(texturesOf('done', { options: options(3) }));

const why = 'The back keeps the brass colour, with no smudges.';
/** Done, with the judge's pick (1 is the final's own texture) */
const judged = (number: number, count = 3) =>
  generation(
    texturesOf('done', { options: options(count), recommended: { number, why, applied: false } }),
  );

afterEach(cleanup);

describe('TextureOptions', () => {
  it('says the textures are being made, without anything to pick yet', () => {
    render(
      <TextureOptions
        generation={generation(texturesOf('running'))}
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
    expect(
      screen.getByText('The same shape with other textures. Pick the one you like best.'),
    ).toBeTruthy();

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
        generation={generation(
          texturesOf('failed', { error: 'texture options need TRELLIS.2 finals' }),
        )}
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

  describe('with the judge (AI_TEXTURE_JUDGE=1)', () => {
    it('marks the recommended texture with a dot, and says why in its tooltip', () => {
      const onChoose = vi.fn();
      render(<TextureOptions generation={judged(3)} sceneUrl={final.url} onChoose={onChoose} />);
      const radios = screen.getAllByRole('radio');
      expect(radios.map((radio) => radio.hasAttribute('data-recommended'))).toEqual([
        false,
        false,
        true,
        false,
      ]);
      expect(radios[2]!.getAttribute('aria-label')).toBe('Texture 3, recommended');
      expect(radios[2]!.getAttribute('title')).toBe(`Recommended: ${why}`);
      expect(radios.filter((radio) => radio.hasAttribute('title'))).toHaveLength(1);
      // Not switched to (yet, or the creator chose another): the line says which it is
      expect(
        screen.getByText('The same shape with other textures. Texture 3 is recommended.'),
      ).toBeTruthy();
      // It is picked like any other
      fireEvent.click(radios[2]!);
      expect(onChoose).toHaveBeenCalledWith({ number: 3, url: optionUrl(2) });
    });

    it('says when the panel switched the model to it', () => {
      const { rerender } = render(
        <TextureOptions
          generation={judged(3)}
          sceneUrl={optionUrl(2)}
          switched
          onChoose={() => {}}
        />,
      );
      const note = screen.getByText(
        'Switched to texture 3, the cleanest of the four. Pick another any time.',
      );
      expect(note.getAttribute('title')).toBe(why);
      expect(screen.getAllByRole('radio')[2]!.getAttribute('aria-pressed')).toBe('true');

      // Fewer textures when some failed: of the three, or the better of two
      rerender(
        <TextureOptions
          generation={judged(3, 2)}
          sceneUrl={optionUrl(2)}
          switched
          onChoose={() => {}}
        />,
      );
      expect(
        screen.getByText(
          'Switched to texture 3, the cleanest of the three. Pick another any time.',
        ),
      ).toBeTruthy();
      rerender(
        <TextureOptions
          generation={judged(2, 1)}
          sceneUrl={optionUrl(1)}
          switched
          onChoose={() => {}}
        />,
      );
      expect(
        screen.getByText('Switched to texture 2, the cleaner of the two. Pick another any time.'),
      ).toBeTruthy();
    });

    it("looks as without the judge when it picked the final's own texture, or gave no pick", () => {
      const own = judged(1);
      const failed = generation(
        texturesOf('done', { options: options(3), judgeError: 'judge failed: timed out' }),
      );
      for (const g of [own, failed]) {
        const { container, unmount } = render(
          <TextureOptions generation={g} sceneUrl={final.url} switched onChoose={() => {}} />,
        );
        expect(container.querySelector('[data-recommended]')).toBeNull();
        expect(screen.getAllByRole('radio').some((radio) => radio.hasAttribute('title'))).toBe(
          false,
        );
        expect(screen.getAllByRole('radio')[0]!.getAttribute('aria-label')).toBe(
          'Texture 1, the final’s own',
        );
        expect(
          screen.getByText('The same shape with other textures. Pick the one you like best.'),
        ).toBeTruthy();
        unmount();
      }
    });
  });
});
