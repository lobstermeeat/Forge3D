import {
  chosenTexture,
  recommendedTexture,
  textureChoices,
  type TextureChoice,
  type TextureGeneration,
} from '@/editor/textureOptions';

/** "the cleanest of the four", for however many textures there are to choose from */
function cleanestOf(count: number): string {
  if (count === 2) return 'the cleaner of the two';
  return `the cleanest of the ${['three', 'four', 'five'][count - 3] ?? count}`;
}

/**
 * A done final's texture options, under it in the AI panel: a quiet line while they are made,
 * then a picker (1 is the final's own texture, and the pressed one is the one in the scene), or
 * a quiet note when none could be made. Nothing here holds up the final. With AI_TEXTURE_JUDGE=1
 * on the server, a dot marks the recommended texture (its tooltip says why), and once the panel has
 * switched the model to it, the line under the picker says so.
 */
export function TextureOptions({
  generation,
  sceneUrl,
  disabled = false,
  switched = false,
  onChoose,
}: {
  generation: TextureGeneration;
  /** The model file the scene shows for this generation, or null when it isn't in the scene */
  sceneUrl: string | null | undefined;
  disabled?: boolean;
  /** The panel put the recommended texture in the scene, and it is still there */
  switched?: boolean;
  onChoose: (choice: TextureChoice) => void;
}) {
  const { textures } = generation;
  if (textures?.status === 'running') {
    return (
      <p className="f3-ai-fine f3-ai-texwait">
        <span className="f3-spin" aria-hidden />
        Making {textures.count} more textures to choose from…
      </p>
    );
  }
  if (textures?.status === 'failed') {
    return (
      <p className="f3-ai-fine" title={textures.error ?? undefined}>
        More textures couldn’t be made this time.
      </p>
    );
  }
  const choices = textureChoices(generation);
  if (!choices.length) return null;
  const chosen = chosenTexture(choices, sceneUrl);
  const recommended = recommendedTexture(generation);
  // The recommended texture's tooltip: the judge's reason
  const tip = recommended?.why ? `Recommended: ${recommended.why}` : 'Recommended';
  return (
    <>
      <div className="f3-ai-textures">
        <span aria-hidden>Texture</span>
        <div className="f3-ai-texpicks" role="radiogroup" aria-label="Texture">
          {choices.map((choice) => {
            const best = choice.number === recommended?.number;
            return (
              <button
                key={choice.url}
                type="button"
                role="radio"
                className="f3-chip"
                aria-checked={chosen === choice.number}
                aria-pressed={chosen === choice.number}
                aria-label={
                  choice.number === 1
                    ? 'Texture 1, the final’s own'
                    : best
                      ? `Texture ${choice.number}, recommended`
                      : `Texture ${choice.number}`
                }
                title={best ? tip : undefined}
                data-recommended={best || undefined}
                disabled={disabled}
                onClick={() => onChoose(choice)}
              >
                {choice.number}
              </button>
            );
          })}
        </div>
      </div>
      {recommended && switched ? (
        <p className="f3-ai-fine" title={recommended.why ?? undefined}>
          Switched to texture {recommended.number}, {cleanestOf(choices.length)}. Pick another any
          time.
        </p>
      ) : recommended ? (
        <p className="f3-ai-fine">
          The same shape with other textures. Texture {recommended.number} is recommended.
        </p>
      ) : (
        <p className="f3-ai-fine">
          The same shape with other textures. Pick the one you like best.
        </p>
      )}
    </>
  );
}
