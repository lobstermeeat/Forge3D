import {
  chosenTexture,
  textureChoices,
  type TextureChoice,
  type TextureGeneration,
} from '@/editor/textureOptions';

/**
 * A done final's texture options, under it in the AI panel: a quiet line while they are made,
 * then a picker (1 is the final's own texture, and the pressed one is the one in the scene), or
 * a quiet note when none could be made. Nothing here holds up the final.
 */
export function TextureOptions({
  generation,
  sceneUrl,
  disabled = false,
  onChoose,
}: {
  generation: TextureGeneration;
  /** The model file the scene shows for this generation, or null when it isn't in the scene */
  sceneUrl: string | null | undefined;
  disabled?: boolean;
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
  return (
    <>
      <div className="f3-ai-textures">
        <span aria-hidden>Texture</span>
        <div className="f3-ai-texpicks" role="radiogroup" aria-label="Texture">
          {choices.map((choice) => (
            <button
              key={choice.url}
              type="button"
              role="radio"
              className="f3-chip"
              aria-checked={chosen === choice.number}
              aria-pressed={chosen === choice.number}
              aria-label={
                choice.number === 1 ? 'Texture 1, the final’s own' : `Texture ${choice.number}`
              }
              disabled={disabled}
              onClick={() => onChoose(choice)}
            >
              {choice.number}
            </button>
          ))}
        </div>
      </div>
      <p className="f3-ai-fine">The same shape with other textures. Pick the one you like best.</p>
    </>
  );
}
