// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi, type Mock } from 'vitest';
import { act, cleanup, fireEvent, render, screen } from '@testing-library/react';
import type { ModelData } from '@forge3d/shared';
import type { EditorActions } from '@/hooks/useEditorActions';
import { useAIStore } from '@/stores/aiStore';
import { useEditorStore } from '@/stores/editorStore';
import type { GenerationView } from '../../../../apps/server/src/services/ai/studio';
import { AIPanel } from './AIPanel';

/** What the server answers: the generation ai.get returns, and the calls the panel makes */
const server = vi.hoisted(() => {
  const mutation = { mutate: () => {}, isPending: false };
  return {
    generation: undefined as unknown,
    chooseTexture: { mutate: (() => {}) as (input: unknown) => void, isPending: false },
    mutation,
    utils: { ai: { get: { setData: () => {} }, recent: { invalidate: async () => {} } } },
  };
});

vi.mock('@/api/trpc', () => ({
  trpc: {
    useUtils: () => server.utils,
    ai: {
      capabilities: {
        useQuery: () => ({
          data: { available: true, prompts: true, photos: true, multiview: false, mock: false },
        }),
      },
      recent: { useQuery: () => ({ data: [] }) },
      get: { useQuery: () => ({ data: server.generation, isError: false }) },
      start: { useMutation: () => server.mutation },
      pick: { useMutation: () => server.mutation },
      keep: { useMutation: () => server.mutation },
      retry: { useMutation: () => server.mutation },
      warm: { useMutation: () => server.mutation },
      chooseTexture: { useMutation: () => server.chooseTexture },
    },
  },
}));
vi.mock('@/auth/client', () => ({
  useSession: () => ({ data: { user: { id: 'u1' } }, isPending: false }),
}));
vi.mock('react-router-dom', () => ({ useNavigate: () => () => {} }));

const final = { url: 'https://files.test/ai/g1/final-42.glb', triangles: 100_000 };
const optionUrl = (k: number) => `https://files.test/ai/g1/final-42-texture-${42 + 1000 * k}.glb`;
const why = 'The back keeps the brass colour, with no smudges.';

/** The generation as ai.get shows it: a done final, its textures made with the judge's `pick` */
function generation(pick: number | null, status: 'running' | 'done' = 'done'): GenerationView {
  return {
    id: 'g1',
    source: 'prompt',
    prompt: 'a brass desk lamp',
    status: 'done',
    references: [],
    recommended: null,
    image: null,
    views: [],
    viewsError: null,
    preview: null,
    final,
    textures: {
      status,
      count: 3,
      options:
        status === 'done'
          ? [1, 2, 3].map((k) => ({
              url: optionUrl(k),
              triangles: 96_000,
              textureSeed: 42 + 1000 * k,
            }))
          : [],
      error: null,
      recommended: pick === null ? null : { number: pick, why, applied: false },
      chosen: null,
      judgeError: null,
    },
    credits: [],
    error: null,
    sceneId: null,
    createdAt: '2026-10-02T00:00:00.000Z',
  };
}

/** A scene with at most the generation's model, and the editor actions the panel calls */
function scene(url: string | null) {
  const model = (file: string): ModelData => ({
    url: file,
    source: 'ai',
    generationId: 'g1',
    quality: 'final',
    credits: [],
  });
  const state = { model: url ? model(url) : null };
  const actions = {
    findModelEntity: (id: string) => (state.model?.generationId === id ? 'e1' : null),
    modelOf: (entityId: string) => (entityId === 'e1' ? state.model : null),
    replaceModel: vi.fn((_entityId: string, next: ModelData, _what?: string) => {
      state.model = next;
    }),
    insertModel: vi.fn((next: ModelData) => {
      state.model = next;
      return 'e1';
    }),
    remove: vi.fn(() => {
      state.model = null;
    }),
  };
  return { state, actions, editor: actions as unknown as EditorActions };
}

let chooseTexture: Mock<(input: unknown) => void>;

beforeEach(() => {
  chooseTexture = vi.fn<(input: unknown) => void>();
  server.chooseTexture = { mutate: chooseTexture, isPending: false };
  useAIStore.setState({
    currentId: 'g1',
    restored: true,
    watched: {},
    placed: {},
    recommendedApplied: {},
    textureChosen: {},
  });
  useEditorStore.setState({ isPlaying: false });
});
afterEach(cleanup);

describe("AIPanel and the judge's pick (AI_TEXTURE_JUDGE=1)", () => {
  it('switches the model to the recommended texture once the options arrive, and says so', () => {
    const { state, actions, editor } = scene(final.url);
    server.generation = generation(null, 'running');
    const { rerender } = render(<AIPanel actions={editor} />);
    expect(screen.getByText('Making 3 more textures to choose from…')).toBeTruthy();

    server.generation = generation(3);
    rerender(<AIPanel actions={editor} />);
    // As picking texture 3 would: in place, and Undo says "Use texture 3"
    expect(actions.replaceModel).toHaveBeenCalledTimes(1);
    expect(actions.replaceModel).toHaveBeenCalledWith(
      'e1',
      expect.objectContaining({ url: optionUrl(2), generationId: 'g1', quality: 'final' }),
      'texture 3',
    );
    expect(state.model?.url).toBe(optionUrl(2));
    expect(chooseTexture).toHaveBeenCalledWith({ id: 'g1', number: 3, by: 'judge' });
    expect(
      screen.getByText('Switched to texture 3, the cleanest of the four. Pick another any time.'),
    ).toBeTruthy();
    expect(screen.getAllByRole('radio')[2]!.getAttribute('aria-pressed')).toBe('true');

    // More renders, and the server's record coming back, change nothing
    server.generation = {
      ...generation(3),
      textures: {
        ...generation(3).textures!,
        recommended: { number: 3, why, applied: true },
      },
    };
    rerender(<AIPanel actions={editor} />);
    rerender(<AIPanel actions={editor} />);
    expect(actions.replaceModel).toHaveBeenCalledTimes(1);
    expect(chooseTexture).toHaveBeenCalledTimes(1);
  });

  it("records the creator's choice, and the pick never goes over it", () => {
    // The options arrive while the model isn't in the scene: nothing to switch
    const { state, actions, editor } = scene(null);
    server.generation = generation(3);
    const { rerender } = render(<AIPanel actions={editor} />);
    expect(actions.replaceModel).not.toHaveBeenCalled();
    expect(
      screen.getByText('The same shape with other textures. Texture 3 is recommended.'),
    ).toBeTruthy();

    // The creator places texture 2 from the picker
    fireEvent.click(screen.getByRole('radio', { name: 'Texture 2' }));
    expect(actions.insertModel).toHaveBeenCalledWith(
      expect.objectContaining({ url: optionUrl(1) }),
      'Brass desk lamp',
    );
    expect(chooseTexture).toHaveBeenCalledWith({ id: 'g1', number: 2, by: 'creator' });
    expect(useAIStore.getState().textureChosen).toEqual({ [final.url]: true });

    // Then swaps back to the final's own texture: the pick stays out
    rerender(<AIPanel actions={editor} />);
    fireEvent.click(screen.getByRole('radio', { name: 'Texture 1, the final’s own' }));
    expect(state.model?.url).toBe(final.url);
    rerender(<AIPanel actions={editor} />);
    expect(actions.replaceModel).toHaveBeenCalledTimes(1);
    expect(actions.replaceModel).toHaveBeenLastCalledWith(
      'e1',
      expect.objectContaining({ url: final.url }),
      'texture 1',
    );
    expect(chooseTexture).not.toHaveBeenCalledWith(expect.objectContaining({ by: 'judge' }));
  });

  it('waits while the scene plays, and switches when it stops', () => {
    const { actions, editor } = scene(final.url);
    useEditorStore.setState({ isPlaying: true });
    server.generation = generation(3);
    render(<AIPanel actions={editor} />);
    expect(actions.replaceModel).not.toHaveBeenCalled();
    expect(
      screen.getAllByRole('radio').every((radio) => (radio as HTMLButtonElement).disabled),
    ).toBe(true);

    act(() => useEditorStore.setState({ isPlaying: false }));
    expect(actions.replaceModel).toHaveBeenCalledTimes(1);
    expect(chooseTexture).toHaveBeenCalledWith({ id: 'g1', number: 3, by: 'judge' });
  });

  it("changes nothing without a pick, or when the pick is the final's own", () => {
    for (const pick of [null, 1]) {
      const { actions, editor } = scene(final.url);
      server.generation = generation(pick);
      const { unmount } = render(<AIPanel actions={editor} />);
      expect(actions.replaceModel).not.toHaveBeenCalled();
      expect(chooseTexture).not.toHaveBeenCalled();
      expect(
        screen.getByText('The same shape with other textures. Pick the one you like best.'),
      ).toBeTruthy();
      unmount();
    }
  });
});
