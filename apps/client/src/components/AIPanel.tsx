import { useEffect, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import type { ModelData } from '@forge3d/shared';
import { trpc } from '@/api/trpc';
import { useSession } from '@/auth/client';
import { Icon } from '@/editor/Icon';
import { modelName } from '@/editor/modelName';
import { photoToDataUrl } from '@/editor/photo';
import type { EditorActions } from '@/hooks/useEditorActions';
import { useAIStore } from '@/stores/aiStore';
import { useEditorStore } from '@/stores/editorStore';
import { logOutput } from '@/stores/outputStore';
// Type only, like AppRouter in api/trpc.ts: no server code is bundled
import type { GenerationView as Generation } from '../../../../apps/server/src/services/ai/studio';

type Status = Generation['status'];

const RUNNING: ReadonlySet<Status> = new Set(['drawing', 'previewing', 'finishing']);
const MAX_PROMPT = 500;
const EXAMPLE = 'a wooden treasure chest with iron bands';
const DEFAULT_CREDITS = [
  'Built with DINOv3',
  '3D: TRELLIS.2 (Microsoft, MIT)',
  'Pictures: FLUX.1 [schnell] (Apache-2.0)',
];

/** What each running step is doing, and how long it usually takes. */
const STEP: Record<'drawing' | 'previewing' | 'finishing', { title: string; detail: string }> = {
  drawing: {
    title: 'Drawing 4 pictures',
    detail: 'Usually 10–40 s. After a quiet spell the first run takes a minute or two.',
  },
  previewing: {
    title: 'Building a 3D preview',
    detail: 'Usually 30–90 s. It appears in your scene when it’s ready.',
  },
  finishing: {
    title: 'Making the final model',
    detail: 'Usually 1–2 min. It replaces the preview in your scene, where it stands.',
  },
};

function modelData(g: Generation, quality: 'preview' | 'final'): ModelData {
  const file = quality === 'final' ? g.final! : g.preview!;
  return { url: file.url, source: 'ai', generationId: g.id, quality, credits: g.credits };
}

/** Seconds since `key` last changed, ticking while `active`. */
function useElapsed(key: string, active: boolean): number {
  const [started, setStarted] = useState(() => Date.now());
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    setStarted(Date.now());
    setNow(Date.now());
  }, [key]);
  useEffect(() => {
    if (!active) return;
    const timer = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(timer);
  }, [active]);
  return Math.max(0, Math.round((now - started) / 1000));
}

function errorText(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}

/** A picture's tooltip: what the worker found that makes it a worse start for 3D, one per line. */
function issuesText(issues: string[] | undefined): string | undefined {
  if (!issues?.length) return undefined;
  return issues.map((issue) => issue.charAt(0).toUpperCase() + issue.slice(1)).join('\n');
}

export function AIPanel({ actions, sceneId }: { actions: EditorActions; sceneId?: string }) {
  const navigate = useNavigate();
  const { data: session, isPending: sessionPending } = useSession();
  const signedIn = !!session?.user;
  const playing = useEditorStore((s) => s.isPlaying);
  const {
    currentId,
    setCurrentId,
    mode,
    setMode,
    prompt,
    setPrompt,
    photo,
    setPhoto,
    watched,
    watch,
    restored,
    markRestored,
  } = useAIStore();
  const utils = trpc.useUtils();
  const photoInput = useRef<HTMLInputElement>(null);
  const [picked, setPicked] = useState<number | null>(null);
  const [photoError, setPhotoError] = useState<string | null>(null);

  const capabilities = trpc.ai.capabilities.useQuery(undefined, { staleTime: 60_000 });
  const recent = trpc.ai.recent.useQuery(undefined, { enabled: signedIn });
  const gen = trpc.ai.get.useQuery(
    { id: currentId ?? '' },
    {
      enabled: signedIn && !!currentId,
      refetchInterval: (query) => (RUNNING.has(query.state.data?.status as Status) ? 2000 : false),
      retry: 1,
    },
  );
  const g = currentId ? gen.data : undefined;

  const show = (next: Generation) => {
    utils.ai.get.setData({ id: next.id }, next);
    if (RUNNING.has(next.status)) watch(next.id);
    setCurrentId(next.id);
    void utils.ai.recent.invalidate();
  };
  const onError = (err: unknown) => logOutput('error', `AI: ${errorText(err)}`);
  const start = trpc.ai.start.useMutation({ onSuccess: show, onError });
  const pick = trpc.ai.pick.useMutation({ onSuccess: show, onError });
  const keep = trpc.ai.keep.useMutation({ onSuccess: show, onError });
  const retry = trpc.ai.retry.useMutation({ onSuccess: show, onError });
  const busy = start.isPending || pick.isPending || keep.isPending || retry.isPending;

  // Start FLUX while the user types: after a quiet spell it takes about 45 s before it can draw
  const { mutate: warm } = trpc.ai.warm.useMutation();
  const warmed = useRef(false);
  const prompts = capabilities.data?.prompts ?? false;
  useEffect(() => {
    if (warmed.current || !signedIn || !prompts) return;
    warmed.current = true;
    warm({ worker: 'references' });
  }, [signedIn, prompts, warm]);

  // After a reload, carry on with this scene's unfinished generation (once per page load)
  useEffect(() => {
    if (restored || !recent.data) return;
    markRestored();
    if (currentId) return;
    const open = recent.data.find(
      (r) => r.status !== 'done' && r.status !== 'failed' && (!sceneId || r.sceneId === sceneId),
    );
    if (open) setCurrentId(open.id);
  }, [recent.data, restored, markRestored, currentId, sceneId, setCurrentId]);

  // Place what finished while the panel watched: the preview, then the final in its place
  const placed = useRef(new Set<string>());
  useEffect(() => {
    if (!g) return;
    if (RUNNING.has(g.status)) {
      watch(g.id);
      return;
    }
    if (!watched[g.id]) return;
    const quality =
      g.status === 'done' && g.final
        ? 'final'
        : g.status === 'reviewing' && g.preview
          ? 'preview'
          : null;
    if (!quality) return;
    const model = modelData(g, quality);
    if (placed.current.has(model.url)) return;
    placed.current.add(model.url);
    const existing = actions.findModelEntity(g.id);
    if (existing) actions.replaceModel(existing, model);
    else actions.insertModel(model, modelName(g.prompt));
    void utils.ai.recent.invalidate();
  }, [g, watched, watch, actions, utils]);

  // Recent shows each generation's step, so refresh it as steps change
  useEffect(() => {
    if (g?.status) void utils.ai.recent.invalidate();
  }, [g?.id, g?.status, utils]);

  // A new step starts with no picture chosen, except that picking starts on the one rated the
  // best start for 3D (if the worker rated them). The user can still choose any.
  const preselected = g?.status === 'picking' ? g.recommended : null;
  useEffect(() => setPicked(preselected), [g?.id, g?.status, preselected]);

  const elapsed = useElapsed(`${g?.id}:${g?.status}`, !!g && RUNNING.has(g.status));

  if (sessionPending) return <div className="f3-ai" aria-busy="true" />;

  if (!signedIn) {
    return (
      <div className="f3-ai">
        <div className="f3-empty" style={{ margin: 0 }}>
          <Icon name="sparkle" size={22} />
          <span>Sign in to make 3D models from a description or a photo.</span>
          <button type="button" className="f3-btn pri" onClick={() => navigate('/login')}>
            Sign in
          </button>
        </div>
      </div>
    );
  }

  const caps = capabilities.data;
  if (caps && !caps.available) {
    return (
      <div className="f3-ai">
        <div className="f3-empty" style={{ margin: 0 }}>
          <Icon name="sparkle" size={22} />
          <span>AI models aren’t set up on this server yet.</span>
          <span className="f3-ai-fine">
            Its admin sets AI_WORKERS_URL and AI_WORKERS_TOKEN (see workers/README.md).
          </span>
        </div>
      </div>
    );
  }
  const photosOnly = caps ? !caps.prompts : false;
  const composeMode = photosOnly ? 'photo' : mode;

  const choosePhoto = async (file: File | undefined) => {
    if (!file) return;
    setPhotoError(null);
    try {
      setPhoto({ dataUrl: await photoToDataUrl(file), name: file.name });
    } catch (err) {
      setPhotoError(errorText(err));
    }
  };

  const composer = (
    <>
      {!photosOnly && (
        <div className="f3-ai-modes" role="radiogroup" aria-label="Start from">
          {(
            [
              ['describe', 'Describe it'],
              ['photo', 'From a photo'],
            ] as const
          ).map(([key, label]) => (
            <button
              key={key}
              type="button"
              role="radio"
              className="f3-chip"
              aria-checked={composeMode === key}
              aria-pressed={composeMode === key}
              onClick={() => setMode(key)}
            >
              {label}
            </button>
          ))}
        </div>
      )}

      {composeMode === 'describe' ? (
        <form
          className="f3-ai-form"
          onSubmit={(e) => {
            e.preventDefault();
            if (prompt.trim()) start.mutate({ prompt: prompt.trim(), sceneId });
          }}
        >
          <label className="f3-field-l" htmlFor="f3-ai-prompt">
            Describe one object
          </label>
          <textarea
            id="f3-ai-prompt"
            className="f3-textfield f3-ai-prompt"
            rows={3}
            maxLength={MAX_PROMPT}
            placeholder={`e.g. ${EXAMPLE}`}
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && (e.metaKey || e.ctrlKey))
                e.currentTarget.form?.requestSubmit();
            }}
          />
          <button type="submit" className="f3-btn pri" disabled={!prompt.trim() || busy || playing}>
            {start.isPending ? (
              <span className="f3-spin" aria-hidden />
            ) : (
              <Icon name="sparkle" size={14} />
            )}
            Draw 4 pictures
          </button>
          <p className="f3-ai-fine">
            You pick the one you like, then get a 3D preview in about a minute.
          </p>
        </form>
      ) : (
        <div className="f3-ai-form">
          <input
            ref={photoInput}
            type="file"
            accept="image/png,image/jpeg,image/webp"
            hidden
            onChange={(e) => {
              void choosePhoto(e.target.files?.[0]);
              e.target.value = '';
            }}
          />
          <button
            type="button"
            className="f3-ai-drop"
            onClick={() => photoInput.current?.click()}
            onDragOver={(e) => e.preventDefault()}
            onDrop={(e) => {
              e.preventDefault();
              void choosePhoto(e.dataTransfer.files[0]);
            }}
          >
            {photo ? (
              <img src={photo.dataUrl} alt={`Chosen photo: ${photo.name}`} />
            ) : (
              <>
                <Icon name="image" size={24} />
                <span>Choose a photo, or drop one here</span>
              </>
            )}
          </button>
          {photoError && <p className="f3-ai-error">{photoError}</p>}
          <button
            type="button"
            className="f3-btn pri"
            disabled={!photo || busy || playing}
            onClick={() => photo && start.mutate({ photo: photo.dataUrl, sceneId })}
          >
            {start.isPending ? (
              <span className="f3-spin" aria-hidden />
            ) : (
              <Icon name="sparkle" size={14} />
            )}
            Make 3D preview
          </button>
          <p className="f3-ai-fine">One object on a plain background works best.</p>
        </div>
      )}
    </>
  );

  let card: React.ReactNode = null;
  if (g) {
    const running = RUNNING.has(g.status);
    const entityId = actions.findModelEntity(g.id);
    const title = g.source === 'photo' ? 'From a photo' : `“${g.prompt}”`;
    card = (
      <section className="f3-ai-card" aria-live="polite" aria-label="Current AI model">
        <header>
          <span className="f3-ai-subject" title={title}>
            {title}
          </span>
          <button
            type="button"
            className="f3-ibtn"
            title="Start something new (this one stays in Recent)"
            aria-label="Start something new"
            onClick={() => setCurrentId(null)}
          >
            <Icon name="close" size={13} />
          </button>
        </header>

        {running && (
          <div className="f3-ai-step">
            <div className="f3-ai-steptitle">
              <span className="f3-spin" aria-hidden />
              {STEP[g.status as keyof typeof STEP].title}
              <span className="f3-ai-time">{elapsed} s</span>
            </div>
            <div className="f3-ai-bar" aria-hidden>
              <span />
            </div>
            <p className="f3-ai-fine">{STEP[g.status as keyof typeof STEP].detail}</p>
          </div>
        )}

        {running && g.image && g.status !== 'drawing' && (
          <img className="f3-ai-source" src={g.image} alt="The picture being made into 3D" />
        )}

        {g.status === 'reviewing' && g.preview && (
          <div className="f3-ai-result">
            <p>
              <Icon name="ok" size={14} /> The preview is in your scene
              {g.preview.triangles ? ` (${g.preview.triangles.toLocaleString()} triangles)` : ''}.
            </p>
            <button
              type="button"
              className="f3-btn pri"
              disabled={busy || playing}
              onClick={() => keep.mutate({ id: g.id })}
            >
              {keep.isPending && <span className="f3-spin" aria-hidden />}
              Keep it: make the final
            </button>
            <p className="f3-ai-fine">The final has about 3× the detail and sharper textures.</p>
            {!entityId && (
              <button
                type="button"
                className="f3-btn ghost"
                onClick={() => actions.insertModel(modelData(g, 'preview'), modelName(g.prompt))}
              >
                Place the preview again
              </button>
            )}
            {entityId && (
              <button
                type="button"
                className="f3-btn ghost"
                onClick={() => {
                  actions.remove(entityId);
                  setCurrentId(null);
                }}
              >
                Discard the preview
              </button>
            )}
          </div>
        )}

        {g.status === 'done' && g.final && (
          <div className="f3-ai-result">
            <p>
              <Icon name="ok" size={14} /> The final model is in your scene
              {g.final.triangles ? ` (${g.final.triangles.toLocaleString()} triangles)` : ''}.
            </p>
            {!entityId && (
              <button
                type="button"
                className="f3-btn ghost"
                onClick={() => actions.insertModel(modelData(g, 'final'), modelName(g.prompt))}
              >
                Place it in the scene
              </button>
            )}
            <button type="button" className="f3-btn ghost" onClick={() => setCurrentId(null)}>
              Make another
            </button>
          </div>
        )}

        {g.status === 'failed' && (
          <div className="f3-ai-result">
            <p className="f3-ai-error">{g.error ?? 'Something went wrong.'}</p>
            <button
              type="button"
              className="f3-btn pri"
              disabled={busy || playing}
              onClick={() => retry.mutate({ id: g.id })}
            >
              {retry.isPending && <span className="f3-spin" aria-hidden />}
              Try again
            </button>
            <button type="button" className="f3-btn ghost" onClick={() => setCurrentId(null)}>
              Start over
            </button>
          </div>
        )}
        {g.references.length > 0 &&
          (g.status === 'picking' || g.status === 'reviewing' || g.status === 'failed') && (
            <div className="f3-ai-pickwrap">
              <p className="f3-ai-label">
                {g.status === 'picking' ? 'Pick a picture' : 'Or try another picture'}
              </p>
              <div
                className="f3-ai-picks"
                data-compact={g.status !== 'picking' || undefined}
                role="radiogroup"
                aria-label="Pictures to make 3D from"
              >
                {g.references.map((ref, i) => {
                  const best = i === g.recommended;
                  return (
                    <button
                      key={ref.url}
                      type="button"
                      role="radio"
                      className="f3-ai-pick"
                      aria-checked={picked === i}
                      aria-label={best ? `Picture ${i + 1}, best for 3D` : `Picture ${i + 1}`}
                      title={issuesText(ref.issues)}
                      data-current={g.image === ref.url || undefined}
                      data-recommended={best || undefined}
                      onClick={() => setPicked(i)}
                      onDoubleClick={() => pick.mutate({ id: g.id, index: i })}
                    >
                      <img src={ref.url} alt="" />
                    </button>
                  );
                })}
              </div>
              <button
                type="button"
                className={g.status === 'picking' ? 'f3-btn pri' : 'f3-btn ghost'}
                disabled={picked === null || busy || playing}
                onClick={() => picked !== null && pick.mutate({ id: g.id, index: picked })}
              >
                {pick.isPending && <span className="f3-spin" aria-hidden />}
                {g.status === 'picking' ? 'Make 3D preview' : 'Preview this picture instead'}
              </button>
            </div>
          )}
      </section>
    );
  }

  const history = (recent.data ?? []).filter((r) => r.id !== currentId).slice(0, 6);

  return (
    <div className="f3-ai">
      {caps?.mock && (
        <p className="f3-ai-mock">
          <Icon name="info" size={13} /> Mock workers: pictures and models are stand-ins.
        </p>
      )}
      {currentId && !g && gen.isError ? (
        <div className="f3-ai-card">
          <p className="f3-ai-error">{errorText(gen.error)}</p>
          <button type="button" className="f3-btn ghost" onClick={() => setCurrentId(null)}>
            Start over
          </button>
        </div>
      ) : (
        (card ?? composer)
      )}

      {history.length > 0 && (
        <section className="f3-ai-recent" aria-label="Recent AI models">
          <p className="f3-ai-label">Recent</p>
          {history.map((r) => {
            const thumb = r.image ?? r.references[r.recommended ?? 0]?.url;
            return (
              <button
                key={r.id}
                type="button"
                className="f3-ai-row"
                onClick={() => setCurrentId(r.id)}
              >
                <span className="f3-ai-rowthumb">
                  {thumb ? <img src={thumb} alt="" /> : <Icon name="model" size={14} />}
                </span>
                <span className="f3-ai-rowname">{r.source === 'photo' ? 'Photo' : r.prompt}</span>
                <span className="f3-ai-rowstate" data-state={r.status}>
                  {r.status === 'done'
                    ? 'Final'
                    : r.status === 'reviewing'
                      ? 'Preview'
                      : r.status === 'failed'
                        ? 'Failed'
                        : r.status === 'picking'
                          ? 'Pick'
                          : 'Working'}
                </span>
              </button>
            );
          })}
        </section>
      )}

      <p className="f3-ai-credits">
        {(g?.credits.length ? g.credits : DEFAULT_CREDITS).join(' · ')}
      </p>
    </div>
  );
}
