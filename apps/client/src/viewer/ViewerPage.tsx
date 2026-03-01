import { useEffect, useRef, useState, useCallback } from 'react';
import { useParams, Link } from 'react-router-dom';
import * as THREE from 'three';
import type { ExperienceData, ExperienceMeta } from '@forge3d/shared';
import { SceneBuilder } from './SceneBuilder';
import { ViewerControls } from './ViewerControls';
import type { FormatController } from './formats/types';
import { TurntableController } from './formats/TurntableController';
import { VideoController } from './formats/VideoController';
import { AnimatedController } from './formats/AnimatedController';
import { InteractiveController } from './formats/InteractiveController';
import { TimelineScrubber } from './components/TimelineScrubber';
import { InfoPopup } from './components/InfoPopup';
import { LikeButton } from '../components/LikeButton';
import { CommentSection } from '../components/CommentSection';
import { ReportButton } from '../components/ReportButton';
import { MOCK_EXPERIENCES, getMockExperienceData, USE_MOCKS } from '../mocks/experiences';

interface InfoPopupData {
  title: string;
  description?: string;
}

// Prevent StrictMode double-fire from incrementing view count twice
const viewedIds = new Set<string>();

export function ViewerPage() {
  const { slug } = useParams<{ slug: string }>();
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [meta, setMeta] = useState<ExperienceMeta | null>(null);
  const [formatType, setFormatType] = useState<string | null>(null);
  const [playing, setPlaying] = useState(true);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [infoPopup, setInfoPopup] = useState<InfoPopupData | null>(null);

  // Store controller in a ref so React re-renders don't affect it
  const controllerRef = useRef<FormatController | null>(null);

  const handleTogglePlay = useCallback(() => {
    const ctrl = controllerRef.current;
    if (ctrl) {
      const nowPlaying = ctrl.togglePlayPause();
      setPlaying(nowPlaying);
    }
  }, []);

  const handleSeek = useCallback((time: number) => {
    const ctrl = controllerRef.current;
    if (ctrl?.seek) {
      ctrl.seek(time);
      setCurrentTime(time);
    }
  }, []);

  useEffect(() => {
    if (!slug || !canvasRef.current) return;

    const canvas = canvasRef.current;
    let disposed = false;
    let animId = 0;
    let renderer: THREE.WebGLRenderer | null = null;
    let controls: ViewerControls | null = null;
    let formatCtrl: FormatController | null = null;
    const sceneBuilder = new SceneBuilder();
    const clock = new THREE.Clock();

    // Throttle time updates to ~10fps to avoid excessive React re-renders
    let lastTimeUpdate = 0;

    async function init() {
      try {
        let resultMeta: ExperienceMeta;
        let experienceData: ExperienceData;

        if (USE_MOCKS) {
          // Resolve mock data by slug
          const mockExp = MOCK_EXPERIENCES.find((e) => e.slug === slug);
          if (!mockExp) {
            setError('Experience not found');
            setLoading(false);
            return;
          }
          resultMeta = mockExp;
          experienceData = getMockExperienceData(mockExp.formatType);
        } else {
          const res = await fetch(`/trpc/experience.getBySlug?input=${encodeURIComponent(JSON.stringify({ slug }))}`);
          const json = await res.json();

          if (!res.ok || json.error) {
            setError(json.error?.message ?? 'Experience not found');
            setLoading(false);
            return;
          }

          const result = json.result?.data;
          if (!result) {
            setError('Experience not found');
            setLoading(false);
            return;
          }

          resultMeta = result.meta;
          experienceData = result.experienceData;

          // Record view once per experience (fire-and-forget)
          if (!viewedIds.has(resultMeta.id)) {
            viewedIds.add(resultMeta.id);
            fetch('/trpc/experience.recordView', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ id: resultMeta.id }),
            }).catch(() => {});
          }
        }

        if (disposed) return;

        setMeta(resultMeta);

        // Renderer
        renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
        renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        renderer.setSize(canvas.clientWidth, canvas.clientHeight);
        renderer.shadowMap.enabled = true;
        renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        renderer.toneMapping = THREE.ACESFilmicToneMapping;

        // Camera
        const cam = experienceData.camera;
        const camera = new THREE.PerspectiveCamera(
          cam.fov,
          canvas.clientWidth / canvas.clientHeight,
          0.1,
          1000,
        );

        // Scene
        const scene = new THREE.Scene();
        sceneBuilder.build(experienceData, scene);

        // Controls
        controls = new ViewerControls({
          camera,
          domElement: canvas,
          initialPosition: cam.position,
          initialTarget: cam.target,
        });

        // Create format controller based on type
        const fmt = experienceData.format;
        setFormatType(fmt.type);

        switch (fmt.type) {
          case 'turntable':
            formatCtrl = new TurntableController(controls, {
              speed: fmt.speed,
              axis: fmt.axis,
            });
            break;

          case 'video':
            formatCtrl = new VideoController(controls, camera, fmt.keyframes);
            setDuration(formatCtrl.getDuration!());
            break;

          case 'animated':
            formatCtrl = new AnimatedController(scene, { duration: fmt.duration });
            setDuration(formatCtrl.getDuration!());
            break;

          case 'interactive':
            formatCtrl = new InteractiveController(
              camera,
              scene,
              canvas,
              fmt.interactions,
              {
                onShowInfo: (_entityId, payload) => {
                  setInfoPopup({
                    title: (payload['title'] as string) ?? 'Info',
                    description: payload['description'] as string | undefined,
                  });
                },
                onNavigate: (url) => {
                  window.location.href = url;
                },
              },
            );
            break;
        }

        controllerRef.current = formatCtrl;

        // Resize handler
        const observer = new ResizeObserver(() => {
          if (!renderer || disposed) return;
          const w = canvas.clientWidth;
          const h = canvas.clientHeight;
          renderer.setSize(w, h);
          camera.aspect = w / h;
          camera.updateProjectionMatrix();
        });
        observer.observe(canvas);

        // Render loop
        clock.start();
        function animate() {
          if (disposed) return;
          animId = requestAnimationFrame(animate);
          const dt = clock.getDelta();

          formatCtrl?.update(dt);
          controls?.update();
          renderer?.render(scene, camera);

          // Update time state for scrubber (throttled)
          if (formatCtrl?.getCurrentTime && formatCtrl?.getDuration) {
            const now = performance.now();
            if (now - lastTimeUpdate > 100) {
              lastTimeUpdate = now;
              setCurrentTime(formatCtrl.getCurrentTime());
            }
          }
        }
        animate();

        setLoading(false);
      } catch (err) {
        if (!disposed) {
          setError(err instanceof Error ? err.message : 'Failed to load experience');
          setLoading(false);
        }
      }
    }

    init();

    return () => {
      disposed = true;
      cancelAnimationFrame(animId);
      formatCtrl?.dispose();
      controllerRef.current = null;
      controls?.dispose();
      renderer?.dispose();
      sceneBuilder.dispose();
    };
  }, [slug]);

  const handleReset = () => {
    window.location.reload();
  };

  const hasTimeline = formatType === 'video' || formatType === 'animated';

  return (
    <div className="min-h-screen bg-[#11111b]">
      {/* 3D Viewport */}
      <div className="relative aspect-video w-full max-h-[70vh] bg-[#11111b]">
        <canvas
          ref={canvasRef}
          className="block h-full w-full"
        />

        {/* Loading overlay */}
        {loading && (
          <div className="absolute inset-0 flex items-center justify-center bg-[#11111b]">
            <div className="text-center">
              <div className="mx-auto mb-4 h-8 w-8 animate-spin rounded-full border-2 border-[#cdd6f4] border-t-transparent" />
              <p className="text-sm text-[#a6adc8]">Loading experience...</p>
            </div>
          </div>
        )}

        {/* Error overlay */}
        {error && (
          <div className="absolute inset-0 flex items-center justify-center bg-[#11111b]">
            <div className="text-center">
              <p className="text-lg text-[#f38ba8]">{error}</p>
              <Link to="/explore" className="mt-4 inline-block text-sm text-[#89b4fa] underline">
                Browse experiences
              </Link>
            </div>
          </div>
        )}

        {/* Info popup for interactive format */}
        {infoPopup && (
          <InfoPopup
            title={infoPopup.title}
            description={infoPopup.description}
            onClose={() => setInfoPopup(null)}
          />
        )}

        {/* Bottom controls bar */}
        {!loading && !error && (
          <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/60 to-transparent px-4 py-3">
            {hasTimeline ? (
              <TimelineScrubber
                currentTime={currentTime}
                duration={duration}
                playing={playing}
                onTogglePlay={handleTogglePlay}
                onSeek={handleSeek}
              />
            ) : (
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  {formatType === 'turntable' && (
                    <button
                      onClick={handleTogglePlay}
                      className="rounded bg-white/10 px-3 py-1.5 text-xs font-medium text-white backdrop-blur-sm hover:bg-white/20"
                    >
                      {playing ? 'Pause' : 'Play'}
                    </button>
                  )}
                  {formatType === 'interactive' && (
                    <span className="text-xs text-white/60">Click objects to interact</span>
                  )}
                  <button
                    onClick={handleReset}
                    className="rounded bg-white/10 px-3 py-1.5 text-xs font-medium text-white backdrop-blur-sm hover:bg-white/20"
                  >
                    Reset View
                  </button>
                </div>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Info section below viewport */}
      {meta && !error && (
        <div className="mx-auto max-w-4xl px-4 py-4">
          {/* Title + actions row */}
          <div className="flex items-start justify-between gap-4">
            <div className="flex-1">
              <h1 className="text-lg font-bold text-[#cdd6f4]">{meta.title}</h1>
              <p className="mt-0.5 text-xs text-[#6c7086]">
                {meta.viewCount.toLocaleString()} views
                {meta.publishedAt && ` · ${new Date(meta.publishedAt).toLocaleDateString()}`}
              </p>
            </div>
            <div className="flex items-center gap-3">
              <LikeButton experienceId={meta.id} initialCount={meta.likeCount} />
              <ReportButton targetType="experience" targetId={meta.id} />
            </div>
          </div>

          {/* Creator info */}
          <div className="mt-4 flex items-center gap-3 border-t border-[#313244] pt-4">
            {meta.creator.avatarUrl ? (
              <img
                src={meta.creator.avatarUrl}
                alt={meta.creator.displayName}
                className="h-10 w-10 rounded-full"
              />
            ) : (
              <div className="flex h-10 w-10 items-center justify-center rounded-full bg-[#45475a] text-sm font-bold text-[#cdd6f4]">
                {(meta.creator.displayName ?? meta.creator.username)[0]?.toUpperCase()}
              </div>
            )}
            <div>
              <Link
                to={`/creator/${meta.creator.username}`}
                className="text-sm font-medium text-[#cdd6f4] hover:text-[#89b4fa]"
              >
                {meta.creator.displayName ?? meta.creator.username}
              </Link>
              <p className="text-xs text-[#6c7086]">@{meta.creator.username}</p>
            </div>
          </div>

          {/* Description */}
          {meta.description && (
            <p className="mt-3 text-sm text-[#a6adc8]">{meta.description}</p>
          )}

          {/* Comments */}
          <CommentSection experienceId={meta.id} />
        </div>
      )}
    </div>
  );
}
