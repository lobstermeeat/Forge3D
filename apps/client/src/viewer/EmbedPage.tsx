import { useEffect, useRef, useState } from 'react';
import { useParams } from 'react-router-dom';
import * as THREE from 'three';
import type { ExperienceData } from '@forge3d/shared';
import { SceneBuilder } from './SceneBuilder';
import { ViewerControls } from './ViewerControls';
import type { FormatController } from './formats/types';
import { TurntableController } from './formats/TurntableController';
import { VideoController } from './formats/VideoController';
import { AnimatedController } from './formats/AnimatedController';
import { InteractiveController } from './formats/InteractiveController';

// Prevent StrictMode double-fire from incrementing view count twice
const viewedIds = new Set<string>();

/**
 * Minimal-chrome viewer for iframe embeds.
 * No title bar, no metadata — just the 3D viewport with basic controls.
 */
export function EmbedPage() {
  const { slug } = useParams<{ slug: string }>();
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

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

    async function init() {
      try {
        const res = await fetch(`/trpc/experience.getBySlug?input=${encodeURIComponent(JSON.stringify({ slug }))}`);
        const json = await res.json();

        if (!res.ok || json.error || !json.result?.data) {
          setError('Not found');
          setLoading(false);
          return;
        }
        if (disposed) return;

        // Record view once per experience (fire-and-forget)
        const expId = json.result.data.meta?.id;
        if (expId && !viewedIds.has(expId)) {
          viewedIds.add(expId);
          fetch('/trpc/experience.recordView', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ id: expId }),
          }).catch(() => {});
        }

        const experienceData: ExperienceData = json.result.data.experienceData;

        renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
        renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        renderer.setSize(canvas.clientWidth, canvas.clientHeight);
        renderer.shadowMap.enabled = true;
        renderer.toneMapping = THREE.ACESFilmicToneMapping;

        const cam = experienceData.camera;
        const camera = new THREE.PerspectiveCamera(cam.fov, canvas.clientWidth / canvas.clientHeight, 0.1, 1000);

        const scene = new THREE.Scene();
        sceneBuilder.build(experienceData, scene);

        controls = new ViewerControls({
          camera,
          domElement: canvas,
          initialPosition: cam.position,
          initialTarget: cam.target,
        });

        // Create format controller
        const fmt = experienceData.format;
        switch (fmt.type) {
          case 'turntable':
            formatCtrl = new TurntableController(controls, {
              speed: fmt.speed,
              axis: fmt.axis,
            });
            break;
          case 'video':
            formatCtrl = new VideoController(controls, camera, fmt.keyframes);
            break;
          case 'animated':
            formatCtrl = new AnimatedController(scene, { duration: fmt.duration });
            break;
          case 'interactive':
            formatCtrl = new InteractiveController(
              camera, scene, canvas, fmt.interactions,
              {
                onShowInfo: () => { /* No UI in embed */ },
                onNavigate: (url) => { window.open(url, '_blank', 'noopener'); },
              },
            );
            break;
        }

        const observer = new ResizeObserver(() => {
          if (!renderer || disposed) return;
          renderer.setSize(canvas.clientWidth, canvas.clientHeight);
          camera.aspect = canvas.clientWidth / canvas.clientHeight;
          camera.updateProjectionMatrix();
        });
        observer.observe(canvas);

        clock.start();
        function animate() {
          if (disposed) return;
          animId = requestAnimationFrame(animate);
          const dt = clock.getDelta();
          formatCtrl?.update(dt);
          controls?.update();
          renderer?.render(scene, camera);
        }
        animate();

        setLoading(false);
      } catch {
        if (!disposed) {
          setError('Failed to load');
          setLoading(false);
        }
      }
    }

    init();

    return () => {
      disposed = true;
      cancelAnimationFrame(animId);
      formatCtrl?.dispose();
      controls?.dispose();
      renderer?.dispose();
      sceneBuilder.dispose();
    };
  }, [slug]);

  return (
    <div className="relative h-screen w-screen bg-[#11111b]">
      <canvas ref={canvasRef} className="block h-full w-full" />

      {loading && (
        <div className="absolute inset-0 flex items-center justify-center bg-[#11111b]">
          <div className="h-6 w-6 animate-spin rounded-full border-2 border-white border-t-transparent" />
        </div>
      )}

      {error && (
        <div className="absolute inset-0 flex items-center justify-center bg-[#11111b]">
          <p className="text-sm text-[#f38ba8]">{error}</p>
        </div>
      )}

      {/* Forge3D watermark */}
      {!loading && !error && (
        <a
          href="/"
          target="_blank"
          rel="noopener noreferrer"
          className="absolute bottom-2 right-2 rounded bg-black/40 px-2 py-1 text-[10px] text-white/50 backdrop-blur-sm hover:text-white/80"
        >
          Forge3D
        </a>
      )}
    </div>
  );
}
