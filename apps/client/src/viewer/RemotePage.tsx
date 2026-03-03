import { useEffect, useRef, useState } from 'react';
import { useParams, Link } from 'react-router-dom';
import * as THREE from 'three';
import type { ExperienceData, ExperienceMeta, RemoteControlCameraSync } from '@forge3d/shared';
import { SceneBuilder } from './SceneBuilder';
import { ViewerControls } from './ViewerControls';
import { RemoteControlController } from './formats/RemoteControlController';
import { useRemoteControl } from '../hooks/useRemoteControl';
import { MOCK_EXPERIENCES, getMockExperienceData, USE_MOCKS } from '../mocks/experiences';

/**
 * Remote-controlled viewer page.
 * Loads the experience at /remote/:sessionId and syncs the camera
 * to the presenter's view in real-time.
 */
export function RemotePage() {
  const { sessionId } = useParams<{ sessionId: string }>();
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [meta, setMeta] = useState<ExperienceMeta | null>(null);
  const controllerRef = useRef<RemoteControlController | null>(null);
  const [slug, setSlug] = useState<string | null>(null);

  // Fetch session info to get the experience slug
  useEffect(() => {
    if (!sessionId) return;
    fetch(`/trpc/remoteControl.getSession?input=${encodeURIComponent(JSON.stringify({ sessionId }))}`)
      .then((res) => res.json())
      .then((json) => {
        const data = json.result?.data;
        if (data?.found) {
          setSlug(data.session.experienceSlug);
        } else {
          setError('Session not found or expired');
          setLoading(false);
        }
      })
      .catch(() => {
        setError('Failed to connect');
        setLoading(false);
      });
  }, [sessionId]);

  // Connect to remote control WebSocket as viewer
  const onCameraSync = (msg: RemoteControlCameraSync) => {
    controllerRef.current?.applyCameraSync(msg);
  };
  const rc = useRemoteControl(sessionId, 'viewer', onCameraSync);

  // Build the 3D scene once we have the slug
  useEffect(() => {
    if (!slug || !canvasRef.current) return;

    const canvas = canvasRef.current;
    let disposed = false;
    let animId = 0;
    let renderer: THREE.WebGLRenderer | null = null;
    let controls: ViewerControls | null = null;
    let remoteCtrl: RemoteControlController | null = null;
    const sceneBuilder = new SceneBuilder();
    const clock = new THREE.Clock();

    async function init() {
      try {
        let resultMeta: ExperienceMeta | null = null;
        let experienceData: ExperienceData;

        if (USE_MOCKS) {
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
          if (!res.ok || json.error || !json.result?.data) {
            setError('Experience not found');
            setLoading(false);
            return;
          }
          resultMeta = json.result.data.meta;
          experienceData = json.result.data.experienceData;
        }

        if (disposed) return;
        setMeta(resultMeta);

        renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
        renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        renderer.setSize(canvas.clientWidth, canvas.clientHeight);
        renderer.shadowMap.enabled = true;
        renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        renderer.toneMapping = THREE.ACESFilmicToneMapping;

        const cam = experienceData.camera;
        const camera = new THREE.PerspectiveCamera(
          cam.fov,
          canvas.clientWidth / canvas.clientHeight,
          0.1,
          1000,
        );

        const scene = new THREE.Scene();
        sceneBuilder.build(experienceData, scene);

        controls = new ViewerControls({
          camera,
          domElement: canvas,
          initialPosition: cam.position,
          initialTarget: cam.target,
        });

        // Disable user orbit controls — camera is driven by the presenter
        controls.setEnabled(false);

        // Create remote control controller
        remoteCtrl = new RemoteControlController(camera, controls.controls.target);
        controllerRef.current = remoteCtrl;

        const observer = new ResizeObserver(() => {
          if (!renderer || disposed) return;
          const w = canvas.clientWidth;
          const h = canvas.clientHeight;
          renderer.setSize(w, h);
          camera.aspect = w / h;
          camera.updateProjectionMatrix();
        });
        observer.observe(canvas);

        clock.start();
        function animate() {
          if (disposed) return;
          animId = requestAnimationFrame(animate);
          const dt = clock.getDelta();
          remoteCtrl?.update(dt);
          controls?.update();
          renderer?.render(scene, camera);
        }
        animate();
        setLoading(false);
      } catch (err) {
        if (!disposed) {
          setError(err instanceof Error ? err.message : 'Failed to load');
          setLoading(false);
        }
      }
    }

    init();

    return () => {
      disposed = true;
      cancelAnimationFrame(animId);
      remoteCtrl?.dispose();
      controllerRef.current = null;
      controls?.dispose();
      renderer?.dispose();
      sceneBuilder.dispose();
    };
  }, [slug]);

  return (
    <div className="min-h-screen bg-[#11111b]">
      <div className="relative aspect-video w-full max-h-[70vh] bg-[#11111b]">
        <canvas ref={canvasRef} className="block h-full w-full" />

        {loading && (
          <div className="absolute inset-0 flex items-center justify-center bg-[#11111b]">
            <div className="text-center">
              <div className="mx-auto mb-4 h-8 w-8 animate-spin rounded-full border-2 border-[#cdd6f4] border-t-transparent" />
              <p className="text-sm text-[#a6adc8]">Joining session...</p>
            </div>
          </div>
        )}

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

        {rc.error && !error && (
          <div className="absolute inset-0 flex items-center justify-center bg-black/60">
            <p className="text-lg text-[#f38ba8]">{rc.error}</p>
          </div>
        )}

        {/* Status bar */}
        {!loading && !error && (
          <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/60 to-transparent px-4 py-3">
            <div className="flex items-center justify-between">
              <span className="text-xs text-white/60">
                {rc.connected ? 'Following presenter' : 'Connecting...'}
              </span>
              <span className="text-xs text-white/40">
                Remote session
              </span>
            </div>
          </div>
        )}
      </div>

      {meta && !error && (
        <div className="mx-auto max-w-4xl px-4 py-4">
          <h1 className="text-lg font-bold text-[#cdd6f4]">{meta.title}</h1>
          {meta.description && (
            <p className="mt-2 text-sm text-[#a6adc8]">{meta.description}</p>
          )}
        </div>
      )}
    </div>
  );
}
