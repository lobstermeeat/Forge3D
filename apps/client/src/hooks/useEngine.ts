import { useRef, useEffect, useState, useCallback } from 'react';
import { Renderer, ViewportControls, SceneManager, CommandHistory } from '@forge3d/engine';
import type { RendererType } from '@forge3d/engine';

export type FrameCallback = (dt: number) => void;

export function useEngine(canvasRef: React.RefObject<HTMLCanvasElement | null>) {
  const rendererRef = useRef<Renderer | null>(null);
  const controlsRef = useRef<ViewportControls | null>(null);
  const sceneManagerRef = useRef(new SceneManager());
  const historyRef = useRef(new CommandHistory());
  const frameCallbacksRef = useRef(new Set<FrameCallback>());
  const [rendererType, setRendererType] = useState<RendererType | null>(null);
  const [backendLabel, setBackendLabel] = useState<string | null>(null);
  const [ready, setReady] = useState(false);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    let disposed = false;

    // Start on the next task, not now. In development React's StrictMode mounts, cleans up and
    // mounts again at once, and two renderers on one canvas fight over its WebGPU context: each
    // configures it with its own GPU device, and if the discarded one does so last, the live one
    // draws nothing (an empty viewport). StrictMode's throwaway mount is cleaned up before this
    // timer fires, so only the live mount makes a renderer.
    const start = window.setTimeout(() => {
      if (disposed) return;
      const renderer = new Renderer({ canvas, preferWebGPU: true });
      rendererRef.current = renderer;

      void renderer.init().then((type) => {
        // Unmounted before init resolved
        if (disposed) {
          renderer.dispose();
          return;
        }

        renderer.addStudioEnvironment();
        setRendererType(type);
        setBackendLabel(renderer.getBackendLabel());

        const controls = new ViewportControls({
          camera: renderer.camera,
          domElement: canvas,
        });
        controlsRef.current = controls;

        renderer.start((dt) => {
          controls.update();
          for (const cb of frameCallbacksRef.current) cb(dt);
        });

        setReady(true);
      });
    }, 0);

    return () => {
      disposed = true;
      window.clearTimeout(start);
      controlsRef.current?.dispose();
      rendererRef.current?.dispose();
      rendererRef.current = null;
      controlsRef.current = null;
      setReady(false);
    };
  }, [canvasRef]);

  const getScene = useCallback(() => rendererRef.current?.scene ?? null, []);
  const getCamera = useCallback(() => rendererRef.current?.camera ?? null, []);
  const getRenderer = useCallback(() => rendererRef.current?.getThreeRenderer() ?? null, []);

  /** Run `cb` every rendered frame, before the frame is drawn. Returns an unsubscribe. */
  const addFrameCallback = useCallback((cb: FrameCallback) => {
    frameCallbacksRef.current.add(cb);
    return () => {
      frameCallbacksRef.current.delete(cb);
    };
  }, []);

  const setGridVisible = useCallback((visible: boolean) => {
    rendererRef.current?.setGridVisible(visible);
  }, []);

  const setPlayPreview = useCallback((active: boolean) => {
    rendererRef.current?.setPlayPreview(active);
  }, []);

  return {
    renderer: rendererRef,
    controls: controlsRef,
    sceneManager: sceneManagerRef.current,
    history: historyRef.current,
    rendererType,
    backendLabel,
    ready,
    getScene,
    getCamera,
    getRenderer,
    addFrameCallback,
    setGridVisible,
    setPlayPreview,
  };
}
