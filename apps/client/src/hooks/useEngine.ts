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
    const renderer = new Renderer({ canvas, preferWebGPU: true });
    rendererRef.current = renderer;

    renderer.init().then((type) => {
      // StrictMode cleanup may have run before init resolved
      if (disposed) {
        renderer.dispose();
        return;
      }

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

    return () => {
      disposed = true;
      controlsRef.current?.dispose();
      rendererRef.current?.dispose();
      rendererRef.current = null;
      controlsRef.current = null;
      setReady(false);
    };
  }, [canvasRef]);

  const getScene = useCallback(() => rendererRef.current?.scene ?? null, []);
  const getCamera = useCallback(() => rendererRef.current?.camera ?? null, []);

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
    addFrameCallback,
    setGridVisible,
    setPlayPreview,
  };
}
