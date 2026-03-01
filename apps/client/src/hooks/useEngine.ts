import { useRef, useEffect, useState, useCallback } from 'react';
import { Renderer, ViewportControls, SceneManager, CommandHistory } from '@forge3d/engine';
import type { RendererType } from '@forge3d/engine';

export function useEngine(canvasRef: React.RefObject<HTMLCanvasElement | null>) {
  const rendererRef = useRef<Renderer | null>(null);
  const controlsRef = useRef<ViewportControls | null>(null);
  const sceneManagerRef = useRef(new SceneManager());
  const historyRef = useRef(new CommandHistory());
  const [rendererType, setRendererType] = useState<RendererType | null>(null);
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

      const controls = new ViewportControls({
        camera: renderer.camera,
        domElement: canvas,
      });
      controlsRef.current = controls;

      renderer.start(() => {
        controls.update();
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

  return {
    renderer: rendererRef,
    controls: controlsRef,
    sceneManager: sceneManagerRef.current,
    history: historyRef.current,
    rendererType,
    ready,
    getScene,
    getCamera,
  };
}
