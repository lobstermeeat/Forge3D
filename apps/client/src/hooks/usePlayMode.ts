import { useCallback, useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import type { SceneBridge, SceneManager, ViewportControls } from '@forge3d/engine';
import type { ExperienceFormatType } from '@forge3d/shared';
import { ViewerControls } from '@/viewer/ViewerControls';
import {
  AnimatedController,
  InteractiveController,
  TurntableController,
  VideoController,
  type FormatController,
} from '@/viewer/formats';
import { useEditorStore } from '@/stores/editorStore';
import { useFormatStore } from '@/stores/formatStore';
import { logOutput } from '@/stores/outputStore';
import type { FrameCallback } from '@/hooks/useEngine';

export const FORMAT_LABEL: Record<ExperienceFormatType, string> = {
  turntable: 'Turntable',
  interactive: 'Interactive experience',
  animated: 'Animated scene',
  video: '3D Video',
};

interface PlayDeps {
  ready: boolean;
  canvas: HTMLCanvasElement | null;
  getCamera: () => THREE.PerspectiveCamera | null;
  getScene: () => THREE.Scene | null;
  controls: React.RefObject<ViewportControls | null>;
  bridge: React.RefObject<SceneBridge | null>;
  sceneManager: SceneManager;
  addFrameCallback: (cb: FrameCallback) => () => void;
  setGridVisible: (visible: boolean) => void;
  setPlayPreview: (active: boolean) => void;
}

interface PlaySession {
  viewer: ViewerControls;
  controller: FormatController;
  savedPosition: THREE.Vector3;
  savedTarget: THREE.Vector3;
  removeFrame: () => void;
}

export interface InfoCard {
  title: string;
  description?: string;
}

/**
 * Play-tests the scene in place: the editor camera is parked, and the same format
 * controller the published viewer uses (turntable, video, animated, interactive)
 * drives the live scene. Stop puts the camera and scene back.
 */
export function usePlayMode(d: PlayDeps) {
  const [info, setInfo] = useState<InfoCard | null>(null);
  const sessionRef = useRef<PlaySession | null>(null);
  const depsRef = useRef(d);
  depsRef.current = d;

  const stop = useCallback(() => {
    const s = sessionRef.current;
    if (!s) return;
    sessionRef.current = null;
    const deps = depsRef.current;
    s.removeFrame();
    s.controller.dispose();
    s.viewer.dispose();

    const camera = deps.getCamera();
    const controls = deps.controls.current;
    if (camera && controls) {
      camera.position.copy(s.savedPosition);
      controls.target.copy(s.savedTarget);
      controls.setEnabled(true);
      controls.update();
    }
    deps.sceneManager.notifyChange(); // re-applies every entity transform
    deps.setPlayPreview(false);
    deps.setGridVisible(useEditorStore.getState().showGrid);
    setInfo(null);
    useEditorStore.getState().setPlaying(false);
    logOutput('info', 'Stopped — the scene is back to how you left it');
  }, []);

  const play = useCallback(() => {
    const deps = depsRef.current;
    if (sessionRef.current || !deps.ready) return;
    const camera = deps.getCamera();
    const scene = deps.getScene();
    const controls = deps.controls.current;
    const bridge = deps.bridge.current;
    const canvas = deps.canvas;
    if (!camera || !scene || !controls || !bridge || !canvas) return;

    const format = useFormatStore.getState().getFormatConfig();
    if (format.type === 'video' && format.keyframes.length < 2) {
      logOutput(
        'warn',
        '3D Video needs at least two camera keyframes — add them in the Timeline first',
      );
      useEditorStore.getState().setDockTab('timeline');
      return;
    }

    // Format controllers find entities through userData
    for (const obj of bridge.getManagedObjects()) {
      const id = bridge.getEntityId(obj);
      if (id) obj.userData.entityId = id;
    }

    const savedPosition = camera.position.clone();
    const savedTarget = controls.target.clone();
    controls.setEnabled(false);
    deps.setGridVisible(false);
    deps.setPlayPreview(true);

    const viewer = new ViewerControls({
      camera,
      domElement: canvas,
      initialPosition: savedPosition.toArray(),
      initialTarget: savedTarget.toArray(),
    });

    let controller: FormatController;
    switch (format.type) {
      case 'video':
        controller = new VideoController(viewer, camera, format.keyframes);
        break;
      case 'animated':
        controller = new AnimatedController(scene, { duration: format.duration });
        break;
      case 'interactive':
        if (format.interactions.length === 0) {
          logOutput(
            'info',
            'No interactions yet — select an object and add one under Properties › Interaction',
          );
        }
        controller = new InteractiveController(camera, scene, canvas, format.interactions, {
          onShowInfo: (_entityId, payload) =>
            setInfo({
              title:
                typeof payload['title'] === 'string' && payload['title']
                  ? payload['title']
                  : 'Info',
              description:
                typeof payload['description'] === 'string' ? payload['description'] : undefined,
            }),
          onNavigate: (url) => logOutput('info', `A viewer would be taken to ${url}`),
        });
        break;
      case 'turntable':
      default:
        controller = new TurntableController(viewer, {
          speed: format.type === 'turntable' ? format.speed : 1,
          axis: format.type === 'turntable' ? format.axis : 'y',
        });
        break;
    }

    const removeFrame = deps.addFrameCallback((dt) => {
      viewer.update();
      controller.update(dt);
    });

    sessionRef.current = { viewer, controller, savedPosition, savedTarget, removeFrame };
    useEditorStore.getState().setPlaying(true);
    logOutput(
      'ok',
      `Playing as ${FORMAT_LABEL[format.type]} — press Esc or Stop to return to editing`,
    );
  }, []);

  // Never leave the editor stuck in Play when it unmounts
  useEffect(() => stop, [stop]);

  return { play, stop, info, closeInfo: useCallback(() => setInfo(null), []) };
}
