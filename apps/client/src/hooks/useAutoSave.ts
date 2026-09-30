import { useCallback, useEffect, useRef, useState } from 'react';
import type { SceneManager } from '@forge3d/engine';
import { trpc } from '@/api/trpc';

export type SaveStatus = 'idle' | 'saving' | 'saved' | 'error';

export function useAutoSave(
  sceneManager: SceneManager,
  sceneId: string | undefined,
  enabled: boolean,
) {
  const saveMutation = trpc.scene.save.useMutation({});
  const mutateRef = useRef(saveMutation.mutateAsync);
  mutateRef.current = saveMutation.mutateAsync;
  const liveRef = useRef({ sceneId, enabled });
  liveRef.current = { sceneId, enabled };
  const timerRef = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);
  const [saveStatus, setSaveStatus] = useState<SaveStatus>('idle');

  /**
   * Save right away. Returns false when there is nothing to save to (a scratch scene)
   * or the scene hasn't finished loading — saving then would overwrite it with an empty one.
   */
  const saveNow = useCallback((): boolean => {
    const { sceneId: id, enabled: loaded } = liveRef.current;
    if (!id || !loaded) return false;
    clearTimeout(timerRef.current);
    setSaveStatus('saving');
    const data = sceneManager.serialize();
    mutateRef
      .current({ id, data: data as unknown as Record<string, unknown> })
      .then(() => setSaveStatus('saved'))
      .catch(() => setSaveStatus('error'));
    return true;
  }, [sceneManager]);

  useEffect(() => {
    if (!sceneId || !enabled) return;

    const unsub = sceneManager.subscribe(() => {
      clearTimeout(timerRef.current);
      setSaveStatus('idle');
      timerRef.current = setTimeout(saveNow, 2000);
    });

    return () => {
      unsub();
      clearTimeout(timerRef.current);
    };
  }, [sceneManager, sceneId, enabled, saveNow]);

  return { saveStatus, saveNow };
}
