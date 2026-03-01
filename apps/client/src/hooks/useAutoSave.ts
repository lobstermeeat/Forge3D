import { useEffect, useRef, useState } from 'react';
import type { SceneManager } from '@forge3d/engine';
import { trpc } from '@/api/trpc';

export type SaveStatus = 'idle' | 'saving' | 'saved' | 'error';

export function useAutoSave(
  sceneManager: SceneManager,
  sceneId: string | undefined,
  enabled: boolean,
) {
  const saveMutation = trpc.scene.save.useMutation({});
  const timerRef = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);
  const [saveStatus, setSaveStatus] = useState<SaveStatus>('idle');

  useEffect(() => {
    if (!sceneId || !enabled) return;

    const unsub = sceneManager.subscribe(() => {
      clearTimeout(timerRef.current);
      setSaveStatus('idle');
      timerRef.current = setTimeout(() => {
        setSaveStatus('saving');
        const data = sceneManager.serialize();
        saveMutation
          .mutateAsync({ id: sceneId, data: data as unknown as Record<string, unknown> })
          .then(() => setSaveStatus('saved'))
          .catch(() => setSaveStatus('error'));
      }, 2000);
    });

    return () => {
      unsub();
      clearTimeout(timerRef.current);
    };
  }, [sceneManager, sceneId, enabled]); // eslint-disable-line react-hooks/exhaustive-deps

  return saveStatus;
}
