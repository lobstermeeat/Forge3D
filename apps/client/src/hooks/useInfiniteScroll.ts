import { useRef, useEffect, useCallback } from 'react';

interface UseInfiniteScrollOptions {
  onLoadMore: () => void;
  enabled: boolean;
  rootMargin?: string;
}

export function useInfiniteScroll({ onLoadMore, enabled, rootMargin = '200px' }: UseInfiniteScrollOptions) {
  const sentinelRef = useRef<HTMLDivElement>(null);
  const callbackRef = useRef(onLoadMore);
  callbackRef.current = onLoadMore;

  const handleIntersect = useCallback(
    (entries: IntersectionObserverEntry[]) => {
      const entry = entries[0];
      if (entry?.isIntersecting && enabled) {
        callbackRef.current();
      }
    },
    [enabled],
  );

  useEffect(() => {
    const node = sentinelRef.current;
    if (!node) return;

    const observer = new IntersectionObserver(handleIntersect, { rootMargin });
    observer.observe(node);

    return () => observer.disconnect();
  }, [handleIntersect, rootMargin]);

  return { sentinelRef };
}
