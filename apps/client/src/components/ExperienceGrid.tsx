import type { ExperienceMeta } from '@forge3d/shared';
import { ExperienceCard } from './ExperienceCard';

interface ExperienceGridProps {
  items: ExperienceMeta[];
  loading?: boolean;
  emptyMessage?: string;
  sentinelRef?: React.RefObject<HTMLDivElement | null>;
}

function SkeletonCard() {
  return (
    <div className="overflow-hidden rounded-lg bg-[#1e1e2e]">
      <div className="aspect-video w-full animate-pulse bg-[#181825]" />
      <div className="p-3">
        <div className="h-4 w-3/4 animate-pulse rounded bg-[#313244]" />
        <div className="mt-2 h-3 w-1/2 animate-pulse rounded bg-[#313244]" />
      </div>
    </div>
  );
}

export function ExperienceGrid({ items, loading, emptyMessage, sentinelRef }: ExperienceGridProps) {
  if (!loading && items.length === 0) {
    return (
      <div className="flex min-h-[200px] items-center justify-center">
        <p className="text-sm text-[#6c7086]">{emptyMessage ?? 'No experiences yet'}</p>
      </div>
    );
  }

  return (
    <>
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
        {items.map((exp) => (
          <ExperienceCard key={exp.id} experience={exp} />
        ))}
        {loading &&
          Array.from({ length: 4 }).map((_, i) => <SkeletonCard key={`skel-${i}`} />)}
      </div>
      {sentinelRef && <div ref={sentinelRef} className="h-1" />}
    </>
  );
}
