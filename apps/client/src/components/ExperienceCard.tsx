import { useNavigate } from 'react-router-dom';
import type { ExperienceMeta } from '@forge3d/shared';

interface ExperienceCardProps {
  experience: ExperienceMeta;
}

const formatBadge: Record<string, string> = {
  turntable: '360',
  animated: 'Anim',
  video: 'Video',
  interactive: 'Interactive',
};

export function ExperienceCard({ experience }: ExperienceCardProps) {
  const navigate = useNavigate();

  return (
    <a
      href={`/view/${experience.slug}`}
      className="group block cursor-pointer overflow-hidden rounded-lg bg-[#1e1e2e] transition-transform hover:scale-[1.02]"
    >
      {/* Thumbnail */}
      <div className="relative aspect-video w-full overflow-hidden bg-[#181825]">
        {experience.thumbnailUrl ? (
          <img
            src={experience.thumbnailUrl}
            alt={experience.title}
            className="h-full w-full object-cover transition-transform group-hover:scale-105"
          />
        ) : (
          <div className="flex h-full w-full items-center justify-center">
            <svg className="h-10 w-10 text-[#45475a]" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M21 7.5l-9-5.25L3 7.5m18 0l-9 5.25m9-5.25v9l-9 5.25M3 7.5l9 5.25M3 7.5v9l9 5.25m0-9v9" />
            </svg>
          </div>
        )}
        {/* Format badge */}
        <span className="absolute right-2 top-2 rounded bg-black/50 px-1.5 py-0.5 text-[10px] font-medium text-white backdrop-blur-sm">
          {formatBadge[experience.formatType] ?? experience.formatType}
        </span>
      </div>

      {/* Info */}
      <div className="p-3">
        <h3 className="truncate text-sm font-medium text-[#cdd6f4]">
          {experience.title}
        </h3>
        <div className="mt-1 flex items-center gap-2">
          {experience.creator.avatarUrl ? (
            <img
              src={experience.creator.avatarUrl}
              alt={experience.creator.displayName}
              className="h-5 w-5 rounded-full"
            />
          ) : (
            <div className="flex h-5 w-5 items-center justify-center rounded-full bg-[#45475a] text-[10px] text-[#cdd6f4]">
              {(experience.creator.displayName ?? experience.creator.username)[0]?.toUpperCase()}
            </div>
          )}
          <a
            className="truncate text-xs text-[#a6adc8] hover:text-[#89b4fa]"
            href={`/creator/${experience.creator.username}`}
            onClick={(e) => {
              e.preventDefault();
              e.stopPropagation();
              navigate(`/creator/${experience.creator.username}`);
            }}
          >
            {experience.creator.displayName ?? experience.creator.username}
          </a>
        </div>
        <div className="mt-1.5 flex items-center gap-3 text-[10px] text-[#6c7086]">
          <span>{experience.viewCount.toLocaleString()} views</span>
          <span>{experience.likeCount.toLocaleString()} likes</span>
        </div>
      </div>
    </a>
  );
}
