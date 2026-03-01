import { useState, useCallback, useEffect, useRef } from 'react';
import { useParams, Link } from 'react-router-dom';
import { trpc } from '@/api/trpc';
import { useSession } from '@/auth/client';
import { ExperienceGrid } from '@/components/ExperienceGrid';
import { useInfiniteScroll } from '@/hooks/useInfiniteScroll';
import type { ExperienceMeta } from '@forge3d/shared';

export function ProfilePage() {
  const { username } = useParams<{ username: string }>();
  const { data: session } = useSession();
  const [cursor, setCursor] = useState<string | undefined>();
  const [items, setItems] = useState<ExperienceMeta[]>([]);
  const [hasMore, setHasMore] = useState(true);
  const cursorRef = useRef(cursor);
  cursorRef.current = cursor;

  const profileQuery = trpc.profile.getByUsername.useQuery(
    { username: username! },
    { enabled: !!username },
  );

  const profile = profileQuery.data;

  const experiencesQuery = trpc.experience.listByCreator.useQuery(
    { userId: profile?.id ?? '', limit: 20, cursor },
    { enabled: !!profile?.id },
  );

  useEffect(() => {
    if (experiencesQuery.data && profile) {
      const mapped = mapExperiences(experiencesQuery.data.items, profile);
      setItems((prev) => (cursorRef.current ? [...prev, ...mapped] : mapped));
      setHasMore(experiencesQuery.data.hasMore);
    }
  }, [experiencesQuery.data, profile]);

  const followMutation = trpc.social.follow.useMutation({
    onSuccess: () => profileQuery.refetch(),
  });
  const unfollowMutation = trpc.social.unfollow.useMutation({
    onSuccess: () => profileQuery.refetch(),
  });

  const loadMore = useCallback(() => {
    if (hasMore && !experiencesQuery.isFetching && items.length > 0) {
      setCursor(items[items.length - 1]?.id);
    }
  }, [hasMore, experiencesQuery.isFetching, items]);

  const { sentinelRef } = useInfiniteScroll({
    onLoadMore: loadMore,
    enabled: hasMore && !experiencesQuery.isFetching,
  });

  const handleFollowToggle = () => {
    if (!profile) return;
    if (profile.isFollowing) {
      unfollowMutation.mutate({ userId: profile.id });
    } else {
      followMutation.mutate({ userId: profile.id });
    }
  };

  if (profileQuery.isLoading) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-[#11111b]">
        <div className="h-8 w-8 animate-spin rounded-full border-2 border-[#cdd6f4] border-t-transparent" />
      </div>
    );
  }

  if (!profile) {
    return (
      <div className="flex min-h-screen flex-col items-center justify-center gap-4 bg-[#11111b]">
        <p className="text-lg text-[#f38ba8]">Creator not found</p>
        <Link to="/explore" className="text-sm text-[#89b4fa] underline">
          Browse experiences
        </Link>
      </div>
    );
  }

  const isOwnProfile = session?.user?.id === profile.id;

  return (
    <div className="min-h-screen bg-[#11111b]">
      {/* Header bar */}
      <header className="border-b border-[#313244] bg-[#1e1e2e]">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-4 py-3">
          <Link to="/explore" className="text-lg font-bold text-[#cdd6f4]">
            Forge3D
          </Link>
        </div>
      </header>

      {/* Profile header */}
      <div className="border-b border-[#313244] bg-[#1e1e2e]">
        {/* Banner area */}
        {profile.headerUrl && (
          <div className="h-40 w-full overflow-hidden">
            <img src={profile.headerUrl} alt="" className="h-full w-full object-cover" />
          </div>
        )}

        <div className="mx-auto max-w-7xl px-4 py-6">
          <div className="flex items-start gap-4">
            {/* Avatar */}
            {profile.avatarUrl ? (
              <img
                src={profile.avatarUrl}
                alt={profile.displayName}
                className="h-20 w-20 rounded-full border-2 border-[#313244]"
              />
            ) : (
              <div className="flex h-20 w-20 items-center justify-center rounded-full bg-[#45475a] text-2xl font-bold text-[#cdd6f4]">
                {profile.displayName[0]?.toUpperCase()}
              </div>
            )}

            {/* Info */}
            <div className="flex-1">
              <div className="flex items-center gap-3">
                <h1 className="text-xl font-bold text-[#cdd6f4]">{profile.displayName}</h1>
                {!isOwnProfile && session?.user && (
                  <button
                    onClick={handleFollowToggle}
                    disabled={followMutation.isPending || unfollowMutation.isPending}
                    className={`rounded-md px-4 py-1.5 text-sm font-medium transition-colors ${
                      profile.isFollowing
                        ? 'bg-[#313244] text-[#cdd6f4] hover:bg-[#45475a]'
                        : 'bg-[#89b4fa] text-[#11111b] hover:bg-[#74c7ec]'
                    }`}
                  >
                    {profile.isFollowing ? 'Unfollow' : 'Follow'}
                  </button>
                )}
                {isOwnProfile && (
                  <Link
                    to="/settings/profile"
                    className="rounded-md bg-[#313244] px-4 py-1.5 text-sm text-[#cdd6f4] hover:bg-[#45475a]"
                  >
                    Edit Profile
                  </Link>
                )}
              </div>
              <p className="mt-0.5 text-sm text-[#6c7086]">@{profile.username}</p>
              {profile.bio && (
                <p className="mt-2 max-w-xl text-sm text-[#a6adc8]">{profile.bio}</p>
              )}
              <div className="mt-2 flex gap-4 text-sm">
                <span className="text-[#a6adc8]">
                  <span className="font-medium text-[#cdd6f4]">{profile.followerCount}</span> followers
                </span>
                <span className="text-[#a6adc8]">
                  <span className="font-medium text-[#cdd6f4]">{profile.followingCount}</span> following
                </span>
              </div>
              {profile.websiteUrl && (
                <a
                  href={profile.websiteUrl}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="mt-1 inline-block text-sm text-[#89b4fa] hover:underline"
                >
                  {profile.websiteUrl.replace(/^https?:\/\//, '')}
                </a>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Experiences grid */}
      <div className="mx-auto max-w-7xl px-4 py-6">
        <h2 className="mb-4 text-sm font-medium text-[#a6adc8]">Published Experiences</h2>
        <ExperienceGrid
          items={items}
          loading={experiencesQuery.isLoading}
          emptyMessage="No published experiences yet"
          sentinelRef={sentinelRef}
        />
      </div>
    </div>
  );
}

function mapExperiences(
  items: Record<string, unknown>[],
  profile: { id: string; username: string; displayName: string; avatarUrl: string | null },
): ExperienceMeta[] {
  return items.map((item) => ({
    id: item.id as string,
    title: item.title as string,
    description: (item.description as string) ?? null,
    slug: item.slug as string,
    status: item.status as ExperienceMeta['status'],
    formatType: item.formatType as ExperienceMeta['formatType'],
    thumbnailUrl: (item.thumbnailUrl as string) ?? null,
    previewUrl: (item.previewUrl as string) ?? null,
    viewCount: (item.viewCount as number) ?? 0,
    likeCount: (item.likeCount as number) ?? 0,
    commentCount: (item.commentCount as number) ?? 0,
    polyCount: (item.polyCount as number) ?? null,
    fileSize: (item.fileSize as number) ?? null,
    creator: {
      id: profile.id,
      username: profile.username,
      displayName: profile.displayName,
      avatarUrl: profile.avatarUrl,
    },
    categoryId: (item.categoryId as string) ?? null,
    publishedAt: item.publishedAt ? String(item.publishedAt) : null,
    createdAt: item.createdAt ? String(item.createdAt) : new Date().toISOString(),
  }));
}
