import { useState, useCallback, useEffect, useRef } from 'react';
import { Link } from 'react-router-dom';
import { trpc } from '@/api/trpc';
import { useSession } from '@/auth/client';
import { ExperienceGrid } from '@/components/ExperienceGrid';
import { useInfiniteScroll } from '@/hooks/useInfiniteScroll';
import { SearchBar } from '@/components/SearchBar';
import { CategoryPicker } from '@/components/CategoryPicker';
import type { ExperienceMeta } from '@forge3d/shared';
import { MOCK_EXPERIENCES, USE_MOCKS } from '@/mocks/experiences';

type Tab = 'explore' | 'trending' | 'following';

export function FeedPage() {
  const { data: session } = useSession();
  const [tab, setTab] = useState<Tab>('explore');
  const [categoryId, setCategoryId] = useState<string | null>(null);
  const [exploreCursor, setExploreCursor] = useState<string | undefined>();
  const [followCursor, setFollowCursor] = useState<string | undefined>();
  const [exploreItems, setExploreItems] = useState<ExperienceMeta[]>([]);
  const [followItems, setFollowItems] = useState<ExperienceMeta[]>([]);
  const [trendingItems, setTrendingItems] = useState<ExperienceMeta[]>([]);
  const [exploreHasMore, setExploreHasMore] = useState(true);
  const [followHasMore, setFollowHasMore] = useState(true);

  const exploreCursorRef = useRef(exploreCursor);
  exploreCursorRef.current = exploreCursor;
  const followCursorRef = useRef(followCursor);
  followCursorRef.current = followCursor;

  // Reset explore when category changes
  useEffect(() => {
    setExploreCursor(undefined);
    setExploreItems([]);
    setExploreHasMore(true);
  }, [categoryId]);

  // Explore query
  const exploreQuery = trpc.experience.listExplore.useQuery(
    { limit: 20, cursor: exploreCursor, categoryId: categoryId ?? undefined },
    { enabled: tab === 'explore' && !USE_MOCKS },
  );

  useEffect(() => {
    if (USE_MOCKS) {
      setExploreItems(MOCK_EXPERIENCES);
      setExploreHasMore(false);
      return;
    }
    if (exploreQuery.data) {
      const mapped = mapExperiences(exploreQuery.data.items);
      setExploreItems((prev) => (exploreCursorRef.current ? [...prev, ...mapped] : mapped));
      setExploreHasMore(exploreQuery.data.hasMore);
    }
  }, [exploreQuery.data]);

  // Trending query
  const trendingQuery = trpc.experience.listTrending.useQuery(
    { limit: 20 },
    { enabled: tab === 'trending' && !USE_MOCKS },
  );

  useEffect(() => {
    if (USE_MOCKS) {
      setTrendingItems(MOCK_EXPERIENCES.slice(0, 4));
      return;
    }
    if (trendingQuery.data) {
      setTrendingItems(trendingQuery.data.items);
    }
  }, [trendingQuery.data]);

  // Following query
  const followQuery = trpc.experience.listFollowing.useQuery(
    { limit: 20, cursor: followCursor },
    { enabled: tab === 'following' && !!session?.user },
  );

  useEffect(() => {
    if (followQuery.data) {
      const mapped = mapExperiences(followQuery.data.items);
      setFollowItems((prev) => (followCursorRef.current ? [...prev, ...mapped] : mapped));
      setFollowHasMore(followQuery.data.hasMore);
    }
  }, [followQuery.data]);

  const loadMoreExplore = useCallback(() => {
    if (exploreHasMore && !exploreQuery.isFetching && exploreItems.length > 0) {
      setExploreCursor(exploreItems[exploreItems.length - 1]?.id);
    }
  }, [exploreHasMore, exploreQuery.isFetching, exploreItems]);

  const loadMoreFollow = useCallback(() => {
    if (followHasMore && !followQuery.isFetching && followItems.length > 0) {
      setFollowCursor(followItems[followItems.length - 1]?.id);
    }
  }, [followHasMore, followQuery.isFetching, followItems]);

  const { sentinelRef: exploreSentinel } = useInfiniteScroll({
    onLoadMore: loadMoreExplore,
    enabled: tab === 'explore' && exploreHasMore && !exploreQuery.isFetching,
  });

  const { sentinelRef: followSentinel } = useInfiniteScroll({
    onLoadMore: loadMoreFollow,
    enabled: tab === 'following' && followHasMore && !followQuery.isFetching,
  });

  return (
    <div className="min-h-screen bg-[#11111b]">
      {/* Header */}
      <header className="border-b border-[#313244] bg-[#1e1e2e]">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-4 py-3">
          <Link to="/" className="text-lg font-bold text-[#cdd6f4]">
            Forge3D
          </Link>
          <div className="flex items-center gap-3">
            <SearchBar className="hidden w-64 sm:block" />
            {session?.user ? (
              <>
                <Link
                  to="/dashboard"
                  className="rounded-md px-3 py-1.5 text-sm text-[#a6adc8] hover:bg-[#313244] hover:text-[#cdd6f4]"
                >
                  Dashboard
                </Link>
                <Link
                  to="/settings/profile"
                  className="rounded-md px-3 py-1.5 text-sm text-[#a6adc8] hover:bg-[#313244] hover:text-[#cdd6f4]"
                >
                  Profile
                </Link>
              </>
            ) : (
              <Link
                to="/login"
                className="rounded-md bg-[#89b4fa] px-3 py-1.5 text-sm font-medium text-[#11111b] hover:bg-[#74c7ec]"
              >
                Sign In
              </Link>
            )}
          </div>
        </div>
      </header>

      {/* Tabs + Categories */}
      <div className="mx-auto max-w-7xl px-4 pt-6">
        {/* Mobile search */}
        <div className="mb-4 sm:hidden">
          <SearchBar />
        </div>

        <div className="flex gap-1 border-b border-[#313244]">
          <button
            className={`px-4 py-2 text-sm font-medium transition-colors ${
              tab === 'explore'
                ? 'border-b-2 border-[#89b4fa] text-[#89b4fa]'
                : 'text-[#6c7086] hover:text-[#a6adc8]'
            }`}
            onClick={() => setTab('explore')}
          >
            Explore
          </button>
          <button
            className={`px-4 py-2 text-sm font-medium transition-colors ${
              tab === 'trending'
                ? 'border-b-2 border-[#89b4fa] text-[#89b4fa]'
                : 'text-[#6c7086] hover:text-[#a6adc8]'
            }`}
            onClick={() => setTab('trending')}
          >
            Trending
          </button>
          {session?.user && (
            <button
              className={`px-4 py-2 text-sm font-medium transition-colors ${
                tab === 'following'
                  ? 'border-b-2 border-[#89b4fa] text-[#89b4fa]'
                  : 'text-[#6c7086] hover:text-[#a6adc8]'
              }`}
              onClick={() => setTab('following')}
            >
              Following
            </button>
          )}
        </div>

        {/* Category filter (only on Explore tab) */}
        {tab === 'explore' && (
          <div className="mt-4">
            <CategoryPicker selected={categoryId} onSelect={setCategoryId} />
          </div>
        )}

        {/* Content */}
        <div className="py-6">
          {tab === 'explore' && (
            <ExperienceGrid
              items={exploreItems}
              loading={exploreQuery.isLoading}
              emptyMessage="No experiences published yet. Be the first!"
              sentinelRef={exploreSentinel}
            />
          )}

          {tab === 'trending' && (
            <ExperienceGrid
              items={trendingItems}
              loading={trendingQuery.isLoading}
              emptyMessage="No trending experiences yet"
            />
          )}

          {tab === 'following' && (
            <ExperienceGrid
              items={followItems}
              loading={followQuery.isLoading}
              emptyMessage="Follow creators to see their work here"
              sentinelRef={followSentinel}
            />
          )}
        </div>
      </div>
    </div>
  );
}

/**
 * Map raw DB experience rows (from listExplore/listFollowing) to ExperienceMeta.
 * These endpoints return the raw Drizzle rows, not typed ExperienceMeta.
 */
function mapExperiences(items: Record<string, unknown>[]): ExperienceMeta[] {
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
      id: (item.creatorId as string) ?? '',
      username: 'creator',
      displayName: 'Creator',
      avatarUrl: null,
    },
    categoryId: (item.categoryId as string) ?? null,
    publishedAt: item.publishedAt ? String(item.publishedAt) : null,
    createdAt: item.createdAt ? String(item.createdAt) : new Date().toISOString(),
  }));
}
