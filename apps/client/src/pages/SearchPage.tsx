import { useState, useEffect } from 'react';
import { useSearchParams, Link } from 'react-router-dom';
import { trpc } from '@/api/trpc';
import { useSession } from '@/auth/client';
import { ExperienceGrid } from '@/components/ExperienceGrid';
import { SearchBar } from '@/components/SearchBar';
import { CategoryPicker } from '@/components/CategoryPicker';
import type { ExperienceMeta } from '@forge3d/shared';

export function SearchPage() {
  const { data: session } = useSession();
  const [searchParams] = useSearchParams();
  const queryParam = searchParams.get('q') ?? '';
  const [categoryId, setCategoryId] = useState<string | null>(null);
  const [items, setItems] = useState<ExperienceMeta[]>([]);

  const searchQuery = trpc.search.experiences.useQuery(
    { query: queryParam, categoryId: categoryId ?? undefined, limit: 40 },
    { enabled: queryParam.length >= 1 },
  );

  useEffect(() => {
    if (searchQuery.data) {
      setItems(searchQuery.data.items);
    }
  }, [searchQuery.data]);

  return (
    <div className="min-h-screen bg-[#11111b]">
      {/* Header */}
      <header className="border-b border-[#313244] bg-[#1e1e2e]">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-4 py-3">
          <Link to="/explore" className="text-lg font-bold text-[#cdd6f4]">
            Forge3D
          </Link>
          <div className="flex items-center gap-3">
            {session?.user ? (
              <>
                <Link
                  to="/dashboard"
                  className="rounded-md px-3 py-1.5 text-sm text-[#a6adc8] hover:bg-[#313244] hover:text-[#cdd6f4]"
                >
                  Dashboard
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

      <div className="mx-auto max-w-7xl px-4 pt-6">
        {/* Search bar */}
        <SearchBar initialQuery={queryParam} className="mb-4 max-w-xl" />

        {/* Category filter */}
        <div className="mb-6">
          <CategoryPicker selected={categoryId} onSelect={setCategoryId} />
        </div>

        {/* Results header */}
        {queryParam && (
          <div className="mb-4 flex items-center justify-between">
            <p className="text-sm text-[#a6adc8]">
              {searchQuery.data
                ? `${searchQuery.data.totalHits} result${searchQuery.data.totalHits !== 1 ? 's' : ''} for "${queryParam}"`
                : 'Searching...'}
            </p>
          </div>
        )}

        {/* Results grid */}
        <ExperienceGrid
          items={items}
          loading={searchQuery.isLoading}
          emptyMessage={queryParam ? `No results found for "${queryParam}"` : 'Enter a search query'}
        />
      </div>
    </div>
  );
}
