import { useState, useRef, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { trpc } from '@/api/trpc';

interface SearchBarProps {
  initialQuery?: string;
  className?: string;
}

export function SearchBar({ initialQuery = '', className = '' }: SearchBarProps) {
  const navigate = useNavigate();
  const [query, setQuery] = useState(initialQuery);
  const [debouncedQuery, setDebouncedQuery] = useState('');
  const [showDropdown, setShowDropdown] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);

  // Debounce query
  useEffect(() => {
    const timer = setTimeout(() => {
      setDebouncedQuery(query.trim());
    }, 300);
    return () => clearTimeout(timer);
  }, [query]);

  const quickSearch = trpc.search.experiences.useQuery(
    { query: debouncedQuery, limit: 5 },
    { enabled: debouncedQuery.length >= 2 },
  );

  // Close dropdown on outside click
  useEffect(() => {
    function handleClick(e: MouseEvent) {
      if (containerRef.current && !containerRef.current.contains(e.target as Node)) {
        setShowDropdown(false);
      }
    }
    document.addEventListener('mousedown', handleClick);
    return () => document.removeEventListener('mousedown', handleClick);
  }, []);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (query.trim()) {
      setShowDropdown(false);
      navigate(`/search?q=${encodeURIComponent(query.trim())}`);
    }
  };

  const handleResultClick = (slug: string) => {
    setShowDropdown(false);
    navigate(`/view/${slug}`);
  };

  return (
    <div ref={containerRef} className={`relative ${className}`}>
      <form onSubmit={handleSubmit}>
        <div className="relative">
          <svg
            className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-[#6c7086]"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"
            />
          </svg>
          <input
            type="text"
            value={query}
            onChange={(e) => {
              setQuery(e.target.value);
              setShowDropdown(true);
            }}
            onFocus={() => setShowDropdown(true)}
            placeholder="Search experiences..."
            className="w-full rounded-lg border border-[#313244] bg-[#181825] py-2 pl-10 pr-4 text-sm text-[#cdd6f4] placeholder-[#6c7086] focus:border-[#89b4fa] focus:outline-none"
          />
        </div>
      </form>

      {/* Quick results dropdown */}
      {showDropdown && debouncedQuery.length >= 2 && quickSearch.data && quickSearch.data.items.length > 0 && (
        <div className="absolute left-0 right-0 top-full z-50 mt-1 rounded-lg border border-[#313244] bg-[#1e1e2e] shadow-lg">
          {quickSearch.data.items.map((item) => (
            <button
              key={item.id}
              onClick={() => handleResultClick(item.slug)}
              className="flex w-full items-center gap-3 px-3 py-2 text-left hover:bg-[#313244] first:rounded-t-lg last:rounded-b-lg"
            >
              {item.thumbnailUrl ? (
                <img
                  src={item.thumbnailUrl}
                  alt=""
                  className="h-8 w-12 rounded object-cover"
                />
              ) : (
                <div className="flex h-8 w-12 items-center justify-center rounded bg-[#45475a] text-xs text-[#6c7086]">
                  3D
                </div>
              )}
              <div className="min-w-0 flex-1">
                <p className="truncate text-sm text-[#cdd6f4]">{item.title}</p>
                <p className="truncate text-xs text-[#6c7086]">
                  by {item.creator.displayName}
                </p>
              </div>
            </button>
          ))}
          <button
            onClick={handleSubmit}
            className="w-full border-t border-[#313244] px-3 py-2 text-center text-xs text-[#89b4fa] hover:bg-[#313244]"
          >
            See all results for &quot;{debouncedQuery}&quot;
          </button>
        </div>
      )}
    </div>
  );
}
