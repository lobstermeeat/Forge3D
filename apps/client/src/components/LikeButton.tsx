import { useState, useEffect } from 'react';
import { trpc } from '@/api/trpc';
import { useSession } from '@/auth/client';

interface LikeButtonProps {
  experienceId: string;
  initialCount: number;
}

export function LikeButton({ experienceId, initialCount }: LikeButtonProps) {
  const { data: session } = useSession();
  const [liked, setLiked] = useState(false);
  const [count, setCount] = useState(initialCount);

  const isLikedQuery = trpc.interaction.isLiked.useQuery(
    { experienceId },
    { enabled: !!session?.user },
  );

  useEffect(() => {
    if (isLikedQuery.data) {
      setLiked(isLikedQuery.data.liked);
    }
  }, [isLikedQuery.data]);

  const likeMutation = trpc.interaction.like.useMutation({
    onMutate: () => {
      setLiked(true);
      setCount((c) => c + 1);
    },
    onError: () => {
      setLiked(false);
      setCount((c) => c - 1);
    },
  });

  const unlikeMutation = trpc.interaction.unlike.useMutation({
    onMutate: () => {
      setLiked(false);
      setCount((c) => c - 1);
    },
    onError: () => {
      setLiked(true);
      setCount((c) => c + 1);
    },
  });

  const handleClick = () => {
    if (!session?.user) return;
    if (liked) {
      unlikeMutation.mutate({ experienceId });
    } else {
      likeMutation.mutate({ experienceId });
    }
  };

  return (
    <button
      onClick={handleClick}
      disabled={!session?.user}
      className="flex items-center gap-1.5 rounded-md px-3 py-1.5 text-sm transition-colors hover:bg-[#313244] disabled:cursor-default disabled:opacity-60"
      title={session?.user ? (liked ? 'Unlike' : 'Like') : 'Sign in to like'}
    >
      <svg
        className={`h-5 w-5 ${liked ? 'fill-[#f38ba8] text-[#f38ba8]' : 'fill-none text-[#a6adc8]'}`}
        viewBox="0 0 24 24"
        stroke="currentColor"
        strokeWidth={liked ? 0 : 1.5}
      >
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          d="M21 8.25c0-2.485-2.099-4.5-4.688-4.5-1.935 0-3.597 1.126-4.312 2.733-.715-1.607-2.377-2.733-4.313-2.733C5.1 3.75 3 5.765 3 8.25c0 7.22 9 12 9 12s9-4.78 9-12z"
        />
      </svg>
      <span className={liked ? 'text-[#f38ba8]' : 'text-[#a6adc8]'}>
        {count.toLocaleString()}
      </span>
    </button>
  );
}
