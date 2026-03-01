import { useState } from 'react';
import { trpc } from '@/api/trpc';
import { useSession } from '@/auth/client';
import type { Comment as CommentType } from '@forge3d/shared';

interface CommentSectionProps {
  experienceId: string;
}

export function CommentSection({ experienceId }: CommentSectionProps) {
  const { data: session } = useSession();
  const [sort, setSort] = useState<'newest' | 'popular'>('newest');
  const [body, setBody] = useState('');
  const [replyTo, setReplyTo] = useState<string | null>(null);
  const [replyBody, setReplyBody] = useState('');

  const commentsQuery = trpc.interaction.listComments.useQuery({
    experienceId,
    sort,
    limit: 20,
  });

  const commentMutation = trpc.interaction.comment.useMutation({
    onSuccess: () => {
      setBody('');
      setReplyTo(null);
      setReplyBody('');
      commentsQuery.refetch();
    },
  });

  const deleteMutation = trpc.interaction.deleteComment.useMutation({
    onSuccess: () => commentsQuery.refetch(),
  });

  const handleSubmit = () => {
    if (!body.trim()) return;
    commentMutation.mutate({ experienceId, body: body.trim() });
  };

  const handleReply = (parentId: string) => {
    if (!replyBody.trim()) return;
    commentMutation.mutate({
      experienceId,
      body: replyBody.trim(),
      parentId,
    });
  };

  const comments = commentsQuery.data?.items ?? [];

  return (
    <div className="mt-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-medium text-[#cdd6f4]">
          Comments ({comments.length})
        </h3>
        <div className="flex gap-1">
          <button
            onClick={() => setSort('newest')}
            className={`rounded px-2 py-1 text-xs ${
              sort === 'newest'
                ? 'bg-[#313244] text-[#cdd6f4]'
                : 'text-[#6c7086] hover:text-[#a6adc8]'
            }`}
          >
            Newest
          </button>
          <button
            onClick={() => setSort('popular')}
            className={`rounded px-2 py-1 text-xs ${
              sort === 'popular'
                ? 'bg-[#313244] text-[#cdd6f4]'
                : 'text-[#6c7086] hover:text-[#a6adc8]'
            }`}
          >
            Popular
          </button>
        </div>
      </div>

      {/* New comment form */}
      {session?.user ? (
        <div className="mt-3 flex gap-2">
          <textarea
            value={body}
            onChange={(e) => setBody(e.target.value)}
            placeholder="Write a comment..."
            maxLength={2000}
            rows={2}
            className="flex-1 resize-none rounded-md border border-[#313244] bg-[#181825] px-3 py-2 text-sm text-[#cdd6f4] placeholder-[#6c7086] focus:border-[#89b4fa] focus:outline-none"
          />
          <button
            onClick={handleSubmit}
            disabled={!body.trim() || commentMutation.isPending}
            className="self-end rounded-md bg-[#89b4fa] px-4 py-2 text-sm font-medium text-[#11111b] hover:bg-[#74c7ec] disabled:opacity-50"
          >
            Post
          </button>
        </div>
      ) : (
        <p className="mt-3 text-sm text-[#6c7086]">
          <a href="/login" className="text-[#89b4fa] hover:underline">Sign in</a> to comment
        </p>
      )}

      {/* Comments list */}
      <div className="mt-4 space-y-4">
        {commentsQuery.isLoading && (
          <div className="flex justify-center py-4">
            <div className="h-5 w-5 animate-spin rounded-full border-2 border-[#cdd6f4] border-t-transparent" />
          </div>
        )}

        {comments.map((comment) => (
          <CommentItem
            key={comment.id}
            comment={comment}
            currentUserId={session?.user?.id}
            replyTo={replyTo}
            replyBody={replyBody}
            onReplyTo={setReplyTo}
            onReplyBodyChange={setReplyBody}
            onReply={handleReply}
            onDelete={(id) => deleteMutation.mutate({ commentId: id })}
            isReplying={commentMutation.isPending}
          />
        ))}

        {!commentsQuery.isLoading && comments.length === 0 && (
          <p className="py-4 text-center text-sm text-[#6c7086]">No comments yet</p>
        )}
      </div>
    </div>
  );
}

interface CommentItemProps {
  comment: CommentType;
  currentUserId?: string;
  replyTo: string | null;
  replyBody: string;
  onReplyTo: (id: string | null) => void;
  onReplyBodyChange: (body: string) => void;
  onReply: (parentId: string) => void;
  onDelete: (id: string) => void;
  isReplying: boolean;
  isReply?: boolean;
}

function CommentItem({
  comment,
  currentUserId,
  replyTo,
  replyBody,
  onReplyTo,
  onReplyBodyChange,
  onReply,
  onDelete,
  isReplying,
  isReply,
}: CommentItemProps) {
  const timeAgo = getTimeAgo(comment.createdAt);

  return (
    <div className={isReply ? 'ml-8' : ''}>
      <div className="flex gap-2.5">
        {/* Avatar */}
        {comment.userAvatar ? (
          <img
            src={comment.userAvatar}
            alt={comment.userName}
            className="h-7 w-7 rounded-full"
          />
        ) : (
          <div className="flex h-7 w-7 items-center justify-center rounded-full bg-[#45475a] text-xs text-[#cdd6f4]">
            {comment.userName[0]?.toUpperCase()}
          </div>
        )}

        <div className="flex-1">
          <div className="flex items-center gap-2">
            <span className="text-sm font-medium text-[#cdd6f4]">{comment.userName}</span>
            <span className="text-xs text-[#6c7086]">{timeAgo}</span>
          </div>
          <p className="mt-0.5 whitespace-pre-wrap text-sm text-[#a6adc8]">{comment.body}</p>
          <div className="mt-1 flex items-center gap-3">
            {!isReply && currentUserId && (
              <button
                onClick={() => onReplyTo(replyTo === comment.id ? null : comment.id)}
                className="text-xs text-[#6c7086] hover:text-[#89b4fa]"
              >
                Reply
              </button>
            )}
            {currentUserId === comment.userId && (
              <button
                onClick={() => onDelete(comment.id)}
                className="text-xs text-[#6c7086] hover:text-[#f38ba8]"
              >
                Delete
              </button>
            )}
          </div>

          {/* Reply form */}
          {replyTo === comment.id && (
            <div className="mt-2 flex gap-2">
              <textarea
                value={replyBody}
                onChange={(e) => onReplyBodyChange(e.target.value)}
                placeholder={`Reply to ${comment.userName}...`}
                maxLength={2000}
                rows={2}
                className="flex-1 resize-none rounded-md border border-[#313244] bg-[#181825] px-3 py-2 text-sm text-[#cdd6f4] placeholder-[#6c7086] focus:border-[#89b4fa] focus:outline-none"
              />
              <div className="flex flex-col gap-1 self-end">
                <button
                  onClick={() => onReply(comment.id)}
                  disabled={!replyBody.trim() || isReplying}
                  className="rounded-md bg-[#89b4fa] px-3 py-1.5 text-xs font-medium text-[#11111b] hover:bg-[#74c7ec] disabled:opacity-50"
                >
                  Reply
                </button>
                <button
                  onClick={() => onReplyTo(null)}
                  className="rounded-md px-3 py-1.5 text-xs text-[#6c7086] hover:text-[#a6adc8]"
                >
                  Cancel
                </button>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Nested replies */}
      {comment.replies && comment.replies.length > 0 && (
        <div className="mt-3 space-y-3">
          {comment.replies.map((reply) => (
            <CommentItem
              key={reply.id}
              comment={reply}
              currentUserId={currentUserId}
              replyTo={replyTo}
              replyBody={replyBody}
              onReplyTo={onReplyTo}
              onReplyBodyChange={onReplyBodyChange}
              onReply={onReply}
              onDelete={onDelete}
              isReplying={isReplying}
              isReply
            />
          ))}
        </div>
      )}
    </div>
  );
}

function getTimeAgo(dateStr: string): string {
  const now = Date.now();
  const then = new Date(dateStr).getTime();
  const diffMs = now - then;
  const diffMin = Math.floor(diffMs / 60000);
  if (diffMin < 1) return 'just now';
  if (diffMin < 60) return `${diffMin}m ago`;
  const diffHr = Math.floor(diffMin / 60);
  if (diffHr < 24) return `${diffHr}h ago`;
  const diffDay = Math.floor(diffHr / 24);
  if (diffDay < 30) return `${diffDay}d ago`;
  const diffMonth = Math.floor(diffDay / 30);
  return `${diffMonth}mo ago`;
}
