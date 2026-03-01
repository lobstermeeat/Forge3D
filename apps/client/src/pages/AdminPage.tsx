import { useState } from 'react';
import { Link } from 'react-router-dom';
import { trpc } from '@/api/trpc';
import type { Report } from '@forge3d/shared';

type TabStatus = 'pending' | 'resolved' | 'dismissed';

export function AdminPage() {
  const [tab, setTab] = useState<TabStatus>('pending');

  const statsQuery = trpc.moderation.getStats.useQuery();
  const reportsQuery = trpc.moderation.listReports.useQuery({ status: tab, limit: 20 });

  const reviewMutation = trpc.moderation.reviewReport.useMutation({
    onSuccess: () => {
      reportsQuery.refetch();
      statsQuery.refetch();
    },
  });

  const stats = statsQuery.data;
  const reports = reportsQuery.data?.items ?? [];

  // If stats query fails with FORBIDDEN, user is not admin
  if (statsQuery.error?.data?.code === 'FORBIDDEN') {
    return (
      <div className="flex min-h-screen flex-col items-center justify-center gap-4 bg-[#11111b]">
        <p className="text-lg text-[#f38ba8]">Access denied</p>
        <Link to="/explore" className="text-sm text-[#89b4fa] underline">
          Go to Explore
        </Link>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-[#11111b]">
      {/* Header */}
      <header className="border-b border-[#313244] bg-[#1e1e2e]">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-4 py-3">
          <Link to="/explore" className="text-lg font-bold text-[#cdd6f4]">
            Forge3D
          </Link>
          <span className="rounded bg-[#f38ba8]/20 px-2 py-0.5 text-xs font-medium text-[#f38ba8]">
            Admin
          </span>
        </div>
      </header>

      <div className="mx-auto max-w-5xl px-4 py-6">
        <h1 className="mb-6 text-xl font-bold text-[#cdd6f4]">Moderation Dashboard</h1>

        {/* Stats bar */}
        {stats && (
          <div className="mb-6 grid grid-cols-3 gap-4">
            <StatCard label="Pending" value={stats.pending} color="#fab387" />
            <StatCard label="Resolved" value={stats.resolved} color="#a6e3a1" />
            <StatCard label="Dismissed" value={stats.dismissed} color="#6c7086" />
          </div>
        )}

        {/* Tabs */}
        <div className="flex gap-1 border-b border-[#313244]">
          {(['pending', 'resolved', 'dismissed'] as const).map((t) => (
            <button
              key={t}
              className={`px-4 py-2 text-sm font-medium capitalize transition-colors ${
                tab === t
                  ? 'border-b-2 border-[#89b4fa] text-[#89b4fa]'
                  : 'text-[#6c7086] hover:text-[#a6adc8]'
              }`}
              onClick={() => setTab(t)}
            >
              {t}
            </button>
          ))}
        </div>

        {/* Reports list */}
        <div className="mt-4 space-y-3">
          {reportsQuery.isLoading && (
            <div className="flex justify-center py-8">
              <div className="h-6 w-6 animate-spin rounded-full border-2 border-[#cdd6f4] border-t-transparent" />
            </div>
          )}

          {reports.map((report) => (
            <ReportCard
              key={report.id}
              report={report}
              onResolve={() => reviewMutation.mutate({ reportId: report.id, action: 'resolve' })}
              onDismiss={() => reviewMutation.mutate({ reportId: report.id, action: 'dismiss' })}
              isPending={tab === 'pending'}
              isActing={reviewMutation.isPending}
            />
          ))}

          {!reportsQuery.isLoading && reports.length === 0 && (
            <p className="py-8 text-center text-sm text-[#6c7086]">
              No {tab} reports
            </p>
          )}
        </div>
      </div>
    </div>
  );
}

function StatCard({ label, value, color }: { label: string; value: number; color: string }) {
  return (
    <div className="rounded-lg bg-[#1e1e2e] p-4">
      <p className="text-xs text-[#6c7086]">{label}</p>
      <p className="mt-1 text-2xl font-bold" style={{ color }}>
        {value}
      </p>
    </div>
  );
}

function ReportCard({
  report,
  onResolve,
  onDismiss,
  isPending,
  isActing,
}: {
  report: Report;
  onResolve: () => void;
  onDismiss: () => void;
  isPending: boolean;
  isActing: boolean;
}) {
  const timeAgo = getTimeAgo(report.createdAt);

  return (
    <div className="rounded-lg bg-[#1e1e2e] p-4">
      <div className="flex items-start justify-between gap-4">
        <div className="flex-1">
          <div className="flex items-center gap-2">
            <span className="rounded bg-[#313244] px-2 py-0.5 text-xs text-[#a6adc8]">
              {report.targetType}
            </span>
            <span className="text-xs text-[#6c7086]">{timeAgo}</span>
          </div>
          <p className="mt-2 text-sm text-[#cdd6f4]">{report.reason}</p>
          <p className="mt-1 text-xs text-[#6c7086]">
            Reported by {report.reporterName} · Target: {report.targetId.slice(0, 8)}...
          </p>
        </div>

        {isPending && (
          <div className="flex shrink-0 gap-2">
            <button
              onClick={onResolve}
              disabled={isActing}
              className="rounded-md bg-[#f38ba8] px-3 py-1.5 text-xs font-medium text-[#11111b] hover:bg-[#eba0ac] disabled:opacity-50"
            >
              Resolve
            </button>
            <button
              onClick={onDismiss}
              disabled={isActing}
              className="rounded-md bg-[#313244] px-3 py-1.5 text-xs text-[#a6adc8] hover:bg-[#45475a] disabled:opacity-50"
            >
              Dismiss
            </button>
          </div>
        )}
      </div>
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
