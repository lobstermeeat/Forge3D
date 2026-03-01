import { useState } from 'react';
import { trpc } from '@/api/trpc';
import { useSession } from '@/auth/client';

interface ReportButtonProps {
  targetType: 'experience' | 'comment' | 'user';
  targetId: string;
}

export function ReportButton({ targetType, targetId }: ReportButtonProps) {
  const { data: session } = useSession();
  const [showModal, setShowModal] = useState(false);
  const [reason, setReason] = useState('');
  const [status, setStatus] = useState<'idle' | 'success' | 'error'>('idle');

  const reportMutation = trpc.moderation.report.useMutation({
    onSuccess: (data) => {
      if (data.reported) {
        setStatus('success');
        setTimeout(() => {
          setShowModal(false);
          setReason('');
          setStatus('idle');
        }, 1500);
      } else {
        setStatus('error');
      }
    },
    onError: () => setStatus('error'),
  });

  if (!session?.user) return null;

  const handleSubmit = () => {
    if (reason.trim().length < 10) return;
    reportMutation.mutate({ targetType, targetId, reason: reason.trim() });
  };

  return (
    <>
      <button
        onClick={() => setShowModal(true)}
        className="flex items-center gap-1 text-xs text-[#6c7086] hover:text-[#f38ba8]"
        title="Report"
      >
        <svg className="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M3 21v-4m0 0V5a2 2 0 012-2h6.5l1 1H21l-3 6 3 6h-8.5l-1-1H5a2 2 0 00-2 2z"
          />
        </svg>
        Report
      </button>

      {/* Modal */}
      {showModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
          <div className="mx-4 w-full max-w-md rounded-lg bg-[#1e1e2e] p-6">
            <h3 className="mb-4 text-sm font-medium text-[#cdd6f4]">
              Report {targetType}
            </h3>

            {status === 'success' ? (
              <p className="py-4 text-center text-sm text-[#a6e3a1]">
                Report submitted. Thank you!
              </p>
            ) : (
              <>
                <textarea
                  value={reason}
                  onChange={(e) => setReason(e.target.value)}
                  placeholder="Describe why you're reporting this (min 10 characters)..."
                  maxLength={1000}
                  rows={4}
                  className="w-full resize-none rounded-md border border-[#313244] bg-[#181825] px-3 py-2 text-sm text-[#cdd6f4] placeholder-[#6c7086] focus:border-[#89b4fa] focus:outline-none"
                />
                <p className="mt-1 text-right text-xs text-[#6c7086]">
                  {reason.length}/1000
                </p>

                {status === 'error' && (
                  <p className="mt-2 text-sm text-[#f38ba8]">
                    Failed to submit report. You may have already reported this.
                  </p>
                )}

                <div className="mt-4 flex justify-end gap-2">
                  <button
                    onClick={() => {
                      setShowModal(false);
                      setReason('');
                      setStatus('idle');
                    }}
                    className="rounded-md px-4 py-2 text-sm text-[#a6adc8] hover:bg-[#313244]"
                  >
                    Cancel
                  </button>
                  <button
                    onClick={handleSubmit}
                    disabled={reason.trim().length < 10 || reportMutation.isPending}
                    className="rounded-md bg-[#f38ba8] px-4 py-2 text-sm font-medium text-[#11111b] hover:bg-[#eba0ac] disabled:opacity-50"
                  >
                    {reportMutation.isPending ? 'Submitting...' : 'Submit Report'}
                  </button>
                </div>
              </>
            )}
          </div>
        </div>
      )}
    </>
  );
}
