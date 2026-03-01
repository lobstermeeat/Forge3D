interface InfoPopupProps {
  title: string;
  description?: string;
  onClose: () => void;
}

export function InfoPopup({ title, description, onClose }: InfoPopupProps) {
  return (
    <div className="absolute left-1/2 top-4 z-10 w-80 -translate-x-1/2 rounded-lg border border-[#313244] bg-[#1e1e2e]/95 p-4 shadow-xl backdrop-blur-sm">
      <div className="flex items-start justify-between gap-2">
        <h3 className="text-sm font-semibold text-[#cdd6f4]">{title}</h3>
        <button
          onClick={onClose}
          className="shrink-0 rounded p-0.5 text-[#6c7086] hover:bg-[#313244] hover:text-[#cdd6f4]"
        >
          <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
      </div>
      {description && (
        <p className="mt-2 text-xs leading-relaxed text-[#a6adc8]">{description}</p>
      )}
    </div>
  );
}
