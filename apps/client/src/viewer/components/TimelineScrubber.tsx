interface TimelineScrubberProps {
  currentTime: number;
  duration: number;
  playing: boolean;
  onTogglePlay: () => void;
  onSeek: (time: number) => void;
}

function formatTime(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, '0')}`;
}

export function TimelineScrubber({
  currentTime,
  duration,
  playing,
  onTogglePlay,
  onSeek,
}: TimelineScrubberProps) {
  return (
    <div className="flex items-center gap-3">
      <button
        onClick={onTogglePlay}
        className="rounded bg-white/10 px-3 py-1.5 text-xs font-medium text-white backdrop-blur-sm hover:bg-white/20"
      >
        {playing ? 'Pause' : 'Play'}
      </button>
      <input
        type="range"
        min={0}
        max={duration}
        step={0.01}
        value={currentTime}
        onChange={(e) => onSeek(parseFloat(e.target.value))}
        className="h-1 flex-1 cursor-pointer appearance-none rounded-full bg-white/20 accent-[#89b4fa]"
      />
      <span className="min-w-[60px] text-right text-[10px] text-white/70">
        {formatTime(currentTime)} / {formatTime(duration)}
      </span>
    </div>
  );
}
