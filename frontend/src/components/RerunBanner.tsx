import { AlertTriangle, RefreshCw } from "lucide-react";

interface Props {
  onRerun: () => void;
  simulating: boolean;
}

export default function RerunBanner({ onRerun, simulating }: Props) {
  return (
    <div className="bg-gradient-to-r from-amber-50 to-orange-50 border-b border-amber-200 px-6 py-3 flex items-center justify-between">
      <div className="flex items-center gap-2 text-amber-800">
        <AlertTriangle className="w-4 h-4 text-amber-500" />
        <span className="text-sm font-medium">
          Parameters changed — results are stale
        </span>
      </div>
      <button
        onClick={onRerun}
        disabled={simulating}
        className="flex items-center gap-1.5 rounded-lg bg-amber-500 hover:bg-amber-600 text-white px-4 py-1.5 text-sm font-semibold transition-colors disabled:opacity-50 shadow-sm"
      >
        <RefreshCw className={`w-3.5 h-3.5 ${simulating ? "animate-spin" : ""}`} />
        Rerun Simulation
      </button>
    </div>
  );
}
