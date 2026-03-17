interface Props {
  icon: string;
  label: string;
  value: string | number;
  accent: "blue" | "red" | "amber" | "green" | "purple" | "cyan";
  delta?: string;
  deltaType?: "positive" | "negative" | "neutral";
}

const accents: Record<string, { border: string; bg: string }> = {
  blue: { border: "border-t-blue-500", bg: "bg-blue-50" },
  red: { border: "border-t-red-500", bg: "bg-red-50" },
  amber: { border: "border-t-amber-500", bg: "bg-amber-50" },
  green: { border: "border-t-green-500", bg: "bg-green-50" },
  purple: { border: "border-t-purple-500", bg: "bg-purple-50" },
  cyan: { border: "border-t-cyan-500", bg: "bg-cyan-50" },
};

const deltaColors: Record<string, string> = {
  positive: "text-green-600",
  negative: "text-red-600",
  neutral: "text-slate-500",
};

export default function KpiCard({ icon, label, value, accent, delta, deltaType }: Props) {
  const a = accents[accent] || accents.blue;
  return (
    <div
      className={`bg-white rounded-xl border border-slate-200 border-t-[3px] ${a.border} p-4 shadow-sm hover:shadow-md hover:-translate-y-0.5 transition-all`}
    >
      <div className="text-2xl mb-1">{icon}</div>
      <div className="text-2xl font-bold text-slate-800 leading-tight">{value}</div>
      <div className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider mt-1">{label}</div>
      {delta && (
        <div className={`text-xs font-medium mt-1 ${deltaColors[deltaType || "neutral"]}`}>
          {delta}
        </div>
      )}
    </div>
  );
}
