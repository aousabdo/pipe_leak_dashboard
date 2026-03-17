interface Props {
  numPipes?: number;
  numEvents?: number;
  hasModel: boolean;
  modelType?: string;
}

const MODEL_LABELS: Record<string, string> = {
  xgboost: "XGBoost",
  lightgbm: "LightGBM",
  random_forest: "Random Forest",
  logistic_regression: "Logistic Reg.",
  gradient_boosting: "Gradient Boost",
  stacking_ensemble: "Stacking",
  voting_ensemble: "Voting",
  blended_boosting: "Blended Boost",
};

export default function Header({ numPipes, numEvents, hasModel, modelType }: Props) {
  return (
    <header className="bg-gradient-to-r from-slate-900 via-blue-900 to-cyan-900 px-6 py-4 shadow-lg">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-xl font-bold text-white tracking-tight flex items-center gap-2">
            <span className="text-2xl">💧</span> Water Network Leak Predictor
          </h1>
          <p className="text-sm text-slate-300 mt-0.5">
            Hydraulic simulation &bull; Weibull deterioration &bull; ML risk scoring
          </p>
        </div>
        {numPipes != null && (
          <div className="flex items-center gap-4">
            <StatusPill label="Pipes" value={numPipes.toLocaleString()} color="blue" />
            <StatusPill label="Events" value={(numEvents ?? 0).toLocaleString()} color="amber" />
            <StatusPill
              label="Model"
              value={hasModel ? (MODEL_LABELS[modelType || ""] || "Trained") : "Not trained"}
              color={hasModel ? "green" : "slate"}
            />
          </div>
        )}
      </div>
    </header>
  );
}

function StatusPill({ label, value, color }: { label: string; value: string; color: string }) {
  const colors: Record<string, string> = {
    blue: "bg-blue-500/20 text-blue-200 border-blue-400/30",
    amber: "bg-amber-500/20 text-amber-200 border-amber-400/30",
    green: "bg-green-500/20 text-green-200 border-green-400/30",
    slate: "bg-slate-500/20 text-slate-300 border-slate-400/30",
  };
  return (
    <div className={`rounded-full border px-3 py-1 text-xs font-medium ${colors[color] || colors.slate}`}>
      <span className="opacity-70">{label}:</span> <span className="font-bold">{value}</span>
    </div>
  );
}
