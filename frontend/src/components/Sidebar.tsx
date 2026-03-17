import { Droplets, Zap, Brain, Settings, ChevronLeft, ChevronRight, Cpu } from "lucide-react";

interface Props {
  params: { num_pipes: number; sim_years: number; seed: number };
  onParamsChange: (p: { num_pipes: number; sim_years: number; seed: number }) => void;
  onSimulate: () => void;
  onTrain: () => void;
  simulating: boolean;
  training: boolean;
  hasData: boolean;
  hasModel: boolean;
  dirty: boolean;
  modelType: string;
  onModelTypeChange: (type: string) => void;
  collapsed: boolean;
  onToggleCollapse: () => void;
}

const MODEL_OPTIONS = [
  { value: "xgboost", label: "XGBoost", group: "Individual" },
  { value: "lightgbm", label: "LightGBM", group: "Individual" },
  { value: "random_forest", label: "Random Forest", group: "Individual" },
  { value: "gradient_boosting", label: "Gradient Boosting", group: "Individual" },
  { value: "logistic_regression", label: "Logistic Regression", group: "Individual" },
  { value: "stacking_ensemble", label: "Stacking Ensemble", group: "Ensemble" },
  { value: "voting_ensemble", label: "Voting Ensemble", group: "Ensemble" },
  { value: "blended_boosting", label: "Blended Boosting", group: "Ensemble" },
];

export default function Sidebar({
  params,
  onParamsChange,
  onSimulate,
  onTrain,
  simulating,
  training,
  hasData,
  hasModel,
  dirty,
  modelType,
  onModelTypeChange,
  collapsed,
  onToggleCollapse,
}: Props) {
  const set = (key: string, val: number) => onParamsChange({ ...params, [key]: val });

  if (collapsed) {
    return (
      <aside className="w-12 flex-shrink-0 bg-white border-r border-slate-200 flex flex-col items-center shadow-sm">
        <button
          onClick={onToggleCollapse}
          className="w-8 h-8 mt-4 rounded-lg bg-slate-100 hover:bg-slate-200 flex items-center justify-center transition-colors"
          title="Expand sidebar"
        >
          <ChevronRight className="w-4 h-4 text-slate-600" />
        </button>
        <div className="mt-4">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-blue-500 to-cyan-500 flex items-center justify-center">
            <Droplets className="w-4 h-4 text-white" />
          </div>
        </div>
      </aside>
    );
  }

  return (
    <aside className="w-72 flex-shrink-0 bg-white border-r border-slate-200 flex flex-col shadow-sm">
      {/* Logo + Collapse */}
      <div className="px-5 py-4 border-b border-slate-100">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-blue-500 to-cyan-500 flex items-center justify-center">
              <Droplets className="w-4 h-4 text-white" />
            </div>
            <div>
              <div className="text-sm font-bold text-slate-800">Leak Predictor</div>
              <div className="text-[10px] text-slate-400 uppercase tracking-wider">Control Panel</div>
            </div>
          </div>
          <button
            onClick={onToggleCollapse}
            className="w-7 h-7 rounded-lg hover:bg-slate-100 flex items-center justify-center transition-colors"
            title="Collapse sidebar"
          >
            <ChevronLeft className="w-4 h-4 text-slate-400" />
          </button>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto px-4 py-4 space-y-5">
        {/* Simulation Parameters */}
        <Section title="Simulation Parameters" icon={<Settings className="w-3.5 h-3.5" />}>
          <SliderField
            label="Number of Pipes"
            value={params.num_pipes}
            min={200}
            max={3000}
            step={100}
            onChange={(v) => set("num_pipes", v)}
          />
          <SliderField
            label="Simulation Years"
            value={params.sim_years}
            min={1}
            max={10}
            step={1}
            onChange={(v) => set("sim_years", v)}
          />
          <div>
            <label className="text-xs font-medium text-slate-500 block mb-1">Random Seed</label>
            <input
              type="number"
              value={params.seed}
              onChange={(e) => set("seed", parseInt(e.target.value) || 0)}
              className="w-full rounded-lg border border-slate-200 px-3 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500/40 focus:border-blue-400"
            />
          </div>
        </Section>

        {/* Model Selection */}
        <Section title="ML Model" icon={<Cpu className="w-3.5 h-3.5" />}>
          <div>
            <label className="text-xs font-medium text-slate-500 block mb-1">Algorithm</label>
            <select
              value={modelType}
              onChange={(e) => onModelTypeChange(e.target.value)}
              className="w-full rounded-lg border border-slate-200 px-3 py-2 text-sm bg-white focus:outline-none focus:ring-2 focus:ring-purple-500/40 focus:border-purple-400"
            >
              <optgroup label="Individual Models">
                {MODEL_OPTIONS.filter((o) => o.group === "Individual").map((opt) => (
                  <option key={opt.value} value={opt.value}>{opt.label}</option>
                ))}
              </optgroup>
              <optgroup label="Ensemble Models">
                {MODEL_OPTIONS.filter((o) => o.group === "Ensemble").map((opt) => (
                  <option key={opt.value} value={opt.value}>{opt.label}</option>
                ))}
              </optgroup>
            </select>
            {modelType.includes("ensemble") || modelType === "blended_boosting" ? (
              <p className="text-[10px] text-purple-500 mt-1">
                Ensembles combine multiple models for higher accuracy. Training takes longer.
              </p>
            ) : null}
          </div>
        </Section>

        {/* Actions */}
        <Section title="Actions" icon={<Zap className="w-3.5 h-3.5" />}>
          <button
            onClick={onSimulate}
            disabled={simulating}
            className={`w-full rounded-lg py-2.5 px-4 text-sm font-semibold text-white transition-all ${
              dirty
                ? "bg-gradient-to-r from-orange-500 to-amber-500 hover:from-orange-600 hover:to-amber-600 shadow-lg shadow-orange-500/25 animate-pulse"
                : "bg-gradient-to-r from-blue-600 to-cyan-600 hover:from-blue-700 hover:to-cyan-700 shadow-md shadow-blue-500/20"
            } disabled:opacity-50 disabled:cursor-not-allowed`}
          >
            {simulating ? (
              <span className="flex items-center justify-center gap-2">
                <Spinner /> Generating...
              </span>
            ) : dirty ? (
              "Regenerate Network"
            ) : (
              "Generate Network"
            )}
          </button>

          <button
            onClick={onTrain}
            disabled={training || !hasData}
            className="w-full rounded-lg py-2.5 px-4 text-sm font-semibold text-white bg-gradient-to-r from-purple-600 to-fuchsia-600 hover:from-purple-700 hover:to-fuchsia-700 shadow-md shadow-purple-500/20 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
          >
            {training ? (
              <span className="flex items-center justify-center gap-2">
                <Spinner /> Training...
              </span>
            ) : (
              <>
                <Brain className="w-4 h-4 inline mr-1" />
                {hasModel ? "Retrain Model" : "Train Model"}
              </>
            )}
          </button>
        </Section>
      </div>

      {/* Footer */}
      <div className="px-4 py-3 border-t border-slate-100 bg-slate-50">
        <p className="text-[10px] text-slate-400 text-center">
          WNTR Simulation &bull; Multi-Model ML &bull; v3.0
        </p>
      </div>
    </aside>
  );
}

function Section({ title, icon, children }: { title: string; icon: React.ReactNode; children: React.ReactNode }) {
  return (
    <div className="bg-slate-50 rounded-xl p-4 border border-slate-100">
      <h3 className="flex items-center gap-1.5 text-[11px] font-semibold text-slate-500 uppercase tracking-wider mb-3">
        {icon} {title}
      </h3>
      <div className="space-y-3">{children}</div>
    </div>
  );
}

function SliderField({
  label,
  value,
  min,
  max,
  step,
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
}) {
  return (
    <div>
      <div className="flex items-center justify-between mb-1">
        <label className="text-xs font-medium text-slate-500">{label}</label>
        <span className="text-xs font-bold text-blue-600 bg-blue-50 px-2 py-0.5 rounded-md">
          {value.toLocaleString()}
        </span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseInt(e.target.value))}
        className="w-full h-1.5 bg-slate-200 rounded-full appearance-none cursor-pointer accent-blue-600"
      />
    </div>
  );
}

function Spinner() {
  return (
    <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24">
      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
      <path
        className="opacity-75"
        fill="currentColor"
        d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
      />
    </svg>
  );
}
