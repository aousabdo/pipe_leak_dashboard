import KpiCard from "../components/KpiCard";
import ChartCard from "../components/ChartCard";
import { api } from "../api";
import {
  AreaChart, Area, BarChart, Bar, LineChart, Line, ScatterChart, Scatter,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine, Cell,
} from "recharts";

interface Props {
  data: any;
  loading: boolean;
  hasModel: boolean;
  onTrain: () => void;
  training: boolean;
}

export default function ModelPage({ data, loading, hasModel, onTrain, training }: Props) {
  if (!hasModel) {
    return (
      <div className="flex h-96 items-center justify-center">
        <div className="text-center">
          <div className="text-6xl mb-4">🧠</div>
          <h2 className="text-xl font-bold text-slate-700 mb-2">No Model Trained Yet</h2>
          <p className="text-slate-500 mb-4 max-w-sm">
            Select a model in the sidebar and click Train to predict pipe leak probabilities.
            Try ensemble models for best accuracy.
          </p>
          <button
            onClick={onTrain}
            disabled={training}
            className="px-6 py-3 bg-gradient-to-r from-purple-600 to-fuchsia-600 text-white rounded-lg font-semibold shadow-lg hover:shadow-xl transition-all disabled:opacity-50"
          >
            {training ? "Training..." : "Train Model"}
          </button>
        </div>
      </div>
    );
  }

  if (loading || !data) {
    return <div className="flex items-center justify-center h-64 text-slate-400">Loading...</div>;
  }

  const { metrics, confusion_matrix, roc_curve, pr_curve, calibration, feature_importance, model_type, optimal_threshold } = data;

  const MODEL_LABELS: Record<string, string> = {
    xgboost: "XGBoost",
    lightgbm: "LightGBM",
    random_forest: "Random Forest",
    logistic_regression: "Logistic Regression",
    gradient_boosting: "Gradient Boosting",
    stacking_ensemble: "Stacking Ensemble (XGB + LGBM + RF + ExtraTrees → LR meta)",
    voting_ensemble: "Voting Ensemble (XGB + LGBM + RF + GB + MLP)",
    blended_boosting: "Blended Boosting (2×XGB + 2×LGBM)",
  };

  // ROC curve data
  const rocData = roc_curve?.fpr?.map((fpr: number, i: number) => ({
    fpr,
    tpr: roc_curve.tpr[i],
  })) || [];

  // PR curve data
  const prData = pr_curve?.recall?.map((recall: number, i: number) => ({
    recall,
    precision: pr_curve.precision[i],
  })) || [];

  // Calibration data
  const calData = calibration?.predicted?.map((pred: number, i: number) => ({
    predicted: pred,
    actual: calibration.actual[i],
    count: calibration.counts[i],
  })) || [];

  // Feature importance data
  const impData = (feature_importance || []).map((f: any) => ({
    ...f,
    importance: parseFloat((f.importance * 100).toFixed(2)),
  }));

  const cm = confusion_matrix || [[0, 0], [0, 0]];
  const cmTotal = cm[0][0] + cm[0][1] + cm[1][0] + cm[1][1];

  return (
    <div className="space-y-6">
      {/* Model Info Banner */}
      <div className="bg-gradient-to-r from-purple-50 to-fuchsia-50 border border-purple-200 rounded-xl px-5 py-3 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <span className="text-sm font-semibold text-purple-800">
            Model: {MODEL_LABELS[model_type] || model_type || "XGBoost"}
          </span>
          <span className="text-xs text-purple-500">|</span>
          <span className="text-sm text-purple-600">
            Optimal threshold: {(optimal_threshold || 0.5).toFixed(3)}
          </span>
          <span className="text-xs text-purple-500">|</span>
          <span className="text-sm text-purple-600">
            {metrics.n_positive || 0} positive / {metrics.n_negative || 0} negative samples
          </span>
        </div>
      </div>

      {/* Metric Cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-7 gap-3">
        {[
          { icon: "🎯", label: "Accuracy", value: `${(metrics.accuracy * 100).toFixed(1)}%`, accent: "blue" as const },
          { icon: "🔍", label: "Precision", value: `${((metrics.precision || 0) * 100).toFixed(1)}%`, accent: "purple" as const },
          { icon: "📡", label: "Recall", value: `${((metrics.recall || 0) * 100).toFixed(1)}%`, accent: "cyan" as const },
          { icon: "⚡", label: "F1 Score", value: `${((metrics.f1 || 0) * 100).toFixed(1)}%`, accent: "green" as const },
          { icon: "📈", label: "ROC AUC", value: `${((metrics.roc_auc || 0) * 100).toFixed(1)}%`, accent: "blue" as const },
          { icon: "📊", label: "PR AUC", value: `${((metrics.pr_auc || 0) * 100).toFixed(1)}%`, accent: "amber" as const },
          { icon: "🎲", label: "Brier Score", value: (metrics.brier_score || 0).toFixed(4), accent: "red" as const },
        ].map((m) => (
          <KpiCard key={m.label} {...m} />
        ))}
      </div>

      {/* Curves Row */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* ROC Curve */}
        <ChartCard title="ROC Curve" subtitle={`AUC = ${(roc_curve?.auc || 0).toFixed(4)}`}>
          <ResponsiveContainer width="100%" height={300}>
            <AreaChart data={rocData}>
              <defs>
                <linearGradient id="rocGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#3b82f6" stopOpacity={0.3} />
                  <stop offset="100%" stopColor="#3b82f6" stopOpacity={0.02} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
              <XAxis dataKey="fpr" tick={{ fontSize: 10 }} label={{ value: "False Positive Rate", position: "insideBottom", offset: -5, fontSize: 11 }} />
              <YAxis tick={{ fontSize: 10 }} label={{ value: "True Positive Rate", angle: -90, position: "insideLeft", offset: 10, fontSize: 11 }} />
              <Tooltip contentStyle={{ borderRadius: 12, border: "1px solid #e2e8f0", fontSize: 12 }} />
              <ReferenceLine stroke="#94a3b8" strokeDasharray="5 5" segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]} />
              <Area type="monotone" dataKey="tpr" stroke="#3b82f6" fill="url(#rocGrad)" strokeWidth={2.5} dot={false} />
            </AreaChart>
          </ResponsiveContainer>
        </ChartCard>

        {/* PR Curve */}
        <ChartCard title="Precision-Recall Curve" subtitle={`PR AUC = ${(pr_curve?.pr_auc || 0).toFixed(4)}`}>
          <ResponsiveContainer width="100%" height={300}>
            <AreaChart data={prData}>
              <defs>
                <linearGradient id="prGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#8b5cf6" stopOpacity={0.3} />
                  <stop offset="100%" stopColor="#8b5cf6" stopOpacity={0.02} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
              <XAxis dataKey="recall" tick={{ fontSize: 10 }} label={{ value: "Recall", position: "insideBottom", offset: -5, fontSize: 11 }} />
              <YAxis tick={{ fontSize: 10 }} label={{ value: "Precision", angle: -90, position: "insideLeft", offset: 10, fontSize: 11 }} />
              <Tooltip contentStyle={{ borderRadius: 12, border: "1px solid #e2e8f0", fontSize: 12 }} />
              <Area type="monotone" dataKey="precision" stroke="#8b5cf6" fill="url(#prGrad)" strokeWidth={2.5} dot={false} />
            </AreaChart>
          </ResponsiveContainer>
        </ChartCard>
      </div>

      {/* Bottom Row */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* Confusion Matrix */}
        <ChartCard title="Confusion Matrix" subtitle={`${cmTotal} test samples`}>
          <div className="grid grid-cols-2 gap-2 p-4 max-w-xs mx-auto">
            <Cell2 value={cm[0][0]} label="True Neg" total={cmTotal} color="bg-green-100 text-green-800 border-green-200" />
            <Cell2 value={cm[0][1]} label="False Pos" total={cmTotal} color="bg-red-50 text-red-600 border-red-200" />
            <Cell2 value={cm[1][0]} label="False Neg" total={cmTotal} color="bg-orange-50 text-orange-600 border-orange-200" />
            <Cell2 value={cm[1][1]} label="True Pos" total={cmTotal} color="bg-blue-100 text-blue-800 border-blue-200" />
          </div>
          <div className="grid grid-cols-2 gap-2 text-center text-[10px] text-slate-400 max-w-xs mx-auto px-4">
            <div>Predicted: No Leak</div>
            <div>Predicted: Leak</div>
          </div>
        </ChartCard>

        {/* Feature Importance */}
        <ChartCard title="Feature Importance" subtitle="Top 15 features by gain" className="lg:col-span-2">
          <ResponsiveContainer width="100%" height={340}>
            <BarChart data={impData} layout="vertical">
              <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
              <XAxis type="number" tick={{ fontSize: 10 }} label={{ value: "Importance (%)", position: "insideBottom", offset: -5, fontSize: 11 }} />
              <YAxis type="category" dataKey="feature" tick={{ fontSize: 9 }} width={150} />
              <Tooltip contentStyle={{ borderRadius: 12, border: "1px solid #e2e8f0", fontSize: 12 }} />
              <Bar dataKey="importance" name="Importance %" radius={[0, 6, 6, 0]}>
                {impData.map((_: any, i: number) => (
                  <Cell key={i} fill={i === 0 ? "#3b82f6" : i < 3 ? "#60a5fa" : i < 6 ? "#93c5fd" : "#bfdbfe"} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </ChartCard>
      </div>

      {/* Calibration Plot */}
      {calData.length > 0 && (
        <ChartCard title="Calibration Plot" subtitle="Predicted vs actual positive rates (reliability diagram)">
          <ResponsiveContainer width="100%" height={280}>
            <LineChart data={calData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
              <XAxis dataKey="predicted" tick={{ fontSize: 10 }} label={{ value: "Predicted Probability", position: "insideBottom", offset: -5, fontSize: 11 }} />
              <YAxis tick={{ fontSize: 10 }} label={{ value: "Actual Fraction", angle: -90, position: "insideLeft", offset: 10, fontSize: 11 }} domain={[0, 1]} />
              <Tooltip contentStyle={{ borderRadius: 12, border: "1px solid #e2e8f0", fontSize: 12 }} />
              <ReferenceLine stroke="#94a3b8" strokeDasharray="5 5" segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]} />
              <Line type="monotone" dataKey="actual" stroke="#06b6d4" strokeWidth={2.5} dot={{ r: 5, fill: "#06b6d4" }} />
            </LineChart>
          </ResponsiveContainer>
        </ChartCard>
      )}

      {/* Download Section */}
      <div className="flex items-center gap-3 pt-2">
        <span className="text-xs text-slate-400 uppercase tracking-wider font-semibold">Export</span>
        <div className="h-px flex-1 bg-slate-200" />
      </div>
      <div className="flex flex-wrap gap-3">
        <DownloadBtn onClick={() => api.downloadModelReport()} color="purple" label="Model Report" sub="Metrics, confusion matrix, features" />
        <DownloadBtn onClick={() => api.downloadPipesCSV()} color="blue" label="Pipes CSV" sub="Pipe data with risk scores" />
        <DownloadBtn onClick={() => api.downloadEventsCSV()} color="amber" label="Events CSV" sub="Leak events data" />
      </div>
    </div>
  );
}

function DownloadBtn({ onClick, color, label, sub }: { onClick: () => void; color: string; label: string; sub: string }) {
  const colors: Record<string, string> = {
    blue: "text-blue-500",
    amber: "text-amber-500",
    purple: "text-purple-500",
  };
  return (
    <button
      onClick={onClick}
      className="flex items-center gap-2 px-4 py-2.5 bg-white border border-slate-200 rounded-lg text-sm font-medium text-slate-700 hover:bg-slate-50 hover:border-slate-300 transition-colors shadow-sm"
    >
      <svg className={`w-5 h-5 ${colors[color] || "text-slate-500"}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
      </svg>
      <div className="text-left">
        <div>{label}</div>
        <div className="text-xs text-slate-400 font-normal">{sub}</div>
      </div>
    </button>
  );
}

function Cell2({ value, label, total, color }: { value: number; label: string; total: number; color: string }) {
  const pct = total > 0 ? ((value / total) * 100).toFixed(1) : "0";
  return (
    <div className={`${color} rounded-xl border p-4 text-center`}>
      <div className="text-2xl font-bold">{value}</div>
      <div className="text-[10px] uppercase font-semibold tracking-wider mt-1 opacity-70">{label}</div>
      <div className="text-xs mt-0.5 opacity-60">{pct}%</div>
    </div>
  );
}
