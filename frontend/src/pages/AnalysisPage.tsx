import { useState } from "react";
import ChartCard from "../components/ChartCard";
import { api } from "../api";
import {
  BarChart, Bar, LineChart, Line, AreaChart, Area,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend, Cell,
} from "recharts";

const SEVERITY_COLORS: Record<string, string> = {
  Minor: "#22c55e",
  Moderate: "#f59e0b",
  Major: "#f97316",
  Critical: "#ef4444",
};

const MATERIAL_COLORS = ["#3b82f6", "#8b5cf6", "#06b6d4", "#f59e0b", "#ef4444", "#22c55e"];

interface Props {
  data: any;
  loading: boolean;
}

function fmt(n: number): string {
  if (n >= 1_000_000) return `$${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `$${(n / 1_000).toFixed(0)}K`;
  return `$${n.toFixed(0)}`;
}

export default function AnalysisPage({ data, loading }: Props) {
  const [activeTab, setActiveTab] = useState<"patterns" | "materials" | "costs">("patterns");

  if (loading || !data) {
    return <div className="flex items-center justify-center h-64 text-slate-400">Loading...</div>;
  }

  const { monthly_severity, material_risk, yearly_trend, soil_analysis, cost_by_severity_year, insights } = data;

  const tabs = [
    { id: "patterns", label: "Patterns & Trends" },
    { id: "materials", label: "Root Causes" },
    { id: "costs", label: "Cost Analysis" },
  ] as const;

  // Pivot monthly severity for stacked area chart
  const monthlySevData = _pivotMonthlySeverity(monthly_severity);

  // Pivot cost data
  const pivotedCost = _pivotCostData(cost_by_severity_year);

  // Compute per-severity cost summary from pivoted data
  const severityCostSummary = ["Minor", "Moderate", "Major", "Critical"].map((sev) => {
    const total = pivotedCost.reduce((acc, row) => acc + (row[sev] || 0), 0);
    return { severity: sev, total };
  }).filter((s) => s.total > 0);
  const grandTotal = severityCostSummary.reduce((acc, s) => acc + s.total, 0);

  return (
    <div className="space-y-4">
      {/* Insights Banner */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        {[
          { label: "Peak Month", value: insights.peak_month, icon: "📅" },
          { label: "Most At-Risk Material", value: insights.most_affected_material, icon: "🔧" },
          { label: "Total Repair Cost", value: fmt(insights.total_cost), icon: "💰" },
          { label: "Avg Detection Time", value: `${insights.avg_detection_hours} hrs`, icon: "⏱️" },
        ].map((item) => (
          <div key={item.label} className="bg-white rounded-lg border border-slate-200 p-3 flex items-center gap-3">
            <span className="text-2xl">{item.icon}</span>
            <div>
              <div className="text-sm font-bold text-slate-800">{item.value}</div>
              <div className="text-[10px] text-slate-400 uppercase tracking-wider">{item.label}</div>
            </div>
          </div>
        ))}
      </div>

      {/* Tab Buttons */}
      <div className="flex gap-2">
        {tabs.map((t) => (
          <button
            key={t.id}
            onClick={() => setActiveTab(t.id)}
            className={`px-4 py-2 text-sm font-medium rounded-lg transition-colors ${
              activeTab === t.id
                ? "bg-blue-600 text-white shadow-md"
                : "bg-white text-slate-600 border border-slate-200 hover:bg-slate-50"
            }`}
          >
            {t.label}
          </button>
        ))}
      </div>

      {/* Tab Content */}
      {activeTab === "patterns" && (
        <div className="space-y-4">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <ChartCard title="Yearly Event Count" subtitle="Leak events per year">
              <ResponsiveContainer width="100%" height={280}>
                <BarChart data={yearly_trend}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                  <XAxis dataKey="year" tick={{ fontSize: 11 }} />
                  <YAxis tick={{ fontSize: 11 }} />
                  <Tooltip contentStyle={{ borderRadius: 12, border: "1px solid #e2e8f0", fontSize: 12 }} />
                  <Bar dataKey="count" fill="#3b82f6" radius={[6, 6, 0, 0]} name="Events" />
                </BarChart>
              </ResponsiveContainer>
            </ChartCard>

            <ChartCard title="Yearly Cost Trend" subtitle="Total repair cost per year">
              <ResponsiveContainer width="100%" height={280}>
                <AreaChart data={yearly_trend}>
                  <defs>
                    <linearGradient id="yearCostGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="#ef4444" stopOpacity={0.3} />
                      <stop offset="100%" stopColor="#ef4444" stopOpacity={0.02} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                  <XAxis dataKey="year" tick={{ fontSize: 11 }} />
                  <YAxis tick={{ fontSize: 11 }} tickFormatter={(v) => fmt(v)} />
                  <Tooltip formatter={(v: any) => fmt(Number(v))} contentStyle={{ borderRadius: 12, border: "1px solid #e2e8f0", fontSize: 12 }} />
                  <Area type="monotone" dataKey="cost" stroke="#ef4444" fill="url(#yearCostGrad)" strokeWidth={2} name="Cost" />
                </AreaChart>
              </ResponsiveContainer>
            </ChartCard>
          </div>

          {/* Monthly Severity Trend */}
          {monthlySevData.length > 0 && (
            <ChartCard title="Monthly Events by Severity" subtitle="Stacked trend over time">
              <ResponsiveContainer width="100%" height={280}>
                <AreaChart data={monthlySevData}>
                  <defs>
                    {["Minor", "Moderate", "Major", "Critical"].map((sev) => (
                      <linearGradient key={sev} id={`sev-${sev}`} x1="0" y1="0" x2="0" y2="1">
                        <stop offset="0%" stopColor={SEVERITY_COLORS[sev]} stopOpacity={0.4} />
                        <stop offset="100%" stopColor={SEVERITY_COLORS[sev]} stopOpacity={0.05} />
                      </linearGradient>
                    ))}
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                  <XAxis dataKey="month" tick={{ fontSize: 9 }} interval="preserveStartEnd" angle={-30} textAnchor="end" height={50} />
                  <YAxis tick={{ fontSize: 11 }} />
                  <Tooltip contentStyle={{ borderRadius: 12, border: "1px solid #e2e8f0", fontSize: 12 }} />
                  <Legend />
                  {["Minor", "Moderate", "Major", "Critical"].map((sev) => (
                    <Area key={sev} type="monotone" dataKey={sev} stackId="1" stroke={SEVERITY_COLORS[sev]} fill={`url(#sev-${sev})`} strokeWidth={1.5} />
                  ))}
                </AreaChart>
              </ResponsiveContainer>
            </ChartCard>
          )}
        </div>
      )}

      {activeTab === "materials" && (
        <div className="space-y-4">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {/* Material Risk Profile */}
            <ChartCard title="Material Risk Profile" subtitle="Events per pipe by material">
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={material_risk} layout="vertical">
                  <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                  <XAxis type="number" tick={{ fontSize: 11 }} />
                  <YAxis type="category" dataKey="material" tick={{ fontSize: 10 }} width={110} />
                  <Tooltip contentStyle={{ borderRadius: 12, border: "1px solid #e2e8f0", fontSize: 12 }} />
                  <Bar dataKey="events_per_pipe" name="Events/Pipe" radius={[0, 6, 6, 0]}>
                    {material_risk.map((_: any, i: number) => (
                      <Cell key={i} fill={MATERIAL_COLORS[i % MATERIAL_COLORS.length]} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </ChartCard>

            {/* Soil Analysis */}
            <ChartCard title="Soil Type Impact" subtitle="Events and avg cost by soil">
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={soil_analysis}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                  <XAxis dataKey="soil_type" tick={{ fontSize: 10 }} />
                  <YAxis yAxisId="left" tick={{ fontSize: 11 }} />
                  <YAxis yAxisId="right" orientation="right" tick={{ fontSize: 11 }} tickFormatter={(v) => fmt(v)} />
                  <Tooltip contentStyle={{ borderRadius: 12, border: "1px solid #e2e8f0", fontSize: 12 }} />
                  <Legend />
                  <Bar yAxisId="left" dataKey="event_count" name="Events" fill="#3b82f6" radius={[4, 4, 0, 0]} />
                  <Bar yAxisId="right" dataKey="avg_cost" name="Avg Cost" fill="#f59e0b" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </ChartCard>
          </div>

          {/* Material Details Table */}
          <ChartCard title="Material Details" subtitle="Full breakdown">
            <div className="overflow-auto">
              <table className="w-full text-sm">
                <thead className="bg-slate-50">
                  <tr>
                    <th className="px-4 py-2 text-left font-semibold text-slate-500">Material</th>
                    <th className="px-4 py-2 text-right font-semibold text-slate-500">Pipes</th>
                    <th className="px-4 py-2 text-right font-semibold text-slate-500">Events</th>
                    <th className="px-4 py-2 text-right font-semibold text-slate-500">Events/Pipe</th>
                    <th className="px-4 py-2 text-right font-semibold text-slate-500">Avg Cost</th>
                    <th className="px-4 py-2 text-right font-semibold text-slate-500">Total Cost</th>
                  </tr>
                </thead>
                <tbody>
                  {material_risk.map((m: any) => (
                    <tr key={m.material} className="border-t border-slate-100 hover:bg-slate-50">
                      <td className="px-4 py-2 font-medium text-slate-700">{m.material}</td>
                      <td className="px-4 py-2 text-right text-slate-600">{m.pipe_count}</td>
                      <td className="px-4 py-2 text-right text-slate-600">{m.event_count}</td>
                      <td className="px-4 py-2 text-right">
                        <span className={`font-semibold ${m.events_per_pipe > 2 ? "text-red-600" : m.events_per_pipe > 1 ? "text-amber-600" : "text-green-600"}`}>
                          {m.events_per_pipe}
                        </span>
                      </td>
                      <td className="px-4 py-2 text-right text-slate-600">{fmt(m.avg_cost)}</td>
                      <td className="px-4 py-2 text-right font-medium text-slate-700">{fmt(m.total_cost)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </ChartCard>
        </div>
      )}

      {activeTab === "costs" && (
        <div className="space-y-4">
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
            <ChartCard title="Cost by Severity Over Time" subtitle="Annual breakdown" className="lg:col-span-2">
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={pivotedCost}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                  <XAxis dataKey="year" tick={{ fontSize: 11 }} />
                  <YAxis tick={{ fontSize: 11 }} tickFormatter={(v) => fmt(v)} />
                  <Tooltip formatter={(v: any) => fmt(Number(v))} contentStyle={{ borderRadius: 12, border: "1px solid #e2e8f0", fontSize: 12 }} />
                  <Legend />
                  {["Minor", "Moderate", "Major", "Critical"].map((sev) => (
                    <Bar key={sev} dataKey={sev} stackId="a" fill={SEVERITY_COLORS[sev]} />
                  ))}
                </BarChart>
              </ResponsiveContainer>
            </ChartCard>

            {/* Cost Summary Card */}
            <ChartCard title="Cost Breakdown" subtitle="By severity level">
              <div className="space-y-3 p-2">
                {severityCostSummary.map((s) => {
                  const pct = grandTotal > 0 ? (s.total / grandTotal) * 100 : 0;
                  return (
                    <div key={s.severity}>
                      <div className="flex justify-between text-sm mb-1">
                        <span className="font-medium text-slate-700">{s.severity}</span>
                        <span className="text-slate-600">{fmt(s.total)}</span>
                      </div>
                      <div className="h-2 bg-slate-100 rounded-full overflow-hidden">
                        <div className="h-full rounded-full" style={{ width: `${pct}%`, backgroundColor: SEVERITY_COLORS[s.severity] }} />
                      </div>
                      <div className="text-[10px] text-slate-400 mt-0.5">{pct.toFixed(1)}% of total</div>
                    </div>
                  );
                })}
                <div className="border-t border-slate-200 pt-2 mt-2 flex justify-between text-sm font-semibold text-slate-800">
                  <span>Total</span>
                  <span>{fmt(grandTotal)}</span>
                </div>
              </div>
            </ChartCard>
          </div>

          {/* Cost per year trend */}
          <ChartCard title="Annual Cost Trend" subtitle="Total repair cost per year">
            <ResponsiveContainer width="100%" height={250}>
              <AreaChart data={yearly_trend}>
                <defs>
                  <linearGradient id="annualCostGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#f97316" stopOpacity={0.3} />
                    <stop offset="100%" stopColor="#f97316" stopOpacity={0.02} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                <XAxis dataKey="year" tick={{ fontSize: 11 }} />
                <YAxis tick={{ fontSize: 11 }} tickFormatter={(v) => fmt(v)} />
                <Tooltip formatter={(v: any) => fmt(Number(v))} contentStyle={{ borderRadius: 12, border: "1px solid #e2e8f0", fontSize: 12 }} />
                <Area type="monotone" dataKey="cost" stroke="#f97316" fill="url(#annualCostGrad)" strokeWidth={2} name="Cost" />
              </AreaChart>
            </ResponsiveContainer>
          </ChartCard>
        </div>
      )}

      {/* Download Section */}
      <div className="flex items-center gap-3 pt-2">
        <span className="text-xs text-slate-400 uppercase tracking-wider font-semibold">Export</span>
        <div className="h-px flex-1 bg-slate-200" />
      </div>
      <div className="flex flex-wrap gap-3">
        <DownloadBtn onClick={() => api.downloadEventsCSV()} color="amber" label="Events CSV" sub="Leak events data" />
        <DownloadBtn onClick={() => api.downloadPipesCSV()} color="blue" label="Pipes CSV" sub="Pipe network data" />
        <DownloadBtn onClick={() => api.downloadReport()} color="purple" label="Full Report" sub="HTML analysis report" />
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

function _pivotCostData(raw: any[]): any[] {
  const map: Record<number, any> = {};
  for (const row of raw) {
    if (!map[row.year]) map[row.year] = { year: row.year };
    map[row.year][row.severity] = row.repair_cost;
  }
  return Object.values(map).sort((a, b) => a.year - b.year);
}

function _pivotMonthlySeverity(raw: any[]): any[] {
  if (!raw || raw.length === 0) return [];
  const map: Record<string, any> = {};
  for (const row of raw) {
    const key = row.month;
    if (!map[key]) map[key] = { month: key, Minor: 0, Moderate: 0, Major: 0, Critical: 0 };
    map[key][row.severity] = (map[key][row.severity] || 0) + row.count;
  }
  return Object.values(map).sort((a, b) => a.month.localeCompare(b.month));
}
