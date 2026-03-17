import { useState } from "react";
import ChartCard from "../components/ChartCard";
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
      )}

      {activeTab === "materials" && (
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

          {/* Material Details Table */}
          <ChartCard title="Material Details" subtitle="Full breakdown" className="lg:col-span-2">
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
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          <ChartCard title="Cost by Severity Over Time" subtitle="Annual breakdown" className="lg:col-span-2">
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={_pivotCostData(cost_by_severity_year)}>
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
        </div>
      )}
    </div>
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
