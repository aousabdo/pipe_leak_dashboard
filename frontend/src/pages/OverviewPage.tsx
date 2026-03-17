import KpiCard from "../components/KpiCard";
import ChartCard from "../components/ChartCard";
import {
  AreaChart, Area, BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend,
} from "recharts";

const SEVERITY_COLORS: Record<string, string> = {
  Minor: "#22c55e",
  Moderate: "#f59e0b",
  Major: "#f97316",
  Critical: "#ef4444",
};

const MATERIAL_COLORS = ["#3b82f6", "#8b5cf6", "#06b6d4", "#f59e0b", "#ef4444", "#22c55e", "#ec4899"];

interface Props {
  data: any;
  loading: boolean;
}

function fmt(n: number): string {
  if (n >= 1_000_000) return `$${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `$${(n / 1_000).toFixed(0)}K`;
  return `$${n.toFixed(0)}`;
}

function fmtNum(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}K`;
  return n.toLocaleString();
}

export default function OverviewPage({ data, loading }: Props) {
  if (loading || !data) {
    return <div className="flex items-center justify-center h-64 text-slate-400">Loading...</div>;
  }

  const { kpis, severity_counts, monthly_trend, material_distribution, cost_by_severity, age_distribution } = data;

  // Pie data for severity
  const severityPie = Object.entries(severity_counts).map(([name, value]) => ({
    name,
    value: value as number,
    color: SEVERITY_COLORS[name] || "#94a3b8",
  }));

  // Pie data for cost by severity
  const costPie = Object.entries(cost_by_severity).map(([name, value]) => ({
    name,
    value: value as number,
    color: SEVERITY_COLORS[name] || "#94a3b8",
  }));

  // Bar data for materials
  const materialBars = Object.entries(material_distribution).map(([name, value], i) => ({
    name,
    count: value as number,
    fill: MATERIAL_COLORS[i % MATERIAL_COLORS.length],
  }));

  // Bar data for age distribution
  const ageBars = Object.entries(age_distribution).map(([range, count]) => ({
    range,
    count: count as number,
  }));

  return (
    <div className="space-y-6">
      {/* KPIs */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <KpiCard icon="🔧" label="Total Pipes" value={fmtNum(kpis.total_pipes)} accent="blue" />
        <KpiCard icon="💥" label="Leak Events" value={fmtNum(kpis.total_events)} accent="red" />
        <KpiCard icon="💰" label="Total Repair Cost" value={fmt(kpis.total_cost)} accent="amber" />
        <KpiCard icon="💧" label="Water Loss" value={`${fmtNum(kpis.total_water_loss)} gal`} accent="cyan" />
        <KpiCard icon="📅" label="Avg Pipe Age" value={`${kpis.avg_pipe_age} yrs`} accent="purple" />
        <KpiCard icon="⚠️" label="High Risk Pipes" value={fmtNum(kpis.high_risk_pipes)} accent="red"
                 delta={`${((kpis.high_risk_pipes / Math.max(kpis.total_pipes, 1)) * 100).toFixed(1)}% of network`}
                 deltaType="negative" />
        <KpiCard icon="📊" label="Avg Cost/Event" value={fmt(kpis.avg_cost_per_event)} accent="amber" />
        <KpiCard icon="📈" label="Events/Pipe" value={kpis.events_per_pipe.toFixed(2)} accent="green" />
      </div>

      {/* Charts Row 1: Trend + Severity */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <ChartCard title="Monthly Leak Trend" subtitle="Events over time" className="lg:col-span-2">
          <ResponsiveContainer width="100%" height={280}>
            <AreaChart data={monthly_trend}>
              <defs>
                <linearGradient id="trendGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#3b82f6" stopOpacity={0.3} />
                  <stop offset="100%" stopColor="#3b82f6" stopOpacity={0.02} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
              <XAxis dataKey="month" tick={{ fontSize: 10 }} interval="preserveStartEnd" angle={-30} textAnchor="end" height={50} />
              <YAxis tick={{ fontSize: 11 }} />
              <Tooltip contentStyle={{ borderRadius: 12, border: "1px solid #e2e8f0", fontSize: 12 }} />
              <Area type="monotone" dataKey="count" stroke="#3b82f6" fill="url(#trendGrad)" strokeWidth={2} name="Events" />
            </AreaChart>
          </ResponsiveContainer>
        </ChartCard>

        <ChartCard title="Severity Distribution" subtitle="Event count by severity">
          <ResponsiveContainer width="100%" height={280}>
            <PieChart>
              <Pie data={severityPie} cx="50%" cy="50%" innerRadius={55} outerRadius={90} paddingAngle={3} dataKey="value" label={({ name, percent }) => `${name} ${((percent ?? 0) * 100).toFixed(0)}%`} labelLine={true} style={{ fontSize: 11 }}>
                {severityPie.map((d, i) => (
                  <Cell key={i} fill={d.color} />
                ))}
              </Pie>
              <Tooltip contentStyle={{ borderRadius: 12, border: "1px solid #e2e8f0", fontSize: 12 }} />
            </PieChart>
          </ResponsiveContainer>
        </ChartCard>
      </div>

      {/* Charts Row 2: Cost + Materials + Age */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <ChartCard title="Cost by Severity" subtitle="Repair costs breakdown">
          <ResponsiveContainer width="100%" height={250}>
            <PieChart>
              <Pie data={costPie} cx="50%" cy="50%" innerRadius={50} outerRadius={85} paddingAngle={3} dataKey="value" label={({ name, percent }) => `${name} ${((percent ?? 0) * 100).toFixed(0)}%`} style={{ fontSize: 11 }}>
                {costPie.map((d, i) => (
                  <Cell key={i} fill={d.color} />
                ))}
              </Pie>
              <Tooltip formatter={(v: any) => fmt(Number(v))} contentStyle={{ borderRadius: 12, border: "1px solid #e2e8f0", fontSize: 12 }} />
            </PieChart>
          </ResponsiveContainer>
        </ChartCard>

        <ChartCard title="Material Distribution" subtitle="Pipes by material type">
          <ResponsiveContainer width="100%" height={250}>
            <BarChart data={materialBars} layout="vertical">
              <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
              <XAxis type="number" tick={{ fontSize: 11 }} />
              <YAxis type="category" dataKey="name" tick={{ fontSize: 10 }} width={100} />
              <Tooltip contentStyle={{ borderRadius: 12, border: "1px solid #e2e8f0", fontSize: 12 }} />
              <Bar dataKey="count" radius={[0, 6, 6, 0]}>
                {materialBars.map((d, i) => (
                  <Cell key={i} fill={d.fill} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </ChartCard>

        <ChartCard title="Pipe Age Distribution" subtitle="Number of pipes by age range">
          <ResponsiveContainer width="100%" height={250}>
            <BarChart data={ageBars}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
              <XAxis dataKey="range" tick={{ fontSize: 10 }} />
              <YAxis tick={{ fontSize: 11 }} />
              <Tooltip contentStyle={{ borderRadius: 12, border: "1px solid #e2e8f0", fontSize: 12 }} />
              <Bar dataKey="count" fill="#8b5cf6" radius={[6, 6, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </ChartCard>
      </div>

      {/* Monthly Cost Trend */}
      <ChartCard title="Monthly Repair Costs" subtitle="Cost trend over simulation period">
        <ResponsiveContainer width="100%" height={250}>
          <AreaChart data={monthly_trend}>
            <defs>
              <linearGradient id="costGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#f59e0b" stopOpacity={0.3} />
                <stop offset="100%" stopColor="#f59e0b" stopOpacity={0.02} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
            <XAxis dataKey="month" tick={{ fontSize: 10 }} interval="preserveStartEnd" angle={-30} textAnchor="end" height={50} />
            <YAxis tick={{ fontSize: 11 }} tickFormatter={(v) => fmt(v)} />
            <Tooltip formatter={(v: any) => fmt(Number(v))} contentStyle={{ borderRadius: 12, border: "1px solid #e2e8f0", fontSize: 12 }} />
            <Area type="monotone" dataKey="cost" stroke="#f59e0b" fill="url(#costGrad)" strokeWidth={2} name="Cost" />
          </AreaChart>
        </ResponsiveContainer>
      </ChartCard>
    </div>
  );
}
