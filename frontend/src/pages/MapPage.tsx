import { useMemo, useState } from "react";
import DeckGL from "@deck.gl/react";
import { LineLayer, ScatterplotLayer } from "@deck.gl/layers";
import { Map as MapLibreMap } from "react-map-gl/maplibre";
import "maplibre-gl/dist/maplibre-gl.css";
import ChartCard from "../components/ChartCard";

interface Props {
  data: any;
  loading: boolean;
}

function riskColor(score: number): [number, number, number, number] {
  // Green -> Yellow -> Red gradient
  if (score < 0.3) {
    const t = score / 0.3;
    return [Math.round(34 + t * 211), Math.round(197 - t * 38), Math.round(94 - t * 69), 200];
  } else if (score < 0.7) {
    const t = (score - 0.3) / 0.4;
    return [Math.round(245 - t * 6), Math.round(159 - t * 91), Math.round(25 - t * 25), 200];
  } else {
    const t = (score - 0.7) / 0.3;
    return [239, Math.round(68 - t * 34), Math.round(0 + t * 68), 220];
  }
}

export default function MapPage({ data, loading }: Props) {
  const [viewMode, setViewMode] = useState<"lines" | "risk">("lines");

  if (loading || !data) {
    return <div className="flex items-center justify-center h-64 text-slate-400">Loading...</div>;
  }

  const pipes = data.pipes || [];

  // Compute center from pipe data
  const center = useMemo(() => {
    if (pipes.length === 0) return { latitude: 38.58, longitude: -121.49 };
    const avgLat = pipes.reduce((s: number, p: any) => s + p.mid_lat, 0) / pipes.length;
    const avgLon = pipes.reduce((s: number, p: any) => s + p.mid_lon, 0) / pipes.length;
    return { latitude: avgLat, longitude: avgLon };
  }, [pipes]);

  const lineLayer = new LineLayer({
    id: "pipe-lines",
    data: pipes,
    getSourcePosition: (d: any) => [d.start_lon, d.start_lat],
    getTargetPosition: (d: any) => [d.end_lon, d.end_lat],
    getColor: (d: any) => riskColor(d.risk_score),
    getWidth: (d: any) => 2 + d.risk_score * 6,
    widthMinPixels: 1,
    pickable: true,
  });

  const scatterLayer = new ScatterplotLayer({
    id: "pipe-dots",
    data: pipes.filter((p: any) => p.risk_score > 0.5),
    getPosition: (d: any) => [d.mid_lon, d.mid_lat],
    getFillColor: (d: any) => riskColor(d.risk_score),
    getRadius: (d: any) => 20 + d.risk_score * 80,
    radiusMinPixels: 3,
    radiusMaxPixels: 15,
    opacity: 0.7,
    pickable: true,
  });

  const layers = viewMode === "lines" ? [lineLayer] : [lineLayer, scatterLayer];

  // Top risky pipes
  const topRisky = [...pipes]
    .sort((a: any, b: any) => b.risk_score - a.risk_score)
    .slice(0, 10);

  // Risk distribution
  const riskBuckets = { low: 0, medium: 0, high: 0, critical: 0 };
  pipes.forEach((p: any) => {
    if (p.risk_score < 0.3) riskBuckets.low++;
    else if (p.risk_score < 0.5) riskBuckets.medium++;
    else if (p.risk_score < 0.7) riskBuckets.high++;
    else riskBuckets.critical++;
  });

  return (
    <div className="space-y-4">
      {/* Controls */}
      <div className="flex items-center gap-3">
        <button
          onClick={() => setViewMode("lines")}
          className={`px-4 py-2 text-sm font-medium rounded-lg transition-colors ${
            viewMode === "lines"
              ? "bg-blue-600 text-white shadow-md"
              : "bg-white text-slate-600 border border-slate-200 hover:bg-slate-50"
          }`}
        >
          Network View
        </button>
        <button
          onClick={() => setViewMode("risk")}
          className={`px-4 py-2 text-sm font-medium rounded-lg transition-colors ${
            viewMode === "risk"
              ? "bg-blue-600 text-white shadow-md"
              : "bg-white text-slate-600 border border-slate-200 hover:bg-slate-50"
          }`}
        >
          Risk Hotspots
        </button>

        {/* Legend */}
        <div className="ml-auto flex items-center gap-3 text-xs text-slate-500">
          <span className="flex items-center gap-1">
            <span className="w-3 h-3 rounded-full bg-green-500" /> Low
          </span>
          <span className="flex items-center gap-1">
            <span className="w-3 h-3 rounded-full bg-yellow-500" /> Medium
          </span>
          <span className="flex items-center gap-1">
            <span className="w-3 h-3 rounded-full bg-orange-500" /> High
          </span>
          <span className="flex items-center gap-1">
            <span className="w-3 h-3 rounded-full bg-red-500" /> Critical
          </span>
        </div>
      </div>

      {/* Map */}
      <div className="rounded-xl overflow-hidden border border-slate-200 shadow-sm" style={{ height: 520 }}>
        <DeckGL
          initialViewState={{
            ...center,
            zoom: 13,
            pitch: viewMode === "risk" ? 45 : 0,
            bearing: 0,
          }}
          controller={true}
          layers={layers}
          getTooltip={({ object }: any) => {
            if (!object) return null;
            return {
              html: `
                <div style="font-family: Inter, sans-serif; padding: 8px; min-width: 180px;">
                  <div style="font-weight: 700; font-size: 13px; margin-bottom: 6px; color: #1e293b;">
                    Pipe ${object.pipe_id}
                  </div>
                  <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 4px; font-size: 11px; color: #64748b;">
                    <span>Risk Score</span><span style="font-weight: 600; color: ${object.risk_score > 0.7 ? '#ef4444' : object.risk_score > 0.3 ? '#f59e0b' : '#22c55e'}">${(object.risk_score * 100).toFixed(1)}%</span>
                    <span>Material</span><span style="font-weight: 500;">${object.material}</span>
                    <span>Age</span><span style="font-weight: 500;">${object.age} yrs</span>
                    <span>Diameter</span><span style="font-weight: 500;">${(object.diameter_m * 39.37).toFixed(0)}" (${object.diameter_category})</span>
                    <span>Pressure</span><span style="font-weight: 500;">${object.pressure_avg_m.toFixed(1)} m</span>
                    <span>Soil</span><span style="font-weight: 500;">${object.soil_type}</span>
                    <span>Prev Repairs</span><span style="font-weight: 500;">${object.prev_repairs}</span>
                  </div>
                </div>
              `,
              style: {
                backgroundColor: "white",
                borderRadius: "12px",
                boxShadow: "0 10px 40px rgba(0,0,0,0.15)",
                border: "1px solid #e2e8f0",
                padding: "0",
              },
            };
          }}
        >
          <MapLibreMap
            mapStyle="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json"
            attributionControl={false}
          />
        </DeckGL>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* Risk Distribution */}
        <ChartCard title="Risk Distribution" subtitle="Pipes by risk category">
          <div className="grid grid-cols-4 gap-3 mt-2">
            {[
              { label: "Low", count: riskBuckets.low, color: "bg-green-500", text: "text-green-700", bg: "bg-green-50" },
              { label: "Medium", count: riskBuckets.medium, color: "bg-yellow-500", text: "text-yellow-700", bg: "bg-yellow-50" },
              { label: "High", count: riskBuckets.high, color: "bg-orange-500", text: "text-orange-700", bg: "bg-orange-50" },
              { label: "Critical", count: riskBuckets.critical, color: "bg-red-500", text: "text-red-700", bg: "bg-red-50" },
            ].map((b) => (
              <div key={b.label} className={`${b.bg} rounded-lg p-3 text-center`}>
                <div className={`text-xl font-bold ${b.text}`}>{b.count}</div>
                <div className="text-xs text-slate-500 font-medium mt-0.5">{b.label}</div>
                <div className={`w-full h-1 ${b.color} rounded-full mt-2 opacity-50`} />
              </div>
            ))}
          </div>
        </ChartCard>

        {/* Top Risky Pipes Table */}
        <ChartCard title="Top 10 Riskiest Pipes" subtitle="Highest predicted leak probability">
          <div className="overflow-auto max-h-56">
            <table className="w-full text-xs">
              <thead className="bg-slate-50">
                <tr>
                  <th className="px-3 py-2 text-left font-semibold text-slate-500">Pipe</th>
                  <th className="px-3 py-2 text-left font-semibold text-slate-500">Risk</th>
                  <th className="px-3 py-2 text-left font-semibold text-slate-500">Material</th>
                  <th className="px-3 py-2 text-left font-semibold text-slate-500">Age</th>
                  <th className="px-3 py-2 text-left font-semibold text-slate-500">Repairs</th>
                </tr>
              </thead>
              <tbody>
                {topRisky.map((p: any) => (
                  <tr key={p.pipe_id} className="border-t border-slate-100 hover:bg-slate-50">
                    <td className="px-3 py-1.5 font-mono font-medium text-slate-700">{p.pipe_id}</td>
                    <td className="px-3 py-1.5">
                      <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-bold ${
                        p.risk_score > 0.7 ? "bg-red-100 text-red-700" :
                        p.risk_score > 0.5 ? "bg-orange-100 text-orange-700" :
                        "bg-yellow-100 text-yellow-700"
                      }`}>
                        {(p.risk_score * 100).toFixed(1)}%
                      </span>
                    </td>
                    <td className="px-3 py-1.5 text-slate-600">{p.material}</td>
                    <td className="px-3 py-1.5 text-slate-600">{p.age} yr</td>
                    <td className="px-3 py-1.5 text-slate-600">{p.prev_repairs}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </ChartCard>
      </div>
    </div>
  );
}
