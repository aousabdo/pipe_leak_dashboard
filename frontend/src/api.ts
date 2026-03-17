const BASE = "/api";

async function request<T>(path: string, opts?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...opts,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || "Request failed");
  }
  return res.json();
}

function downloadFile(path: string, fallbackName: string) {
  const link = document.createElement("a");
  link.href = `${BASE}${path}`;
  link.download = fallbackName;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}

export const api = {
  getStatus: () => request<any>("/status"),
  simulate: (params: { num_pipes: number; sim_years: number; seed: number }) =>
    request<any>("/simulate", { method: "POST", body: JSON.stringify(params) }),
  train: (modelType: string = "xgboost") =>
    request<any>("/train", { method: "POST", body: JSON.stringify({ model_type: modelType }) }),
  getOverview: () => request<any>("/overview"),
  getPipes: () => request<any>("/pipes"),
  getEvents: (filters?: any) =>
    request<any>("/events", { method: "POST", body: JSON.stringify(filters || {}) }),
  getAnalysis: () => request<any>("/analysis"),
  getModel: () => request<any>("/model"),
  getFilters: () => request<any>("/filters"),
  downloadPipesCSV: () => downloadFile("/download/pipes", "pipe_network_data.csv"),
  downloadEventsCSV: () => downloadFile("/download/events", "leak_events_data.csv"),
  downloadReport: () => downloadFile("/download/report", "overview_report.html"),
  downloadAnalysisReport: () => downloadFile("/download/report/analysis", "analysis_report.html"),
  downloadModelReport: () => downloadFile("/download/report/model", "model_report.html"),
};
