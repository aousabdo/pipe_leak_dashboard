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

export const api = {
  getStatus: () => request<any>("/status"),
  simulate: (params: { num_pipes: number; sim_years: number; seed: number }) =>
    request<any>("/simulate", { method: "POST", body: JSON.stringify(params) }),
  train: () => request<any>("/train", { method: "POST" }),
  getOverview: () => request<any>("/overview"),
  getPipes: () => request<any>("/pipes"),
  getEvents: (filters?: any) =>
    request<any>("/events", { method: "POST", body: JSON.stringify(filters || {}) }),
  getAnalysis: () => request<any>("/analysis"),
  getModel: () => request<any>("/model"),
  getFilters: () => request<any>("/filters"),
};
