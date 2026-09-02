const BASE = "http://localhost:8012";

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`, { cache: "no-store" });
  if (!res.ok) throw new Error(`${path} -> ${res.status}`);
  return res.json();
}

export const api = {
  summary: () => get<any>("/api/summary"),
  prepSummary: () => get<any>("/api/prep-summary"),
  modelingBaseline: () => get<any>("/api/modeling-baseline"),
  autoresearch: () => get<any>("/api/autoresearch"),
  evaluation: () => get<any>("/api/evaluation"),
};
