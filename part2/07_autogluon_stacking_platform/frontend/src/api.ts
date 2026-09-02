const BASE = "http://localhost:8007";

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`, { cache: "no-store" });
  if (!res.ok) throw new Error(`${path} -> ${res.status}`);
  return res.json();
}

export const api = {
  summary: () => get<any>("/api/summary"),
  prepSummary: () => get<any>("/api/prep-summary"),
  modelingResults: () => get<any>("/api/modeling-results"),
  evaluation: () => get<any>("/api/evaluation"),
};
