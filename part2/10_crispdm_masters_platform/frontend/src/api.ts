const BASE = "http://localhost:8010";

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`, { cache: "no-store" });
  if (!res.ok) throw new Error(`${path} -> ${res.status}`);
  return res.json();
}

export const api = {
  prepSummary: () => get<any>("/api/prep-summary"),
  clustering: () => get<any>("/api/clustering"),
  anomaly: () => get<any>("/api/anomaly"),
  supervised: () => get<any>("/api/supervised"),
  association: () => get<any>("/api/association"),
  lsh: () => get<any>("/api/lsh"),
  synthesis: () => get<any>("/api/synthesis"),
};
