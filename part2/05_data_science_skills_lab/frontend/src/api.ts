const BASE = "http://localhost:8005";

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`, { cache: "no-store" });
  if (!res.ok) throw new Error(`${path} -> ${res.status}`);
  return res.json();
}

export const api = {
  catalog: () => get<any>("/api/catalog"),
  executionResults: () => get<any>("/api/execution-results"),
};
