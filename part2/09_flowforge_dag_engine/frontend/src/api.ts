const BASE = "http://localhost:8009";

export interface TaskDef {
  id: string; label: string; dependsOn: string[]; durationMs: number; simulateFailure?: boolean;
}
export interface WorkflowSummary { id: string; name: string; description: string; nTasks: number; }
export interface WorkflowDef extends WorkflowSummary { tasks: TaskDef[]; }

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`, { cache: "no-store" });
  if (!res.ok) throw new Error(`${path} -> ${res.status}`);
  return res.json();
}

export const api = {
  workflows: () => get<WorkflowSummary[]>("/api/workflows"),
  workflow: (id: string) => get<WorkflowDef>(`/api/workflows/${id}`),
  runWorkflow: async (id: string) => {
    const res = await fetch(`${BASE}/api/workflows/${id}/run`, { method: "POST" });
    return res.json();
  },
  wsUrl: "ws://localhost:8009/ws",
};
