import type { FilterStatus, SortMode, Stats, Todo, TodoInput, TodoUpdate } from "../types";

const BASE = "/api/todos";

async function request<T>(url: string, options?: RequestInit): Promise<T> {
  const res = await fetch(url, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({ error: res.statusText }));
    throw new Error(body.error?.formErrors?.join(", ") || body.error || "Request failed");
  }
  return res.json();
}

export interface FetchTodosParams {
  search?: string;
  status?: FilterStatus;
  priority?: string;
  tag?: string;
  sort?: SortMode;
}

export function fetchTodos(params: FetchTodosParams = {}): Promise<Todo[]> {
  const query = new URLSearchParams();
  if (params.search) query.set("search", params.search);
  if (params.status && params.status !== "all") query.set("completed", String(params.status === "completed"));
  if (params.priority) query.set("priority", params.priority);
  if (params.tag) query.set("tag", params.tag);
  if (params.sort) query.set("sort", params.sort);
  const qs = query.toString();
  return request<Todo[]>(`${BASE}${qs ? `?${qs}` : ""}`);
}

export function fetchStats(): Promise<Stats> {
  return request<Stats>(`${BASE}/stats`);
}

export function createTodo(input: TodoInput): Promise<Todo> {
  return request<Todo>(BASE, { method: "POST", body: JSON.stringify(input) });
}

export function updateTodo(id: string, update: TodoUpdate): Promise<Todo> {
  return request<Todo>(`${BASE}/${id}`, { method: "PATCH", body: JSON.stringify(update) });
}

export function deleteTodo(id: string): Promise<void> {
  return request<void>(`${BASE}/${id}`, { method: "DELETE" });
}

export function reorderTodos(orderedIds: string[]): Promise<void> {
  return request<void>(`${BASE}/reorder`, { method: "POST", body: JSON.stringify({ orderedIds }) });
}

export function clearCompleted(): Promise<{ deleted: number }> {
  return request<{ deleted: number }>(`${BASE}/completed`, { method: "DELETE" });
}
