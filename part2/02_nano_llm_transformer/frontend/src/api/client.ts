import type { AutoResearchResponse, ChatMessage, ChatResponse, ModelInfo, SystemInfo } from "../types";

async function request<T>(url: string, options?: RequestInit): Promise<T> {
  const res = await fetch(url, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(body.detail || "Request failed");
  }
  return res.json();
}

export function sendChat(
  messages: ChatMessage[],
  opts?: { temperature?: number; top_p?: number; max_new_tokens?: number }
): Promise<ChatResponse> {
  return request<ChatResponse>("/api/chat", {
    method: "POST",
    body: JSON.stringify({ messages, ...opts }),
  });
}

export function fetchSystemInfo(): Promise<SystemInfo> {
  return request<SystemInfo>("/api/system");
}

export function fetchModelInfo(): Promise<ModelInfo> {
  return request<ModelInfo>("/api/model-info");
}

export function fetchAutoResearch(): Promise<AutoResearchResponse> {
  return request<AutoResearchResponse>("/api/autoresearch");
}
