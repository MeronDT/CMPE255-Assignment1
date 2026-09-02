import type { AutoResearchResponse, ModelInfo, PredictResponse, ZoneInfo } from "../types";

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

export function fetchZones(): Promise<ZoneInfo[]> {
  return request<ZoneInfo[]>("/api/zones");
}

export function fetchModelInfo(): Promise<ModelInfo> {
  return request<ModelInfo>("/api/model-info");
}

export function fetchAutoResearch(): Promise<AutoResearchResponse> {
  return request<AutoResearchResponse>("/api/autoresearch");
}

export function predictTrip(input: {
  pickup_location_id: number;
  dropoff_location_id: number;
  pickup_datetime: string;
  passenger_count: number;
}): Promise<PredictResponse> {
  return request<PredictResponse>("/api/predict", {
    method: "POST",
    body: JSON.stringify(input),
  });
}
