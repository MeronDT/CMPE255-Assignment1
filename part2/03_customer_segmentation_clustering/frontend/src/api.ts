const BASE = "http://localhost:8003";

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`, { cache: "no-store" });
  if (!res.ok) throw new Error(`${path} -> ${res.status}`);
  return res.json();
}

export interface ClusterProfile {
  cluster: number;
  persona: string;
  n_customers: number;
  pct_of_customers: number;
  total_revenue: number;
  pct_of_revenue: number;
  avg_recency_days: number;
  avg_frequency: number;
  avg_monetary: number;
  avg_basket_value: number;
  avg_tenure_days: number;
  avg_cancellation_rate: number;
}

export interface Customer {
  CustomerID: number;
  Recency: number;
  Frequency: number;
  Monetary: number;
  AvgBasketValue: number;
  DistinctProducts: number;
  TenureDays: number;
  CancellationRate: number;
  PrimaryCountry: string;
  Cluster: number;
}

export const api = {
  summary: () => get<any>("/api/summary"),
  eda: () => get<any>("/api/eda"),
  prepSummary: () => get<any>("/api/prep-summary"),
  modelingBaseline: () => get<any>("/api/modeling-baseline"),
  autoresearch: () => get<any>("/api/autoresearch"),
  clusterProfiles: () => get<{ n_customers: number; n_clusters: number; revenue_concentration_check: string; profiles: ClusterProfile[] }>("/api/cluster-profiles"),
  customers: () => get<Customer[]>("/api/customers"),
};
