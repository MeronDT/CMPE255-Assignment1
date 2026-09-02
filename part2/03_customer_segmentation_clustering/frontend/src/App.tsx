import { useEffect, useState } from "react";
import {
  ResponsiveContainer, ScatterChart, Scatter, XAxis, YAxis, ZAxis, Tooltip, CartesianGrid,
  BarChart, Bar, LineChart, Line, Legend, Cell,
} from "recharts";
import { api, type ClusterProfile, type Customer } from "./api";

const CLUSTER_COLORS = ["#6c8cff", "#e65c5c", "#4dd6b6", "#e6a24d", "#b06cff", "#5cc2e6"];

type Tab = "overview" | "segments" | "explorer" | "modeling" | "autoresearch" | "eda";

function useApi<T>(fn: () => Promise<T>) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    fn().then(setData).catch((e) => setError(String(e)));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  return { data, error };
}

function Stat({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className="card">
      <h3>{label}</h3>
      <div className="stat-value">{value}</div>
      {sub && <div className="stat-sub">{sub}</div>}
    </div>
  );
}

function Overview() {
  const { data: summary, error } = useApi(api.summary);
  const { data: profiles } = useApi(api.clusterProfiles);
  if (error) return <div className="error">Failed to load: {error}</div>;
  if (!summary) return <div className="loading">Loading...</div>;
  return (
    <>
      <div className="grid cols-4">
        <Stat label="Customers" value={summary.n_customers?.toLocaleString()} />
        <Stat label="Segments Found" value={String(summary.n_clusters)} sub="via AutoResearch hill-climb" />
        <Stat label="Silhouette Score" value={summary.winning_silhouette?.toFixed(3)} sub="cluster separation quality" />
        <Stat label="Algorithm" value="K-Means" sub="winner of 4-algorithm tournament" />
      </div>

      <div className="section-title">Business Reading</div>
      <div className="finding">{summary.revenue_concentration_check}</div>

      {profiles && (
        <>
          <div className="section-title">Segment Summary</div>
          <div className="grid cols-3">
            {profiles.profiles.map((p: ClusterProfile) => (
              <div key={p.cluster} className={`card persona-card ${p.persona.includes("At-Risk") ? "at-risk" : p.persona.includes("New") ? "new" : ""}`}>
                <h3>{p.persona}</h3>
                <div className="stat-value">{p.pct_of_customers}%</div>
                <div className="stat-sub">{p.n_customers.toLocaleString()} customers · {p.pct_of_revenue}% of revenue</div>
              </div>
            ))}
          </div>
        </>
      )}
    </>
  );
}

function Segments() {
  const { data } = useApi(api.clusterProfiles);
  const { data: customers } = useApi(api.customers);
  if (!data || !customers) return <div className="loading">Loading...</div>;
  return (
    <>
      <div className="section-title">Recency vs. Monetary by Segment</div>
      <div className="card">
        <ResponsiveContainer width="100%" height={380}>
          <ScatterChart margin={{ top: 10, right: 20, bottom: 10, left: 0 }}>
            <CartesianGrid stroke="#2a3550" />
            <XAxis type="number" dataKey="Recency" name="Recency (days)" stroke="#8a97b5" />
            <YAxis type="number" dataKey="LogMonetary" name="Monetary ($, log scale)" stroke="#8a97b5"
              tickFormatter={(v: number) => Math.round(Math.exp(v)).toLocaleString()} />
            <ZAxis range={[30, 30]} />
            <Tooltip cursor={{ strokeDasharray: "3 3" }} contentStyle={{ background: "#161d2e", border: "1px solid #2a3550" }} labelStyle={{ color: "#e6ebf5" }} itemStyle={{ color: "#e6ebf5" }}
              formatter={(value: number, name: string) => name === "LogMonetary" ? [`$${Math.round(Math.exp(value)).toLocaleString()}`, "Monetary"] : [value, name]} />
            <Legend />
            {data.profiles.map((p) => (
              <Scatter
                key={p.cluster}
                name={p.persona}
                data={customers.filter((c) => c.Cluster === p.cluster).map((c) => ({ ...c, LogMonetary: Math.log(Math.max(c.Monetary, 0.01)) }))}
                fill={CLUSTER_COLORS[p.cluster % CLUSTER_COLORS.length]}
                isAnimationActive={false}
              />
            ))}
          </ScatterChart>
        </ResponsiveContainer>
      </div>

      <div className="section-title">Segment Detail</div>
      <div className="grid cols-1">
        <table className="card" style={{ display: "table" }}>
          <thead>
            <tr>
              <th>Persona</th><th>Customers</th><th>% Revenue</th><th>Avg Recency</th>
              <th>Avg Frequency</th><th>Avg Monetary</th><th>Avg Basket</th><th>Cancel Rate</th>
            </tr>
          </thead>
          <tbody>
            {data.profiles.map((p) => (
              <tr key={p.cluster}>
                <td><span className="badge" style={{ color: CLUSTER_COLORS[p.cluster % CLUSTER_COLORS.length] }}>{p.persona}</span></td>
                <td>{p.n_customers.toLocaleString()}</td>
                <td>{p.pct_of_revenue}%</td>
                <td>{p.avg_recency_days}d</td>
                <td>{p.avg_frequency}</td>
                <td>${p.avg_monetary.toLocaleString()}</td>
                <td>${p.avg_basket_value}</td>
                <td>{(p.avg_cancellation_rate * 100).toFixed(1)}%</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  );
}

function Explorer() {
  const { data } = useApi(api.customers);
  const [q, setQ] = useState("");
  if (!data) return <div className="loading">Loading...</div>;
  const filtered = data.filter((c) =>
    String(c.CustomerID).includes(q) || c.PrimaryCountry.toLowerCase().includes(q.toLowerCase())
  ).slice(0, 200);
  return (
    <>
      <input type="text" placeholder="Search by Customer ID or Country..." value={q} onChange={(e) => setQ(e.target.value)} />
      <div className="card" style={{ overflowX: "auto" }}>
        <table>
          <thead>
            <tr>
              <th>ID</th><th>Segment</th><th>Country</th><th>Recency</th><th>Frequency</th>
              <th>Monetary</th><th>Basket</th><th>Products</th><th>Tenure</th>
            </tr>
          </thead>
          <tbody>
            {filtered.map((c: Customer) => (
              <tr key={c.CustomerID}>
                <td>{c.CustomerID}</td>
                <td><span className="badge" style={{ color: CLUSTER_COLORS[c.Cluster % CLUSTER_COLORS.length] }}>{c.Cluster}</span></td>
                <td>{c.PrimaryCountry}</td>
                <td>{c.Recency}d</td>
                <td>{c.Frequency}</td>
                <td>${c.Monetary.toFixed(0)}</td>
                <td>${c.AvgBasketValue.toFixed(0)}</td>
                <td>{c.DistinctProducts}</td>
                <td>{c.TenureDays}d</td>
              </tr>
            ))}
          </tbody>
        </table>
        <div className="stat-sub" style={{ marginTop: 8 }}>Showing {filtered.length} of {data.length.toLocaleString()} customers</div>
      </div>
    </>
  );
}

function Modeling() {
  const { data } = useApi(api.modelingBaseline);
  if (!data) return <div className="loading">Loading...</div>;
  return (
    <>
      <div className="finding">
        <strong>Algorithm tournament</strong> (K-Means, Agglomerative/Ward, Gaussian Mixture, DBSCAN)
        compared at the unconstrained-optimal k={data.best_k_by_silhouette}. K-Means was deployed
        as the baseline for its interpretable convex segments, then improved further by AutoResearch
        (see AutoResearch tab) once a business-actionable k range was enforced.
      </div>
      <div className="section-title">Silhouette vs. k (K-Means)</div>
      <div className="card">
        <ResponsiveContainer width="100%" height={280}>
          <LineChart data={data.k_sweep}>
            <CartesianGrid stroke="#2a3550" />
            <XAxis dataKey="k" stroke="#8a97b5" />
            <YAxis stroke="#8a97b5" />
            <Tooltip contentStyle={{ background: "#161d2e", border: "1px solid #2a3550" }} labelStyle={{ color: "#e6ebf5" }} itemStyle={{ color: "#e6ebf5" }} cursor={{ fill: "rgba(108,140,255,0.12)", stroke: "#6c8cff", strokeOpacity: 0.4 }} />
            <Line type="monotone" dataKey="silhouette" stroke="#6c8cff" strokeWidth={2} dot={{ r: 3 }} isAnimationActive={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>
      <div className="section-title">Algorithm Comparison (at k={data.best_k_by_silhouette})</div>
      <table className="card" style={{ display: "table" }}>
        <thead><tr><th>Algorithm</th><th>Silhouette</th><th>Davies-Bouldin</th><th>Calinski-Harabasz</th></tr></thead>
        <tbody>
          {Object.entries(data.algorithm_comparison).map(([name, m]: [string, any]) => (
            <tr key={name}>
              <td>{name}</td><td>{m?.silhouette?.toFixed(3)}</td><td>{m?.davies_bouldin?.toFixed(3)}</td><td>{m?.calinski_harabasz?.toFixed(0)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </>
  );
}

function AutoResearch() {
  const { data } = useApi(api.autoresearch);
  if (!data) return <div className="loading">Loading...</div>;
  return (
    <>
      <div className="finding">
        <strong>Phase 1 — unconstrained optimum:</strong> pure silhouette maximization picks k=2
        (silhouette {data.phase1_unconstrained_optimum.silhouette.toFixed(3)}) — statistically
        best-separated, but "big spenders vs. everyone else" isn't something a marketing team can
        act on. {data.phase1_unconstrained_optimum.note}.
      </div>
      <div className="finding" style={{ marginTop: 10 }}>
        <strong>Phase 4 — finalized winner:</strong> k={data.phase4_finalize.k} with{" "}
        {data.phase4_finalize.scaler} scaling, silhouette {data.phase4_finalize.silhouette.toFixed(3)} —
        genuinely <em>better separated</em> than the unconstrained k=2 result
        ({data.phase4_finalize.vs_unconstrained_k2_silhouette.toFixed(3)}), while also being
        business-actionable. Not a compromise — an outright improvement found by applying the
        RFM-literature-informed k∈[3,8] constraint before searching.
      </div>

      <div className="section-title">Phase 3: Hill-Climb over k (business-constrained)</div>
      <div className="card">
        <ResponsiveContainer width="100%" height={260}>
          <BarChart data={data.phase3_hill_climb}>
            <CartesianGrid stroke="#2a3550" />
            <XAxis dataKey="k" stroke="#8a97b5" />
            <YAxis stroke="#8a97b5" />
            <Tooltip contentStyle={{ background: "#161d2e", border: "1px solid #2a3550" }} labelStyle={{ color: "#e6ebf5" }} itemStyle={{ color: "#e6ebf5" }} cursor={{ fill: "rgba(108,140,255,0.12)", stroke: "#6c8cff", strokeOpacity: 0.4 }} />
            <Bar dataKey="silhouette" isAnimationActive={false}>
              {data.phase3_hill_climb.map((entry: any, i: number) => (
                <Cell key={i} fill={entry.k === data.phase4_finalize.k ? "#4dd6b6" : "#6c8cff"} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div className="section-title">Phase 2: Feature/Transform Search</div>
      <table className="card" style={{ display: "table" }}>
        <thead><tr><th>Feature Set</th><th>Scaler</th><th>Silhouette</th><th>Davies-Bouldin</th></tr></thead>
        <tbody>
          {data.phase2_feature_transform_search.map((r: any, i: number) => (
            <tr key={i}><td>{r.feature_set}</td><td>{r.scaler}</td><td>{r.silhouette.toFixed(3)}</td><td>{r.davies_bouldin.toFixed(3)}</td></tr>
          ))}
        </tbody>
      </table>
    </>
  );
}

function EDA() {
  const { data: eda } = useApi(api.eda);
  const { data: prep } = useApi(api.prepSummary);
  if (!eda || !prep) return <div className="loading">Loading...</div>;
  return (
    <>
      <div className="grid cols-4">
        <Stat label="Raw Transactions" value={eda.n_transactions.toLocaleString()} />
        <Stat label="Missing Customer ID" value={`${eda.pct_missing_customer_id}%`} sub="excluded from clustering" />
        <Stat label="Cancellation Rows" value={`${eda.pct_cancellation}%`} sub="netted into totals, not dropped" />
        <Stat label="After Cleaning" value={prep.n_customers.toLocaleString()} sub="customers with valid RFM features" />
      </div>
      <div className="section-title">Country Distribution (Top 10)</div>
      <table className="card" style={{ display: "table" }}>
        <thead><tr><th>Country</th><th>Transactions</th></tr></thead>
        <tbody>
          {Object.entries(eda.top_countries).map(([c, n]: [string, any]) => (
            <tr key={c}><td>{c}</td><td>{n.toLocaleString()}</td></tr>
          ))}
        </tbody>
      </table>
    </>
  );
}

export default function App() {
  const [tab, setTab] = useState<Tab>("overview");
  return (
    <div className="app">
      <header className="top">
        <div>
          <h1>Customer Intelligence &amp; Segmentation</h1>
          <p>RFM clustering · Online Retail dataset · CRISP-DM + AutoResearch</p>
        </div>
      </header>
      <nav className="tabs">
        {(["overview", "segments", "explorer", "modeling", "autoresearch", "eda"] as Tab[]).map((t) => (
          <button key={t} className={tab === t ? "active" : ""} onClick={() => setTab(t)}>
            {t === "eda" ? "Data & EDA" : t.charAt(0).toUpperCase() + t.slice(1)}
          </button>
        ))}
      </nav>
      {tab === "overview" && <Overview />}
      {tab === "segments" && <Segments />}
      {tab === "explorer" && <Explorer />}
      {tab === "modeling" && <Modeling />}
      {tab === "autoresearch" && <AutoResearch />}
      {tab === "eda" && <EDA />}
    </div>
  );
}
