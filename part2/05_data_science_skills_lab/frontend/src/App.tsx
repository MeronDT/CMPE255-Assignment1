import { useEffect, useState } from "react";
import {
  ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, CartesianGrid, Cell, PieChart, Pie, Legend,
} from "recharts";
import { api } from "./api";

type Tab = "overview" | "catalog" | "live";

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

const COLORS = ["#6c8cff", "#4dd6b6", "#e6a24d", "#e65c5c", "#b06cff", "#5cc2e6"];

function Overview() {
  const { data, error } = useApi(api.catalog);
  if (error) return <div className="error">Failed to load: {error}</div>;
  if (!data) return <div className="loading">Loading...</div>;
  const phaseData = Object.entries(data.by_crispdm_phase).map(([name, value]) => ({ name, value }));
  return (
    <>
      <h2>Data Science Skills Mastery Lab</h2>
      <p className="lede">46 vetted skills installed from two third-party repos, cataloged and demonstrated live against real data — see the Catalog and Live Execution tabs.</p>
      <div className="grid cols-3">
        <Stat label="Total Skills Installed" value={String(data.n_total_skills)} />
        <Stat label="Live-Demonstrated" value={String(data.n_live_demonstrated)} sub="genuine execution, real data" />
        <Stat label="Source Repos" value="2" sub="both vetted before install" />
      </div>
      <div className="grid cols-2" style={{ marginTop: 16 }}>
        <div className="card">
          <h4>By Source Repo</h4>
          <ResponsiveContainer width="100%" height={220}>
            <PieChart>
              <Pie data={Object.entries(data.by_source_repo).map(([name, value]) => ({ name, value }))} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={70} isAnimationActive={false}>
                {Object.entries(data.by_source_repo).map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
              </Pie>
              <Tooltip contentStyle={{ background: "#161d2e", border: "1px solid #2a3550" }} labelStyle={{ color: "#e6ebf5" }} itemStyle={{ color: "#e6ebf5" }} cursor={{ fill: "rgba(108,140,255,0.12)", stroke: "#6c8cff", strokeOpacity: 0.4 }} />
              <Legend />
            </PieChart>
          </ResponsiveContainer>
        </div>
        <div className="card">
          <h4>By CRISP-DM Phase</h4>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={phaseData} layout="vertical" margin={{ left: 90 }}>
              <CartesianGrid stroke="#2a3550" />
              <XAxis type="number" stroke="#8a97b5" />
              <YAxis type="category" dataKey="name" stroke="#8a97b5" width={90} tick={{ fontSize: 10 }} />
              <Tooltip contentStyle={{ background: "#161d2e", border: "1px solid #2a3550" }} labelStyle={{ color: "#e6ebf5" }} itemStyle={{ color: "#e6ebf5" }} cursor={{ fill: "rgba(108,140,255,0.12)", stroke: "#6c8cff", strokeOpacity: 0.4 }} />
              <Bar dataKey="value" fill="#6c8cff" isAnimationActive={false} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
      <div className="finding" style={{ marginTop: 16 }}>
        Vetted before installation: both repos' install mechanisms were read in full (pure file-copy operations,
        zero dependencies, no network calls or eval), then cross-referenced against the professor's own reference
        repo's skill-name index for an independent legitimacy signal. Full vetting report in{" "}
        <code>docs/00_skill_vetting.md</code>.
      </div>
    </>
  );
}

function Catalog() {
  const { data } = useApi(api.catalog);
  const [filter, setFilter] = useState("");
  const [sourceFilter, setSourceFilter] = useState<string>("all");
  if (!data) return <div className="loading">Loading...</div>;
  const filtered = data.skills.filter((s: any) =>
    (sourceFilter === "all" || s.source_repo === sourceFilter) &&
    (s.name.toLowerCase().includes(filter.toLowerCase()) || s.description.toLowerCase().includes(filter.toLowerCase()))
  );
  return (
    <>
      <h2>Full Skill Catalog</h2>
      <input type="text" placeholder="Search skills..." value={filter} onChange={(e) => setFilter(e.target.value)} />
      <div className="sim-controls" style={{ marginBottom: 12 }}>
        {["all", "param087/agent-ml-skills", "nimrodfisher/data-analytics-skills"].map((s) => (
          <button key={s} className={sourceFilter === s ? "btn" : "btn secondary"} onClick={() => setSourceFilter(s)}>{s === "all" ? "All Sources" : s.split("/")[0]}</button>
        ))}
      </div>
      <div className="card" style={{ overflowX: "auto" }}>
        <table>
          <thead><tr><th>Skill</th><th>Description</th><th>Source</th><th>CRISP-DM Phase</th><th>Live Demo</th></tr></thead>
          <tbody>
            {filtered.map((s: any) => (
              <tr key={s.name}>
                <td><code>{s.name}</code></td>
                <td style={{ maxWidth: 320, fontSize: 12.5 }}>{s.description}</td>
                <td style={{ fontSize: 12 }}>{s.source_repo.split("/")[0]}</td>
                <td><span className="badge">{s.crispdm_phase}</span></td>
                <td>{s.live_demonstrated ? <span className="badge" style={{ color: "#4dd6b6" }}>✓ Live</span> : "—"}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <div className="stat-sub" style={{ marginTop: 8 }}>Showing {filtered.length} of {data.skills.length} skills</div>
      </div>
    </>
  );
}

// ---------- Custom, skill-specific visualizations (replacing raw JSON) ----------

function EdaView({ r }: { r: any }) {
  const missingData = Object.entries(r.missing_pct).filter(([, v]: any) => v > 0).map(([name, value]) => ({ name, value }));
  const corrData = Object.entries(r.correlation_with_target).map(([name, value]) => ({ name, value })).sort((a: any, b: any) => Math.abs(b.value) - Math.abs(a.value));
  return (
    <>
      <div className="grid cols-3">
        <Stat label="Rows" value={String(r.n_rows)} />
        <Stat label="Columns" value={String(r.n_cols)} />
        <Stat label="Survival Rate" value={`${r.survival_rate_pct}%`} />
      </div>
      <h4 style={{ marginTop: 14 }}>Missing Data by Column</h4>
      <ResponsiveContainer width="100%" height={180}>
        <BarChart data={missingData}>
          <CartesianGrid stroke="#2a3550" /><XAxis dataKey="name" stroke="#8a97b5" tick={{ fontSize: 10 }} /><YAxis stroke="#8a97b5" />
          <Tooltip contentStyle={{ background: "#161d2e", border: "1px solid #2a3550" }} labelStyle={{ color: "#e6ebf5" }} itemStyle={{ color: "#e6ebf5" }} cursor={{ fill: "rgba(108,140,255,0.12)", stroke: "#6c8cff", strokeOpacity: 0.4 }} />
          <Bar dataKey="value" fill="#e6a24d" isAnimationActive={false} />
        </BarChart>
      </ResponsiveContainer>
      <h4>Correlation with Survival</h4>
      <ResponsiveContainer width="100%" height={180}>
        <BarChart data={corrData}>
          <CartesianGrid stroke="#2a3550" /><XAxis dataKey="name" stroke="#8a97b5" tick={{ fontSize: 10 }} /><YAxis stroke="#8a97b5" />
          <Tooltip contentStyle={{ background: "#161d2e", border: "1px solid #2a3550" }} labelStyle={{ color: "#e6ebf5" }} itemStyle={{ color: "#e6ebf5" }} cursor={{ fill: "rgba(108,140,255,0.12)", stroke: "#6c8cff", strokeOpacity: 0.4 }} />
          <Bar dataKey="value" isAnimationActive={false}>
            {corrData.map((e: any, i: number) => <Cell key={i} fill={e.value > 0 ? "#4dd6b6" : "#e65c5c"} />)}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </>
  );
}

function CleaningView({ r }: { r: any }) {
  return (
    <div className="grid cols-2">
      <Stat label="Missing Values Before" value={String(r.missing_values_before)} />
      <Stat label="Missing Values After" value={String(r.missing_values_after)} />
      <div className="card" style={{ gridColumn: "1 / -1" }}>
        <h4>Imputation Strategy</h4>
        <table className="mini" style={{ width: "100%" }}>
          <tbody>{Object.entries(r.imputation_strategy).map(([k, v]: any) => <tr key={k}><td style={{ textAlign: "left" }}>{k}</td><td>{v}</td></tr>)}</tbody>
        </table>
        <p style={{ fontSize: 13, color: "#8a97b5", marginTop: 8 }}>Dropped (&gt;70% missing): {r.columns_dropped_high_missing.join(", ")}</p>
      </div>
    </div>
  );
}

function FeatureEngView({ r }: { r: any }) {
  const titleData = Object.entries(r.title_distribution).map(([name, value]) => ({ name, value }));
  return (
    <>
      <div className="grid cols-2">
        <Stat label="Avg Family Size" value={String(r.avg_family_size)} />
        <Stat label="% Traveling Alone" value={`${r.pct_traveling_alone}%`} />
      </div>
      <h4 style={{ marginTop: 14 }}>New Features: {r.new_features.join(", ")}</h4>
      <h4>Extracted Titles</h4>
      <ResponsiveContainer width="100%" height={200}>
        <BarChart data={titleData}>
          <CartesianGrid stroke="#2a3550" /><XAxis dataKey="name" stroke="#8a97b5" /><YAxis stroke="#8a97b5" />
          <Tooltip contentStyle={{ background: "#161d2e", border: "1px solid #2a3550" }} labelStyle={{ color: "#e6ebf5" }} itemStyle={{ color: "#e6ebf5" }} cursor={{ fill: "rgba(108,140,255,0.12)", stroke: "#6c8cff", strokeOpacity: 0.4 }} />
          <Bar dataKey="value" fill="#b06cff" isAnimationActive={false} />
        </BarChart>
      </ResponsiveContainer>
    </>
  );
}

function ModelEvalView({ r }: { r: any }) {
  const cm = r.confusion_matrix;
  return (
    <>
      <div className="grid cols-3">
        <Stat label="Accuracy" value={`${(r.accuracy * 100).toFixed(1)}%`} />
        <Stat label="ROC-AUC" value={String(r.roc_auc)} />
        <Stat label="F1" value={String(r.f1)} />
      </div>
      <h4 style={{ marginTop: 14 }}>Confusion Matrix</h4>
      <table className="mini">
        <thead><tr><th></th><th>Pred: {r.confusion_matrix_labels[0]}</th><th>Pred: {r.confusion_matrix_labels[1]}</th></tr></thead>
        <tbody>
          <tr><th>Actual: {r.confusion_matrix_labels[0]}</th><td>{cm[0][0]}</td><td>{cm[0][1]}</td></tr>
          <tr><th>Actual: {r.confusion_matrix_labels[1]}</th><td>{cm[1][0]}</td><td>{cm[1][1]}</td></tr>
        </tbody>
      </table>
    </>
  );
}

function HyperparamView({ r }: { r: any }) {
  const chartData = r.all_results.map((row: any, i: number) => ({ name: `#${i + 1}`, roc_auc: row.mean_roc_auc, params: JSON.stringify(row.params) }));
  return (
    <>
      <Stat label="Best CV ROC-AUC" value={String(r.best_cv_roc_auc)} sub={`${r.cv_folds}-fold CV`} />
      <h4 style={{ marginTop: 14 }}>Grid Search Results</h4>
      <ResponsiveContainer width="100%" height={200}>
        <BarChart data={chartData}>
          <CartesianGrid stroke="#2a3550" /><XAxis dataKey="name" stroke="#8a97b5" /><YAxis domain={[0.7, 0.95]} stroke="#8a97b5" />
          <Tooltip contentStyle={{ background: "#161d2e", border: "1px solid #2a3550" }} labelStyle={{ color: "#e6ebf5" }} itemStyle={{ color: "#e6ebf5" }} cursor={{ fill: "rgba(108,140,255,0.12)", stroke: "#6c8cff", strokeOpacity: 0.4 }} />
          <Bar dataKey="roc_auc" isAnimationActive={false}>
            {chartData.map((e: any, i: number) => <Cell key={i} fill={e.roc_auc === r.best_cv_roc_auc ? "#4dd6b6" : "#6c8cff"} />)}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
      <p style={{ fontSize: 12.5, color: "#8a97b5" }}>Best params: {JSON.stringify(r.best_params)}</p>
    </>
  );
}

function ImbalancedView({ r }: { r: any }) {
  if (r.skipped) return <div className="finding">Skipped: {r.error}</div>;
  return (
    <>
      <div className="grid cols-3">
        <Stat label="Transactions" value={r.n_transactions.toLocaleString()} />
        <Stat label="Fraud Rate" value={`${r.fraud_rate_pct}%`} />
        <Stat label="Naive 'Always Normal' Accuracy" value={`${r.naive_always_predict_normal_accuracy_pct}%`} sub="the accuracy trap" />
      </div>
      <div className="finding" style={{ marginTop: 12 }}>{r.lesson}</div>
    </>
  );
}

function SegmentationView({ r }: { r: any }) {
  return (
    <>
      <p style={{ fontSize: 13, color: "#8a97b5" }}>{r.algorithm}</p>
      <ResponsiveContainer width="100%" height={220}>
        <BarChart data={r.segments}>
          <CartesianGrid stroke="#2a3550" /><XAxis dataKey="segment" stroke="#8a97b5" /><YAxis stroke="#8a97b5" />
          <Tooltip contentStyle={{ background: "#161d2e", border: "1px solid #2a3550" }} labelStyle={{ color: "#e6ebf5" }} itemStyle={{ color: "#e6ebf5" }} cursor={{ fill: "rgba(108,140,255,0.12)", stroke: "#6c8cff", strokeOpacity: 0.4 }} />
          <Bar dataKey="avg_monetary" fill="#4dd6b6" isAnimationActive={false} />
        </BarChart>
      </ResponsiveContainer>
      <table className="mini" style={{ width: "100%", marginTop: 10 }}>
        <thead><tr><th>Segment</th><th>N Customers</th><th>Avg Monetary</th><th>Avg Frequency</th></tr></thead>
        <tbody>{r.segments.map((s: any) => <tr key={s.segment}><td>{s.segment}</td><td>{s.n_customers}</td><td>${s.avg_monetary}</td><td>{s.avg_frequency}</td></tr>)}</tbody>
      </table>
    </>
  );
}

function BusinessMetricsView({ r }: { r: any }) {
  return (
    <div className="grid cols-2">
      <Stat label="Avg Order Value" value={`$${r.avg_order_value_usd}`} />
      <Stat label="Repeat Purchase Rate" value={`${r.repeat_purchase_rate_pct}%`} />
      <Stat label="Avg Customer LTV" value={`$${r.avg_customer_lifetime_value_usd}`} />
      <Stat label="Total Customers" value={String(r.total_customers)} />
    </div>
  );
}

function AbTestView({ r }: { r: any }) {
  const chartData = [
    { name: "Control", rate: r.conversion_rate_control_pct },
    { name: "Treatment", rate: r.conversion_rate_treatment_pct },
  ];
  return (
    <>
      <div className="grid cols-3">
        <Stat label="Relative Lift" value={`${r.relative_lift_pct}%`} />
        <Stat label="p-value" value={String(r.p_value)} sub={r.statistically_significant_at_5pct ? "significant at 5%" : "not significant"} />
        <Stat label="z-statistic" value={String(r.z_statistic)} />
      </div>
      <ResponsiveContainer width="100%" height={180}>
        <BarChart data={chartData}>
          <CartesianGrid stroke="#2a3550" /><XAxis dataKey="name" stroke="#8a97b5" /><YAxis stroke="#8a97b5" />
          <Tooltip contentStyle={{ background: "#161d2e", border: "1px solid #2a3550" }} labelStyle={{ color: "#e6ebf5" }} itemStyle={{ color: "#e6ebf5" }} cursor={{ fill: "rgba(108,140,255,0.12)", stroke: "#6c8cff", strokeOpacity: 0.4 }} />
          <Bar dataKey="rate" isAnimationActive={false}>
            <Cell fill="#6c8cff" /><Cell fill="#4dd6b6" />
          </Bar>
        </BarChart>
      </ResponsiveContainer>
      <div className="finding" style={{ marginTop: 10 }}>{r.note}</div>
    </>
  );
}

function TimeSeriesView({ r }: { r: any }) {
  if (r.skipped) return <div className="finding">Skipped: source data not found</div>;
  const weekdayNames = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];
  const chartData = weekdayNames.map((name, i) => ({ name, revenue: r.weekly_seasonality_avg_by_weekday[String(i)] ?? 0 }));
  return (
    <>
      <div className="grid cols-3">
        <Stat label="Days Analyzed" value={String(r.n_days)} />
        <Stat label="Trend" value={r.overall_trend_direction} />
        <Stat label="Coefficient of Variation" value={String(r.coefficient_of_variation)} />
      </div>
      <ResponsiveContainer width="100%" height={200}>
        <BarChart data={chartData}>
          <CartesianGrid stroke="#2a3550" /><XAxis dataKey="name" stroke="#8a97b5" /><YAxis stroke="#8a97b5" />
          <Tooltip contentStyle={{ background: "#161d2e", border: "1px solid #2a3550" }} labelStyle={{ color: "#e6ebf5" }} itemStyle={{ color: "#e6ebf5" }} cursor={{ fill: "rgba(108,140,255,0.12)", stroke: "#6c8cff", strokeOpacity: 0.4 }} />
          <Bar dataKey="revenue" fill="#6c8cff" isAnimationActive={false} />
        </BarChart>
      </ResponsiveContainer>
    </>
  );
}

function CohortView({ r }: { r: any }) {
  const months = Object.keys(r.month_0_to_6_retention_pct).slice(0, 8);
  return (
    <>
      <Stat label="Cohorts Tracked" value={String(r.n_cohorts)} />
      <div className="card" style={{ overflowX: "auto", marginTop: 12 }}>
        <table className="mini">
          <thead><tr><th>Cohort</th>{[0, 1, 2, 3, 4, 5, 6].map((p) => <th key={p}>M{p}</th>)}</tr></thead>
          <tbody>
            {months.map((m) => (
              <tr key={m}>
                <td>{m}</td>
                {[0, 1, 2, 3, 4, 5, 6].map((p) => {
                  const v = r.month_0_to_6_retention_pct[m][String(p)];
                  return <td key={p} style={{ background: v ? `rgba(108,140,255,${v / 100})` : "transparent" }}>{v ?? ""}</td>;
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p style={{ fontSize: 12.5, color: "#8a97b5", marginTop: 8 }}>% of each cohort's customers still purchasing in month M (relative to their first purchase).</p>
    </>
  );
}

function GenericView({ r }: { r: any }) {
  return (
    <table className="mini" style={{ width: "100%" }}>
      <tbody>
        {Object.entries(r).map(([k, v]: any) => (
          <tr key={k}><td style={{ textAlign: "left" }}>{k}</td><td style={{ textAlign: "left" }}>{typeof v === "object" ? JSON.stringify(v) : String(v)}</td></tr>
        ))}
      </tbody>
    </table>
  );
}

const CUSTOM_VIEWS: Record<string, any> = {
  "exploratory-data-analysis": EdaView, "programmatic-eda": EdaView,
  "data-cleaning": CleaningView, "feature-engineering": FeatureEngView,
  "model-evaluation": ModelEvalView, "hyperparameter-tuning": HyperparamView,
  "imbalanced-data": ImbalancedView, "segmentation-analysis": SegmentationView,
  "business-metrics-calculator": BusinessMetricsView, "ab-test-analysis": AbTestView,
  "time-series-analysis": TimeSeriesView, "cohort-analysis": CohortView,
};

function LiveExecution() {
  const { data: catalog } = useApi(api.catalog);
  const { data: results } = useApi(api.executionResults);
  const [selected, setSelected] = useState<string | null>(null);
  if (!catalog || !results) return <div className="loading">Loading...</div>;
  const demonstrated = catalog.skills.filter((s: any) => s.live_demonstrated);
  const active = selected || demonstrated[0]?.name;
  const View = CUSTOM_VIEWS[active] || GenericView;

  return (
    <>
      <h2>Live Skill Execution</h2>
      <p className="lede">Every result below is genuine computation against real data — not mocked, not raw JSON.</p>
      <div className="sim-controls" style={{ flexWrap: "wrap" }}>
        {demonstrated.map((s: any) => (
          <button key={s.name} className={active === s.name ? "btn" : "btn secondary"} onClick={() => setSelected(s.name)}>{s.name}</button>
        ))}
      </div>
      <div className="card" style={{ marginTop: 12 }}>
        <h4>{active}</h4>
        <p style={{ fontSize: 13, color: "#8a97b5" }}>{demonstrated.find((s: any) => s.name === active)?.description}</p>
        <View r={results[active]} />
      </div>
    </>
  );
}

const TABS: { id: Tab; label: string }[] = [
  { id: "overview", label: "Overview" },
  { id: "catalog", label: "Full Catalog" },
  { id: "live", label: "Live Execution" },
];

export default function App() {
  const [tab, setTab] = useState<Tab>("overview");
  return (
    <div className="app">
      <header className="top">
        <div>
          <h1>Data Science Skills Mastery Lab</h1>
          <p>46 vetted skills · 2 source repos · CRISP-DM mapped</p>
        </div>
      </header>
      <nav className="tabs">
        {TABS.map((t) => (
          <button key={t.id} className={tab === t.id ? "active" : ""} onClick={() => setTab(t.id)}>{t.label}</button>
        ))}
      </nav>
      {tab === "overview" && <Overview />}
      {tab === "catalog" && <Catalog />}
      {tab === "live" && <LiveExecution />}
    </div>
  );
}
