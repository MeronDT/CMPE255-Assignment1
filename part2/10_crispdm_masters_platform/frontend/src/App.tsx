import { useEffect, useState } from "react";
import {
  ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, CartesianGrid, Cell, Legend,
} from "recharts";
import { api } from "./api";
import Quiz, { type QuizQuestion } from "./Quiz";

type Tab = "overview" | "eda" | "clustering" | "anomaly" | "supervised" | "association" | "lsh" | "synthesis";

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

const COLORS = ["#6c8cff", "#4dd6b6", "#e6a24d", "#e65c5c", "#b06cff"];

// ---------- Quizzes ----------
const CLUSTERING_QUIZ: QuizQuestion[] = [
  { question: "Why was a business constraint (k in [3,8]) applied before searching for k, instead of letting silhouette maximization pick freely?",
    options: ["It's faster to compute", "Unconstrained maximization picked k=2, statistically valid but not actionable for a marketing team", "K-Means requires k >= 3 mathematically", "It matches the number of CPU cores"],
    correctIndex: 1, explanation: "Same finding as Project 03: the statistical optimum and the business-useful answer aren't always the same thing." },
  { question: "What does silhouette score measure?", options: ["Model training speed", "How well-separated and internally cohesive the clusters are", "The number of clusters", "Feature importance"],
    correctIndex: 1, explanation: "Silhouette combines intra-cluster cohesion and inter-cluster separation into one score, roughly -1 to 1." },
];
const ANOMALY_QUIZ: QuizQuestion[] = [
  { question: "Why were synthetic outliers injected instead of using real fraud labels?", options: ["Real labels were too expensive to obtain", "This dataset has no natural anomaly ground truth to evaluate against", "Synthetic data trains faster", "It's required by Isolation Forest"],
    correctIndex: 1, explanation: "Unlike Project 06's Credit Card Fraud data, Online Retail has no 'this is fraud' labels -- synthetic outliers let us sanity-check the detector works, honestly scoped as that, not a real evaluation." },
  { question: "What does it mean that all synthetic outliers ranked in the top 5% by anomaly score?", options: ["The model is overfit", "The detector correctly identifies extreme, multi-dimensional outliers as anomalous", "5% of all customers are fraudulent", "The contamination parameter was set to 5%"],
    correctIndex: 1, explanation: "This is the validation passing: deliberately extreme synthetic points were scored as highly anomalous, as they should be." },
];
const SUPERVISED_QUIZ: QuizQuestion[] = [
  { question: "Why do features only use the first 90 days, while the label uses full lifetime spend?", options: ["To make the model train faster", "To simulate a genuine early-prediction business scenario, not one that peeks at future data the model wouldn't have at prediction time", "Because more than 90 days of data wasn't available", "It's a coding convenience"],
    correctIndex: 1, explanation: "A real early-warning system only has early behavior to work with -- using later data as a feature would be unrealistic (and would leak information)." },
  { question: "41.5% of lifetime spend occurs within the first 90 days on average. Is this a leakage problem?", options: ["Yes, the model is cheating", "No -- it's expected correlation between early and total behavior, quantified explicitly rather than hidden", "It means the model is useless", "It means the label should be changed"],
    correctIndex: 1, explanation: "Leakage would mean using literally-future information the model shouldn't have access to. Early behavior correlating with total behavior is exactly the real-world signal a predictive model should pick up on." },
];
const ASSOCIATION_QUIZ: QuizQuestion[] = [
  { question: "What does 'lift' measure in an association rule?", options: ["How often the rule appears", "How much more often the itemset co-occurs than random chance would predict", "The rule's confidence", "The number of products in the rule"],
    correctIndex: 1, explanation: "Lift > 1 means genuine positive association; lift = 1 means no better than chance; lift < 1 means negative association." },
];
const LSH_QUIZ: QuizQuestion[] = [
  { question: "Why did the first LSH threshold attempt (0.25) give only 28% recall?", options: ["A bug in the MinHash implementation", "The threshold was miscalibrated -- true top-5 similarities in this data mostly fall below 0.25", "The dataset was too small", "LSH doesn't work on customer data"],
    correctIndex: 1, explanation: "This data's genuine most-similar-customer pairs mostly share only 5-25% of products (Jaccard similarity) -- a threshold above that range structurally excludes real matches, independent of the algorithm's correctness." },
  { question: "What's the fundamental trade-off LSH makes?", options: ["Memory for speed", "Exactness for speed -- may miss some true nearest neighbors in exchange for not scanning every item", "Accuracy for interpretability", "Training time for inference time"],
    correctIndex: 1, explanation: "LSH is an approximate method: sub-linear query time in exchange for a chance of missing some true near-neighbors, quantified here via recall@5." },
];

// ---------- Sections ----------
function Overview() {
  const { data: synth, error } = useApi(api.synthesis);
  if (error) return <div className="error">Failed to load: {error}</div>;
  if (!synth) return <div className="loading">Loading...</div>;
  return (
    <>
      <h2>CRISP-DM Master's Data Science Platform</h2>
      <p className="lede">{synth.dataset}. One dataset, five techniques, one CRISP-DM lifecycle.</p>
      <div className="grid cols-1">
        {synth.cross_technique_findings.map((f: any, i: number) => (
          <div className="card" key={i}>
            <h4 style={{ color: COLORS[i % COLORS.length] }}>{f.technique}</h4>
            <p style={{ margin: 0 }}>{f.headline}</p>
          </div>
        ))}
      </div>
      <div className="section-title">The Pattern Across All Five Techniques</div>
      <div className="finding">{synth.the_pattern_across_all_five}</div>
    </>
  );
}

function EDA() {
  const { data } = useApi(api.prepSummary);
  if (!data) return <div className="loading">Loading...</div>;
  return (
    <>
      <h2>Data Understanding &amp; Preparation</h2>
      <div className="grid cols-4">
        <Stat label="Raw Transactions" value={data.n_raw_transactions.toLocaleString()} />
        <Stat label="Missing Customer ID" value={data.n_missing_customer_id.toLocaleString()} sub="excluded, guest checkouts" />
        <Stat label="Customers (full history)" value={data.n_customers_full_history.toLocaleString()} />
        <Stat label="Eligible for Supervised Task" value={data.n_customers_eligible_for_supervised.toLocaleString()} sub=">=90 days tenure" />
      </div>
      <div className="grid cols-3" style={{ marginTop: 16 }}>
        <Stat label="Supervised Label Threshold" value={`$${data.supervised_label_threshold_monetary.toLocaleString()}`} sub="top-quartile lifetime spend" />
        <Stat label="Positive Rate" value={`${(data.supervised_positive_rate * 100).toFixed(1)}%`} />
        <Stat label="Multi-Item Baskets" value={data.n_multi_item_baskets.toLocaleString()} sub={`top ${data.n_products_in_basket_matrix} products`} />
      </div>
      <div className="finding" style={{ marginTop: 16 }}>
        On average, <strong>{data.pct_of_lifetime_spend_in_first_90_days_avg}%</strong> of a customer's lifetime spend occurs within
        their first 90 days — the expected-correlation figure discussed in the Supervised Learning tab's leakage note.
      </div>
    </>
  );
}

function Clustering() {
  const { data } = useApi(api.clustering);
  if (!data) return <div className="loading">Loading...</div>;
  return (
    <>
      <h2>1. Unsupervised Learning — Clustering</h2>
      <div className="grid cols-2">
        <Stat label="Winning k" value={String(data.winning_k)} sub="business-constrained, k in [3,8]" />
        <Stat label="Silhouette Score" value={String(data.winning_silhouette)} />
      </div>
      <div className="section-title">Segment Profiles</div>
      <table className="card" style={{ display: "table" }}>
        <thead><tr><th>Cluster</th><th>% Customers</th><th>% Revenue</th><th>Avg Recency</th><th>Avg Frequency</th><th>Avg Monetary</th></tr></thead>
        <tbody>
          {data.profiles.map((p: any) => (
            <tr key={p.cluster}><td>{p.cluster}</td><td>{p.pct_of_customers}%</td><td>{p.pct_of_revenue}%</td><td>{p.avg_recency}d</td><td>{p.avg_frequency}</td><td>${p.avg_monetary.toLocaleString()}</td></tr>
          ))}
        </tbody>
      </table>
      <div className="section-title">Silhouette vs. k</div>
      <div className="card">
        <ResponsiveContainer width="100%" height={240}>
          <BarChart data={data.sweep}>
            <CartesianGrid stroke="#2a3550" />
            <XAxis dataKey="k" stroke="#8a97b5" />
            <YAxis stroke="#8a97b5" />
            <Tooltip contentStyle={{ background: "#161d2e", border: "1px solid #2a3550" }} labelStyle={{ color: "#e6ebf5" }} itemStyle={{ color: "#e6ebf5" }} cursor={{ fill: "rgba(108,140,255,0.12)", stroke: "#6c8cff", strokeOpacity: 0.4 }} />
            <Bar dataKey="silhouette" isAnimationActive={false}>
              {data.sweep.map((e: any, i: number) => <Cell key={i} fill={e.k === data.winning_k ? "#4dd6b6" : "#6c8cff"} />)}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
      <Quiz title="Clustering" questions={CLUSTERING_QUIZ} />
    </>
  );
}

function Anomaly() {
  const { data } = useApi(api.anomaly);
  if (!data) return <div className="loading">Loading...</div>;
  return (
    <>
      <h2>2. Anomaly / Outlier Detection</h2>
      <div className="finding">{data.validation_note}</div>
      <div className="grid cols-3" style={{ marginTop: 14 }}>
        <Stat label="Synthetic Outliers Injected" value={String(data.n_synthetic_outliers_injected)} />
        <Stat label="Mean Percentile Rank" value={`${data.synthetic_outlier_mean_percentile_rank}th`} />
        <Stat label="All in Top 5%?" value={String(data.all_synthetic_outliers_in_top_5pct)} />
      </div>
      <div className="section-title">Top Real Anomalies</div>
      <table className="card" style={{ display: "table" }}>
        <thead><tr><th>Customer</th><th>Score</th><th>Monetary</th><th>Frequency</th><th>Distinct Products</th></tr></thead>
        <tbody>
          {data.top_real_anomalies.slice(0, 10).map((r: any) => (
            <tr key={r.CustomerID}><td>{r.CustomerID}</td><td>{r.AnomalyScore.toFixed(3)}</td><td>${r.Monetary.toLocaleString()}</td><td>{r.Frequency}</td><td>{r.DistinctProducts}</td></tr>
          ))}
        </tbody>
      </table>
      <Quiz title="Anomaly Detection" questions={ANOMALY_QUIZ} />
    </>
  );
}

function Supervised() {
  const { data } = useApi(api.supervised);
  if (!data) return <div className="loading">Loading...</div>;
  const chartData = Object.entries(data.model_comparison).map(([name, m]: [string, any]) => ({ name, roc_auc: m.roc_auc }));
  return (
    <>
      <h2>3. Supervised Learning</h2>
      <p>Predict whether a customer becomes high-value using only their first-90-day behavior.</p>
      <div className="finding">{data.leakage_note}</div>
      <div className="section-title">Model Comparison (ROC-AUC)</div>
      <div className="card">
        <ResponsiveContainer width="100%" height={240}>
          <BarChart data={chartData}>
            <CartesianGrid stroke="#2a3550" />
            <XAxis dataKey="name" stroke="#8a97b5" tick={{ fontSize: 11 }} />
            <YAxis domain={[0.8, 1]} stroke="#8a97b5" />
            <Tooltip contentStyle={{ background: "#161d2e", border: "1px solid #2a3550" }} labelStyle={{ color: "#e6ebf5" }} itemStyle={{ color: "#e6ebf5" }} cursor={{ fill: "rgba(108,140,255,0.12)", stroke: "#6c8cff", strokeOpacity: 0.4 }} />
            <Bar dataKey="roc_auc" isAnimationActive={false}>
              {chartData.map((e, i) => <Cell key={i} fill={e.name === data.winner ? "#4dd6b6" : "#6c8cff"} />)}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
      <table className="card" style={{ display: "table" }}>
        <thead><tr><th>Model</th><th>ROC-AUC</th><th>Precision</th><th>Recall</th><th>F1</th></tr></thead>
        <tbody>
          {Object.entries(data.model_comparison).map(([name, m]: [string, any]) => (
            <tr key={name} style={name === data.winner ? { background: "rgba(77,214,182,0.1)" } : {}}>
              <td>{name}</td><td>{m.roc_auc}</td><td>{m.precision}</td><td>{m.recall}</td><td>{m.f1}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <div className="section-title">Feature Importance ({data.winner})</div>
      <table className="card" style={{ display: "table" }}>
        <thead><tr><th>Feature</th><th>Importance</th></tr></thead>
        <tbody>{data.feature_importance.map((f: any, i: number) => <tr key={i}><td>{f.feature}</td><td>{f.importance}</td></tr>)}</tbody>
      </table>
      <Quiz title="Supervised Learning" questions={SUPERVISED_QUIZ} />
    </>
  );
}

function Association() {
  const { data } = useApi(api.association);
  if (!data) return <div className="loading">Loading...</div>;
  return (
    <>
      <h2>4. Associative Rule Mining</h2>
      <div className="grid cols-3">
        <Stat label="Rules Found" value={String(data.n_rules)} sub="business-constrained 5-50 band" />
        <Stat label="Avg Lift" value={`${data.avg_lift}x`} />
        <Stat label="Winning Config" value={`s=${data.winning_support}, c=${data.winning_confidence}`} />
      </div>
      <div className="section-title">Top Rules by Lift</div>
      <table className="card" style={{ display: "table" }}>
        <thead><tr><th>If Bought</th><th>Then Also Buy</th><th>Support</th><th>Confidence</th><th>Lift</th></tr></thead>
        <tbody>
          {data.top_rules.slice(0, 12).map((r: any, i: number) => (
            <tr key={i}><td>{r.antecedents.join(" + ")}</td><td>{r.consequents.join(" + ")}</td><td>{(r.support * 100).toFixed(1)}%</td><td>{(r.confidence * 100).toFixed(0)}%</td><td><span className="badge">{r.lift.toFixed(2)}x</span></td></tr>
          ))}
        </tbody>
      </table>
      <Quiz title="Association Rules" questions={ASSOCIATION_QUIZ} />
    </>
  );
}

function LSH() {
  const { data } = useApi(api.lsh);
  if (!data) return <div className="loading">Loading...</div>;
  return (
    <>
      <h2>5. Sub-linear Search — Locality-Sensitive Hashing</h2>
      <div className="grid cols-4">
        <Stat label="Customers Indexed" value={data.n_customers_indexed.toLocaleString()} />
        <Stat label="Speedup vs. Brute Force" value={`${data.speedup_factor}x`} />
        <Stat label="Recall@5" value={`${(data.avg_recall_at_5 * 100).toFixed(0)}%`} />
        <Stat label="LSH Query Time" value={`${data.avg_lsh_query_ms}ms`} sub={`vs. ${data.avg_brute_force_query_ms}ms brute force`} />
      </div>
      <div className="finding" style={{ marginTop: 14 }}>{data.method_note}</div>
      <div className="section-title">Example Query: Customer {data.example_queries[1]?.query_customer}</div>
      {data.example_queries[1] && (
        <div className="grid cols-2">
          <div className="card">
            <h4>LSH Approximate Top-5</h4>
            <table className="mini">
              <thead><tr><th>Customer</th><th>Jaccard Sim.</th></tr></thead>
              <tbody>{data.example_queries[1].lsh_top_k.map((r: any, i: number) => <tr key={i}><td>{r.customer}</td><td>{r.jaccard_similarity}</td></tr>)}</tbody>
            </table>
          </div>
          <div className="card">
            <h4>Brute-Force Exact Top-5</h4>
            <table className="mini">
              <thead><tr><th>Customer</th><th>Jaccard Sim.</th></tr></thead>
              <tbody>{data.example_queries[1].brute_force_top_k.map((r: any, i: number) => <tr key={i}><td>{r.customer}</td><td>{r.jaccard_similarity}</td></tr>)}</tbody>
            </table>
          </div>
        </div>
      )}
      <Quiz title="LSH & Sub-linear Search" questions={LSH_QUIZ} />
    </>
  );
}

function Synthesis() {
  const { data } = useApi(api.synthesis);
  if (!data) return <div className="loading">Loading...</div>;
  return (
    <>
      <h2>Conclusion &amp; Synthesis</h2>
      <p className="lede">What did applying every major mining paradigm to one dataset actually teach us?</p>
      {data.cross_technique_findings.map((f: any, i: number) => (
        <div className="card" key={i}>
          <h4 style={{ color: COLORS[i % COLORS.length] }}>{f.technique}</h4>
          <p style={{ margin: 0 }}>{f.headline}</p>
        </div>
      ))}
      <div className="section-title">The Cross-Cutting Lesson</div>
      <div className="finding" style={{ fontSize: 14.5 }}>{data.the_pattern_across_all_five}</div>
    </>
  );
}

const TABS: { id: Tab; label: string }[] = [
  { id: "overview", label: "Overview" },
  { id: "eda", label: "Data & EDA" },
  { id: "clustering", label: "1. Clustering" },
  { id: "anomaly", label: "2. Anomaly Detection" },
  { id: "supervised", label: "3. Supervised ML" },
  { id: "association", label: "4. Association Rules" },
  { id: "lsh", label: "5. LSH Search" },
  { id: "synthesis", label: "Conclusion" },
];

export default function App() {
  const [tab, setTab] = useState<Tab>("overview");
  return (
    <div className="app">
      <header className="top">
        <div>
          <h1>CRISP-DM Master's Data Science Platform</h1>
          <p>Every major mining paradigm, one dataset, one lifecycle</p>
        </div>
      </header>
      <nav className="tabs">
        {TABS.map((t) => (
          <button key={t.id} className={tab === t.id ? "active" : ""} onClick={() => setTab(t.id)}>{t.label}</button>
        ))}
      </nav>
      {tab === "overview" && <Overview />}
      {tab === "eda" && <EDA />}
      {tab === "clustering" && <Clustering />}
      {tab === "anomaly" && <Anomaly />}
      {tab === "supervised" && <Supervised />}
      {tab === "association" && <Association />}
      {tab === "lsh" && <LSH />}
      {tab === "synthesis" && <Synthesis />}
    </div>
  );
}
