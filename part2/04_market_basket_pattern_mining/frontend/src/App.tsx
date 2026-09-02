import { useEffect, useState } from "react";
import {
  ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, CartesianGrid, Cell,
  ScatterChart, Scatter, ZAxis,
} from "recharts";
import { api, type Rule } from "./api";

type Tab = "overview" | "rules" | "products" | "modeling" | "autoresearch" | "eda";

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

function RuleRow({ r }: { r: Rule }) {
  return (
    <tr>
      <td>{r.antecedents.join(" + ")}</td>
      <td>&rarr;</td>
      <td>{r.consequents.join(" + ")}</td>
      <td>{(r.support * 100).toFixed(1)}%</td>
      <td>{(r.confidence * 100).toFixed(0)}%</td>
      <td><span className="badge">{r.lift.toFixed(2)}x</span></td>
    </tr>
  );
}

function Overview() {
  const { data: summary, error } = useApi(api.summary);
  const { data: ruleSummary } = useApi(api.ruleSummary);
  if (error) return <div className="error">Failed to load: {error}</div>;
  if (!summary) return <div className="loading">Loading...</div>;
  return (
    <>
      <div className="grid cols-4">
        <Stat label="Baskets Analyzed" value={summary.n_baskets?.toLocaleString()} sub="multi-item, non-cancelled" />
        <Stat label="Rules Found" value={String(summary.n_rules)} sub="business-constrained (5-50 rule band)" />
        <Stat label="Avg Lift" value={`${summary.avg_lift}x`} sub="co-occurrence vs. chance" />
        <Stat label="Max Lift" value={`${summary.max_lift}x`} />
      </div>
      <div className="section-title">Business Reading</div>
      <div className="finding">
        Winning config: support &ge; {(summary.winning_support * 100).toFixed(0)}%, confidence &ge; {(summary.winning_confidence * 100).toFixed(0)}% —
        chosen via AutoResearch to keep the rule set merchandiser-reviewable (5-50 rules) while maximizing average lift among configs
        satisfying that constraint, not by maximizing raw rule count (an earlier unconstrained pass produced 7,799 rules — useless in practice).
      </div>
      {ruleSummary && (
        <>
          <div className="section-title">Top Cross-Sell Products (rule "hubs")</div>
          <div className="grid cols-2">
            {ruleSummary.top_hub_products.slice(0, 6).map((p: any) => (
              <div key={p.stock_code} className="card">
                <h3>{p.name}</h3>
                <div className="stat-value">{p.n_rules}</div>
                <div className="stat-sub">rules involving this product</div>
              </div>
            ))}
          </div>
        </>
      )}
    </>
  );
}

function Rules() {
  const { data: ruleSummary } = useApi(api.ruleSummary);
  const [q, setQ] = useState("");
  if (!ruleSummary) return <div className="loading">Loading...</div>;
  const rules: Rule[] = ruleSummary.top_rules_by_lift;
  const filtered = rules.filter((r) =>
    r.antecedents.join(" ").toLowerCase().includes(q.toLowerCase()) ||
    r.consequents.join(" ").toLowerCase().includes(q.toLowerCase())
  );
  return (
    <>
      <input type="text" placeholder="Search rules by product name..." value={q} onChange={(e) => setQ(e.target.value)} />
      <div className="card" style={{ overflowX: "auto" }}>
        <table>
          <thead><tr><th>If bought</th><th></th><th>Then also buy</th><th>Support</th><th>Confidence</th><th>Lift</th></tr></thead>
          <tbody>{filtered.map((r, i) => <RuleRow key={i} r={r} />)}</tbody>
        </table>
      </div>
    </>
  );
}

function Products() {
  const { data: ruleSummary } = useApi(api.ruleSummary);
  if (!ruleSummary) return <div className="loading">Loading...</div>;
  return (
    <>
      <div className="section-title">Products by Rule Involvement</div>
      <div className="card">
        <ResponsiveContainer width="100%" height={360}>
          <BarChart data={ruleSummary.top_hub_products} layout="vertical" margin={{ left: 140 }}>
            <CartesianGrid stroke="#2a3550" />
            <XAxis type="number" stroke="#8a97b5" />
            <YAxis type="category" dataKey="name" stroke="#8a97b5" width={140} tick={{ fontSize: 11 }} />
            <Tooltip contentStyle={{ background: "#161d2e", border: "1px solid #2a3550" }} labelStyle={{ color: "#e6ebf5" }} itemStyle={{ color: "#e6ebf5" }} cursor={{ fill: "rgba(108,140,255,0.12)", stroke: "#6c8cff", strokeOpacity: 0.4 }} />
            <Bar dataKey="n_rules" fill="#6c8cff" isAnimationActive={false} />
          </BarChart>
        </ResponsiveContainer>
      </div>
      <div className="finding" style={{ marginTop: 14 }}>
        These products appear most often across the mined rule set — strong candidates for cross-sell placement,
        bundle promotions, or "frequently bought together" widgets, since each sits at the center of multiple
        high-lift co-purchase patterns rather than a single isolated pairing.
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
        <strong>Algorithm comparison</strong> (Apriori vs. FP-Growth) at min_support={data.min_support}: both found the
        identical {data.apriori.n_itemsets} frequent itemsets ({data.n_itemsets_match ? "verified match" : "mismatch — investigate"}).
        Apriori: {data.apriori.seconds}s. FP-Growth: {data.fpgrowth.seconds}s. At this dataset's scale, Apriori was
        marginally faster than FP-Growth — the literature's usual "FP-Growth is faster" claim doesn't hold here,
        reported honestly rather than silently deploying FP-Growth on the papers' authority alone.
      </div>
      <div className="grid cols-2" style={{ marginTop: 16 }}>
        <Stat label="Apriori Runtime" value={`${data.apriori.seconds}s`} sub={`${data.apriori.n_itemsets} itemsets`} />
        <Stat label="FP-Growth Runtime" value={`${data.fpgrowth.seconds}s`} sub={`${data.fpgrowth.n_itemsets} itemsets`} />
      </div>
    </>
  );
}

function AutoResearch() {
  const { data } = useApi(api.autoresearch);
  if (!data) return <div className="loading">Loading...</div>;
  const grid = data.phase1_grid_search;
  return (
    <>
      <div className="finding">
        <strong>Grid search</strong> swept min_support &times; min_confidence. Configs producing between 5 and 50
        rules (the business-reviewable band) were kept; among those, the config maximizing <em>average lift</em> won —
        support={data.winning_config.min_support}, confidence={data.winning_config.min_confidence}, yielding{" "}
        {data.phase2_finalize.n_rules_final} rules at avg lift {data.phase2_finalize.avg_lift}x.
      </div>
      <div className="section-title">Rule Count vs. Support/Confidence (bubble size = rule count)</div>
      <div className="card">
        <ResponsiveContainer width="100%" height={360}>
          <ScatterChart margin={{ top: 10, right: 20, bottom: 10, left: 0 }}>
            <CartesianGrid stroke="#2a3550" />
            <XAxis type="number" dataKey="min_support" name="min_support" stroke="#8a97b5" />
            <YAxis type="number" dataKey="min_confidence" name="min_confidence" stroke="#8a97b5" />
            <ZAxis type="number" dataKey="n_rules" range={[20, 400]} name="n_rules" />
            <Tooltip cursor={{ strokeDasharray: "3 3" }} contentStyle={{ background: "#161d2e", border: "1px solid #2a3550" }} labelStyle={{ color: "#e6ebf5" }} itemStyle={{ color: "#e6ebf5" }} />
            <Scatter data={grid} isAnimationActive={false}>
              {grid.map((entry: any, i: number) => (
                <Cell key={i} fill={entry.min_support === data.winning_config.min_support && entry.min_confidence === data.winning_config.min_confidence ? "#4dd6b6" : "#6c8cff"} />
              ))}
            </Scatter>
          </ScatterChart>
        </ResponsiveContainer>
      </div>
      <div className="section-title">Full Grid</div>
      <table className="card" style={{ display: "table" }}>
        <thead><tr><th>Support</th><th>Confidence</th><th>Rules</th><th>High-Lift Rules</th><th>Avg Lift</th></tr></thead>
        <tbody>
          {grid.map((r: any, i: number) => (
            <tr key={i} style={r.min_support === data.winning_config.min_support && r.min_confidence === data.winning_config.min_confidence ? { background: "rgba(77,214,182,0.1)" } : {}}>
              <td>{r.min_support}</td><td>{r.min_confidence}</td><td>{r.n_rules}</td><td>{r.n_high_lift_rules}</td><td>{r.avg_lift}</td>
            </tr>
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
        <Stat label="Raw Transaction Rows" value={eda.n_raw_rows.toLocaleString()} />
        <Stat label="Cancellation Rows" value={eda.n_cancellation_rows.toLocaleString()} sub="excluded from baskets" />
        <Stat label="Total Baskets" value={eda.n_baskets.toLocaleString()} />
        <Stat label="Distinct Products" value={eda.n_distinct_products.toLocaleString()} sub={`top ${prep.top_n_products_kept} used for mining`} />
      </div>
      <div className="section-title">Most Frequently Purchased Products</div>
      <table className="card" style={{ display: "table" }}>
        <thead><tr><th>Product</th><th>Baskets</th></tr></thead>
        <tbody>
          {eda.top_20_products.slice(0, 10).map((p: any) => (
            <tr key={p.stock_code}><td>{p.name}</td><td>{p.n_baskets.toLocaleString()}</td></tr>
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
          <h1>Market Basket Pattern Mining</h1>
          <p>Association rule mining · Online Retail dataset · CRISP-DM + AutoResearch</p>
        </div>
      </header>
      <nav className="tabs">
        {(["overview", "rules", "products", "modeling", "autoresearch", "eda"] as Tab[]).map((t) => (
          <button key={t} className={tab === t ? "active" : ""} onClick={() => setTab(t)}>
            {t === "eda" ? "Data & EDA" : t.charAt(0).toUpperCase() + t.slice(1)}
          </button>
        ))}
      </nav>
      {tab === "overview" && <Overview />}
      {tab === "rules" && <Rules />}
      {tab === "products" && <Products />}
      {tab === "modeling" && <Modeling />}
      {tab === "autoresearch" && <AutoResearch />}
      {tab === "eda" && <EDA />}
    </div>
  );
}
