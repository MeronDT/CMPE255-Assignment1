import { useEffect, useState } from "react";
import {
  ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, CartesianGrid, Cell,
  RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
} from "recharts";
import { api } from "./api";

type Tab = "overview" | "scorecard" | "methodology";

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

function scoreColor(score: number | null) {
  if (score === null) return "#8a97b5";
  if (score >= 4.5) return "#4dd6b6";
  if (score >= 3.5) return "#6c8cff";
  if (score >= 2.5) return "#e6a24d";
  return "#e65c5c";
}

function Overview() {
  const { data, error } = useApi(api.audit);
  if (error) return <div className="error">Failed to load: {error}</div>;
  if (!data) return <div className="loading">Loading...</div>;

  const radarData = Object.entries(data.dimension_averages).map(([dim, val]) => ({
    dimension: dim.replace(/_/g, " "),
    score: val,
  }));

  return (
    <>
      <h2>Enterprise Data Science Audit Platform</h2>
      <p className="lede">An evidence-backed audit of all {data.n_projects_audited} completed projects in this repository — every score derived from an actual filesystem/git check, not asserted from memory.</p>
      <div className="grid cols-3">
        <Stat label="Projects Audited" value={String(data.n_projects_audited)} />
        <Stat label="Average Overall Score" value={`${data.avg_overall_score}/5`} />
        <Stat label="Flagged Issues" value={data.flagged_issues[0] === "None -- every audited project passed all 5 dimensions cleanly." ? "0" : String(data.flagged_issues.length)} />
      </div>

      <div className="section-title">Repo-Wide Score by Dimension</div>
      <div className="card">
        <ResponsiveContainer width="100%" height={340}>
          <RadarChart data={radarData}>
            <PolarGrid stroke="#2a3550" />
            <PolarAngleAxis dataKey="dimension" tick={{ fill: "#8a97b5", fontSize: 11 }} />
            <PolarRadiusAxis domain={[0, 5]} tick={{ fill: "#8a97b5" }} />
            <Radar dataKey="score" stroke="#6c8cff" fill="#6c8cff" fillOpacity={0.35} isAnimationActive={false} />
            <Tooltip contentStyle={{ background: "#161d2e", border: "1px solid #2a3550" }} labelStyle={{ color: "#e6ebf5" }} itemStyle={{ color: "#e6ebf5" }} cursor={{ fill: "rgba(108,140,255,0.12)", stroke: "#6c8cff", strokeOpacity: 0.4 }} />
          </RadarChart>
        </ResponsiveContainer>
      </div>

      <div className="section-title">Flagged Issues</div>
      <div className="finding">
        {data.flagged_issues.map((issue: string, i: number) => <div key={i}>{issue}</div>)}
      </div>

      <div className="section-title">Root README Completeness Check</div>
      <div className="finding" style={{ borderColor: data.root_readme_check.ok ? undefined : "#e65c5c" }}>
        {data.root_readme_check.ok
          ? "✓ Every project directory is linked from the repo's root README.md."
          : `✗ Missing from root README.md: ${data.root_readme_check.missing.join(", ")}`}
        <div style={{ marginTop: 6, color: "#8a97b5" }}>{data.root_readme_check.note}</div>
      </div>

      <div className="section-title">Highest / Lowest Scoring</div>
      <div className="grid cols-2">
        <Stat label="Highest" value={data.highest_scoring.name} sub={`${data.highest_scoring.score}/5`} />
        <Stat label="Lowest" value={data.lowest_scoring.name} sub={`${data.lowest_scoring.score}/5 (still passing — see methodology)`} />
      </div>
    </>
  );
}

function Scorecard() {
  const { data } = useApi(api.audit);
  if (!data) return <div className="loading">Loading...</div>;
  const chartData = data.projects.map((p: any) => ({ name: p.name.length > 20 ? p.name.slice(0, 20) + "…" : p.name, score: p.overall_score }));
  return (
    <>
      <h2>Project Scorecard</h2>
      <div className="card">
        <ResponsiveContainer width="100%" height={320}>
          <BarChart data={chartData} layout="vertical" margin={{ left: 170 }}>
            <CartesianGrid stroke="#2a3550" />
            <XAxis type="number" domain={[0, 5]} stroke="#8a97b5" />
            <YAxis type="category" dataKey="name" stroke="#8a97b5" width={170} tick={{ fontSize: 11 }} />
            <Tooltip contentStyle={{ background: "#161d2e", border: "1px solid #2a3550" }} labelStyle={{ color: "#e6ebf5" }} itemStyle={{ color: "#e6ebf5" }} cursor={{ fill: "rgba(108,140,255,0.12)", stroke: "#6c8cff", strokeOpacity: 0.4 }} />
            <Bar dataKey="score" isAnimationActive={false}>
              {chartData.map((e: any, i: number) => <Cell key={i} fill={scoreColor(e.score)} />)}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      {data.projects.map((p: any) => (
        <div className="card" key={p.directory}>
          <h4>{p.name} <span className="badge" style={{ color: scoreColor(p.overall_score), marginLeft: 8 }}>{p.overall_score}/5</span></h4>
          <table className="mini" style={{ width: "100%" }}>
            <thead><tr><th>Dimension</th><th>Score</th></tr></thead>
            <tbody>
              {Object.entries(p.scores).map(([dim, score]: [string, any]) => (
                <tr key={dim}><td style={{ textAlign: "left" }}>{dim.replace(/_/g, " ")}</td><td style={{ color: scoreColor(score) }}>{score === null ? "N/A" : `${score}/5`}</td></tr>
              ))}
            </tbody>
          </table>
          <ul style={{ fontSize: 13, color: "#8a97b5", marginTop: 10 }}>
            {p.findings.map((f: string, i: number) => <li key={i}>{f}</li>)}
          </ul>
        </div>
      ))}
    </>
  );
}

function Methodology() {
  return (
    <>
      <h2>Audit Methodology</h2>
      <div className="card">
        <h4>What This Audit Actually Checks</h4>
        <p>Every score is derived from a concrete, machine-checked signal — not a subjective read of each project. Five dimensions, each 0-5:</p>
        <ol>
          <li><strong>CRISP-DM Completeness</strong> — real file-existence checks for a business-understanding doc, scripts, backend, frontend, and EDA/results JSON. Project types that were never meant to follow CRISP-DM (a full-stack todo app; a client-side educational site) are scored against what actually applies to them, not penalized for missing phases outside their own stated scope.</li>
          <li><strong>Documentation Quality</strong> — a byte-size floor on DESIGN_DOC.md/README.md/RESEARCH_REPORT.md, so an empty stub file can't pass as "documentation present."</li>
          <li><strong>Verification Evidence</strong> — actual PNG count in each project's docs/screenshots/ directory.</li>
          <li><strong>Honest-Reporting Discipline</strong> — a real text scan for language patterns (e.g. "honest", "limitation", "deliberately", "caught and fixed") across every markdown doc in each project, not just the three root files.</li>
          <li><strong>Repository Hygiene</strong> — actual <code>git check-ignore</code> calls against any large raw-data file found on disk, confirming it's truly excluded rather than assuming the .gitignore is correct.</li>
        </ol>
      </div>
      <div className="card">
        <h4>Known Limitations of This Audit Tool Itself</h4>
        <p>
          Two real bugs were caught and fixed while building this audit, worth disclosing rather than
          hiding behind a clean final scorecard: an early version penalized Project 00 (a full-stack app,
          not a CRISP-DM project by design) and Project 08 (a client-side educational site with a
          different folder structure than every other project) for missing phases that were never
          applicable to them — fixed by adding type-aware scoring rather than applying one rigid rubric
          to every project regardless of its actual scope. The honest-reporting text scan is a keyword
          heuristic, not true language understanding — it can under-count genuine honest disclosures that
          happen to use different phrasing than the marker list expects (this happened once, for Project
          08's GitHub-Pages-privacy caveat, before the marker list was broadened) and can, in principle,
          be gamed by a project that sprinkles the marker words without genuine substance behind them.
          Treat these scores as a useful, evidence-backed signal — not an infallible ground truth.
        </p>
      </div>
    </>
  );
}

const TABS: { id: Tab; label: string }[] = [
  { id: "overview", label: "Overview" },
  { id: "scorecard", label: "Project Scorecard" },
  { id: "methodology", label: "Methodology" },
];

export default function App() {
  const [tab, setTab] = useState<Tab>("overview");
  return (
    <div className="app">
      <header className="top">
        <div>
          <h1>Enterprise Data Science Audit Platform</h1>
          <p>Evidence-backed audit of every completed project in this repository</p>
        </div>
      </header>
      <nav className="tabs">
        {TABS.map((t) => (
          <button key={t.id} className={tab === t.id ? "active" : ""} onClick={() => setTab(t.id)}>{t.label}</button>
        ))}
      </nav>
      {tab === "overview" && <Overview />}
      {tab === "scorecard" && <Scorecard />}
      {tab === "methodology" && <Methodology />}
    </div>
  );
}
