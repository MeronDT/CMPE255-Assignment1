import { Bar, BarChart, CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { useAutoResearch } from "../hooks/useAutoResearch";
import { StatTile } from "./StatTile";

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div>
      <div className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-400">{title}</div>
      {children}
    </div>
  );
}

export function AutoResearchPage() {
  const { data, loading } = useAutoResearch();

  if (loading) return <div className="py-16 text-center text-sm text-slate-400">Loading AutoResearch telemetry…</div>;
  if (!data?.history) return <div className="py-16 text-center text-sm text-slate-400">No AutoResearch run yet.</div>;

  const h = data.history;
  const archData = h.phase1_architecture_tournament.results.map((r) => ({ name: r.variant.replace(/_/g, " "), ppl: Number(r.val_perplexity.toFixed(2)) }));
  const shapeData = h.phase2_shape_search.results.map((r) => ({ name: r.shape, ppl: Number(r.val_perplexity.toFixed(2)), params: r.n_params }));
  const hillClimbData = h.phase3_hill_climbing.path.map((p) => ({ iter: `#${p.iteration}`, val_loss: Number(p.val_loss.toFixed(4)) }));

  return (
    <div className="space-y-6">
      <section>
        <h2 className="mb-3 text-sm font-semibold uppercase tracking-wide text-slate-500">AutoResearch · 4-Phase Search</h2>
        <p className="mb-4 text-xs text-slate-400">
          Proxy budget: {h.proxy_budget.iters} iters, batch {h.proxy_budget.batch_size}, context {h.proxy_budget.context_length} — every
          trial below is a short run, not the full training budget, so the whole search finishes in minutes.
        </p>

        <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
          <Section title="Phase 1 · Architecture-Primitive Tournament (each vs. the NanoLlama default)">
            <ResponsiveContainer width="100%" height={180}>
              <BarChart data={archData} margin={{ top: 5, right: 10, left: -10, bottom: 40 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                <XAxis dataKey="name" tick={{ fontSize: 9 }} interval={0} angle={-20} textAnchor="end" height={60} />
                <YAxis tick={{ fontSize: 10 }} label={{ value: "val perplexity", angle: -90, fontSize: 9, position: "insideLeft" }} />
                <Tooltip />
                <Bar dataKey="ppl" fill="#6366f1" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
            <p className="mt-1 text-xs text-slate-400">
              Winner: <strong>{h.phase1_architecture_tournament.winner.replace(/_/g, " ")}</strong>.{" "}
              {h.phase1_architecture_tournament.winner.startsWith("nanollama")
                ? "The default (RoPE + RMSNorm + SwiGLU) beat every single-primitive swap at this scale."
                : "A single-primitive swap nominally beat the default here — a single 250-step run at one seed isn't enough signal to overturn literature-established results (see RESEARCH_REPORT.md §4.3); the deployed model keeps the default primitives."}
            </p>
          </Section>
        </div>

        <div className="mt-5 rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
          <Section title="Phase 2 · Width vs. Depth Shape Search">
            <ResponsiveContainer width="100%" height={180}>
              <BarChart data={shapeData} margin={{ top: 5, right: 10, left: -10, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                <XAxis dataKey="name" tick={{ fontSize: 10 }} />
                <YAxis tick={{ fontSize: 10 }} />
                <Tooltip />
                <Bar dataKey="ppl" fill="#f59e0b" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
            <p className="mt-1 text-xs text-slate-400">
              Winner: <strong>{h.phase2_shape_search.winner}</strong> ({JSON.stringify(h.phase2_shape_search.winner_dims)})
            </p>
          </Section>
        </div>

        <div className="mt-5 rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
          <Section title="Phase 3 · Hyperparameter Hill-Climbing (greedy local search)">
            <ResponsiveContainer width="100%" height={180}>
              <LineChart data={hillClimbData} margin={{ top: 5, right: 10, left: -10, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                <XAxis dataKey="iter" tick={{ fontSize: 10 }} />
                <YAxis tick={{ fontSize: 10 }} domain={["auto", "auto"]} />
                <Tooltip />
                <Line type="stepAfter" dataKey="val_loss" stroke="#ec4899" strokeWidth={2} dot={{ r: 3 }} />
              </LineChart>
            </ResponsiveContainer>
            <p className="mt-1 text-xs text-slate-400">
              {h.phase3_hill_climbing.path.length - 1} accepted moves from seed {JSON.stringify(h.phase3_hill_climbing.seed_params)} to{" "}
              {JSON.stringify(h.phase3_hill_climbing.best_params)}.
            </p>
          </Section>
        </div>

        <div className="mt-5 grid grid-cols-1 gap-3 sm:grid-cols-3">
          <StatTile label="Model A (hill-climbed)" value={h.phase4_model_soup.model_a_val_loss.toFixed(4)} sub="val loss" />
          <StatTile label="Model B (seed config)" value={h.phase4_model_soup.model_b_val_loss.toFixed(4)} sub="val loss" />
          <StatTile
            label="Soup (avg weights)"
            value={h.phase4_model_soup.soup_val_loss.toFixed(4)}
            sub={h.phase4_model_soup.soup_beats_both ? "beats both ingredients" : "did not beat both ingredients"}
          />
        </div>

        <div className="mt-5 rounded-2xl border border-indigo-200 bg-indigo-50 p-4 text-sm text-indigo-900">
          <strong>Final recommended config:</strong> {JSON.stringify(h.final_recommended_config)}
        </div>
      </section>
    </div>
  );
}
