import { Bar, BarChart, CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { AutoResearchFinalizeEntry, AutoResearchTarget } from "../types";

const TARGET_META: Record<string, { title: string; unit: string; color: string }> = {
  duration_min: { title: "Trip Duration", unit: "min", color: "#6366f1" },
  fare_amount: { title: "Fare Amount", unit: "$", color: "#22c55e" },
};

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div>
      <div className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-400">{title}</div>
      {children}
    </div>
  );
}

export function AutoResearchTargetCard({
  data,
  finalize,
}: {
  data: AutoResearchTarget;
  finalize: AutoResearchFinalizeEntry;
}) {
  const meta = TARGET_META[data.target] ?? { title: data.target, unit: "", color: "#6366f1" };

  const tournamentData = data.phase1_tournament.results.map((r) => ({ name: r.backbone, rmse: Number(r.rmse.toFixed(3)) }));
  const featureData = data.phase2_feature_transform.results.map((r) => ({ name: r.variant.replace(/_/g, " "), rmse: Number(r.rmse.toFixed(3)) }));
  const hillClimbData = data.phase3_hill_climbing.path.map((p) => ({ iter: `#${p.iteration}`, rmse: Number(p.rmse.toFixed(3)) }));
  const blendData = data.phase4_blending.results.map((b) => ({ weight: b.weight_a.toFixed(1), rmse: Number(b.rmse.toFixed(3)) }));

  return (
    <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
      <div className="mb-4 flex flex-wrap items-center justify-between gap-2">
        <h3 className="text-base font-semibold text-slate-800">{meta.title} — AutoResearch</h3>
        <div className="flex gap-2">
          <span className="rounded-full bg-indigo-50 px-2.5 py-1 text-xs font-semibold text-indigo-600">
            Winner: {data.phase1_tournament.winning_backbone} + {data.feature_transform_added.replace(/_/g, " ")}
          </span>
          <span
            className={`rounded-full px-2.5 py-1 text-xs font-semibold ${finalize.redeployed ? "bg-emerald-50 text-emerald-600" : "bg-slate-100 text-slate-500"}`}
          >
            {finalize.redeployed ? "Redeployed to production" : "Kept existing production model"}
          </span>
        </div>
      </div>

      <div className="grid grid-cols-1 gap-5 md:grid-cols-2">
        <Section title="Phase 1 · Multi-Backbone Tournament">
          <ResponsiveContainer width="100%" height={160}>
            <BarChart data={tournamentData} margin={{ top: 5, right: 10, left: -10, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
              <XAxis dataKey="name" tick={{ fontSize: 10 }} />
              <YAxis tick={{ fontSize: 10 }} />
              <Tooltip />
              <Bar dataKey="rmse" fill={meta.color} radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </Section>

        <Section title="Phase 2 · Feature-Transform Search">
          <ResponsiveContainer width="100%" height={160}>
            <BarChart data={featureData} margin={{ top: 5, right: 10, left: -10, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
              <XAxis dataKey="name" tick={{ fontSize: 9 }} interval={0} angle={-15} textAnchor="end" height={40} />
              <YAxis tick={{ fontSize: 10 }} domain={["dataMin - 0.01", "dataMax + 0.01"]} />
              <Tooltip />
              <Bar dataKey="rmse" fill="#f59e0b" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </Section>

        <Section title="Phase 3 · Hyperparameter Hill-Climbing (greedy local search)">
          <ResponsiveContainer width="100%" height={160}>
            <LineChart data={hillClimbData} margin={{ top: 5, right: 10, left: -10, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
              <XAxis dataKey="iter" tick={{ fontSize: 10 }} />
              <YAxis tick={{ fontSize: 10 }} domain={["dataMin - 0.02", "dataMax + 0.02"]} />
              <Tooltip />
              <Line type="stepAfter" dataKey="rmse" stroke={meta.color} strokeWidth={2} dot={{ r: 3 }} />
            </LineChart>
          </ResponsiveContainer>
          <p className="mt-1 text-xs text-slate-400">
            {data.phase3_hill_climbing.path.length - 1} accepted moves from seed config to local optimum ({data.search_sample_size.toLocaleString()}-row search sample).
          </p>
        </Section>

        <Section title="Phase 4 · Blending (hill-climbed × seed model)">
          <ResponsiveContainer width="100%" height={160}>
            <LineChart data={blendData} margin={{ top: 5, right: 10, left: -10, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
              <XAxis dataKey="weight" tick={{ fontSize: 10 }} label={{ value: "weight on hill-climbed model", fontSize: 9, position: "insideBottom", offset: -2 }} />
              <YAxis tick={{ fontSize: 10 }} domain={["auto", "auto"]} />
              <Tooltip />
              <Line type="monotone" dataKey="rmse" stroke="#ec4899" strokeWidth={2} dot={{ r: 3 }} />
            </LineChart>
          </ResponsiveContainer>
        </Section>
      </div>

      <div className="mt-4 grid grid-cols-1 gap-3 border-t border-slate-100 pt-4 sm:grid-cols-2">
        <div className="rounded-lg bg-slate-50 p-3 text-xs">
          <div className="font-semibold text-slate-500">Full-data finalization check</div>
          <div className="mt-1 text-slate-600">
            Production test RMSE: <strong>{finalize.previous_test_rmse.toFixed(3)}</strong> {meta.unit} · AutoResearch config retrained on full data:{" "}
            <strong>{finalize.full_data_retrain_test_metrics.rmse.toFixed(3)}</strong> {meta.unit}
          </div>
          <div className="mt-1 text-slate-400">
            {finalize.redeployed
              ? "Full-data retrain beat production — redeployed."
              : "Subsample-tuned config didn't beat the model already tuned on the full training set — a reminder that search-sample tuning can overfit to the sample. Production model kept as-is."}
          </div>
        </div>
        <div className="rounded-lg bg-slate-50 p-3 text-xs">
          <div className="font-semibold text-slate-500">Literature benchmark</div>
          <div className="mt-1 text-slate-600">{data.paper_benchmark.source}</div>
          <div className="mt-1 text-slate-400">{data.paper_benchmark.note}</div>
          {data.paper_benchmark.r2 !== null && (
            <div className="mt-1 font-semibold text-slate-600">
              Paper R²: {data.paper_benchmark.r2} · Ours: {finalize.full_data_retrain_test_metrics.r2.toFixed(3)}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
