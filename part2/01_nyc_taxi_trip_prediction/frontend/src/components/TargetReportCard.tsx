import {
  Bar,
  BarChart,
  CartesianGrid,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { TargetReport } from "../types";
import { StatTile } from "./StatTile";

const TARGET_LABELS: Record<string, { title: string; unit: string; color: string }> = {
  duration_min: { title: "Trip Duration Model", unit: "min", color: "#6366f1" },
  fare_amount: { title: "Fare Amount Model", unit: "$", color: "#22c55e" },
};

export function TargetReportCard({ report }: { report: TargetReport }) {
  const meta = TARGET_LABELS[report.target] ?? { title: report.target, unit: "", color: "#6366f1" };
  const improvementPct =
    ((report.baseline_val_metrics.rmse - report.best_val_rmse) / report.baseline_val_metrics.rmse) * 100;

  const searchData = report.search_trials.map((t) => ({
    trial: `#${t.trial}`,
    rmse: Number(t.val_rmse.toFixed(3)),
    leaves: t.params.num_leaves,
  }));

  const importanceData = [...report.feature_importance].slice(0, 8).reverse();

  return (
    <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
      <div className="mb-4 flex items-center justify-between">
        <h3 className="text-base font-semibold text-slate-800">{meta.title}</h3>
        <span className="rounded-full bg-slate-100 px-2.5 py-1 text-xs font-semibold text-slate-500">
          {improvementPct.toFixed(0)}% better than baseline
        </span>
      </div>

      <div className="mb-5 grid grid-cols-3 gap-3">
        <StatTile label="Test RMSE" value={`${report.test_metrics.rmse.toFixed(2)} ${meta.unit}`} />
        <StatTile label="Test MAE" value={`${report.test_metrics.mae.toFixed(2)} ${meta.unit}`} />
        <StatTile label="Test R²" value={report.test_metrics.r2.toFixed(3)} />
      </div>

      <div className="grid grid-cols-1 gap-5 md:grid-cols-2">
        <div>
          <div className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-400">
            Hyperparameter Search ("Hill Climbing")
          </div>
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={searchData} margin={{ top: 5, right: 10, left: -10, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
              <XAxis dataKey="trial" tick={{ fontSize: 11 }} />
              <YAxis tick={{ fontSize: 11 }} domain={[0, Math.ceil(report.baseline_val_metrics.rmse * 1.15)]} />
              <Tooltip />
              <ReferenceLine
                y={report.baseline_val_metrics.rmse}
                stroke="#ef4444"
                strokeDasharray="4 4"
                label={{ value: "baseline", fontSize: 10, fill: "#ef4444" }}
              />
              <Line type="monotone" dataKey="rmse" stroke={meta.color} strokeWidth={2} dot={{ r: 3 }} />
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div>
          <div className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-400">Feature Importance</div>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={importanceData} layout="vertical" margin={{ top: 5, right: 10, left: 10, bottom: 0 }}>
              <XAxis type="number" tick={{ fontSize: 10 }} />
              <YAxis type="category" dataKey="feature" tick={{ fontSize: 10 }} width={90} />
              <Tooltip />
              <Bar dataKey="importance" fill={meta.color} radius={[0, 4, 4, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      <details className="mt-4 text-xs text-slate-500">
        <summary className="cursor-pointer font-semibold text-slate-400">Best hyperparameters</summary>
        <pre className="mt-2 overflow-x-auto rounded-lg bg-slate-50 p-3">{JSON.stringify(report.best_params, null, 2)}</pre>
      </details>
    </div>
  );
}
