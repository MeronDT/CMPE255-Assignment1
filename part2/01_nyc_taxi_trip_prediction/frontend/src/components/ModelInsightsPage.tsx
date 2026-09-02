import { useModelInfo } from "../hooks/useModelInfo";
import { useAutoResearch } from "../hooks/useAutoResearch";
import { StatTile } from "./StatTile";
import { TargetReportCard } from "./TargetReportCard";
import { AutoResearchTargetCard } from "./AutoResearchPanel";

export function ModelInsightsPage() {
  const { data, loading, error } = useModelInfo();
  const { data: autoResearch, loading: autoResearchLoading } = useAutoResearch();

  if (loading) return <div className="py-16 text-center text-sm text-slate-400">Loading model insights…</div>;
  if (error || !data) return <div className="py-16 text-center text-sm text-red-500">{error ?? "No data"}</div>;

  const { eda, data_preparation: prep, model_search } = data;

  return (
    <div className="space-y-6">
      <section>
        <h2 className="mb-3 text-sm font-semibold uppercase tracking-wide text-slate-500">
          CRISP-DM · Data Understanding &amp; Preparation
        </h2>
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
          <StatTile label="Raw Rows" value={eda.n_rows_raw.toLocaleString()} sub="Jan 2024 Yellow Taxi" />
          <StatTile
            label="After Cleaning"
            value={prep.n_rows_after_cleaning.toLocaleString()}
            sub={`${prep.pct_dropped}% dropped`}
          />
          <StatTile label="Train / Test" value={`${prep.n_train.toLocaleString()} / ${prep.n_test.toLocaleString()}`} sub="time-based split" />
          <StatTile label="OD Pairs w/ History" value={prep.n_od_pairs_with_history.toLocaleString()} sub="≥5 trips" />
        </div>

        <div className="mt-4 rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
          <div className="mb-3 text-xs font-semibold uppercase tracking-wide text-slate-400">Data Quality Issues Found</div>
          <table className="w-full text-sm">
            <tbody>
              {eda.data_quality_issues.map((issue) => (
                <tr key={issue.issue} className="border-t border-slate-100 first:border-t-0">
                  <td className="py-1.5 pr-4 text-slate-600">{issue.issue}</td>
                  <td className="py-1.5 text-right font-semibold text-slate-800">{issue.count.toLocaleString()} rows</td>
                </tr>
              ))}
            </tbody>
          </table>
          <div className="mt-3 text-xs text-slate-400">
            Feature columns used ({prep.feature_cols.length}): {prep.feature_cols.join(", ")}
          </div>
        </div>
      </section>

      <section>
        <h2 className="mb-3 text-sm font-semibold uppercase tracking-wide text-slate-500">
          CRISP-DM · Modeling &amp; Evaluation
        </h2>
        <div className="space-y-5">
          {Object.values(model_search.targets).map((report) => (
            <TargetReportCard key={report.target} report={report} />
          ))}
        </div>
      </section>

      <section>
        <h2 className="mb-3 text-sm font-semibold uppercase tracking-wide text-slate-500">
          AutoResearch · 4-Phase Hill-Climbing Search
        </h2>
        {autoResearchLoading || !autoResearch ? (
          <div className="py-8 text-center text-sm text-slate-400">Loading AutoResearch telemetry…</div>
        ) : (
          <div className="space-y-5">
            {Object.values(autoResearch.history.targets).map((t) => (
              <AutoResearchTargetCard key={t.target} data={t} finalize={autoResearch.finalize[t.target]} />
            ))}
          </div>
        )}
      </section>
    </div>
  );
}
