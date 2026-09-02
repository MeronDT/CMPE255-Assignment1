import type { PredictResponse } from "../types";

export function ResultCard({ result }: { result: PredictResponse }) {
  return (
    <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
      <div className="mb-4 flex items-center justify-between text-sm text-slate-500">
        <span>
          {result.pickup_zone} <span className="text-slate-300">→</span> {result.dropoff_zone}
        </span>
        <div className="flex gap-1.5">
          {result.is_rush_hour && (
            <span className="rounded-full bg-amber-100 px-2 py-0.5 text-xs font-semibold text-amber-700">Rush hour</span>
          )}
          {result.is_airport_trip && (
            <span className="rounded-full bg-sky-100 px-2 py-0.5 text-xs font-semibold text-sky-700">Airport</span>
          )}
        </div>
      </div>

      <div className="grid grid-cols-2 gap-4">
        <div className="rounded-xl bg-indigo-50 p-4">
          <div className="text-xs font-semibold uppercase tracking-wide text-indigo-500">Est. Duration</div>
          <div className="mt-1 text-3xl font-bold text-indigo-700">{result.predicted_duration_min}<span className="text-base font-medium"> min</span></div>
        </div>
        <div className="rounded-xl bg-emerald-50 p-4">
          <div className="text-xs font-semibold uppercase tracking-wide text-emerald-600">Est. Fare</div>
          <div className="mt-1 text-3xl font-bold text-emerald-700">${result.predicted_fare_amount}</div>
        </div>
      </div>

      <div className="mt-4 flex flex-wrap gap-x-6 gap-y-1 text-xs text-slate-500">
        <span>Distance (haversine): {result.haversine_km} km</span>
        <span>{result.pickup_borough} → {result.dropoff_borough}</span>
      </div>

      {result.historical && (
        <div className="mt-4 border-t border-slate-100 pt-4">
          <div className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-400">
            Historical (Jan 2024, {result.historical.n_trips.toLocaleString()} trips this route)
          </div>
          <div className="grid grid-cols-3 gap-3 text-sm">
            <div>
              <div className="text-slate-400">Avg duration</div>
              <div className="font-semibold text-slate-700">{result.historical.avg_duration_min.toFixed(1)} min</div>
            </div>
            <div>
              <div className="text-slate-400">Avg fare</div>
              <div className="font-semibold text-slate-700">${result.historical.avg_fare_amount.toFixed(2)}</div>
            </div>
            <div>
              <div className="text-slate-400">Avg distance</div>
              <div className="font-semibold text-slate-700">{result.historical.avg_trip_distance_mi.toFixed(1)} mi</div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
