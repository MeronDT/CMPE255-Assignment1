import { useState } from "react";
import { useZones } from "../hooks/useZones";
import { predictTrip } from "../api/client";
import { TripMap } from "./TripMap";
import { ZoneSelect } from "./ZoneSelect";
import { ResultCard } from "./ResultCard";
import type { PredictResponse } from "../types";

function defaultDatetimeLocal(): string {
  const now = new Date(Date.now() - new Date().getTimezoneOffset() * 60000);
  return now.toISOString().slice(0, 16);
}

export function EstimatorPage() {
  const { zones, loading } = useZones();
  const [pickupId, setPickupId] = useState<number | null>(null);
  const [dropoffId, setDropoffId] = useState<number | null>(null);
  const [datetime, setDatetime] = useState(defaultDatetimeLocal());
  const [passengers, setPassengers] = useState(1);
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  const handleMapSelect = (locationId: number) => {
    if (pickupId === null) setPickupId(locationId);
    else if (dropoffId === null && locationId !== pickupId) setDropoffId(locationId);
    else {
      setPickupId(locationId);
      setDropoffId(null);
      setResult(null);
    }
  };

  const canEstimate = pickupId !== null && dropoffId !== null;

  const handleEstimate = async () => {
    if (!canEstimate) return;
    setSubmitting(true);
    setError(null);
    try {
      const res = await predictTrip({
        pickup_location_id: pickupId!,
        dropoff_location_id: dropoffId!,
        pickup_datetime: datetime,
        passenger_count: passengers,
      });
      setResult(res);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Estimate failed");
    } finally {
      setSubmitting(false);
    }
  };

  const handleClear = () => {
    setPickupId(null);
    setDropoffId(null);
    setResult(null);
    setError(null);
  };

  return (
    <div className="grid grid-cols-1 gap-5 lg:grid-cols-[1fr_380px]">
      <div className="h-[520px] overflow-hidden rounded-2xl border border-slate-200 shadow-sm lg:h-[640px]">
        {loading ? (
          <div className="flex h-full items-center justify-center text-sm text-slate-400">Loading map…</div>
        ) : (
          <TripMap zones={zones} pickupId={pickupId} dropoffId={dropoffId} onSelectZone={handleMapSelect} />
        )}
      </div>

      <div className="flex flex-col gap-4">
        <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
          <h2 className="mb-3 text-sm font-semibold text-slate-700">Plan your trip</h2>
          <p className="mb-3 text-xs text-slate-400">Click two zones on the map (pickup, then dropoff), or pick them below.</p>
          <div className="space-y-3">
            <ZoneSelect label="Pickup" zones={zones} value={pickupId} onChange={setPickupId} accent="green" />
            <ZoneSelect label="Dropoff" zones={zones} value={dropoffId} onChange={setDropoffId} accent="red" />

            <div className="grid grid-cols-2 gap-3">
              <label className="block">
                <span className="mb-1 block text-xs font-semibold uppercase tracking-wide text-slate-500">Pickup time</span>
                <input
                  type="datetime-local"
                  value={datetime}
                  onChange={(e) => setDatetime(e.target.value)}
                  className="w-full rounded-lg border border-slate-200 bg-white px-2 py-2 text-sm text-slate-800 focus:border-indigo-400 focus:outline-none"
                />
              </label>
              <label className="block">
                <span className="mb-1 block text-xs font-semibold uppercase tracking-wide text-slate-500">Passengers</span>
                <input
                  type="number"
                  min={1}
                  max={6}
                  value={passengers}
                  onChange={(e) => setPassengers(Number(e.target.value))}
                  className="w-full rounded-lg border border-slate-200 bg-white px-2 py-2 text-sm text-slate-800 focus:border-indigo-400 focus:outline-none"
                />
              </label>
            </div>

            <div className="flex gap-2 pt-1">
              <button
                onClick={handleEstimate}
                disabled={!canEstimate || submitting}
                className="flex-1 rounded-lg bg-indigo-600 px-4 py-2 text-sm font-semibold text-white transition-colors hover:bg-indigo-500 disabled:cursor-not-allowed disabled:bg-slate-200 disabled:text-slate-400"
              >
                {submitting ? "Estimating…" : "Get Estimate"}
              </button>
              <button
                onClick={handleClear}
                className="rounded-lg border border-slate-200 px-3 py-2 text-sm font-medium text-slate-500 hover:bg-slate-50"
              >
                Clear
              </button>
            </div>
          </div>
        </div>

        {error && (
          <div className="rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">{error}</div>
        )}

        {result && <ResultCard result={result} />}
      </div>
    </div>
  );
}
