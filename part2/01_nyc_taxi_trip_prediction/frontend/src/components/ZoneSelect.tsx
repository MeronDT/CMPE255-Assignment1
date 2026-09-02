import type { ZoneInfo } from "../types";

interface Props {
  label: string;
  zones: ZoneInfo[];
  value: number | null;
  onChange: (locationId: number | null) => void;
  accent: "green" | "red";
}

export function ZoneSelect({ label, zones, value, onChange, accent }: Props) {
  const byBorough = new Map<string, ZoneInfo[]>();
  for (const z of zones) {
    if (!byBorough.has(z.borough)) byBorough.set(z.borough, []);
    byBorough.get(z.borough)!.push(z);
  }

  return (
    <label className="block">
      <span className="mb-1 flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wide text-slate-500">
        <span className={`h-2 w-2 rounded-full ${accent === "green" ? "bg-emerald-500" : "bg-red-500"}`} />
        {label}
      </span>
      <select
        value={value ?? ""}
        onChange={(e) => onChange(e.target.value ? Number(e.target.value) : null)}
        className="w-full rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm text-slate-800 focus:border-indigo-400 focus:outline-none focus:ring-2 focus:ring-indigo-100"
      >
        <option value="">Select a zone or click the map…</option>
        {[...byBorough.entries()].map(([borough, list]) => (
          <optgroup key={borough} label={borough}>
            {list.map((z) => (
              <option key={z.location_id} value={z.location_id}>
                {z.zone}
              </option>
            ))}
          </optgroup>
        ))}
      </select>
    </label>
  );
}
