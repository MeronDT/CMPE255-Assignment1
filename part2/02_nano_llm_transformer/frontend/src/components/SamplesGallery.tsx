import type { GenerationSample } from "../types";

export function SamplesGallery({ title, samples }: { title: string; samples: GenerationSample[] }) {
  return (
    <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
      <h3 className="mb-3 text-base font-semibold text-slate-800">{title}</h3>
      <div className="space-y-3">
        {samples.map((s, i) => (
          <div key={i} className="rounded-lg bg-slate-50 p-3 text-sm">
            <div className="mb-1 font-semibold text-indigo-600">{s.prompt}</div>
            <div className="whitespace-pre-wrap text-slate-600">{s.completion}</div>
          </div>
        ))}
      </div>
    </div>
  );
}
