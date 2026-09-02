import { useModelInfo } from "../hooks/useModelInfo";
import { useSystemInfo } from "../hooks/useSystemInfo";
import { StatTile } from "./StatTile";
import { TrainingCurveCard } from "./TrainingCurveCard";
import { SamplesGallery } from "./SamplesGallery";

export function ModelInsightsPage() {
  const { data, loading, error } = useModelInfo();
  const system = useSystemInfo();

  if (loading) return <div className="py-16 text-center text-sm text-slate-400">Loading model insights…</div>;
  if (error || !data) return <div className="py-16 text-center text-sm text-red-500">{error ?? "No data"}</div>;

  const { eda, data_preparation: prep, train_history_base, train_history_sft, evaluation } = data;

  return (
    <div className="space-y-6">
      <section>
        <h2 className="mb-3 text-sm font-semibold uppercase tracking-wide text-slate-500">System &amp; Hardware</h2>
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
          <StatTile label="Device" value={system?.gpu_name ?? system?.device ?? "…"} />
          <StatTile
            label="VRAM"
            value={system?.gpu_total_memory_mb ? `${(system.gpu_total_memory_mb / 1000).toFixed(1)} GB` : "—"}
            sub={system?.gpu_allocated_memory_mb ? `${system.gpu_allocated_memory_mb.toFixed(0)}MB allocated now` : undefined}
          />
          <StatTile label="PyTorch" value={system?.torch_version ?? "…"} sub={system?.cuda_version ? `CUDA ${system.cuda_version}` : undefined} />
          <StatTile label="bf16 support" value={system?.bf16_supported ? "Yes" : "No"} />
        </div>
      </section>

      <section>
        <h2 className="mb-3 text-sm font-semibold uppercase tracking-wide text-slate-500">
          CRISP-DM · Data Understanding &amp; Preparation
        </h2>
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
          <StatTile label="TinyStories" value={eda.tinystories.n_stories.toLocaleString()} sub={`avg ${eda.tinystories.avg_words_per_story.toFixed(0)} words/story`} />
          <StatTile label="Alpaca (used subset)" value={prep.sft.n_train_examples.toLocaleString()} sub={`of ${prep.sft.n_available_total.toLocaleString()} available`} />
          <StatTile label="Vocab Size" value={prep.vocab_size.toLocaleString()} sub="byte-level BPE" />
          <StatTile label="Pretrain Tokens" value={`${(prep.pretrain.n_train_tokens / 1e6).toFixed(1)}M`} sub={`context length ${prep.context_length}`} />
        </div>
      </section>

      <section>
        <h2 className="mb-3 text-sm font-semibold uppercase tracking-wide text-slate-500">CRISP-DM · Modeling — Base Pretraining</h2>
        <TrainingCurveCard title="Base Model (TinyStories)" history={train_history_base} color="#6366f1" />
      </section>

      <section>
        <h2 className="mb-3 text-sm font-semibold uppercase tracking-wide text-slate-500">CRISP-DM · Modeling — SFT</h2>
        <TrainingCurveCard title="SFT Model (Alpaca instructions)" history={train_history_sft} color="#22c55e" />
      </section>

      <section>
        <h2 className="mb-3 text-sm font-semibold uppercase tracking-wide text-slate-500">CRISP-DM · Evaluation — Qualitative Samples</h2>
        <div className="grid grid-cols-1 gap-5 md:grid-cols-2">
          <SamplesGallery title="Base model (free continuation)" samples={evaluation.base.samples} />
          <SamplesGallery title="SFT model (instruction-following)" samples={evaluation.sft.samples} />
        </div>
      </section>
    </div>
  );
}
