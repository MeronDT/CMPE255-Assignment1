import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { TrainHistory } from "../types";
import { StatTile } from "./StatTile";

export function TrainingCurveCard({ title, history, color }: { title: string; history: TrainHistory; color: string }) {
  const data = history.steps.map((s) => ({
    iter: s.iter,
    train_loss: Number(s.train_loss.toFixed(3)),
    val_loss: Number(s.val_loss.toFixed(3)),
    val_ppl: Number(s.val_perplexity.toFixed(2)),
    lr: s.lr,
  }));
  const final = history.steps[history.steps.length - 1];

  return (
    <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
      <div className="mb-4 flex items-center justify-between">
        <h3 className="text-base font-semibold text-slate-800">{title}</h3>
        <span className="text-xs text-slate-400">
          {history.n_params ? `${(history.n_params / 1e6).toFixed(1)}M params · ` : ""}
          {history.total_train_seconds.toFixed(0)}s on {history.device}
        </span>
      </div>

      <div className="mb-4 grid grid-cols-3 gap-3">
        <StatTile label="Final Train Loss" value={final.train_loss.toFixed(3)} />
        <StatTile label="Final Val Loss" value={final.val_loss.toFixed(3)} />
        <StatTile label="Val Perplexity" value={final.val_perplexity.toFixed(2)} />
      </div>

      <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
        <div>
          <div className="mb-1 text-xs font-semibold uppercase tracking-wide text-slate-400">Loss</div>
          <ResponsiveContainer width="100%" height={170}>
            <LineChart data={data} margin={{ top: 5, right: 10, left: -10, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
              <XAxis dataKey="iter" tick={{ fontSize: 10 }} />
              <YAxis tick={{ fontSize: 10 }} />
              <Tooltip />
              <Line type="monotone" dataKey="train_loss" stroke={color} strokeWidth={2} dot={false} name="train" />
              <Line type="monotone" dataKey="val_loss" stroke="#ef4444" strokeWidth={2} strokeDasharray="4 4" dot={false} name="val" />
            </LineChart>
          </ResponsiveContainer>
        </div>
        <div>
          <div className="mb-1 text-xs font-semibold uppercase tracking-wide text-slate-400">Validation Perplexity (log scale)</div>
          <ResponsiveContainer width="100%" height={170}>
            <LineChart data={data} margin={{ top: 5, right: 10, left: -10, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
              <XAxis dataKey="iter" tick={{ fontSize: 10 }} />
              <YAxis tick={{ fontSize: 10 }} scale="log" domain={["auto", "auto"]} allowDataOverflow />
              <Tooltip />
              <Line type="monotone" dataKey="val_ppl" stroke={color} strokeWidth={2} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
        <div>
          <div className="mb-1 text-xs font-semibold uppercase tracking-wide text-slate-400">LR Schedule</div>
          <ResponsiveContainer width="100%" height={170}>
            <LineChart data={data} margin={{ top: 5, right: 10, left: -10, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
              <XAxis dataKey="iter" tick={{ fontSize: 10 }} />
              <YAxis tick={{ fontSize: 10 }} />
              <Tooltip />
              <Line type="monotone" dataKey="lr" stroke="#f59e0b" strokeWidth={2} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}
