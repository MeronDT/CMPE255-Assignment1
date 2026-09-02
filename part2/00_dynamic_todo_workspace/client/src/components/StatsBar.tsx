import type { Stats } from "../types";

export function StatsBar({ stats }: { stats: Stats }) {
  const pct = stats.total === 0 ? 0 : Math.round((stats.completed / stats.total) * 100);

  return (
    <div className="rounded-2xl border border-slate-200 bg-white/70 p-4 shadow-sm backdrop-blur-sm dark:border-slate-700 dark:bg-slate-800/50">
      <div className="flex items-center justify-between text-sm">
        <span className="font-medium text-slate-600 dark:text-slate-300">
          {stats.completed} of {stats.total} tasks complete
        </span>
        <div className="flex items-center gap-3 text-xs text-slate-500 dark:text-slate-400">
          {stats.overdue > 0 && (
            <span className="flex items-center gap-1 font-semibold text-red-600 dark:text-red-400">
              <span className="h-1.5 w-1.5 rounded-full bg-red-500" />
              {stats.overdue} overdue
            </span>
          )}
          <span>{pct}%</span>
        </div>
      </div>
      <div className="mt-2 h-2 w-full overflow-hidden rounded-full bg-slate-100 dark:bg-slate-700">
        <div
          className="h-full rounded-full bg-gradient-to-r from-indigo-500 to-violet-500 transition-all duration-500 ease-out"
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
}
