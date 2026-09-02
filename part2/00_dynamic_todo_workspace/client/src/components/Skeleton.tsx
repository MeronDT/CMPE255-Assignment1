export function TodoListSkeleton() {
  return (
    <ul className="space-y-2">
      {[0, 1, 2, 3].map((i) => (
        <li
          key={i}
          className="flex items-center gap-3 rounded-2xl border border-slate-200 bg-white p-4 dark:border-slate-700 dark:bg-slate-800"
        >
          <div className="h-5 w-5 shrink-0 animate-pulse rounded-full bg-slate-200 dark:bg-slate-700" />
          <div className="flex-1 space-y-2">
            <div
              className="h-3.5 animate-pulse rounded bg-slate-200 dark:bg-slate-700"
              style={{ width: `${55 + i * 8}%` }}
            />
            <div className="h-2.5 w-24 animate-pulse rounded bg-slate-100 dark:bg-slate-700/60" />
          </div>
        </li>
      ))}
    </ul>
  );
}
