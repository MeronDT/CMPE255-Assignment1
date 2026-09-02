const content = {
  none: {
    title: "Nothing on your plate",
    body: "Add your first task above to get started.",
    icon: "📝",
  },
  "no-results": {
    title: "No matching tasks",
    body: "Try a different search term or clear your filters.",
    icon: "🔍",
  },
  "all-done": {
    title: "All caught up!",
    body: "Every task in this view is complete. Nice work.",
    icon: "🎉",
  },
} as const;

export function EmptyState({ kind }: { kind: keyof typeof content }) {
  const { title, body, icon } = content[kind];
  return (
    <div className="flex flex-col items-center justify-center rounded-2xl border border-dashed border-slate-200 bg-white/50 px-6 py-14 text-center dark:border-slate-700 dark:bg-slate-800/30">
      <span className="mb-3 text-4xl">{icon}</span>
      <p className="text-sm font-semibold text-slate-700 dark:text-slate-200">{title}</p>
      <p className="mt-1 text-sm text-slate-400 dark:text-slate-500">{body}</p>
    </div>
  );
}
