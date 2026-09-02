export function TagPill({ tag, onClick, active }: { tag: string; onClick?: () => void; active?: boolean }) {
  const Comp = onClick ? "button" : "span";
  return (
    <Comp
      onClick={onClick}
      className={`inline-flex items-center rounded-full px-2 py-0.5 text-xs font-medium transition-colors ${
        active
          ? "bg-indigo-600 text-white"
          : "bg-indigo-50 text-indigo-700 hover:bg-indigo-100 dark:bg-indigo-500/10 dark:text-indigo-300 dark:hover:bg-indigo-500/20"
      }`}
    >
      #{tag}
    </Comp>
  );
}
