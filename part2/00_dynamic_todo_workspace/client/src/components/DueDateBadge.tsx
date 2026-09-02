import { format, isPast, isToday, isTomorrow, parseISO } from "date-fns";

export function DueDateBadge({ dueDate, completed }: { dueDate: string; completed: boolean }) {
  const date = parseISO(dueDate);
  const overdue = !completed && isPast(date) && !isToday(date);

  let label: string;
  if (isToday(date)) label = "Today";
  else if (isTomorrow(date)) label = "Tomorrow";
  else label = format(date, "MMM d");

  return (
    <span
      className={`inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-xs font-semibold ${
        overdue
          ? "bg-red-100 text-red-700 dark:bg-red-500/15 dark:text-red-300"
          : "bg-slate-100 text-slate-600 dark:bg-slate-700/60 dark:text-slate-300"
      }`}
    >
      <svg className="h-3 w-3" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z" />
      </svg>
      {overdue ? `Overdue · ${label}` : label}
    </span>
  );
}
