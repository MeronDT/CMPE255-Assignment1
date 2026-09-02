import { forwardRef } from "react";
import type { FilterStatus, SortMode } from "../types";

interface Props {
  search: string;
  onSearchChange: (v: string) => void;
  status: FilterStatus;
  onStatusChange: (v: FilterStatus) => void;
  priority: string | null;
  onPriorityChange: (v: string | null) => void;
  sort: SortMode;
  onSortChange: (v: SortMode) => void;
  allTags: string[];
  activeTag: string | null;
  onTagChange: (v: string | null) => void;
}

const statusTabs: { value: FilterStatus; label: string }[] = [
  { value: "all", label: "All" },
  { value: "active", label: "Active" },
  { value: "completed", label: "Completed" },
];

const sortOptions: { value: SortMode; label: string }[] = [
  { value: "manual", label: "Manual order" },
  { value: "dueDate", label: "Due date" },
  { value: "priority", label: "Priority" },
  { value: "created", label: "Recently added" },
  { value: "alphabetical", label: "Alphabetical" },
];

export const FilterBar = forwardRef<HTMLInputElement, Props>(function FilterBar(
  { search, onSearchChange, status, onStatusChange, priority, onPriorityChange, sort, onSortChange, allTags, activeTag, onTagChange },
  ref
) {
  return (
    <div className="space-y-3">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-center">
        <div className="relative flex-1">
          <svg
            className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-400"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
            strokeWidth={2}
          >
            <path strokeLinecap="round" strokeLinejoin="round" d="M21 21l-4.35-4.35M17 10a7 7 0 11-14 0 7 7 0 0114 0z" />
          </svg>
          <input
            ref={ref}
            value={search}
            onChange={(e) => onSearchChange(e.target.value)}
            placeholder="Search tasks… (press / to focus)"
            className="w-full rounded-xl border border-slate-200 bg-white py-2 pl-9 pr-3 text-sm text-slate-800 placeholder:text-slate-400 focus:border-indigo-400 focus:outline-none focus:ring-2 focus:ring-indigo-100 dark:border-slate-700 dark:bg-slate-800 dark:text-slate-100 dark:focus:ring-indigo-500/20"
          />
          {search && (
            <button
              onClick={() => onSearchChange("")}
              aria-label="Clear search"
              className="absolute right-2 top-1/2 -translate-y-1/2 text-slate-400 hover:text-slate-600 dark:hover:text-slate-200"
            >
              ✕
            </button>
          )}
        </div>

        <select
          value={sort}
          onChange={(e) => onSortChange(e.target.value as SortMode)}
          className="rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm text-slate-600 focus:border-indigo-400 focus:outline-none dark:border-slate-700 dark:bg-slate-800 dark:text-slate-300"
        >
          {sortOptions.map((o) => (
            <option key={o.value} value={o.value}>
              Sort: {o.label}
            </option>
          ))}
        </select>
      </div>

      <div className="flex flex-wrap items-center gap-2">
        <div className="flex rounded-xl bg-slate-100 p-1 dark:bg-slate-800">
          {statusTabs.map((tab) => (
            <button
              key={tab.value}
              onClick={() => onStatusChange(tab.value)}
              className={`rounded-lg px-3 py-1.5 text-sm font-medium transition-colors ${
                status === tab.value
                  ? "bg-white text-indigo-600 shadow-sm dark:bg-slate-700 dark:text-indigo-300"
                  : "text-slate-500 hover:text-slate-700 dark:text-slate-400 dark:hover:text-slate-200"
              }`}
            >
              {tab.label}
            </button>
          ))}
        </div>

        <div className="flex items-center gap-1">
          {(["high", "medium", "low"] as const).map((p) => (
            <button
              key={p}
              onClick={() => onPriorityChange(priority === p ? null : p)}
              className={`rounded-full px-2.5 py-1 text-xs font-semibold capitalize transition-colors ${
                priority === p
                  ? "bg-indigo-600 text-white"
                  : "bg-slate-100 text-slate-500 hover:bg-slate-200 dark:bg-slate-800 dark:text-slate-400 dark:hover:bg-slate-700"
              }`}
            >
              {p}
            </button>
          ))}
        </div>

        {allTags.length > 0 && (
          <div className="flex flex-wrap items-center gap-1.5 border-l border-slate-200 pl-2 dark:border-slate-700">
            {allTags.map((t) => (
              <button
                key={t}
                onClick={() => onTagChange(activeTag === t ? null : t)}
                className={`rounded-full px-2 py-0.5 text-xs font-medium transition-colors ${
                  activeTag === t
                    ? "bg-indigo-600 text-white"
                    : "bg-indigo-50 text-indigo-700 hover:bg-indigo-100 dark:bg-indigo-500/10 dark:text-indigo-300 dark:hover:bg-indigo-500/20"
                }`}
              >
                #{t}
              </button>
            ))}
          </div>
        )}
      </div>
    </div>
  );
});
