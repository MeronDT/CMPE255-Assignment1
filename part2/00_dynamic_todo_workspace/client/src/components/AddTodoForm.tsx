import { forwardRef, useState, type FormEvent } from "react";
import type { Priority, TodoInput } from "../types";
import { PriorityBadge } from "./PriorityBadge";

interface Props {
  onAdd: (input: TodoInput) => void;
}

const priorities: Priority[] = ["low", "medium", "high"];

export const AddTodoForm = forwardRef<HTMLInputElement, Props>(function AddTodoForm({ onAdd }, ref) {
  const [title, setTitle] = useState("");
  const [expanded, setExpanded] = useState(false);
  const [priority, setPriority] = useState<Priority>("medium");
  const [dueDate, setDueDate] = useState("");
  const [tagsInput, setTagsInput] = useState("");

  const reset = () => {
    setTitle("");
    setPriority("medium");
    setDueDate("");
    setTagsInput("");
    setExpanded(false);
  };

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault();
    const trimmed = title.trim();
    if (!trimmed) return;
    const tags = tagsInput
      .split(",")
      .map((t) => t.trim())
      .filter(Boolean);
    onAdd({ title: trimmed, priority, dueDate: dueDate || null, tags });
    reset();
  };

  return (
    <form
      onSubmit={handleSubmit}
      className="rounded-2xl border border-slate-200 bg-white p-3 shadow-sm transition-shadow focus-within:shadow-md dark:border-slate-700 dark:bg-slate-800"
    >
      <div className="flex items-center gap-2">
        <svg className="h-5 w-5 shrink-0 text-slate-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M12 4v16m8-8H4" />
        </svg>
        <input
          ref={ref}
          value={title}
          onChange={(e) => setTitle(e.target.value)}
          onFocus={() => setExpanded(true)}
          placeholder="Add a task… (press N to focus)"
          className="min-w-0 flex-1 bg-transparent text-sm text-slate-800 placeholder:text-slate-400 focus:outline-none dark:text-slate-100"
        />
        {title.trim() && (
          <button
            type="submit"
            className="shrink-0 rounded-lg bg-indigo-600 px-3 py-1.5 text-sm font-semibold text-white transition-colors hover:bg-indigo-500"
          >
            Add
          </button>
        )}
      </div>

      {expanded && (
        <div className="mt-3 flex flex-wrap items-center gap-3 border-t border-slate-100 pt-3 dark:border-slate-700">
          <div className="flex items-center gap-1.5">
            {priorities.map((p) => (
              <button
                type="button"
                key={p}
                onClick={() => setPriority(p)}
                className={`rounded-full transition-transform ${priority === p ? "scale-105 ring-2 ring-indigo-400" : "opacity-60 hover:opacity-100"}`}
              >
                <PriorityBadge priority={p} />
              </button>
            ))}
          </div>

          <input
            type="date"
            value={dueDate}
            onChange={(e) => setDueDate(e.target.value)}
            className="rounded-lg border border-slate-200 bg-white px-2 py-1 text-xs text-slate-600 focus:border-indigo-400 focus:outline-none dark:border-slate-600 dark:bg-slate-700 dark:text-slate-200"
          />

          <input
            value={tagsInput}
            onChange={(e) => setTagsInput(e.target.value)}
            placeholder="tags, comma, separated"
            className="min-w-[10rem] flex-1 rounded-lg border border-slate-200 bg-white px-2 py-1 text-xs text-slate-600 placeholder:text-slate-400 focus:border-indigo-400 focus:outline-none dark:border-slate-600 dark:bg-slate-700 dark:text-slate-200"
          />

          {(title || dueDate || tagsInput) && (
            <button
              type="button"
              onClick={reset}
              className="ml-auto text-xs font-medium text-slate-400 hover:text-slate-600 dark:hover:text-slate-200"
            >
              Cancel
            </button>
          )}
        </div>
      )}
    </form>
  );
});
