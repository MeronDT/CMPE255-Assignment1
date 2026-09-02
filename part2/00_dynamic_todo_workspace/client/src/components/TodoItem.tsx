import { useSortable } from "@dnd-kit/sortable";
import { CSS } from "@dnd-kit/utilities";
import { useEffect, useRef, useState, type KeyboardEvent } from "react";
import type { Priority, Todo, TodoUpdate } from "../types";
import { PriorityBadge } from "./PriorityBadge";
import { DueDateBadge } from "./DueDateBadge";
import { TagPill } from "./TagPill";

interface Props {
  todo: Todo;
  onUpdate: (id: string, update: TodoUpdate) => void;
  onDelete: (id: string) => void;
  dragDisabled: boolean;
}

const priorities: Priority[] = ["low", "medium", "high"];

export function TodoItem({ todo, onUpdate, onDelete, dragDisabled }: Props) {
  const { attributes, listeners, setNodeRef, transform, transition, isDragging } = useSortable({
    id: todo.id,
    disabled: dragDisabled,
  });

  const [expanded, setExpanded] = useState(false);
  const [editingTitle, setEditingTitle] = useState(false);
  const [draftTitle, setDraftTitle] = useState(todo.title);
  const [draftNotes, setDraftNotes] = useState(todo.notes);
  const titleInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (editingTitle) titleInputRef.current?.focus();
  }, [editingTitle]);

  const style = {
    transform: CSS.Transform.toString(transform),
    transition,
    opacity: isDragging ? 0.5 : 1,
  };

  const commitTitle = () => {
    const trimmed = draftTitle.trim();
    setEditingTitle(false);
    if (trimmed && trimmed !== todo.title) onUpdate(todo.id, { title: trimmed });
    else setDraftTitle(todo.title);
  };

  const commitNotes = () => {
    if (draftNotes !== todo.notes) onUpdate(todo.id, { notes: draftNotes });
  };

  const handleTitleKeyDown = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter") titleInputRef.current?.blur();
    if (e.key === "Escape") {
      setDraftTitle(todo.title);
      setEditingTitle(false);
    }
  };

  const isOverdue = !todo.completed && todo.dueDate && todo.dueDate < new Date().toISOString().slice(0, 10);

  return (
    <li
      ref={setNodeRef}
      style={style}
      className={`group rounded-2xl border bg-white shadow-sm transition-colors dark:bg-slate-800 ${
        isOverdue ? "border-red-200 dark:border-red-500/30" : "border-slate-200 dark:border-slate-700"
      } ${isDragging ? "z-10 shadow-lg" : ""}`}
    >
      <div className="flex items-start gap-2 p-3">
        <button
          {...attributes}
          {...listeners}
          disabled={dragDisabled}
          aria-label="Drag to reorder"
          className={`mt-1.5 shrink-0 touch-none rounded p-1 text-slate-300 transition-opacity dark:text-slate-600 ${
            dragDisabled ? "cursor-not-allowed opacity-0" : "cursor-grab opacity-0 hover:text-slate-500 group-hover:opacity-100 active:cursor-grabbing dark:hover:text-slate-400"
          }`}
        >
          <svg className="h-4 w-4" fill="currentColor" viewBox="0 0 20 20">
            <path d="M7 4a1 1 0 11-2 0 1 1 0 012 0zM7 10a1 1 0 11-2 0 1 1 0 012 0zM7 16a1 1 0 11-2 0 1 1 0 012 0zM15 4a1 1 0 11-2 0 1 1 0 012 0zM15 10a1 1 0 11-2 0 1 1 0 012 0zM15 16a1 1 0 11-2 0 1 1 0 012 0z" />
          </svg>
        </button>

        <button
          onClick={() => onUpdate(todo.id, { completed: !todo.completed })}
          aria-label={todo.completed ? "Mark incomplete" : "Mark complete"}
          className={`mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center rounded-full border-2 transition-colors ${
            todo.completed
              ? "border-indigo-500 bg-indigo-500 text-white"
              : "border-slate-300 hover:border-indigo-400 dark:border-slate-600"
          }`}
        >
          {todo.completed && (
            <svg className="h-3 w-3" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={3}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
            </svg>
          )}
        </button>

        <div className="min-w-0 flex-1">
          {editingTitle ? (
            <input
              ref={titleInputRef}
              value={draftTitle}
              onChange={(e) => setDraftTitle(e.target.value)}
              onBlur={commitTitle}
              onKeyDown={handleTitleKeyDown}
              className="w-full rounded bg-transparent text-sm font-medium text-slate-800 focus:outline-none dark:text-slate-100"
            />
          ) : (
            <button
              onClick={() => setEditingTitle(true)}
              className={`block w-full truncate text-left text-sm font-medium transition-colors ${
                todo.completed ? "text-slate-400 line-through dark:text-slate-500" : "text-slate-800 dark:text-slate-100"
              }`}
            >
              {todo.title}
            </button>
          )}

          <div className="mt-1.5 flex flex-wrap items-center gap-1.5">
            <PriorityBadge priority={todo.priority} />
            {todo.dueDate && <DueDateBadge dueDate={todo.dueDate} completed={todo.completed} />}
            {todo.tags.map((t) => (
              <TagPill key={t} tag={t} />
            ))}
          </div>
        </div>

        <div className="flex shrink-0 items-center gap-1">
          <button
            onClick={() => setExpanded((v) => !v)}
            aria-label="Expand details"
            className="rounded-lg p-1.5 text-slate-400 opacity-0 transition-opacity hover:bg-slate-100 hover:text-slate-600 group-hover:opacity-100 dark:hover:bg-slate-700 dark:hover:text-slate-200"
          >
            <svg
              className={`h-4 w-4 transition-transform ${expanded ? "rotate-180" : ""}`}
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
              strokeWidth={2}
            >
              <path strokeLinecap="round" strokeLinejoin="round" d="M19 9l-7 7-7-7" />
            </svg>
          </button>
          <button
            onClick={() => onDelete(todo.id)}
            aria-label="Delete task"
            className="rounded-lg p-1.5 text-slate-400 opacity-0 transition-opacity hover:bg-red-50 hover:text-red-500 group-hover:opacity-100 dark:hover:bg-red-500/10"
          >
            <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6M9 7V4a1 1 0 011-1h4a1 1 0 011 1v3M4 7h16" />
            </svg>
          </button>
        </div>
      </div>

      {expanded && (
        <div className="space-y-3 border-t border-slate-100 px-3 pb-3 pt-3 dark:border-slate-700">
          <textarea
            value={draftNotes}
            onChange={(e) => setDraftNotes(e.target.value)}
            onBlur={commitNotes}
            placeholder="Add notes…"
            rows={2}
            className="w-full resize-none rounded-lg border border-slate-200 bg-slate-50 px-2 py-1.5 text-sm text-slate-600 placeholder:text-slate-400 focus:border-indigo-400 focus:bg-white focus:outline-none dark:border-slate-600 dark:bg-slate-900/40 dark:text-slate-300 dark:focus:bg-slate-900"
          />
          <div className="flex flex-wrap items-center gap-3 text-xs">
            <span className="text-slate-400 dark:text-slate-500">Priority</span>
            <div className="flex items-center gap-1">
              {priorities.map((p) => (
                <button
                  key={p}
                  onClick={() => onUpdate(todo.id, { priority: p })}
                  className={`rounded-full transition-transform ${todo.priority === p ? "scale-105 ring-2 ring-indigo-400" : "opacity-50 hover:opacity-100"}`}
                >
                  <PriorityBadge priority={p} />
                </button>
              ))}
            </div>
            <span className="text-slate-400 dark:text-slate-500">Due</span>
            <input
              type="date"
              value={todo.dueDate ?? ""}
              onChange={(e) => onUpdate(todo.id, { dueDate: e.target.value || null })}
              className="rounded-lg border border-slate-200 bg-white px-2 py-1 text-xs text-slate-600 focus:border-indigo-400 focus:outline-none dark:border-slate-600 dark:bg-slate-700 dark:text-slate-200"
            />
            <span className="text-slate-400 dark:text-slate-500">
              Updated {new Date(todo.updatedAt).toLocaleDateString()}
            </span>
          </div>
        </div>
      )}
    </li>
  );
}
