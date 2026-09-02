import { useCallback, useEffect, useMemo, useState } from "react";
import * as api from "../api/client";
import type { FilterStatus, SortMode, Todo, TodoInput, TodoUpdate } from "../types";
import { useToast } from "../context/ToastContext";

export function useTodos() {
  const [todos, setTodos] = useState<Todo[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [search, setSearch] = useState("");
  const [status, setStatus] = useState<FilterStatus>("all");
  const [priority, setPriority] = useState<string | null>(null);
  const [tag, setTag] = useState<string | null>(null);
  const [sort, setSort] = useState<SortMode>("manual");

  const { showToast } = useToast();

  const load = useCallback(async () => {
    try {
      setError(null);
      const data = await api.fetchTodos({ search, status, priority: priority ?? undefined, tag: tag ?? undefined, sort });
      setTodos(data);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load todos");
    } finally {
      setLoading(false);
    }
  }, [search, status, priority, tag, sort]);

  useEffect(() => {
    const handle = window.setTimeout(load, search ? 250 : 0);
    return () => window.clearTimeout(handle);
  }, [load, search]);

  const addTodo = useCallback(
    async (input: TodoInput) => {
      const tempId = `temp-${Date.now()}`;
      const now = new Date().toISOString();
      const optimistic: Todo = {
        id: tempId,
        title: input.title,
        notes: input.notes ?? "",
        completed: false,
        priority: input.priority ?? "medium",
        dueDate: input.dueDate ?? null,
        tags: input.tags ?? [],
        position: (todos.at(-1)?.position ?? 0) + 1,
        createdAt: now,
        updatedAt: now,
      };
      setTodos((prev) => [...prev, optimistic]);
      try {
        const created = await api.createTodo(input);
        setTodos((prev) => prev.map((t) => (t.id === tempId ? created : t)));
      } catch (e) {
        setTodos((prev) => prev.filter((t) => t.id !== tempId));
        showToast(e instanceof Error ? e.message : "Couldn't add task", { tone: "error" });
      }
    },
    [todos, showToast]
  );

  const updateTodoLocal = useCallback(
    async (id: string, update: TodoUpdate, options?: { silent?: boolean }) => {
      const prevTodos = todos;
      setTodos((prev) => prev.map((t) => (t.id === id ? { ...t, ...update, updatedAt: new Date().toISOString() } : t)));
      try {
        const saved = await api.updateTodo(id, update);
        setTodos((prev) => prev.map((t) => (t.id === id ? saved : t)));
      } catch (e) {
        setTodos(prevTodos);
        if (!options?.silent) showToast(e instanceof Error ? e.message : "Couldn't save changes", { tone: "error" });
      }
    },
    [todos, showToast]
  );

  const removeTodo = useCallback(
    async (id: string) => {
      const removed = todos.find((t) => t.id === id);
      const prevTodos = todos;
      setTodos((prev) => prev.filter((t) => t.id !== id));
      try {
        await api.deleteTodo(id);
        if (removed) {
          showToast(`Deleted "${removed.title}"`, {
            tone: "default",
            action: {
              label: "Undo",
              onClick: async () => {
                const restored = await api.createTodo({
                  title: removed.title,
                  notes: removed.notes,
                  priority: removed.priority,
                  dueDate: removed.dueDate,
                  tags: removed.tags,
                });
                if (removed.completed) await api.updateTodo(restored.id, { completed: true });
                load();
              },
            },
          });
        }
      } catch (e) {
        setTodos(prevTodos);
        showToast(e instanceof Error ? e.message : "Couldn't delete task", { tone: "error" });
      }
    },
    [todos, showToast, load]
  );

  const reorder = useCallback(
    async (orderedIds: string[]) => {
      const prevTodos = todos;
      const byId = new Map(todos.map((t) => [t.id, t]));
      setTodos(orderedIds.map((id, i) => ({ ...byId.get(id)!, position: i + 1 })));
      try {
        await api.reorderTodos(orderedIds);
      } catch (e) {
        setTodos(prevTodos);
        showToast(e instanceof Error ? e.message : "Couldn't reorder tasks", { tone: "error" });
      }
    },
    [todos, showToast]
  );

  const clearCompleted = useCallback(async () => {
    const prevTodos = todos;
    const count = todos.filter((t) => t.completed).length;
    if (count === 0) return;
    setTodos((prev) => prev.filter((t) => !t.completed));
    try {
      await api.clearCompleted();
      showToast(`Cleared ${count} completed task${count === 1 ? "" : "s"}`, { tone: "success" });
    } catch (e) {
      setTodos(prevTodos);
      showToast(e instanceof Error ? e.message : "Couldn't clear completed tasks", { tone: "error" });
    }
  }, [todos, showToast]);

  const allTags = useMemo(() => {
    const set = new Set<string>();
    todos.forEach((t) => t.tags.forEach((tag) => set.add(tag)));
    return Array.from(set).sort();
  }, [todos]);

  return {
    todos,
    loading,
    error,
    filters: { search, status, priority, tag, sort },
    setSearch,
    setStatus,
    setPriority,
    setTag,
    setSort,
    addTodo,
    updateTodo: updateTodoLocal,
    removeTodo,
    reorder,
    clearCompleted,
    allTags,
    reload: load,
  };
}
