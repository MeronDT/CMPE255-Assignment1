import { useEffect, useMemo, useRef } from "react";
import { useTodos } from "./hooks/useTodos";
import { useStats } from "./hooks/useStats";
import { useTheme } from "./hooks/useTheme";
import { AddTodoForm } from "./components/AddTodoForm";
import { FilterBar } from "./components/FilterBar";
import { TodoList } from "./components/TodoList";
import { StatsBar } from "./components/StatsBar";
import { ThemeToggle } from "./components/ThemeToggle";
import { ToastContainer } from "./components/ToastContainer";
import { TodoListSkeleton } from "./components/Skeleton";

function App() {
  const {
    todos,
    loading,
    error,
    filters,
    setSearch,
    setStatus,
    setPriority,
    setTag,
    setSort,
    addTodo,
    updateTodo,
    removeTodo,
    reorder,
    clearCompleted,
    allTags,
    reload,
  } = useTodos();

  const stats = useStats(todos.length + todos.filter((t) => t.completed).length);
  const { theme, toggleTheme } = useTheme();

  const addInputRef = useRef<HTMLInputElement>(null);
  const searchInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement;
      const isTyping = ["INPUT", "TEXTAREA"].includes(target.tagName);
      if (isTyping) {
        if (e.key === "Escape") target.blur();
        return;
      }
      if (e.key === "n" || e.key === "N") {
        e.preventDefault();
        addInputRef.current?.focus();
      } else if (e.key === "/") {
        e.preventDefault();
        searchInputRef.current?.focus();
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, []);

  const emptyKind = useMemo(() => {
    if (filters.search || filters.priority || filters.tag || filters.status !== "all") return "no-results" as const;
    return "none" as const;
  }, [filters]);

  const hasCompletedInView = todos.some((t) => t.completed);

  return (
    <div className="min-h-screen bg-gradient-to-b from-slate-50 to-slate-100 pb-24 dark:from-slate-950 dark:to-slate-900">
      <div className="mx-auto max-w-2xl px-4 pt-10 sm:pt-16">
        <header className="mb-6 flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold tracking-tight text-slate-900 dark:text-white">Flow</h1>
            <p className="text-sm text-slate-500 dark:text-slate-400">Your day, organized.</p>
          </div>
          <ThemeToggle theme={theme} onToggle={toggleTheme} />
        </header>

        <div className="mb-5">
          <StatsBar stats={stats} />
        </div>

        <div className="mb-5">
          <AddTodoForm ref={addInputRef} onAdd={addTodo} />
        </div>

        <div className="mb-5">
          <FilterBar
            ref={searchInputRef}
            search={filters.search}
            onSearchChange={setSearch}
            status={filters.status}
            onStatusChange={setStatus}
            priority={filters.priority}
            onPriorityChange={setPriority}
            sort={filters.sort}
            onSortChange={setSort}
            allTags={allTags}
            activeTag={filters.tag}
            onTagChange={setTag}
          />
        </div>

        {error && (
          <div className="mb-4 flex items-center justify-between rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700 dark:border-red-500/30 dark:bg-red-950/40 dark:text-red-300">
            <span>{error}</span>
            <button onClick={reload} className="font-semibold underline underline-offset-2">
              Retry
            </button>
          </div>
        )}

        {loading ? (
          <TodoListSkeleton />
        ) : (
          <TodoList
            todos={todos}
            onUpdate={updateTodo}
            onDelete={removeTodo}
            onReorder={reorder}
            dragDisabled={filters.sort !== "manual"}
            emptyKind={emptyKind}
          />
        )}

        {hasCompletedInView && (
          <div className="mt-5 flex justify-center">
            <button
              onClick={clearCompleted}
              className="text-sm font-medium text-slate-400 transition-colors hover:text-red-500 dark:text-slate-500 dark:hover:text-red-400"
            >
              Clear completed tasks
            </button>
          </div>
        )}

        <footer className="mt-10 text-center text-xs text-slate-400 dark:text-slate-600">
          <kbd className="rounded border border-slate-300 px-1 dark:border-slate-600">N</kbd> new task ·{" "}
          <kbd className="rounded border border-slate-300 px-1 dark:border-slate-600">/</kbd> search
        </footer>
      </div>

      <ToastContainer />
    </div>
  );
}

export default App;
