import { useToast } from "../context/ToastContext";

export function ToastContainer() {
  const { toasts, dismissToast } = useToast();

  return (
    <div className="pointer-events-none fixed inset-x-0 bottom-4 z-50 flex flex-col items-center gap-2 px-4 sm:bottom-6 sm:items-end sm:pr-6">
      {toasts.map((toast) => (
        <div
          key={toast.id}
          role="status"
          className={`pointer-events-auto flex w-full max-w-sm items-center justify-between gap-3 rounded-xl border px-4 py-3 shadow-lg backdrop-blur-md animate-toast-in ${
            toast.tone === "success"
              ? "border-emerald-500/30 bg-emerald-50/95 text-emerald-900 dark:bg-emerald-950/90 dark:text-emerald-100"
              : toast.tone === "error"
                ? "border-red-500/30 bg-red-50/95 text-red-900 dark:bg-red-950/90 dark:text-red-100"
                : "border-slate-200 bg-white/95 text-slate-800 dark:border-slate-700 dark:bg-slate-800/95 dark:text-slate-100"
          }`}
        >
          <span className="text-sm font-medium">{toast.message}</span>
          <div className="flex shrink-0 items-center gap-2">
            {toast.action && (
              <button
                onClick={() => {
                  toast.action?.onClick();
                  dismissToast(toast.id);
                }}
                className="rounded-lg px-2 py-1 text-sm font-semibold text-indigo-600 hover:bg-indigo-50 dark:text-indigo-300 dark:hover:bg-indigo-500/10"
              >
                {toast.action.label}
              </button>
            )}
            <button
              onClick={() => dismissToast(toast.id)}
              aria-label="Dismiss"
              className="rounded-lg p-1 text-slate-400 hover:bg-slate-100 hover:text-slate-600 dark:hover:bg-slate-700 dark:hover:text-slate-200"
            >
              ✕
            </button>
          </div>
        </div>
      ))}
    </div>
  );
}
