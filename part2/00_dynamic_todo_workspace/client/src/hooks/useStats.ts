import { useCallback, useEffect, useState } from "react";
import * as api from "../api/client";
import type { Stats } from "../types";

const EMPTY: Stats = { total: 0, completed: 0, active: 0, overdue: 0 };

export function useStats(refreshKey: unknown) {
  const [stats, setStats] = useState<Stats>(EMPTY);

  const load = useCallback(() => {
    api.fetchStats().then(setStats).catch(() => {});
  }, []);

  useEffect(() => {
    load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [refreshKey]);

  return stats;
}
