import { useEffect, useState } from "react";
import { fetchZones } from "../api/client";
import type { ZoneInfo } from "../types";

export function useZones() {
  const [zones, setZones] = useState<ZoneInfo[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchZones()
      .then(setZones)
      .finally(() => setLoading(false));
  }, []);

  return { zones, loading };
}
