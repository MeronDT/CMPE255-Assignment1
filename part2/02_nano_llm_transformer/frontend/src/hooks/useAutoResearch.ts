import { useEffect, useState } from "react";
import { fetchAutoResearch } from "../api/client";
import type { AutoResearchResponse } from "../types";

export function useAutoResearch() {
  const [data, setData] = useState<AutoResearchResponse | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchAutoResearch()
      .then(setData)
      .finally(() => setLoading(false));
  }, []);

  return { data, loading };
}
