import { useEffect, useState } from "react";
import { fetchModelInfo } from "../api/client";
import type { ModelInfo } from "../types";

export function useModelInfo() {
  const [data, setData] = useState<ModelInfo | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchModelInfo()
      .then(setData)
      .catch((e) => setError(e instanceof Error ? e.message : "Failed to load"))
      .finally(() => setLoading(false));
  }, []);

  return { data, loading, error };
}
