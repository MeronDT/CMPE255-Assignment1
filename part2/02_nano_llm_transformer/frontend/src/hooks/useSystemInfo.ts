import { useEffect, useState } from "react";
import { fetchSystemInfo } from "../api/client";
import type { SystemInfo } from "../types";

export function useSystemInfo() {
  const [data, setData] = useState<SystemInfo | null>(null);

  useEffect(() => {
    fetchSystemInfo().then(setData).catch(() => {});
  }, []);

  return data;
}
