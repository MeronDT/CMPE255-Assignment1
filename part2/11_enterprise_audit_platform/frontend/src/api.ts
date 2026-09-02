const BASE = "http://localhost:8011";

async function get<T>(path: string): Promise<T> {
  // cache: "no-store" is not cosmetic here -- this dashboard's entire purpose
  // is reporting the CURRENT audit state, and a stale cached GET silently
  // showing an old scorecard is exactly the kind of untrustworthy-tool bug
  // this project's own Methodology tab warns against. Found by direct
  // reproduction (a fresh fetch with cache disabled returned different,
  // correct data than the app was rendering), not assumed.
  const res = await fetch(`${BASE}${path}`, { cache: "no-store" });
  if (!res.ok) throw new Error(`${path} -> ${res.status}`);
  return res.json();
}

export const api = {
  audit: () => get<any>("/api/audit"),
};
