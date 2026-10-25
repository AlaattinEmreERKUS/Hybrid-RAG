import type { GraphPayload, QueryResponse } from "../types";

const BASE = "/api";

async function check<T>(rsp: Response): Promise<T> {
  if (!rsp.ok) throw new Error(`${rsp.status} ${await rsp.text()}`);
  return rsp.json() as Promise<T>;
}

export const fetchGraph = (limit = 300): Promise<GraphPayload> =>
  fetch(`${BASE}/graph?limit=${limit}`).then((r) => check<GraphPayload>(r));

export const postQuery = (question: string): Promise<QueryResponse> =>
  fetch(`${BASE}/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
  }).then((r) => check<QueryResponse>(r));

export const uploadFile = (file: File): Promise<{ docId: string; status: string; chunks: number }> => {
  const form = new FormData();
  form.append("file", file);
  return fetch(`${BASE}/ingest`, { method: "POST", body: form }).then((r) => check(r));
};
