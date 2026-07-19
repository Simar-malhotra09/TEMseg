export const BASE_URL = "http://localhost:8080";

// Tracks in-flight requests to the backend so the UI can show a blocking
// indicator. Every fetch below goes through trackedFetch instead of the
// global fetch, so this stays accurate without callers opting in per call.
type RequestActivityListener = () => void;

let activeRequestCount = 0;
const requestActivityListeners = new Set<RequestActivityListener>();

export function subscribeToRequestActivity(listener: RequestActivityListener): () => void {
  requestActivityListeners.add(listener);
  return () => requestActivityListeners.delete(listener);
}

export function getActiveRequestCount(): number {
  return activeRequestCount;
}

async function trackedFetch(input: RequestInfo | URL, init?: RequestInit): Promise<Response> {
  activeRequestCount += 1;
  requestActivityListeners.forEach((listener) => listener());
  try {
    return await fetch(input, init);
  } finally {
    activeRequestCount -= 1;
    requestActivityListeners.forEach((listener) => listener());
  }
}

// These are mirrors of the pydantic model the server defines
export interface Box {
  id: string;
  x: number;
  y: number;
  width: number;
  height: number;
}

// this already exists in BlackOutCanvas. low priority fix
interface BlackoutRect {
  id: string;
  x: number;
  y: number;
  width: number;
  height: number;
}

// mirrors the server's Stroke pydantic model ie  a freehand scribble
export interface Stroke {
  id: string;
  points: number[]; // flat [x1, y1, x2, y2, ...] in image coordinates
  strokeWidth: number;
}

export interface StatsResult {
  // scale info
  pixel_size: number | null;
  pixel_unit: string | null;
  unit: string;
  has_scale: boolean;

  // aggregate (backward compat)
  particle_count: number;
  coverage: number;
  avg_size: number;
  avg_circularity: number;
  avg_aspect_ratio: number;

  // detailed aggregate
  avg_area_px: number;
  avg_diameter_px: number;
  avg_area_real: number | null;
  avg_diameter_real: number | null;

  // distributions
  size_stats: {
    area_mean: number;
    area_std: number;
    area_min: number;
    area_max: number;
    area_median: number;
    diameter_mean: number;
    diameter_std: number;
    diameter_min: number;
    diameter_max: number;
    diameter_median: number;
    unit: string;
  };

  shape_distribution: Record<string, { count: number; fraction: number }>;

  distribution_fits_diameter: {
      reliable: boolean;
      reason?: string;
      best_model?: string;
      fits?: Record<string, {
        params: Record<string, number>;
        ks_statistic: number;
        ks_pvalue: number;
      }>;
  };

  distribution_fits_area: {
      reliable: boolean;
      reason?: string;
      best_model?: string;
      fits?: Record<string, {
        params: Record<string, number>;
        ks_statistic: number;
        ks_pvalue: number;
      }>;
  };

  // per-particle
  particles: ParticleStats[];
}

export interface ParticleStats {
  id: number;
  area_px: number;
  area_real?: number;
  perimeter_px: number;
  perimeter_real?: number;
  diameter_px: number;
  diameter_real?: number;
  major_axis_px: number;
  major_axis_real?: number;
  minor_axis_px: number;
  minor_axis_real?: number;
  circularity: number;
  solidity: number;
  convexity: number;
  rectangularity: number;
  aspect_ratio: number;
  n_vertices: number;
  shape: string;
  bbox: { x: number; y: number; w: number; h: number };
}

// Client side source of truth for which computed metrics are selectable in
// the stats table and the refine-mode hover tooltip. Both derive their
// column/field list from this instead of keeping their own copy.
export const PARTICLE_METRIC_FIELDS = [
  "diameter", "area", "circularity", "solidity", "aspect_ratio", "n_vertices", "shape",
] as const;
export type ParticleMetricField = typeof PARTICLE_METRIC_FIELDS[number];

export interface Metadata {
  image_shape?: number[];
  original_format?: string;
  pixel_size: number | string | null;
  pixel_unit?: string | null;
  axes?: { scale: number; size: number; units: string }[];
}

export interface SegmentResponse {
  model: string;
  mask_url: string;
  metadata: Metadata | null;
  stats: StatsResult;
  time_elapsed: number;
  debug_boxes_url?: string;
}

export interface GTResponse {
  warnings: string[];
  scores: {
    iou: number;
    dice: number;
    pixel_acc: number;
  } | null;
}

export interface UploadResponse {
  session_id: string;
  filename: string;
  preview_url: string;
  image_info: Record<string, string>;
}

export interface Instance {
  id: number;
  contour: [number, number][];
  bbox: { x: number; y: number; w: number; h: number };
  area: number;
  // Optional: present on instances created via /from-points or /propose-similar
  sam_score?: number;
  source?: string;
  similarity?: number;
  seed?: [number, number];
}


/** True when running inside the PyWebView desktop window */
export function isPyWebView(): boolean {
  return typeof window !== "undefined" && !!(window as any).pywebview?.api?.export_zip;
}

/**
 * Export via PyWebView native save dialog.
 * Returns { success, path?, error? }
 */
export async function exportViaPyWebView(
  sessionId: string,
  items: string[]
): Promise<{ success: boolean; path?: string; error?: string }> {
  return (window as any).pywebview.api.export_zip(sessionId, items);
}


export async function getModels(): Promise<string[]> {
  const res = await trackedFetch(`${BASE_URL}/models`);

  if (!res.ok) {
    throw new Error("Failed to fetch models");
  }

  const data = await res.json();

  if (Array.isArray(data)) {
    return data;
  }

  return data.models ?? [];
}

export async function getSessionMetadata(
  sessionId: string,
): Promise<Metadata | null> {
  // Used to validate a session exists on refresh (?session=... restore).
  // Returns metadata dict on 200, null on 404 (session evicted/deleted).
  const res = await trackedFetch(`${BASE_URL}/images/${sessionId}/metadata`);
  if (res.status === 404) return null;
  if (!res.ok) throw new Error(`getSessionMetadata: ${res.status}`);
  return res.json();
}

export async function updatePixelSize(
  sessionId: string,
  pixelSize: number,
  pixelUnit: string,
): Promise<{ metadata: Metadata; stats?: StatsResult }> {
  const res = await trackedFetch(`${BASE_URL}/images/${sessionId}/metadata`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ pixel_size: pixelSize, pixel_unit: pixelUnit }),
  });
  if (!res.ok) throw new Error(`updatePixelSize failed: ${res.status}`);
  return res.json();
}

export async function getStats(sessionId: string): Promise<StatsResult | null> {
  // Returns cached stats.json for a session, or null if /segment hasn't run.
  const res = await trackedFetch(`${BASE_URL}/masks/${sessionId}/stats`);
  if (res.status === 404) return null;
  if (!res.ok) throw new Error(`getStats: ${res.status}`);
  return res.json();
}

export interface FromPointsResponse {
  proposals: Instance[];
  rejected: { index: number; reason: string }[];
  elapsed: number;
}

export interface ProposeSimilarResponse {
  proposals: Instance[];
  seed_count?: number;
  median_area?: number;
  sim_threshold?: number;
  method?: "ncc" | "cosine";
  message?: string;
  elapsed?: number;
  debug_overlay_url?: string | null;
}

export async function proposeSimilar(
  sessionId: string,
  method: "ncc" | "cosine" | undefined,
): Promise<ProposeSimilarResponse> {
  // Find in-context-similar particles across the image using the existing
  // annotations as a prior. Returns proposals only — caller commits accepted
  // ones via saveInstances (same flow as fromPoints). `method` selects the
  // seed-finding algorithm; omit to use the backend default (currently NCC).
  const body: Record<string, unknown> = {};
  if (method) body.method = method;
  const res = await trackedFetch(`${BASE_URL}/masks/${sessionId}/propose-similar`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const err = await res.text();
    throw new Error(`proposeSimilar failed (${res.status}): ${err}`);
  }
  return res.json();
}

export interface FromBoxesResponse {
  proposals: Instance[];
  rejected: { index: number; reason: string }[];
  elapsed: number;
}

export async function fromBoxes(
  sessionId: string,
  boxes: [number, number, number, number][],
  pending: Instance[],
): Promise<FromBoxesResponse> {
  // Drag-box-to-segment via SAM box prompt. Same proposal lifecycle as
  // fromPoints — caller commits accepted proposals via saveInstances.
  const res = await trackedFetch(`${BASE_URL}/masks/${sessionId}/from-boxes`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ boxes, pending }),
  });
  if (!res.ok) {
    const err = await res.text();
    throw new Error(`fromBoxes failed (${res.status}): ${err}`);
  }
  return res.json();
}

export async function fromPoints(
  sessionId: string,
  points: [number, number][],
  pending: Instance[],
): Promise<FromPointsResponse> {
  // Each click becomes a *proposal* via SAM single-point predict on the backend.
  // Nothing is persisted — caller commits accepted proposals via saveInstances.
  // `pending` is the list of not-yet-committed proposals from prior clicks in
  // this bootstrap session; the backend paints them into its dedup mask so a
  // new click on/near one is rejected instead of producing an overlap.
  const res = await trackedFetch(`${BASE_URL}/masks/${sessionId}/from-points`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ points, pending }),
  });
  if (!res.ok) {
    const err = await res.text();
    throw new Error(`fromPoints failed (${res.status}): ${err}`);
  }
  return res.json();
}

export async function uploadImage(file: File) {
  console.log("[uploadImage] uploading:", file.name);

  const form = new FormData();
  form.append("file", file);

  const res = await trackedFetch(`${BASE_URL}/images/upload`, {
    method: "POST",
    body: form,
  });

  if (!res.ok) {
    console.error("[uploadImage] failed:", res.status);
    throw new Error("Upload failed");
  }

  const data = await res.json(); 
  console.log("[uploadImage] success:", data);

  return data;
}

export async function segmentImage(
  sessionId: string,
  model: string,
  regions: Box[],
  blackout: boolean = false,
  inverseBlackout: boolean = false,
  colorize: boolean= true
) : Promise<SegmentResponse> {

  console.log("Calling [segmentImage]", sessionId, model, blackout, inverseBlackout, regions);
  

  const res = await trackedFetch(`${BASE_URL}/segment/`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      session_id: sessionId,
      model: model,
      blackout: blackout,
      inverse_blackout: inverseBlackout,
      regions: regions ,
      colorize: colorize,
    })
  });

  if (!res.ok){
    console.error("[segmentImage] failed:", res.status);
    throw new Error("Segmentation failed");
  } 
  const data= await res.json()
  console.log("[segmentImage] success:", data);
  return data;
}



export async function uploadGroundTruth(
  sessionId: string, 
  file: File,
) {
  const form = new FormData();
  form.append("file", file);
  const res = await trackedFetch(`${BASE_URL}/gt/${sessionId}`, { method: "POST", body: form });
  if (!res.ok) throw new Error("GT upload failed");
  return res.json();
}

export async function computeGTScore(
  sessionId: string, 
  regions: BlackoutRect[],
  blackout: boolean = false,
  inverseBlackout: boolean = false,
) {
  const res = await trackedFetch(`${BASE_URL}/gt/${sessionId}/compute`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      session_id: sessionId,
      regions: regions, 
      blackout: blackout, 
      inverse_blackout: inverseBlackout
    }),
  });
  if (!res.ok) throw new Error("GT compute failed");
  return res.json(); 
}


export async function getInstances(sessionId: string) {
  const res = await trackedFetch(`${BASE_URL}/masks/${sessionId}/instances`, { method: "POST" });
  if (!res.ok) throw new Error("Failed to get instances");
  return res.json();
}

export async function saveInstances(sessionId: string, instances: Instance[]) {
  const res = await trackedFetch(`${BASE_URL}/masks/${sessionId}/instances`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId, instances }),
  });
  if (!res.ok) throw new Error("Failed to save instances");
  return res.json();
}

export async function splitInstances(
  sessionId: string,
  instanceId: number,
  points: [number, number][]
): Promise<{ instances: Instance[] }> {
  const res = await trackedFetch(`${BASE_URL}/masks/${sessionId}/split`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ instance_id: instanceId, points }),
  });
  if (!res.ok) throw new Error("Split failed");
  return res.json();
}

export interface RFProposeResponse {
  proposals: Instance[];
  message?: string;
  error?: string;
  elapsed?: number;
}

export async function rfPropose(
  sessionId: string,
  topN: number = 5,
  bgScribbles: Stroke[] = [],
): Promise<RFProposeResponse> {
  const res = await trackedFetch(`${BASE_URL}/rf/propose`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId, top_n: topN, bg_scribbles: bgScribbles }),
  });
  if (!res.ok) {
    const err = await res.text();
    throw new Error(`rfPropose failed (${res.status}): ${err}`);
  }
  return res.json();
}

export async function exportSession(
  sessionId: string,
  items: string[]
): Promise<Blob> {
  const res = await trackedFetch(`${BASE_URL}/export/${sessionId}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ items }),
  });
  if (!res.ok) throw new Error("Export failed");
  return res.blob();
}
