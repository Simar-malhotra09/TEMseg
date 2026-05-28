export const BASE_URL = "http://localhost:8080";

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
  particles: {
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
  }[];
}

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
}

/* ------------------------------------------------------------------ */
/*  PyWebView bridge detection                                        */
/* ------------------------------------------------------------------ */

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

/* ------------------------------------------------------------------ */
/*  Standard API functions                                            */
/* ------------------------------------------------------------------ */

export async function getModels(): Promise<string[]> {
  const res = await fetch(`${BASE_URL}/models`);

  if (!res.ok) {
    throw new Error("Failed to fetch models");
  }

  const data = await res.json();

  if (Array.isArray(data)) {
    return data;
  }

  return data.models ?? [];
}

export async function uploadImage(file: File) {
  console.log("[uploadImage] uploading:", file.name);

  const form = new FormData();
  form.append("file", file);

  const res = await fetch(`${BASE_URL}/images/upload`, {
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
  

  const res = await fetch(`${BASE_URL}/segment/`, {
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
  const res = await fetch(`${BASE_URL}/gt/${sessionId}`, { method: "POST", body: form });
  if (!res.ok) throw new Error("GT upload failed");
  return res.json();
}

export async function computeGTScore(
  sessionId: string, 
  regions: BlackoutRect[],
  blackout: boolean = false,
  inverseBlackout: boolean = false,
) {
  const res = await fetch(`${BASE_URL}/gt/${sessionId}/compute`, {
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
  const res = await fetch(`${BASE_URL}/masks/${sessionId}/instances`, { method: "POST" });
  if (!res.ok) throw new Error("Failed to get instances");
  return res.json();
}

export async function saveInstances(sessionId: string, instances: Instance[]) {
  const res = await fetch(`${BASE_URL}/masks/${sessionId}/instances`, {
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
  const res = await fetch(`${BASE_URL}/masks/${sessionId}/split`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ instance_id: instanceId, points }),
  });
  if (!res.ok) throw new Error("Split failed");
  return res.json();
}

export async function exportSession(
  sessionId: string,
  items: string[]
): Promise<Blob> {
  const res = await fetch(`${BASE_URL}/export/${sessionId}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ items }),
  });
  if (!res.ok) throw new Error("Export failed");
  return res.blob();
}
