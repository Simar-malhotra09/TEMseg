export const BASE_URL = "http://localhost:8000";

// These are mirrors of the pydantic model the server defines
export interface Box {
  id: string;
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface StatsResult {
  particle_count: number;
  avg_size: number;
  avg_circularity: number;
  coverage: number;
}

export interface SegmentResponse {
  model: string;
  mask_url: string;
  metadata: Record<string, any>;
  stats: StatsResult;
  time_elapsed: number;
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
}

export interface Instance {
  id: number;
  contour: [number, number][];
  bbox: { x: number; y: number; w: number; h: number };
  area: number;
}

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

  const data = await res.json(); //{ session_id, filename, preview_url} }
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
  regions: any[],
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

export async function saveInstances(sessionId: string, instances: any[]) {
  const res = await fetch(`${BASE_URL}/masks/${sessionId}/instances`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId, instances }),
  });
  if (!res.ok) throw new Error("Failed to save instances");
  return res.json();
}

