export const BASE_URL = "http://localhost:8000";

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

  const data = await res.json(); //{ session_id, filename } }
  console.log("[uploadImage] success:", data);

  return data;
}

export async function SegmentImage(sessionId: string) {
  console.log("[SegmentImage] trying to segment image with session id:", sessionId);

  const res = await fetch(`${BASE_URL}/segment`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      session_id: sessionId,
      model: "yolosam",
    }),
  });

  if (!res.ok) throw new Error("Segmentation failed");

  return res.json();
}

