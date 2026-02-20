const BASE_URL = "http://localhost:8000";

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
