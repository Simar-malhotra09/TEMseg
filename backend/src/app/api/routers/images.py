from typing import Annotated
import uuid, shutil
from pathlib import Path 
from fastapi import APIRouter, File, UploadFile
from fastapi.responses import FileResponse
import logging
import numpy as np 
import cv2 as cv


router = APIRouter(prefix="/images")
logger = logging.getLogger("routes.images")
SESSIONS_DIR = Path("sessions")

'''
User sends an image file over this endpoint.

UploadFile has the following attributes:

filename: A str with the original file name that was uploaded (e.g. myimage.jpg).
content_type: A str with the content type (MIME type / media type) (e.g. image/jpeg).
file: A SpooledTemporaryFile (a file-like object). 
(all async methods): 
write(data): Writes data (str or bytes) to the file.
read(size): Reads size (int) bytes/characters of the file.
seek(offset): Goes to the byte position offset (int) in the file.
close(): Closes the file.

'''

@router.get("/{session_id}/preview")
async def get_preview(session_id: str):
    path = SESSIONS_DIR / session_id / "original_preview.png"
    if not path.exists():
        return {"error": "No preview found"}
    return FileResponse(path)

@router.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    session_id="test"
    # session_id = str(uuid.uuid4())
    session_dir = SESSIONS_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    dest = session_dir / f"org_{file.filename}"
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    logger.info("Upload — session: %s, filename: %s", session_id, file.filename)

    is_npy = file.filename.endswith(".npy")
    preview_url = f"/images/{session_id}/file"

    if is_npy:
        arr = np.load(str(dest))
        # normalize to full 0-255 range for display
        arr_min, arr_max = arr.min(), arr.max()
        if arr_max > arr_min:
            display = ((arr - arr_min) / (arr_max - arr_min) * 255).astype("uint8")
        else:
            display = np.zeros_like(arr, dtype="uint8")
        preview_path = session_dir / "original_preview.png"
        cv.imwrite(str(preview_path), display)
        preview_url = f"/images/{session_id}/preview"

    return {
        "session_id": session_id,
        "filename": file.filename,
        "preview_url": preview_url,
    }


''' 
Send response back to user for a given session id. 
Just sends the first file for now.
'''
@router.get("/{session_id}/file")
async def get_image(session_id: str):
    session_dir = SESSIONS_DIR / session_id
    files = list(session_dir.glob("*"))
    if files is None:
        return {"No file exits for session id:": session_id}
    logger.info(
        "Tried to get an image. session_id: %s, filename: %s",
        session_id,
        file.filename,
    )
    return FileResponse(files[0])

@router.get("/{session_id}/mask")
async def get_mask(session_id: str):
    mask_path = SESSIONS_DIR / session_id / "mask.png"
    return FileResponse(mask_path)


@router.post("/{session_id}/gt")
async def upload_ground_truth(session_id:str, file:UploadFile=File(...)):
    session_dir = SESSIONS_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    
    dest = session_dir / file.filename
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    logger.info(
        "Tried to upload an ground truth for session_id: %s, filename: %s",
        session_id,
        file.filename,
    )
    return {"session_id": session_id, "filename": file.filename}

