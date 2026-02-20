from typing import Annotated
import uuid, shutil
from pathlib import Path 
from fastapi import APIRouter, File, UploadFile
from fastapi.responses import FileResponse
import logging



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


@router.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    session_id = str(uuid.uuid4())
    session_dir = SESSIONS_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    
    dest = session_dir / file.filename
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    
    logger.info(
        "Tried to upload an image. session_id: %s, filename: %s",
        session_id,
        file.filename,
    )
    return {"session_id": session_id, "filename": file.filename}


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
