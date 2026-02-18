from typing import Annotated
from fastapi import APIRouter, File, UploadFile

router = APIRouter()

@router.post("/files/")
async def create_file(file: Annotated[bytes | None, File()] = None):
    if not file:
        return {"message": "No file sent"}
    return {"file_size": len(file)}

'''
User sends an image file over this endpoint.
file is a 
'''
@router.post("/uploadfile/")
async def create_upload_file(file: UploadFile | None = None):
    if not file:
        return {"message": "No upload file sent"}
    return {
            "filename": file.filename, 
            "content_type": file.content_type}
