from fastapi import FastAPI
from app.api.routers import images

app = FastAPI()

app.include_router(images.router)

@app.get("/")
async def root():
    return {"message": "Hello World"}
