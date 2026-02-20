from fastapi import FastAPI
from app.api.routers import images
from app.api.logger import get_route_logger


app = FastAPI()
global_logger = get_route_logger()
app.include_router(images.router)


@app.get("/")
async def root():
    return {"message": "Hello World"}
