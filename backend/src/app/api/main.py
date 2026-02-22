from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routers import images
from app.api.routers import segment
from app.api.live_models import AvailableModels
from typing import List
import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

# parent logger for all routes
routes_logger = logging.getLogger("routes")
routes_logger.setLevel(logging.INFO)

# file handler for routes
file_handler = logging.FileHandler("routes.log")
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
file_handler.setFormatter(formatter)
routes_logger.addHandler(file_handler)


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(images.router)
app.include_router(segment.router)


@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/models", response_model=List[str])
def get_models():
    return [model.value for model in AvailableModels]
