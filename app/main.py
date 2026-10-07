import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from app.api.routes.prediction import router as prediction_router
load_dotenv()

app = FastAPI(
    title="Fraud Detection API",
    version="1.0.0"
)

frontend_url = os.getenv("FRONTEND_URL")

if not frontend_url:
    raise ValueError("FRONTEND_URL is not set")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[frontend_url],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "Hello, FastAPI"}

@app.get("/health")
def health():
    return {"status": "ok"}

app.include_router(prediction_router)