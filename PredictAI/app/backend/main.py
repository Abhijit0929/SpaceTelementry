from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.backend.api.predictions import router as predictions_router
from app.backend.api.diagnosis import router as diagnosis_router
from app.backend.api.simulation import router as simulation_router
from app.backend.api.channels import router as channels_router
from app.backend.api.health import router as health_router


app = FastAPI(
    title="PredictAI Spacecraft Telemetry API",
    description="Backend API for spacecraft telemetry anomaly detection and RAG-based diagnosis.",
    version="1.0.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(predictions_router)
app.include_router(diagnosis_router)
app.include_router(simulation_router)
app.include_router(channels_router)
app.include_router(health_router)


@app.get("/")
def root():
    return {
        "project": "PredictAI",
        "status": "online",
        "version": "1.0.0"
    }


@app.get("/api/health")
def health():
    return {
        "status": "healthy",
        "service": "PredictAI Backend"
    }


@app.get("/api/spacecrafts")
def spacecrafts():
    return {
        "spacecrafts": [
            {
                "id": "SMAP",
                "model": "random_forest_SMAP.joblib"
            },
            {
                "id": "MSL",
                "model": "random_forest_MSL.joblib"
            }
        ]
    }
