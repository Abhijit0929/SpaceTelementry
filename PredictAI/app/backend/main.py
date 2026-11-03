from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware

from app.backend.api.predictions import router as predictions_router
from app.backend.api.diagnosis import router as diagnosis_router
from app.backend.api.simulation import router as simulation_router
from app.backend.api.channels import router as channels_router
from app.backend.api.health import router as health_router

from app.backend.auth.routes import router as auth_router
from app.backend.auth.security import get_current_user
from app.backend.auth.bootstrap import initialize_auth


# --------------------------------------------------
# FASTAPI APPLICATION
# --------------------------------------------------

app = FastAPI(
    title="PredictAI Spacecraft Telemetry API",
    description=(
        "Backend API for spacecraft telemetry "
        "anomaly detection and RAG-based diagnosis."
    ),
    version="1.0.0",
)


# --------------------------------------------------
# CORS CONFIGURATION
# --------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------
# DATABASE INITIALIZATION
# --------------------------------------------------

@app.on_event("startup")
def startup():

    initialize_auth()


# --------------------------------------------------
# AUTHENTICATION ROUTER
# --------------------------------------------------

app.include_router(auth_router)


# --------------------------------------------------
# PROTECTED API ROUTERS
# --------------------------------------------------

protected = [
    Depends(get_current_user)
]


app.include_router(
    predictions_router,
    dependencies=protected
)

app.include_router(
    diagnosis_router,
    dependencies=protected
)

app.include_router(
    simulation_router,
    dependencies=protected
)

app.include_router(
    channels_router,
    dependencies=protected
)

app.include_router(
    health_router,
    dependencies=protected
)


# --------------------------------------------------
# PUBLIC ROOT ENDPOINT
# --------------------------------------------------

@app.get("/")
def root():

    return {
        "project": "PredictAI",
        "status": "online",
        "version": "1.0.0"
    }


# --------------------------------------------------
# PUBLIC BACKEND STATUS
# --------------------------------------------------

@app.get("/api/health")
def health():

    return {
        "status": "healthy",
        "service": "PredictAI Backend"
    }


# --------------------------------------------------
# SPACECRAFT INFORMATION
# --------------------------------------------------

@app.get(
    "/api/spacecrafts",
    dependencies=protected
)
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