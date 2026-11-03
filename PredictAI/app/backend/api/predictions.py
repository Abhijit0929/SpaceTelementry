from fastapi import APIRouter, HTTPException

from app.backend.services.rf_service import run_prediction


router = APIRouter(
    prefix="/api/predictions",
    tags=["Predictions"]
)


@router.get("/{channel}")
def get_predictions(channel: str):

    try:

        result = run_prediction(channel)

        return {
            "success": True,
            "data": result
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )