from fastapi import APIRouter, HTTPException

from app.backend.services.rag_service import run_diagnosis


router = APIRouter(
    prefix="/api/diagnosis",
    tags=["Diagnosis"]
)


@router.get("/{channel}")
def get_diagnosis(
    channel: str,
    top_k: int = 3
):

    try:

        result = run_diagnosis(
            channel=channel,
            top_k=top_k
        )

        return {
            "success": True,
            "data": result
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )