from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.utils import (
    validate_upload,
    save_upload_file,
    cleanup_file,
    InvalidFileError,
    EmptyFileError,
    CorruptedFileError,
    logger,
)
from app.preprocessing import preprocess_pdf
from app.predictor import run_prediction
from app.model_loader import get_model

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="AI-powered Brain Tumor Diagnosis Support API using Bio_ClinicalBERT "
                "for structured information extraction from radiology reports.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    logger.info("Starting NeuroX ClinicalBERT API...")
    try:
        get_model()
        logger.info("Model preloaded successfully.")
    except Exception as e:
        logger.error(f"Failed to preload model: {e}")


@app.get("/", tags=["Health"])
async def root():
    return {
        "success": True,
        "message": f"{settings.APP_NAME} v{settings.APP_VERSION} is running.",
        "docs": "/docs",
    }


@app.get("/health", tags=["Health"])
async def health_check():
    return {"success": True, "status": "ok"}


@app.post("/predict", tags=["Prediction"])
async def predict(file: UploadFile = File(...)):
    saved_path = None
    try:
        validate_upload(file)
        saved_path = save_upload_file(file)

        cleaned_text = preprocess_pdf(saved_path)

        result = run_prediction(cleaned_text)

        return JSONResponse(content=result, status_code=200)

    except InvalidFileError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except EmptyFileError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except CorruptedFileError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Unexpected error during prediction")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    finally:
        if saved_path:
            cleanup_file(saved_path)


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"success": False, "error": exc.detail},
    )
