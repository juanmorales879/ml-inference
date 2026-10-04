"""
ML Inference API - Main Application

FastAPI application for serving ML model predictions.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import structlog
from typing import Dict, List

from src.models.classifier import ImageClassifier
from src.preprocessing.image import preprocess_image

# Configure logging
logger = structlog.get_logger()

# Initialize FastAPI app
app = FastAPI(
    title="ML Inference API",
    description="REST API for image classification",
    version="0.1.0"
)

# Initialize classifier (will be loaded on startup)
classifier = None


@app.on_event("startup")
async def startup_event():
    """Load model on application startup."""
    global classifier
    logger.info("Loading model...")
    classifier = ImageClassifier()
    await classifier.load_model()
    logger.info("Model loaded successfully")


@app.get("/")
async def root():
    """Root endpoint - API information."""
    return {
        "name": "ML Inference API",
        "version": "0.1.0",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    if classifier is None or not classifier.is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded")

    return {
        "status": "healthy",
        "model_loaded": True
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> JSONResponse:
    """
    Predict image classification.

    Args:
        file: Uploaded image file

    Returns:
        JSON response with predictions
    """
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="File must be an image"
        )

    try:
        # Read and preprocess image
        image_bytes = await file.read()
        processed_image = await preprocess_image(image_bytes)

        # Get predictions
        predictions = await classifier.predict(processed_image)

        logger.info(
            "Prediction made",
            filename=file.filename,
            top_class=predictions[0]["class"]
        )

        return JSONResponse(content={
            "success": True,
            "predictions": predictions
        })

    except Exception as e:
        logger.error("Prediction failed", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@app.get("/models")
async def list_models() -> Dict[str, str]:
    """List available models."""
    if classifier is None:
        return {"models": []}

    return {
        "current_model": classifier.model_name,
        "version": classifier.version
    }
