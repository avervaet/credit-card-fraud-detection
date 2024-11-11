import logging
import time

import numpy as np
from fastapi import APIRouter, HTTPException
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.responses import Response

from ..training.model import CreditCardFraudModel
from .models import ModelInfo, PredictionRequest, PredictionResponse
from .monitoring import MetricsTracker

router = APIRouter()
logger = logging.getLogger(__name__)

# Create a single instance of MetricsTracker
metrics_tracker = MetricsTracker()


class ModelService:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.metadata = None
        self.load_model()

    def load_model(self):
        """Load the latest model version with timing."""
        try:
            start_time = time.time()
            model_pipeline = CreditCardFraudModel()
            self.model, self.scaler, self.metadata = model_pipeline.load_model_artifacts()

            # Track model loading time and update info
            loading_time = time.time() - start_time
            metrics_tracker.model_loading_time.set(loading_time)
            metrics_tracker.update_model_info(self.metadata)

            logger.info(f"Loaded model version: {self.metadata['version']}")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise RuntimeError("Failed to load model")

    async def predict(self, features: list[float]) -> PredictionResponse:
        """Make prediction using loaded model with metrics tracking."""
        try:
            features_array = np.array(features).reshape(1, -1)
            scaled_features = self.scaler.transform(features_array)

            prediction = int(self.model.predict(scaled_features)[0])
            probability = float(self.model.predict_proba(scaled_features)[0][1])

            # Track prediction metrics
            try:
                metrics_tracker.track_prediction(prediction, probability)
            except Exception as e:
                logger.warning(f"Failed to track prediction metrics: {e}")

            return PredictionResponse(prediction=prediction, probability=probability, version=self.metadata["version"])
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise HTTPException(status_code=500, detail="Prediction failed")


# Create global model service instance
model_service = ModelService()


@router.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Endpoint for making predictions."""
    try:
        with metrics_tracker.track_request("predict"):
            return await model_service.predict(request.features)
    except Exception as e:
        logger.error(f"Prediction endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/model/info", response_model=ModelInfo)
async def get_model_info():
    """Get information about the currently loaded model."""
    try:
        with metrics_tracker.track_request("model_info"):
            return ModelInfo(
                version=model_service.metadata["version"],
                metrics=model_service.metadata["metrics"],
                timestamp=model_service.metadata["timestamp"],
            )
    except Exception as e:
        logger.error(f"Model info endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics")
async def metrics():
    """Endpoint for Prometheus metrics."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
