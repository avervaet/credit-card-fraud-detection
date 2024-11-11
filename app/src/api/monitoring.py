# src/api/monitoring.py
from prometheus_client import Counter, Histogram, Gauge, Info
from typing import Dict
import time
import logging

logger = logging.getLogger(__name__)

class MetricsTracker:
    def __init__(self):
        # Request metrics
        self.request_count = Counter(
            'ml_request_total',
            'Total ML prediction requests',
            ['endpoint', 'status']
        )
        
        self.request_latency = Histogram(
            'ml_request_latency_seconds',
            'Request latency in seconds',
            ['endpoint'],
            buckets=(0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0)
        )
        
        # Model metrics
        self.prediction_distribution = Counter(
            'ml_prediction_class_total',
            'Distribution of predictions',
            ['prediction_class']
        )
        
        self.prediction_probability = Histogram(
            'ml_prediction_probability',
            'Distribution of prediction probabilities',
            buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
        )
        
        # Model info
        self.model_info = Info('ml_model', 'ML model information')
        
        # System metrics
        self.model_loading_time = Gauge(
            'ml_model_loading_seconds',
            'Time taken to load the model'
        )

    def track_request(self, endpoint: str):
        """Context manager to track request latency and count."""
        class RequestTracker:
            def __init__(self, metrics, endpoint):
                self.metrics = metrics
                self.endpoint = endpoint
                self.start_time = None
                
            def __enter__(self):
                self.start_time = time.time()
                return self
                
            def __exit__(self, exc_type, exc_val, exc_tb):
                duration = time.time() - self.start_time
                self.metrics.request_latency.labels(endpoint=self.endpoint).observe(duration)
                status = "error" if exc_type else "success"
                self.metrics.request_count.labels(endpoint=self.endpoint, status=status).inc()
                
        return RequestTracker(self, endpoint)

    def track_prediction(self, prediction: int, probability: float) -> None:
        """
        Track prediction distribution and probabilities.
        
        Args:
            prediction (int): The predicted class (0 or 1)
            probability (float): The prediction probability
            
        Raises:
            ValueError: If prediction is not 0 or 1 or if probability is not between 0 and 1
        """
        try:
            # Validate inputs
            if prediction not in (0, 1):
                raise ValueError(f"Invalid prediction value: {prediction}. Must be 0 or 1.")
            
            if not 0 <= probability <= 1:
                raise ValueError(f"Invalid probability value: {probability}. Must be between 0 and 1.")
            
            # Track metrics
            self.prediction_distribution.labels(prediction_class=str(prediction)).inc()
            self.prediction_probability.observe(probability)
            
        except Exception as e:
            logger.error(f"Failed to track prediction metrics: {str(e)}")
            raise

    def update_model_info(self, metadata: Dict):
        """Update model information metrics."""
        try:
            self.model_info.info({
                'version': str(metadata['version']),
                'timestamp': str(metadata['timestamp']),
                'accuracy': str(metadata['metrics'].get('accuracy', 'N/A')),
                'roc_auc': str(metadata['metrics'].get('roc_auc', 'N/A'))
            })
        except Exception as e:
            logger.error(f"Failed to update model info metrics: {str(e)}")