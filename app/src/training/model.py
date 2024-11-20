import json
import logging
from datetime import datetime
from pathlib import Path

import joblib
import mlflow
import mlflow.xgboost
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.utils.config import Config
from xgboost import XGBClassifier


class CreditCardFraudModel:
    def __init__(self):
        """Initialize the model with configuration."""
        self.config = Config().config
        self.model_version = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.model_artifacts_path = Path(self.config["paths"]["model_artifacts"])
        self.model_artifacts_path.mkdir(parents=True, exist_ok=True)

        # Set up logging
        self.setup_logging()

        # Initialize MLflow
        mlflow.set_tracking_uri(self.config["mlflow"]["tracking_uri"])
        mlflow.set_experiment(self.config["mlflow"]["experiment_name"])

    def setup_logging(self):
        """Set up logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler(f"logs/training_{self.model_version}.log"), logging.StreamHandler()],
        )
        self.logger = logging.getLogger(__name__)

    def load_data(self):
        """Load and preprocess the dataset."""
        self.logger.info("Loading dataset...")
        data = pd.read_csv(self.config["paths"]["data"])

        # Separate features and target
        X = data.drop(columns=["Class"])
        y = data["Class"]

        return X, y

    def preprocess_data(self, X, y):
        """Preprocess the data including scaling and train-test split."""
        self.logger.info("Preprocessing data...")

        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled,
            y,
            test_size=self.config["training"]["test_size"],
            random_state=self.config["training"]["random_state"],
            stratify=y,
        )

        # Apply SMOTE
        smote = SMOTE(random_state=self.config["training"]["random_state"])
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

        return X_train_res, X_test, y_train_res, y_test

    def train_model(self, X_train, X_test, y_train, y_test):
        """Train the model and log metrics with MLflow."""
        self.logger.info("Training model...")

        with mlflow.start_run(run_name=f"training_{self.model_version}"):
            # Log parameters
            mlflow.log_params(self.config["model_params"])

            # Initialize and train model
            model = XGBClassifier(
                **self.config["model_params"], scale_pos_weight=len(y_train[y_train == 0]) / len(y_train[y_train == 1])
            )

            eval_set = [(X_train, y_train), (X_test, y_test)]
            model.fit(X_train, y_train, eval_set=eval_set, verbose=True)

            # Generate predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]

            # Calculate metrics
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            metrics = {
                "sensitivity": tp / (tp + fn),
                "specificity": tn / (tn + fp),
                "roc_auc": roc_auc_score(y_test, y_pred_proba),
            }

            # Log metrics
            self.logger.info(f"Model metrics: {metrics}")
            mlflow.log_metrics(metrics)

            # Log model
            mlflow.xgboost.log_model(model, "model")

            return model, metrics

    def save_artifacts(self, model, metrics):
        """Save model artifacts and metadata."""
        artifacts_dir = self.model_artifacts_path / self.model_version
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        model_path = artifacts_dir / "model.joblib"
        joblib.dump(model, model_path)

        # Save scaler
        scaler_path = artifacts_dir / "scaler.joblib"
        joblib.dump(self.scaler, scaler_path)

        # Save metadata
        metadata = {
            "version": self.model_version,
            "metrics": metrics,
            "config": self.config,
            "timestamp": datetime.now().isoformat(),
        }

        metadata_path = artifacts_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)

        self.logger.info(f"Saved model artifacts to {artifacts_dir}")

    def load_model_artifacts(self, version=None):
        """Load model artifacts for a specific version."""
        if version is None:
            # Load latest version
            versions = [d for d in self.model_artifacts_path.iterdir() if d.is_dir()]
            if not versions:
                raise ValueError("No model versions found")
            version_path = max(versions, key=lambda x: x.name)
        else:
            version_path = self.model_artifacts_path / version

        if not version_path.exists():
            raise ValueError(f"Model version {version} not found")

        # Load artifacts
        model = joblib.load(version_path / "model.joblib")
        scaler = joblib.load(version_path / "scaler.joblib")

        with open(version_path / "metadata.json") as f:
            metadata = json.load(f)

        return model, scaler, metadata

    def run_training_pipeline(self):
        """Run the complete training pipeline."""
        self.logger.info(f"Starting training pipeline for version {self.model_version}")

        # Load and preprocess data
        X, y = self.load_data()
        X_train, X_test, y_train, y_test = self.preprocess_data(X, y)

        # Train model
        model, metrics = self.train_model(X_train, X_test, y_train, y_test)

        # Save artifacts
        self.save_artifacts(model, metrics)

        self.logger.info("Training pipeline completed successfully")
        return model, metrics


def main():
    # Run pipeline
    model_pipeline = CreditCardFraudModel()
    model, metrics = model_pipeline.run_training_pipeline()


if __name__ == "__main__":
    main()
