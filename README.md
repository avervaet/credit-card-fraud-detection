# Credit Card Fraud Detection System

## Problem Statement

### Business Context
SaaS vendors handling large volumes of credit card transactions must proactively identify and prevent fraudulent activities to safeguard both customers and merchants. The challenge is to develop a reliable and efficient fraud detection system that accurately distinguishes between legitimate and suspicious transactions in real time, minimizing false positives while swiftly blocking potentially fraudulent activities.

### Problem Definition
The goal is to develop and deploy a system that can:

- Accurately distinguish between legitimate and fraudulent credit card transactions in real time
- Process high transaction volumes with minimal latency
- Maintain a low false positive rate to avoid disrupting legitimate customer transactions
- Adapt to evolving fraud patterns

### Dataset

We used the Kaggle Credit Card Fraud Detection dataset hosted on Kaggle: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/data

## Running this Project

### Downloading the dataset
Dowload the dataset from Kaggle: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/data, unzip and store it under data/raw/ at the root of this project.

### Training the Model
```bash
cd app
python -m src.training.model
```

### Deploying in Production

You can deploy this project using Docker Compose, located at the root.

```bash
docker-compose up -d --build
```

### Sending Requests

#### 1. Checking the Model Version

```bash
curl "http://localhost:8000/api/v1/model/info"
```

You should receive something like this:
```
{"version":"20241111_110829","metrics":{"sensitivity":0.8040540540540541,"specificity":0.9995193153174278,"roc_auc":0.9738266873474096},"timestamp":"2024-11-11T11:08:33.725403"}
```

#### 2. Sending a Prediction Request

```bash
curl -X POST "http://localhost:8000/api/v1/predict" \
     -H "Content-Type: application/json" \
     -d '{"features": [0, -1.3598071336738, -0.0727811733098497, 2.53634673796914, 1.37815522427443, -0.338320769942518, 0.462387777762292, 0.239598554061257, 0.0986979012610507, 0.363786969611213, 0.0907941719789316, -0.551599533260813, -0.617800855762348, -0.991389847235408, -0.311169353699879, 1.46817697209427, -0.470400525259478, 0.207971241929242, 0.0257905801985591, 0.403992960255733, 0.251412098239705, -0.018306777944153, 0.277837575558899, -0.110473910188767, 0.0669280749146731, 0.128539358273528, -0.189114843888824, 0.133558376740387, -0.0210530534538215, 149.62]}'
```

The response will look like this:
```
{"prediction":0,"probability":1.228905330208363e-05,"version":"20241111_110829"}
```

## Monitoring
Access the monitoring interfaces:

- **Prometheus:** http://localhost:9090
- **Grafana:** http://localhost:3000 (admin/admin)
- **Raw metrics:** http://localhost:8000/api/v1/metrics

## About Our Approach

### Technical Challenges

The development of this system encountered several key technical challenges:

#### 1. Class Imbalance

The dataset exhibits extreme imbalance, with only 0.172% fraudulent transactions (492 fraud cases out of 284,807 transactions). Standard machine learning approaches may fail to detect fraud due to this imbalance.

#### 2. Real-time Processing Requirements

Decisions must be made within milliseconds to maintain a smooth customer experience. The system must handle peak transaction volumes without degradation.

#### 3. Feature Privacy and Dimensionality

The dataset contains 28 PCA-transformed features (V1-V28) due to confidentiality. Only 'Time' and 'Amount' are non-transformed features, making it challenging to interpret business logic. We chose not to focus extensively on feature engineering, as this is a toy dataset.

#### 4. Production Deployment Challenges

- Need for robust monitoring and alerting systems
- Regular model retraining and validation
- Version control and rollback capabilities

### Developing the Anomaly Detection Model

To build a reliable fraud detection model, we implemented a comprehensive training pipeline with the following steps:

1. **Configuration and Initialization**:
   - The `CreditCardFraudModel` class initializes with a configuration file specifying parameters for paths, training settings, and model hyperparameters.
   - The model version is timestamped, and a structured directory is created for storing artifacts, enabling model version control.

2. **Data Loading and Preprocessing**:
   - The model loads the credit card transaction dataset, with `Class` as the target variable indicating fraud (1) or non-fraud (0).
   - Features are scaled using `StandardScaler` to improve model performance.
   - We split the dataset into training and testing sets, ensuring a stratified split to maintain class balance proportions.
   - Given the class imbalance, we applied **Synthetic Minority Oversampling Technique (SMOTE)** to balance the training data, creating synthetic samples of the minority (fraudulent) class.

3. **Model Training and Hyperparameter Logging**:
   - An **XGBoost classifier** is trained with `scale_pos_weight` adjustment to address class imbalance.
   - Using **MLflow**, the training process logs model parameters, evaluation metrics, and artifacts, ensuring transparency and reproducibility.
   - We evaluate the model on the test set, calculating key metrics such as **sensitivity**, **specificity**, and **ROC AUC** to gauge performance on fraud detection.

4. **Artifact Management**:
   - Upon completion, the trained model and scaler are saved as artifacts using `joblib`, allowing for quick loading during inference or retraining.
   - Additionally, metadata including model version, metrics, and configuration are saved in a `metadata.json` file. This comprehensive artifact management facilitates traceability and version control for production deployments.

5. **Model Deployment**:
   - The model can be deployed as a REST API, and version control allows for quick rollback or model updates as fraud patterns evolve.
   - Monitoring and metrics interfaces such as **Prometheus** and **Grafana** provide insights into model performance and operational metrics in production.

This pipeline enables efficient model training, versioned deployment, and robust monitoring, providing a foundation for scalable fraud detection in real-time transaction environments.

> **Note**: Given that this is a dummy dataset, we focused on simplicity and efficiency over exhaustive fine-tuning or complex approaches. Instead, we opted to implement **proven, state-of-the-art techniques**—specifically, **SMOTE** for handling class imbalance and **XGBoost** for robust classification performance.

### Deployment and Monitoring

In the production environment, this credit card fraud detection system is deployed using **Docker Compose** to ensure consistency, scalability, and ease of deployment. Docker Compose allows us to define and manage all the necessary services in isolated containers, ensuring that dependencies, configurations, and the application stack are consistent across different environments. This setup includes not only the model’s API but also monitoring and logging services to maintain robust operational oversight.

To monitor the model’s performance and system health, we integrated **Grafana** and **Prometheus**. Prometheus is configured to collect real-time metrics on model latency, response times, and throughput, while Grafana provides a customizable dashboard to visualize these metrics, making it easy to detect and address potential issues proactively. Additionally, all model predictions and errors are logged, giving us a comprehensive view of both normal and anomalous behavior in production.

## Going Further

We had really limited time to work on this project (~6 hours), so we focused on having something functional that can easily be scaled.

In the future, we would like to add the following functionality:

- **Analysis of Feature Distribution:** Statistical comparison (e.g., using a Kolmogorov-Smirnov or Chi-squared test) between the feature distributions of new data and those of the training data. If the distributions differ significantly, this indicates that the model may no longer generalize well.
- **Monitoring Performance Drift:** Continuous evaluation of predictions—collect the performance metrics of predictions in production to measure deviations from historical performance. A drop in performance, such as a decrease in accuracy or an increase in mean squared error, can indicate performance drift.
- **Blue/Green or Canary Deployment:** Before replacing the model in production, a blue/green or canary deployment strategy can be used to test the retrained model on a portion of the requests, allowing the switch to the new model only if its performance exceeds that of the current model.
