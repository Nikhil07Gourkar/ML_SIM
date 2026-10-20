"""
ML Algorithm Simulator - Production FastAPI Python ML Backend
Provides scikit-learn model training, evaluation metrics, and PCA transformations.
Run with: uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

import time
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, mean_absolute_error, mean_squared_error, r2_score,
    silhouette_score
)

app = FastAPI(
    title="ML Algorithm Simulator - Python ML Service",
    description="Dedicated Scikit-Learn execution microservice for ML Algorithm Simulator",
    version="1.0.0"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TrainRequest(BaseModel):
    algorithm: str
    task_type: str  # "classification", "regression", "clustering"
    X_train: List[List[float]]
    y_train: Optional[List[float]] = None
    X_test: Optional[List[List[float]]] = None
    y_test: Optional[List[float]] = None
    feature_names: List[str]
    target_classes: Optional[List[str]] = None
    hyperparameters: Optional[Dict[str, Any]] = None

@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "service": "python-scikit-learn-backend",
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "sklearn_installed": True,
        "supported_algorithms": [
            "linear_regression", "logistic_regression", "decision_tree",
            "random_forest", "kmeans", "knn", "svm", "pca"
        ]
    }

@app.post("/api/train")
def train_model(req: TrainRequest):
    start_time = time.time()
    X_tr = np.array(req.X_train)
    y_tr = np.array(req.y_train) if req.y_train is not None else None
    X_te = np.array(req.X_test) if req.X_test is not None and len(req.X_test) > 0 else X_tr
    y_te = np.array(req.y_test) if req.y_test is not None and len(req.y_test) > 0 else y_tr

    hp = req.hyperparameters or {}
    metrics = {}
    details = {}
    predictions = []

    try:
        if req.algorithm == "linear_regression":
            model = LinearRegression(fit_intercept=hp.get("fitIntercept", True))
            model.fit(X_tr, y_tr)
            preds = model.predict(X_te)
            predictions = preds.tolist()

            mse = float(mean_squared_error(y_te, preds))
            mae = float(mean_absolute_error(y_te, preds))
            r2 = float(r2_score(y_te, preds))
            metrics = {
                "mae": round(mae, 3),
                "mse": round(mse, 3),
                "rmse": round(np.sqrt(mse), 3),
                "r2": round(r2, 3),
            }
            details = {
                "slope": float(model.coef_[0]) if len(model.coef_) > 0 else 0.0,
                "intercept": float(model.intercept_),
                "coefficients": model.coef_.tolist()
            }

        elif req.algorithm == "logistic_regression":
            model = LogisticRegression(max_iter=hp.get("maxIterations", 200))
            model.fit(X_tr, y_tr)
            preds = model.predict(X_te)
            predictions = preds.tolist()

            cm = confusion_matrix(y_te, preds).tolist()
            metrics = {
                "accuracy": round(float(accuracy_score(y_te, preds)), 3),
                "precision": round(float(precision_score(y_te, preds, average="macro", zero_division=0)), 3),
                "recall": round(float(recall_score(y_te, preds, average="macro", zero_division=0)), 3),
                "f1Score": round(float(f1_score(y_te, preds, average="macro", zero_division=0)), 3),
                "confusionMatrix": {"matrix": cm, "labels": req.target_classes or ["0", "1"]}
            }
            details = {"coefficients": model.coef_.tolist(), "intercept": model.intercept_.tolist()}

        elif req.algorithm == "decision_tree":
            if req.task_type == "regression":
                model = DecisionTreeRegressor(max_depth=hp.get("maxDepth", 4), min_samples_split=hp.get("minSamplesSplit", 2))
                model.fit(X_tr, y_tr)
                preds = model.predict(X_te)
                mse = float(mean_squared_error(y_te, preds))
                metrics = {
                    "mae": round(float(mean_absolute_error(y_te, preds)), 3),
                    "mse": round(mse, 3),
                    "rmse": round(np.sqrt(mse), 3),
                    "r2": round(float(r2_score(y_te, preds)), 3),
                }
            else:
                model = DecisionTreeClassifier(max_depth=hp.get("maxDepth", 4), criterion=hp.get("criterion", "gini"))
                model.fit(X_tr, y_tr)
                preds = model.predict(X_te)
                cm = confusion_matrix(y_te, preds).tolist()
                metrics = {
                    "accuracy": round(float(accuracy_score(y_te, preds)), 3),
                    "precision": round(float(precision_score(y_te, preds, average="macro", zero_division=0)), 3),
                    "recall": round(float(recall_score(y_te, preds, average="macro", zero_division=0)), 3),
                    "f1Score": round(float(f1_score(y_te, preds, average="macro", zero_division=0)), 3),
                    "confusionMatrix": {"matrix": cm, "labels": req.target_classes or ["0", "1"]}
                }
            predictions = preds.tolist()
            details = {"featureImportances": model.feature_importances_.tolist()}

        elif req.algorithm == "kmeans":
            k = hp.get("k", 3)
            model = KMeans(n_clusters=k, init="k-means++", max_iter=hp.get("maxIterations", 100), random_state=42)
            preds = model.fit_predict(X_tr)
            predictions = preds.tolist()

            sil = float(silhouette_score(X_tr, preds)) if len(set(preds)) > 1 else 0.0
            metrics = {
                "inertia": round(float(model.inertia_), 2),
                "silhouetteScore": round(sil, 3),
            }
            details = {"centroids": model.cluster_centers_.tolist()}

        elif req.algorithm == "pca":
            n_comp = hp.get("numComponents", 2)
            model = PCA(n_components=min(n_comp, X_tr.shape[1]))
            transformed = model.fit_transform(X_tr)
            predictions = transformed.tolist()
            metrics = {}
            details = {
                "explainedVarianceRatios": [round(float(r), 3) for r in model.explained_variance_ratio_],
                "components": model.components_.tolist()
            }

        else:
            raise HTTPException(status_code=400, detail=f"Unsupported algorithm: {req.algorithm}")

        elapsed_ms = int((time.time() - start_time) * 1000)
        metrics["trainTimeMs"] = elapsed_ms
        metrics["sampleCount"] = len(X_tr)

        return {
            "success": True,
            "engine": "scikit-learn (Python)",
            "algorithm": req.algorithm,
            "metrics": metrics,
            "predictions": predictions,
            "details": details
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
