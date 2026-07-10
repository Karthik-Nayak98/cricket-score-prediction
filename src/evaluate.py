import pickle
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.data.data_ingestion import train_test_data, load_params
import mlflow
from mlflow.models import infer_signature
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from src.logging_setup import setup_logging
from dotenv import load_dotenv

# Use the centralized pipeline logger
logger = setup_logging()  # Initialize logging configuration

client = mlflow.MlflowClient()

MODEL_NAME = (
    "cricket-score-prediction"  # Centralized model name for MLflow Model Registry
)

load_dotenv()

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


def load_model(model_path: str):
    """Load the trained model."""
    try:
        with open(model_path, "rb") as file:
            model = pickle.load(file)
        logger.debug("Model loaded from %s", model_path)
        return model
    except Exception as e:
        logger.error("Error loading model from %s: %s", model_path, e)
        raise


def eval_metrics(actual, prediction):
    """Calculate core regression metrics."""
    r2 = r2_score(actual, prediction)
    mae = mean_absolute_error(actual, prediction)
    rmse = np.sqrt(mean_squared_error(actual, prediction))
    return {"r2": r2, "mae": mae, "rmse": rmse}


def log_residual_plot(y_test, y_pred, dataset_name):
    """
    Log a Residual vs. Predicted plot as an MLflow artifact.
    This replaces the Confusion Matrix plot for regression models.
    """
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_pred, y=(y_test - y_pred), alpha=0.5, color="purple")
    plt.axhline(y=0, color="r", linestyle="--")
    plt.title(f"Residual Plot for {dataset_name}")
    plt.xlabel("Predicted Final Score")
    plt.ylabel("Residuals (Actual - Predicted)")

    # Save and log to MLflow
    plot_file_path = f'residual_plot_{dataset_name.lower().replace(" ", "_")}.png'
    plt.savefig(plot_file_path)
    mlflow.log_artifact(plot_file_path)
    plt.close()

    # # Clean up local file system
    # if os.path.exists(plot_file_path):
    #     os.remove(plot_file_path)


def save_model_info(model_info: dict, file_path: str) -> None:
    """Save the model run ID and path metadata to a JSON file."""
    try:
        with open(file_path, "w") as file:
            json.dump(model_info, file, indent=4)
        logger.debug("Model metadata run info saved to %s", file_path)
    except Exception as e:
        logger.error("Error occurred while saving model info metadata: %s", e)
        raise


def main():
    # 1. Connect to your tracking server
    mlflow.set_experiment("cricket_score_prediction")

    path = os.path.join(os.getcwd(), "config", "params.yaml")
    params = load_params(path)

    algorithm = params["algorithm"]
    model_params = params[algorithm]

    pipeline = load_model("./models/score_prediction.pkl")


    # 3. Load your data (Assuming processed files from data_preprocessing stage)
    x_train, x_test, y_train, y_test = train_test_data()

    # 4. Wrap execution inside the MLflow Run context
    with mlflow.start_run(run_name=f"{algorithm}"):

        # Log active tags
        mlflow.set_tags({"dataset": "cricsheet", "environment": "dev"})

        # Generate predictions for tracking metrics
        # preprocessor = get_preprocessor()
        # pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])


        y_train_pred = pipeline.predict(x_train)
        y_test_pred = pipeline.predict(x_test)

        # Calculate metrics using your eval_metrics dict function
        train_metrics = eval_metrics(y_train, y_train_pred)
        test_metrics = eval_metrics(y_test, y_test_pred)

        # Log active model parameters dynamically
        mlflow.log_param("model_type", algorithm)
        for param_name, param_val in model_params.items():
            mlflow.log_param(f"hp_{param_name}", param_val)

        # Log training and testing evaluation metrics
        mlflow.log_metrics(
            {
                "train_r2": train_metrics["r2"],
                "train_mae": train_metrics["mae"],
                "train_rmse": train_metrics["rmse"],
                "test_r2": test_metrics["r2"],
                "test_mae": test_metrics["mae"],
                "test_rmse": test_metrics["rmse"],
            }
        )

        # Infer model signature schema and log the actual model binary artifact
        signature = infer_signature(x_test, y_test_pred)


        mlflow.sklearn.log_model(sk_model=pipeline, name=algorithm, signature=signature)

        log_residual_plot(y_test, y_test_pred, dataset_name="cricsheet")

        # save_model_info(run_id: str, model_path: str, file_path: str, model_version: str) -> None:
        # model_info = {"run_id": run_id, "model_path": model_path, "model_version": model_version}

        model_info = {
            "run_id": mlflow.active_run().info.run_id,
            "model_path": algorithm,
        }
        save_model_info(model_info, "./models/experiments_info.json")


if __name__ == "__main__":
    main()
