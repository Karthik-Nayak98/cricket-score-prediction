import mlflow
import json
from logging_setup import setup_logging
import os
from dotenv import load_dotenv

load_dotenv()

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

client = mlflow.MlflowClient()
logger = setup_logging()  # Initialize logging configuration

MODEL_NAME = "cricket-score-prediction"


def load_model_info(file_path: str) -> dict:
    """Load the model info from a JSON file."""
    try:
        with open(file_path, "r") as file:
            model_info = json.load(file)
        logger.debug("Model info loaded from %s", file_path)
        return model_info
    except FileNotFoundError:
        logger.error("File not found: %s", file_path)
        raise
    except Exception as e:
        logger.error("Unexpected error occurred while loading the model info: %s", e)
        raise


def get_model_metrics(run_id: str):

    model_run = client.get_run(run_id)

    return model_run.data.metrics


def promote_model_to_champion(model_version):

    client.set_registered_model_alias(
        name=MODEL_NAME, alias="champion", version=model_version
    )
    logger.info(
        f"Model version {model_version} promoted to champion for model {MODEL_NAME}."
    )


def main():
    try:
        model_info = load_model_info("./models/experiments_info.json")

        candidate_run_id = model_info.get("run_id")
        candidate_model_version = model_info.get("model_version")

        logger.info(mlflow.get_tracking_uri())

        champion_model = client.get_model_version_by_alias(MODEL_NAME, "champion")

        model_info = load_model_info("./models/experiments_info.json")

        candidate_run_id = model_info.get("run_id")
        candidate_model_version = model_info.get("model_version")

        candidate_model_metrics = get_model_metrics(candidate_run_id)

        champion_model_metrics = get_model_metrics(champion_model.run_id)

        if champion_model_metrics.get(
            "test_rmse", float("inf")
        ) > candidate_model_metrics.get("test_rmse", float("inf")):
            logger.info(
                "Candidate model has better performance. Promoting to champion."
            )
            promote_model_to_champion(model_version=candidate_model_version)
        else:
            logger.info(
                "Champion model has better or equal performance. No promotion needed."
            )

    except Exception as _:
        logger.info("No champion model found.")
        promote_model_to_champion(model_version=candidate_model_version)


if __name__ == "__main__":
    main()
