import mlflow
from dotenv import load_dotenv
import json
from src.logging_setup import setup_logging
from src.evaluate import save_model_info
import os


load_dotenv()

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

print("MLFLOW_TRACKING_URI", MLFLOW_TRACKING_URI)

logger = setup_logging()  # Initialize logging configuration

MODEL_NAME = "cricket-score-prediction"  # Centralized model name for MLflow Model Registry

def load_model_info(file_path: str) -> dict:
    """Load the model info from a JSON file."""
    try:
        with open(file_path, 'r') as file:
            model_info = json.load(file)
        logger.debug('Model info loaded from %s', file_path)
        return model_info
    except FileNotFoundError:
        logger.error('File not found: %s', file_path)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the model info: %s', e)
        raise

def register_champion_model(run_id, model_path):
    """
    Registers the run model to the central MLflow Model Registry
    """
    model_uri = f"runs:/{run_id}/{model_path}"
    print(model_uri)
    logger.info("Registering model from run_id: %s with model_uri: %s", run_id, model_uri)

    result = mlflow.register_model(model_uri, MODEL_NAME)
    logger.info(f"Model registered successfully. Version: {result.version}")
    return result.version


def main():
    model_info_path = "./models/experiments_info.json"
    model_info = load_model_info(model_info_path)
    run_id = model_info.get("run_id")
    model_path = model_info.get("model_path")

    model_version = register_champion_model(run_id=run_id, model_path=model_path)

    model_info['model_version'] = model_version
    save_model_info(model_info, model_info_path)  # Save the model info back to the JSON file

if __name__ == "__main__":
    main()