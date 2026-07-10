import pickle
import yaml
import os
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from src.logging_setup import setup_logging
from sklearn.preprocessing import OrdinalEncoder
from src.data.data_ingestion import train_test_data
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

logger = setup_logging()  # Initialize logging configuration


def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, "r") as file:
            params = yaml.safe_load(file)
        logger.debug("Parameters retrieved from %s", params_path)
        return params
    except FileNotFoundError:
        logger.error("File not found: %s", params_path)
        raise
    except yaml.YAMLError as e:
        logger.error("YAML error: %s", e)
        raise
    except Exception as e:
        logger.error("Unexpected error: %s", e)
        raise


def get_algorithm(algorithm, model_params):
    models = {
        "linear_regression": LinearRegression,
        "random_forest": RandomForestRegressor,
        "xgboost": XGBRegressor,
    }

    return models[algorithm](**model_params)


def save_model(model, file_path: str) -> None:
    """Save the trained model to a file."""
    try:
        with open(file_path, "wb") as file:
            pickle.dump(model, file)
        logger.debug("Model saved to %s", file_path)
    except Exception as e:
        logger.error("Error occurred while saving the model: %s", e)
        raise


def get_preprocessor():
    numerical_cols = [
        "current_score",
        "balls_left",
        "wickets_left",
        "current_run_rate",
    ]
    categorical_cols = ["batting_team", "bowling_team", "city"]

    return ColumnTransformer(
        transformers=[
            ("num", "passthrough", numerical_cols),
            ("cat", OrdinalEncoder(), categorical_cols),
        ]
    )


def main():
    path = os.path.join(os.getcwd(), "config", "params.yaml")
    params = load_params(path)

    algorithm = params["algorithm"]
    model_params = params[algorithm]

    logger.info("Starting model training with algorithm: %s", algorithm)

    # 3. Load your data (Assuming processed files from data_preprocessing stage)
    x_train, x_test, y_train, y_test = train_test_data()

    # Initialize and fit
    model = get_algorithm(algorithm, model_params)

    preprocessor = get_preprocessor()

    pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])

    pipeline.fit(x_train, y_train)
    # mlflow.sklearn.log_model(sk_model=pipeline, name=algorithm, signature=signature)

    # model.fit(x_train, y_train)

    logger.info("Model training completed for algorithm")

    # 5. Export the model locally as a generic placeholder file required by DVC
    save_model(pipeline, f"./models/score_prediction.pkl")


if __name__ == "__main__":
    main()
