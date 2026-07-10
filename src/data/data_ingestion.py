import os
from sklearn.model_selection import GroupShuffleSplit
import yaml
import pandas as pd
from src.logging_setup import setup_logging
from sklearn.preprocessing import OrdinalEncoder
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


def load_data(data_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(data_path)
        logger.debug("Data loaded from %s", data_path)
        return df
    except FileNotFoundError:
        logger.error("File not found: %s", data_path)
        raise
    except pd.errors.EmptyDataError:
        logger.error("No data: %s is empty", data_path)
        raise
    except pd.errors.ParserError as e:
        logger.error("Parsing error: %s", e)
        raise
    except Exception as e:
        logger.error("Unexpected error: %s", e)
        raise


def get_preprocessor(numerical_cols, categorical_cols):
    return ColumnTransformer(
        transformers=[
            ("num", "passthrough", numerical_cols),
            ("cat", OrdinalEncoder(), categorical_cols),
        ]
    )


def train_test_data():

    params = load_params("./config/params.yaml")
    test_size = params["data_ingestion"]["test_size"]

    df = load_data("./data/match_summary.csv")

    numerical_cols = [
        # "match_id",
        "current_score",
        # "runs",
        "balls_left",
        "wickets_left",
        # "last_5_overs_runs",
        "current_run_rate",
    ]

    categorical_cols = ["batting_team", "bowling_team", "city"]

    feature_cols = numerical_cols + categorical_cols

    X_raw = df[feature_cols]
    y_raw = df["total_runs"]

    # X_encoded = X_raw.copy()

    # one hot encoding for only non numeric values
    # X_encoded = pd.get_dummies(X_raw, columns=categorical_cols, drop_first=True)

    # encoder = OrdinalEncoder()

    # X_encoded[categorical_cols] = encoder.fit_transform(X_encoded[categorical_cols])

    groups = df["match_id"]

    gss = GroupShuffleSplit(n_splits=2, test_size=test_size, random_state=42)
    # train_idx, test_idx = next(gss.split(X_encoded, y_raw, groups=groups))
    train_idx, test_idx = next(gss.split(X_raw, y_raw, groups=groups))

    x_train = X_raw.iloc[train_idx]
    x_test = X_raw.iloc[test_idx]
    y_train = y_raw.iloc[train_idx]
    y_test = y_raw.iloc[test_idx]

    return x_train, x_test, y_train, y_test


def save_train_test(x_train, x_test, y_train, y_test, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    x_train.to_csv(os.path.join(output_dir, "x_train.csv"), index=False)
    x_test.to_csv(os.path.join(output_dir, "x_test.csv"), index=False)
    y_train.to_csv(os.path.join(output_dir, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(output_dir, "y_test.csv"), index=False)


def main():
    x_train, x_test, y_train, y_test = train_test_data()
    save_train_test(x_train, x_test, y_train, y_test, os.path.join("data", "raw"))


if __name__ == "__main__":
    main()
