import pandas as pd
from bankruptcy_model.processing.data_manager import load_pipeline
from bankruptcy_model.processing.features import get_features_and_target

SAVE_PATH = "bankruptcy_model/trained_models/lgbm_pipeline.pkl"


def make_prediction(df: pd.DataFrame) -> pd.DataFrame:
    X, _ = get_features_and_target(df)
    pipeline = load_pipeline(SAVE_PATH)
    predictions = pipeline.predict(X)
    probabilities = pipeline.predict_proba(X)[:, 1]

    results = df[["Bankrupt?"]].copy()
    results["predicted"] = predictions
    results["bankruptcy_probability"] = probabilities

    return results


if __name__ == "__main__":
    df = pd.read_csv("data/data.csv")
    results = make_prediction(df)
    print(results.head())