import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report

from bankruptcy_model.processing.data_manager import load_raw_data, save_pipeline
from bankruptcy_model.processing.features import get_features_and_target
from bankruptcy_model.pipeline import create_pipeline

DATA_PATH = "data/data.csv"
SAVE_PATH = "bankruptcy_model/trained_models/lgbm_pipeline.pkl"


def run_training():
    # Load data
    print("Loading data...")
    df = load_raw_data(DATA_PATH)

    # Build features
    print("Building features...")
    X, y = get_features_and_target(df)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train
    print("Training model...")
    pipeline = create_pipeline()
    pipeline.fit(X_train, y_train)

    # Evaluate
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    y_pred = pipeline.predict(X_test)
    print(f"AUC: {roc_auc_score(y_test, y_prob):.4f}")
    print(classification_report(y_test, y_pred))

    # Save
    save_pipeline(pipeline, SAVE_PATH)


if __name__ == "__main__":
    run_training()