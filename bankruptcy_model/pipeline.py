from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier


def create_pipeline() -> Pipeline:
    return Pipeline([
        ("model", LGBMClassifier(
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
            verbose=-1,
            max_depth=7,
            num_leaves=31,
            min_child_samples=50,
            learning_rate=0.05
        ))
    ])