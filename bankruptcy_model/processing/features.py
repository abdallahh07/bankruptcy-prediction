import pandas as pd


def drop_constant_features(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=[" Net Income Flag"], errors="ignore")


def get_features_and_target(df: pd.DataFrame):
    X = df.drop(columns=["Bankrupt?", " Net Income Flag"], errors="ignore")
    y = df["Bankrupt?"]
    return X, y