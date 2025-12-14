import pandas as pd

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create rolling statistics used for failure prediction.
    """
    df["vibration_mean"] = df["vibration"].rolling(5).mean()
    df["temp_mean"] = df["temperature"].rolling(5).mean()
    df = df.dropna()
    return df
