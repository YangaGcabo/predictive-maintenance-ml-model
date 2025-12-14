import pandas as pd

def load_sensor_data(path: str) -> pd.DataFrame:
    """
    Load sensor data for predictive maintenance.
    Expected columns: timestamp, vibration, temperature, pressure, failure
    """
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.sort_values("timestamp")
    return df
