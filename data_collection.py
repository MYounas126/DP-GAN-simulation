import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer

def collect_data(samples=1000):
    """
    Simulates collecting data from PX4 uORB topics.
    Includes fields for roll, pitch, yaw, throttle, and other control parameters.
    """
    np.random.seed(42)
    data = {
        'roll': np.random.normal(0, 1, samples),
        'pitch': np.random.normal(0, 1, samples),
        'yaw': np.random.normal(0, 1, samples),
        'throttle': np.random.normal(0, 1, samples),
        'actuator_controls': np.random.normal(0, 1, samples),
        'gps': np.random.normal(0, 1, samples),
        'air_speed': np.random.normal(0, 1, samples),
        'battery': np.random.normal(0, 1, samples)
    }
    df = pd.DataFrame(data)
    return df

def preprocess_data(df, window_size=20):
    """
    Synchronizes and preprocesses the flight data.
    1. Applies sliding window to downsample the data.
    2. Fills missing values using KNN Imputer.
    3. Standardizes the data for model training.
    """
    # Downsample data
    df_downsampled = df.groupby(df.index // window_size).mean()

    # Fill missing values
    imputer = KNNImputer(n_neighbors=3)
    imputed_data = imputer.fit_transform(df_downsampled)
    df_imputed = pd.DataFrame(imputed_data, columns=df.columns)

    # Standardize data
    scaler = StandardScaler()
    standardized_data = scaler.fit_transform(df_imputed)
    df_standardized = pd.DataFrame(standardized_data, columns=df.columns)
    
    return df_standardized, scaler

if __name__ == "__main__":
    # Simulate data collection
    raw_data = collect_data(samples=1000)

    # Preprocess the collected data
    processed_data, scaler = preprocess_data(raw_data)

    # Save processed data for further use
    processed_data.to_csv("processed_data.csv", index=False)
    print("Processed data saved as 'processed_data.csv'")
