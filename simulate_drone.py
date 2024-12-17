import numpy as np
import pandas as pd

def simulate_flight_data(samples=1000, constrained=False):
    """
    Simulates flight data for a PX4-based drone.
    Args:
        samples (int): Number of data points to generate.
        constrained (bool): If True, simulates constrained environments.
    Returns:
        pd.DataFrame: Simulated flight data.
    """
    np.random.seed(42)

    # Simulate parameters
    roll = np.random.uniform(-30, 30, samples)  # Degrees
    pitch = np.random.uniform(-30, 30, samples)  # Degrees
    yaw = np.random.uniform(-180, 180, samples)  # Degrees
    throttle = np.random.uniform(0, 100, samples)  # Percentage

    # Add constraints or noise for constrained environments
    if constrained:
        roll += np.random.normal(0, 2, samples)  # Add small noise
        pitch += np.random.normal(0, 2, samples)
        throttle = np.clip(throttle, 20, 80)  # Throttle limited in constrained environments

    # Additional parameters
    actuator_controls = np.random.uniform(-1, 1, samples)  # Control outputs
    gps = np.random.uniform(0, 10, samples)  # Simulated GPS drift
    air_speed = np.random.uniform(5, 50, samples)  # m/s
    battery = np.random.uniform(10, 100, samples)  # Percentage

    # Combine into a DataFrame
    flight_data = pd.DataFrame({
        "roll": roll,
        "pitch": pitch,
        "yaw": yaw,
        "throttle": throttle,
        "actuator_controls": actuator_controls,
        "gps": gps,
        "air_speed": air_speed,
        "battery": battery
    })

    return flight_data

if __name__ == "__main__":
    # Simulate flight data for both natural and constrained environments
    natural_data = simulate_flight_data(samples=1000, constrained=False)
    constrained_data = simulate_flight_data(samples=1000, constrained=True)

    # Save to CSV for analysis
    natural_data.to_csv("natural_flight_data.csv", index=False)
    constrained_data.to_csv("constrained_flight_data.csv", index=False)

    print("Simulated flight data saved successfully.")


import json

# Convert flight data to JSON
def save_data_as_json(df, filename):
    """
    Save a DataFrame as a JSON file.
    Args:
        df (pd.DataFrame): DataFrame to save.
        filename (str): Output JSON filename.
    """
    df.to_json(filename, orient="records", lines=False)
    print(f"Data saved to {filename}")

if __name__ == "__main__":
    # Load simulated data
    natural_data = pd.read_csv("natural_flight_data.csv")
    constrained_data = pd.read_csv("constrained_flight_data.csv")

    # Save as JSON
    save_data_as_json(natural_data, "natural_flight_data.json")
    save_data_as_json(constrained_data, "constrained_flight_data.json")
