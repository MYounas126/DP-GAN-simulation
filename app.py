from flask import Flask, jsonify, request
import os
import logging
import pandas as pd
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Set up logging
logging.basicConfig(
    filename="app.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Global configurations
DATA_DIR = "static"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Load models (Example: Generator and Discriminator)
generator_model_path = "generator.pth"
discriminator_model_path = "discriminator.pth"

def initialize_models():
    """
    Load models during application startup.
    """
    try:
        logging.info("Loading models...")
        # Mocking model loading for simplicity; replace with actual code
        if not os.path.exists(generator_model_path) or not os.path.exists(discriminator_model_path):
            raise FileNotFoundError("Model files not found.")
        logging.info("Models loaded successfully.")
    except Exception as e:
        logging.error(f"Error loading models: {e}")
        raise

# Initialize models before the first request
with app.app_context():
    initialize_models()


@app.route("/", methods=["GET"])
def home():
    """
    Home route.
    """
    return jsonify({"message": "Welcome to the Drone Simulation API"})


@app.route("/generate", methods=["POST"])
def generate():
    """
    Generates synthetic flight data and saves it to a CSV file.
    """
    try:
        num_samples = int(request.form.get("num_samples", 100))

        # Mock data generation (replace with actual generation logic)
        synthetic_data = np.random.rand(num_samples, 4)  # Roll, Pitch, Yaw, Throttle
        legality_scores = np.random.rand(num_samples)  # Mock legality scores

        # Convert to DataFrame
        df = pd.DataFrame(
            synthetic_data, columns=["roll", "pitch", "yaw", "throttle"]
        )
        df["legality_score"] = legality_scores

        # Save to file
        output_path = os.path.join(DATA_DIR, "synthetic_data.csv")
        df.to_csv(output_path, index=False)

        logging.info(f"Synthetic data generated and saved to {output_path}")
        return jsonify({"message": "Data generated successfully!", "path": output_path})

    except ValueError:
        logging.error("Invalid input: num_samples must be an integer.")
        return jsonify({"error": "Invalid input: num_samples must be an integer."}), 400

    except Exception as e:
        logging.error(f"Error during data generation: {e}")
        return jsonify({"error": f"Error during data generation: {e}"}), 500


@app.route("/stats", methods=["GET"])
def stats():
    """
    Returns basic statistics about the synthetic flight data.
    """
    try:
        file_path = os.path.join(DATA_DIR, "synthetic_data.csv")
        if not os.path.exists(file_path):
            raise FileNotFoundError("Synthetic data file not found.")

        # Read data
        df = pd.read_csv(file_path)

        # Compute statistics
        stats = df.describe().to_dict()
        logging.info("Statistics fetched successfully.")
        return jsonify({"statistics": stats})

    except FileNotFoundError as e:
        logging.error(f"Error: {e}")
        return jsonify({"error": str(e)}), 404

    except Exception as e:
        logging.error(f"Error fetching statistics: {e}")
        return jsonify({"error": f"Error fetching statistics: {e}"}), 500


@app.route("/logs", methods=["GET"])
def fetch_logs():
    """
    Returns the latest server logs.
    """
    try:
        with open("app.log", "r") as log_file:
            logs = log_file.readlines()[-50:]  # Return last 50 log entries
        return jsonify({"logs": logs})
    except Exception as e:
        logging.error(f"Error fetching logs: {e}")
        return jsonify({"error": f"Error fetching logs: {e}"}), 500


if __name__ == "__main__":
    app.run(debug=True)
