import torch
import json
import matplotlib.pyplot as plt
import pandas as pd
from generator import Generator
from discriminator import Discriminator

# Load models
def load_models(generator_path="generator.pth", discriminator_path="discriminator.pth"):
    """
    Load pre-trained generator and discriminator models.
    """
    generator = Generator(input_dim=4, hidden_dim=128, output_dim=4)
    discriminator = Discriminator(input_dim=4, hidden_dim=128, output_dim=10)
    generator.load_state_dict(torch.load(generator_path))
    discriminator.load_state_dict(torch.load(discriminator_path))
    generator.eval()
    discriminator.eval()
    return generator, discriminator

# Generate synthetic data
def generate_synthetic_data(generator, num_samples=100):
    """
    Generate synthetic drone flight data using the trained generator.
    """
    pilot_ids = torch.randint(0, 10, (num_samples, 1), dtype=torch.float32)
    noise = torch.randn(num_samples, 4)  # Random noise
    synthetic_data = generator(pilot_ids, noise).detach().numpy()
    return synthetic_data, pilot_ids.numpy()

# Evaluate generated data
def evaluate_synthetic_data(discriminator, synthetic_data):
    """
    Evaluate the legality and pilot identity of synthetic data using the discriminator.
    """
    synthetic_data_tensor = torch.tensor(synthetic_data, dtype=torch.float32)
    pilot_classes, legality_scores = discriminator(synthetic_data_tensor)
    return pilot_classes.detach().numpy(), legality_scores.detach().numpy()

# Visualization
def visualize_results(synthetic_data, pilot_ids, legality_scores):
    """
    Visualize synthetic flight data and legality scores.
    """
    fig, ax = plt.subplots(2, 1, figsize=(10, 8))

    # Scatter plot for synthetic flight data
    ax[0].scatter(range(len(synthetic_data)), synthetic_data[:, 0], label="Roll", alpha=0.7)
    ax[0].scatter(range(len(synthetic_data)), synthetic_data[:, 1], label="Pitch", alpha=0.7)
    ax[0].scatter(range(len(synthetic_data)), synthetic_data[:, 2], label="Yaw", alpha=0.7)
    ax[0].scatter(range(len(synthetic_data)), synthetic_data[:, 3], label="Throttle", alpha=0.7)
    ax[0].set_title("Synthetic Flight Data")
    ax[0].set_xlabel("Sample Index")
    ax[0].set_ylabel("Values")
    ax[0].legend()

    # Bar plot for legality scores
    ax[1].bar(range(len(legality_scores)), legality_scores.flatten(), alpha=0.7)
    ax[1].set_title("Legality Scores of Synthetic Data")
    ax[1].set_xlabel("Sample Index")
    ax[1].set_ylabel("Legality Score")

    plt.tight_layout()
    plt.show()

# Save synthetic data and legality scores to JSON
def save_data_to_json(synthetic_data, pilot_ids, legality_scores, filename="synthetic_data.json"):
    data = {
        "synthetic_data": synthetic_data.tolist(),
        "pilot_ids": pilot_ids.tolist(),
        "legality_scores": legality_scores.tolist()
    }
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

if __name__ == "__main__":
    # Load models
    generator, discriminator = load_models()

    # Generate synthetic data
    synthetic_data, pilot_ids = generate_synthetic_data(generator, num_samples=100)

    # Evaluate synthetic data
    pilot_classes, legality_scores = evaluate_synthetic_data(discriminator, synthetic_data)

    # Save the data to a JSON file
    save_data_to_json(synthetic_data, pilot_ids, legality_scores)

    # Visualize results
    visualize_results(synthetic_data, pilot_ids, legality_scores)
