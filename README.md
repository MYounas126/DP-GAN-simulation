# DP-GAN: Drone Pilot Identification

This project implements the DP-GAN framework described in the article, including data collection, preprocessing, generator and discriminator models, adversarial training, and evaluation metrics.

## Features
- **Data Collection and Preprocessing**: Simulates drone flight data and applies preprocessing steps.
- **Generator and Discriminator**: Implements LSTM-based generator and multi-task discriminator.
- **Adversarial Training**: Trains DP-GAN using a three-stage adversarial strategy.
- **Evaluation Metrics**: Calculates ACC, FAR, MAR, and AUC for model performance.
- **Simulation**: Generates synthetic drone flight data and visualizes results.

## Prerequisites
- Python 3.9 or higher
- Docker (for containerization)

## Installation

### Option 1: Using Docker
1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd dp-gan-project
