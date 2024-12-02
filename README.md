# Drone Pilot Generative Adversarial Network (DP-GAN)

**DP-GAN** is a machine learning project designed to identify drone pilots based on their unique flight patterns. Using **Unity** for simulation and **TensorFlow** for training a **Generative Adversarial Network (GAN)**, the system can classify pilots by their control behaviors, without the need for real-world drone data.

## Summary:

1. **Unity Setup**: Simulates drone behavior and generates control input data (throttle, pitch, yaw, roll).
2. **Training**: Uses TensorFlow to train a GAN on the generated data.
3. **Evaluation**: After training, the model generates new drone flight data that can be used for further analysis or pilot identification.

## Key Features
- **Unity Simulation**: Generate realistic flight data (throttle, pitch, yaw, roll) using Unity's physics engine.
- **TensorFlow**: Train a GAN model to learn drone pilot behavior from the simulated data.
- **Pilot Identification**: Classify drone pilots based on the unique flight patterns captured by the model.

## Requirements
Unity (for simulation).
TensorFlow (for model training).
Python (3.7+).

## Setup & Usage

### 1. Unity Setup
- Clone the repository and open the Unity project.
- Press **Play** in Unity to simulate drone behavior and generate control input data (throttle, pitch, roll, yaw).

### 2. Model Training
Install TensorFlow using pip:
  ```bash
  pip install tensorflow
  ```

Train the GAN model with the following script:
  ```bash
  python train_model.py
  ```

### 3. Evaluate the Model
After training, evaluate the model with:
  ```bash
  python evaluate_model.py
  ```
