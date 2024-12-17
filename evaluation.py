import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, classification_report
import matplotlib.pyplot as plt
import torch


def calculate_metrics(y_true, y_pred):
    """
    Calculate evaluation metrics for impersonation attack detection.
    Args:
        y_true (array): Ground truth labels.
        y_pred (array): Predicted labels.
    Returns:
        dict: Evaluation metrics (ACC, MAR, FAR, AUC).
    """
    # Confusion matrix elements
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Metrics calculation
    acc = (tp + tn) / (tp + tn + fp + fn)  # Accuracy
    mar = fn / (tp + fn)                  # Miss Alarm Rate
    far = fp / (tn + fp)                  # False Alarm Rate
    auc = roc_auc_score(y_true, y_pred)   # Area Under Curve

    return {"ACC": acc, "MAR": mar, "FAR": far, "AUC": auc}


def plot_confusion_matrix(y_true, y_pred, labels):
    """
    Plot confusion matrix.
    Args:
        y_true (array): Ground truth labels.
        y_pred (array): Predicted labels.
        labels (list): List of class labels.
    """
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    cax = ax.matshow(cm, cmap="Blues")
    plt.colorbar(cax)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(np.arange(len(labels)), labels, rotation=45)
    plt.yticks(np.arange(len(labels)), labels)
    plt.show()


def prepare_combined_input(pilot_emb, noise):
    """
    Prepares the combined input for the generator or discriminator by concatenating
    pilot embeddings and noise along the feature dimension.
    Args:
        pilot_emb (Tensor): Pilot embeddings (shape: [batch_size, hidden_dim, feature_dim]).
        noise (Tensor): Noise input (shape: [batch_size, feature_dim]).
    Returns:
        Tensor: Combined input (shape: [batch_size, hidden_dim, combined_feature_dim]).
    """
    # Ensure pilot_emb has the correct shape
    pilot_emb = pilot_emb.squeeze(1)  # Remove extra dimension -> Shape: [batch_size, hidden_dim, feature_dim]

    # Expand noise to match pilot_emb's last dimension
    noise = noise.unsqueeze(-1).expand(-1, -1, pilot_emb.size(2))  # Shape: [batch_size, hidden_dim, feature_dim]

    # Debugging: Print shapes to confirm alignment
    print("Pilot Embedding Shape (after squeeze):", pilot_emb.shape)
    print("Noise Shape (after expand):", noise.shape)

    # Concatenate along the last dimension
    combined_input = torch.cat((pilot_emb, noise), dim=2)  # Shape: [batch_size, hidden_dim, combined_feature_dim]

    return combined_input


if __name__ == "__main__":
    # Example inputs for evaluation
    y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 0])  # Ground truth
    y_pred = np.array([1, 0, 1, 1, 0, 0, 0, 1, 1, 0])  # Predictions

    # Calculate metrics
    metrics = calculate_metrics(y_true, y_pred)
    print("Evaluation Metrics:", metrics)

    # Classification report
    print("\nClassification Report:\n", classification_report(y_true, y_pred))

    # Plot confusion matrix
    labels = ["Legitimate", "Impostor"]
    plot_confusion_matrix(y_true, y_pred, labels)

    # Tensor compatibility demonstration
    pilot_emb = torch.randn(10, 1, 16, 32)  # Example pilot embeddings
    noise = torch.randn(10, 16)             # Example noise
    combined_input = prepare_combined_input(pilot_emb, noise)
    print("Combined input shape:", combined_input.shape)
