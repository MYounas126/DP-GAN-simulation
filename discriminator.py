import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        """
        Discriminator for DP-GAN.
        Args:
            input_dim (int): Dimension of the input features.
            hidden_dim (int): Number of hidden units in intermediate layers.
            output_dim (int): Number of pilot identities (classification output).
        """
        super(Discriminator, self).__init__()
        # Behavioral feature extraction
        self.conv1 = nn.Conv1d(1, hidden_dim, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        
        # Task specification
        self.fc1 = nn.Linear(input_dim * hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)  # Pilot classification
        self.fc3 = nn.Linear(hidden_dim, 1)          # Input sequence legality

        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass for the Discriminator.
        Args:
            x (Tensor): Input sequence of drone flight data.
        Returns:
            Tuple: Classification output (pilot identity) and sequence legality score.
        """
        # Behavioral feature extraction
        x = x.unsqueeze(1)  # Add channel dimension for Conv1D
        x = self.relu(self.bn1(self.conv1(x)))
        x = x.view(x.size(0), -1)  # Flatten the output
        
        # Task specification
        features = self.relu(self.fc1(x))
        pilot_class = self.softmax(self.fc2(features))
        legality = self.sigmoid(self.fc3(features))
        
        return pilot_class, legality

if __name__ == "__main__":
    # Test the Discriminator
    discriminator = Discriminator(input_dim=28, hidden_dim=64, output_dim=10)
    test_input = torch.randn(5, 28)  # Batch size of 5, feature dimension of 28
    pilot_class, legality = discriminator(test_input)
    print("Pilot Class Probabilities:", pilot_class)
    print("Input Sequence Legality:", legality)
