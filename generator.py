import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        """
        LSTM-based Generator for DP-GAN.
        Args:
            input_dim (int): Dimension of the input (noise + pilot embedding).
            hidden_dim (int): Number of hidden units in LSTM.
            output_dim (int): Dimension of the output (generated flight data).
        """
        super(Generator, self).__init__()
        self.embedding = nn.Linear(1, hidden_dim)  # Pilot ID embedding layer
        self.lstm = nn.LSTM(hidden_dim + input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, pilot_id, noise):
        """
        Forward pass for the Generator.
        Args:
            pilot_id (Tensor): Pilot IDs to embed (shape: [batch_size, 1]).
            noise (Tensor): Noise input for generating synthetic data (shape: [batch_size, input_dim]).
        Returns:
            Tensor: Generated flight data (shape: [batch_size, output_dim]).
        """
        # Embed pilot ID
        pilot_emb = self.embedding(pilot_id)  # Shape: [batch_size, hidden_dim]

        # Reshape pilot_emb to match noise dimensions
        pilot_emb = pilot_emb.unsqueeze(1)  # Add time dimension -> Shape: [batch_size, 1, hidden_dim]

        # Ensure noise has the same time dimension
        noise = noise.unsqueeze(1)  # Shape: [batch_size, 1, input_dim]

        # Combine pilot embedding and noise
        combined_input = torch.cat((pilot_emb, noise), dim=2)  # Shape: [batch_size, 1, hidden_dim + input_dim]

        # LSTM output
        lstm_out, _ = self.lstm(combined_input)  # Shape: [batch_size, 1, hidden_dim]

        # Fully connected layer to generate flight data
        generated_data = self.fc(lstm_out[:, -1, :])  # Use last output -> Shape: [batch_size, output_dim]
        return generated_data

if __name__ == "__main__":
    # Testing the generator
    generator = Generator(input_dim=4, hidden_dim=128, output_dim=4)

    # Example inputs
    pilot_ids = torch.randint(0, 10, (5, 1))  # Batch of 5 pilot IDs with shape [5, 1]
    noise = torch.randn(5, 4)  # Batch of 5 noise vectors (input_dim=4)

    # Generate synthetic data
    synthetic_data = generator(pilot_ids.float(), noise)
    print("Synthetic Data:")
    print(synthetic_data)
