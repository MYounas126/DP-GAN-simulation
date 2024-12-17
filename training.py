import torch
import torch.nn as nn
import torch.optim as optim
from generator import Generator
from discriminator import Discriminator
from data_collection import preprocess_data

# Define the loss functions
def loss_idf(predicted_identity, true_identity):
    return torch.mean(torch.abs(predicted_identity - true_identity))

def loss_adv(discriminator_output, is_real):
    labels = torch.ones_like(discriminator_output) if is_real else torch.zeros_like(discriminator_output)
    criterion = nn.BCELoss()
    return criterion(discriminator_output, labels)

# Training function
def train(generator, discriminator, data_loader, epochs=100, batch_size=64, lr=0.0001, lambda_factor=0.5):
    optimizer_G = optim.Adam(generator.parameters(), lr=lr)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr)

    for epoch in range(epochs):
        for i in range(0, len(data_loader), batch_size):
            # Extract real data batch
            current_batch = data_loader[i:i + batch_size]
            real_data = torch.tensor(current_batch[['roll', 'pitch', 'yaw', 'throttle']].values, dtype=torch.float32)
            pilot_ids = torch.randint(0, 10, (len(real_data), 1), dtype=torch.float32)  # Corrected shape: [batch_size, 1]
            noise = torch.randn(len(real_data), 4)  # Random noise

            # Train Discriminator
            optimizer_D.zero_grad()
            real_classes, real_legality = discriminator(real_data)
            d_loss_real = loss_idf(real_classes, pilot_ids) + loss_adv(real_legality, is_real=True)

            fake_data = generator(pilot_ids, noise)
            fake_classes, fake_legality = discriminator(fake_data.detach())
            d_loss_fake = loss_idf(fake_classes, pilot_ids) + loss_adv(fake_legality, is_real=False)

            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            optimizer_D.step()

            # Train Generator
            optimizer_G.zero_grad()
            fake_classes, fake_legality = discriminator(fake_data)
            g_loss_idf = loss_idf(fake_classes, pilot_ids)
            g_loss_adv = loss_adv(fake_legality, is_real=True)

            g_loss = g_loss_idf + lambda_factor * g_loss_adv
            g_loss.backward()
            optimizer_G.step()

        if epoch % 10 == 0:
            print(f"Epoch [{epoch}/{epochs}], D Loss: {d_loss.item()}, G Loss: {g_loss.item()}")

if __name__ == "__main__":
    import pandas as pd
    data = pd.read_csv("processed_data.csv")
    
    generator = Generator(input_dim=4, hidden_dim=128, output_dim=4)
    discriminator = Discriminator(input_dim=4, hidden_dim=128, output_dim=10)

    train(generator, discriminator, data_loader=data, epochs=100, batch_size=64)
    
    torch.save(generator.state_dict(), "generator.pth")
    torch.save(discriminator.state_dict(), "discriminator.pth")
    print("Models saved successfully.")
