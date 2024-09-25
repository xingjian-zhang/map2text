import os
import argparse
import logging
import datetime
import yaml
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split

def main():
    # Load the configuration,
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="Path to the configuration file.")
    args = parser.parse_args()

    with open(args.config, "r") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    train_kwargs = config["train_kwargs"]
    model_kwargs = config["model_kwargs"]
    config_basename_no_ext = os.path.splitext(os.path.basename(args.config))[0]
    log_dir = f"logs/{config_basename_no_ext}"
    time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"{log_dir}/{time_str}.log"

    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, filename=log_file, format="%(asctime)s - %(message)s"
    )
    logging.info(f"Configuration: {config}")  # NOTE: logging may expose api keys.

    dim_in, dim_out = 2, 1536
    model = nn.Sequential(
        nn.Linear(dim_in, model_kwargs["hidden_dim"]),
        nn.ReLU(),
        nn.Linear(model_kwargs["hidden_dim"], dim_out),
    )
    max_similarity = 0.0
    best_state_dict = model.state_dict()
    best_epoch = 0

    # Load data
    data = np.load("data/key_ideas.npz")
    low_dim_embeddings = data["low_dim_embeddings"]
    high_dim_embeddings = data["high_dim_embeddings"]

    # Convert numpy arrays to PyTorch tensors
    low_dim_embeddings_tensor = torch.tensor(low_dim_embeddings, dtype=torch.float32)
    high_dim_embeddings_tensor = torch.tensor(high_dim_embeddings, dtype=torch.float32)

    # Create a dataset containing both low_dim and high_dim embeddings as input-output pairs
    dataset = TensorDataset(low_dim_embeddings_tensor, high_dim_embeddings_tensor)

    # Split dataset into training and testing
    train_size = int(train_kwargs["train_prop"] * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    batch_size = train_kwargs["batch_size"]
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_kwargs["lr"],
        betas=(train_kwargs["beta1"], train_kwargs["beta2"]),
    )

    for epoch in range(train_kwargs["num_epochs"]):    
        model.train()
        running_loss = 0.0
        for low_dim, high_dim in train_loader:
            optimizer.zero_grad()
            # Forward pass: input low_dim embeddings, predict high_dim embeddings
            outputs = model(low_dim)
            loss = criterion(outputs, high_dim)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        logging.info(f"Epoch [{epoch+1}/{train_kwargs['num_epochs']}], Loss: {running_loss / len(train_loader):.4f}")

        model.eval()
        total_loss = 0.0
        total_cosine_similarity = 0.0
        with torch.no_grad():
            for low_dim, high_dim in test_loader:
                outputs = model(low_dim)
                loss = criterion(outputs, high_dim)
                total_loss += loss.item()

                cosine_sim = F.cosine_similarity(
                    outputs, high_dim, dim=1).mean().item()
                total_cosine_similarity += cosine_sim

        avg_test_loss = total_loss / len(test_loader)
        avg_test_cosine_similarity = total_cosine_similarity / len(test_loader)
        logging.info(f"Test Loss: {avg_test_loss:.4f}, Test Cosine Similarity: {avg_test_cosine_similarity:.4f}")
        if avg_test_cosine_similarity > max_similarity:
            max_similarity = avg_test_cosine_similarity
            best_state_dict = model.state_dict()
            best_epoch = epoch + 1

    save_file = os.path.join(log_dir, "model.pth")
    logging.info(f"Best model: Epoch {best_epoch}, Simialrity {max_similarity}")
    logging.info(f"Saving model to {save_file}")
    torch.save(best_state_dict, save_file)

if __name__ == "__main__":
    main()
