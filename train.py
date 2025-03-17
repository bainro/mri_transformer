import csv
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from vit_pytorch.cct_3d import CCT
from torch.utils.data import Dataset, DataLoader

class MRIDataset(Dataset):
    def __init__(self, split='train'):
        # Load train.csv or test.csv
        self.data = pd.read_csv(f"subjects/{split}.csv")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get the path to the numpy file
        npy_path = f"subjects/{self.data.iloc[idx, 0]}" # t1_path column (first col)
        voxel_data = np.load(npy_path) # Load 3D MRI scan

        # Get the label (gmv value)
        label = torch.tensor(self.data.iloc[idx, 1], dtype=torch.float32)
        label = label / 100 # this is a hyperparamter. Lazy man's normalization/scaling

        # Convert voxel data to tensor (float32 for neural networks)
        voxel_tensor = torch.tensor(voxel_data, dtype=torch.float32)

        return voxel_tensor.squeeze(), label

# Training function
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for voxels, labels in loader:
        voxels, labels = voxels.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(voxels.unsqueeze(1))  # Add channel dimension
        loss = criterion(outputs.squeeze(), labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)

# Testing function
def test_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for voxels, labels in loader:
            voxels, labels = voxels.to(device), labels.to(device)
            outputs = model(voxels.unsqueeze(1))
            loss = criterion(outputs.squeeze(), labels)
            total_loss += loss.item()
    
    return total_loss / len(loader)

if __name__ == "__main__":
    # Load dataset
    train_loader = DataLoader(MRIDataset(split="train"), batch_size=8, shuffle=True)
    test_loader = DataLoader(MRIDataset(split="test"), batch_size=8, shuffle=False)

    # Model, loss, optimizer
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model = CCT(
        img_size = 128,
        num_frames = 128,
        embedding_dim = 384,
        n_conv_layers = 5, # increase after things are working!
        n_input_channels=1,
        frame_kernel_size = 3, # decrease after things are working
        kernel_size = 3, # decrease after things are working
        stride = 2,
        frame_stride = 2,
        padding = 3,
        frame_padding= 3,
        pooling_kernel_size = 3,
        frame_pooling_kernel_size = 3,
        pooling_stride = 2,
        frame_pooling_stride = 2,
        pooling_padding = 1,
        frame_pooling_padding = 1,
        num_layers = 10, # default was 14
        num_heads = 6, # parallel attn fxs. Same input can go thru different W_k & W_q's.
        mlp_ratio = 2., # how many neurons in a Transformer block's FC layers. Bigger = more neurons. 
        num_classes = 1, # we're going to regress (e.g. white matter) & use MSE loss
        positional_embedding = 'sine'
    ).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Training loop
    num_epochs = 200
    best_loss = float("inf") # Initialize best loss as infinity
    save_path = "best_cct_model.pth"  # File to save the best model
    train_losses = []
    test_losses = []
    
    dt_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        test_loss = test_epoch(model, test_loader, criterion, device)
        test_losses.append(test_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")

        # Save model if test loss improves
        if test_loss < best_loss:
            best_loss = test_loss
            torch.save(model.state_dict(), save_path)
            print(f"Saved Best Model (Epoch {epoch+1})")

            with open(f"train_loss_{dt_str}.csv", 'w', newline='') as csv_file:
                wr = csv.writer(csv_file)
                wr.writerow(train_losses)

            with open(f"test_loss_{dt_str}.csv", 'w', newline='') as csv_file:
                wr = csv.writer(csv_file)
                wr.writerow(test_losses)