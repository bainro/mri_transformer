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
        # @TODO see if rm'ing unsqueeze and later squeeze works
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
    train_loader = DataLoader(MRIDataset(split="train"), batch_size=16, shuffle=True)
    test_loader = DataLoader(MRIDataset(split="test"), batch_size=8, shuffle=False)

    # Model, loss, optimizer
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    best_grid_loss = 1e9
    best_emb_dim = None
    best_n_c_l = None
    best_n_l = None
    best_mlp_ratio = None

    j = -1
    for emb_dim in [420, 504]:
        for n_c_l in [7, 9]:
            for n_l in [12, 14]:
                for mlp_ratio in [3., 4.]:
                    j += 1
                    model = CCT(
                        img_size = 128,
                        num_frames = 128,
                        embedding_dim = emb_dim, # 384
                        n_conv_layers = n_c_l, # 5
                        n_input_channels=1,
                        frame_kernel_size = 3, 
                        kernel_size = 3, 
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
                        num_layers = n_l, # 10
                        num_heads = 6, # parallel attn fxs. Same input can go thru different W_k & W_q's.
                        mlp_ratio = mlp_ratio, # 2.0 before. how many neurons in a Transformer block's FC layers.
                        num_classes = 1, # we're going to regress (e.g. white matter) & use MSE loss
                        positional_embedding = 'sine'
                    ).to(device)
                    criterion = nn.MSELoss()
                    optimizer = optim.Adam(model.parameters(), lr=1e-4)

                    # Training loop
                    num_epochs = 15
                    dt_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    save_path = f"best_cct_model_{dt_str}_{j}.pth"  # File to save the best model
                    best_loss = float("inf") # Initialize best loss as infinity
                    train_losses = []
                    test_losses = []
                    
                    for epoch in range(num_epochs):
                        try:
                            train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
                        except:
                            print(":c")
                            continue
                        train_losses.append(train_loss)
                        test_loss = test_epoch(model, test_loader, criterion, device)
                        test_losses.append(test_loss)
                        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")

                        # Save model if test loss improves
                        if test_loss < best_loss:
                            # check if best loss in whole grid search too
                            if test_loss < best_grid_loss:
                                best_grid_loss = test_loss
                                # keep the values to print at the end
                                best_emb_dim = emb_dim
                                best_n_c_l = n_c_l
                                best_n_l = n_l
                                best_mlp_ratio = mlp_ratio
                            best_loss = test_loss
                            torch.save(model.state_dict(), save_path)
                            print(f"Saved Best Model (Epoch {epoch+1})")

                            with open(f"train_loss_{dt_str}_{j}.csv", 'w', newline='') as csv_file:
                                wr = csv.writer(csv_file)
                                wr.writerow(train_losses)

                            with open(f"test_loss_{dt_str}_{j}.csv", 'w', newline='') as csv_file:
                                wr = csv.writer(csv_file)
                                wr.writerow(test_losses)

    print(f"best grid pms: emb_dim: {best_emb_dim}; num_conv_layers: {best_n_c_l}; num_layers: {best_n_l}; mlp_ratio: {best_mlp_ratio}")
