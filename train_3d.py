import torch
import torch.nn as nn
import torch.optim as optim
import os
from my_vit_models import MyViT3D, MyViT3D_V2
DATA_DIR = "sample_data/3d/"
BATCH_SIZE = 8
EPOCHS = 5
LEARNING_RATE = 0.001
PATCH_SIZE = 16
EMBED_DIM = 768
IMAGE_HEIGHT, IMAGE_WIDTH = 900, 600
IN_CHANNELS = 1  # Single 2D variable zeta

def load_data():
    #file_paths = sorted([os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith(".pth")])
    file_paths = ["sample_data/3d/0000.pth"]
    dataset = []
    for path in file_paths:
        data = torch.load(path).to(torch.float32)
        print(data.shape)
        data = data.permute(0, 4, 1, 2, 3) 
        print(data.shape)
        dataset.append(data)
    return torch.cat(dataset, dim=0) 

def main():
    #Prepare dataset + load model + optimizer + loss
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_data()
    train_loader = torch.utils.data.DataLoader(data, batch_size=BATCH_SIZE, shuffle=True)
    print(data.shape)
    model = MyViT3D(in_channels=IN_CHANNELS, embed_dim=EMBED_DIM, image_size=(900, 600, 24), patch_size=(60, 50, 12))
    model = model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    #Training loop
    for epoch in range(EPOCHS):
        running_loss = 0.0
        for inputs in train_loader:
            inputs = inputs.to(device, dtype=torch.float32) 
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs) 
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {running_loss/len(train_loader):.4f}")
    torch.save(model.state_dict(), "3d_vit.pth")
    print("Training complete.")

if __name__ == "__main__":
    main()