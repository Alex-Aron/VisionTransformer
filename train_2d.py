import torch
import torch.nn as nn
import torch.optim as optim
import os
from my_vit_models import MyViT2D
DATA_DIR = "sample_data/2d/"
BATCH_SIZE = 8
EPOCHS = 10
LEARNING_RATE = 0.001
PATCH_SIZE = 16
EMBED_DIM = 768
IMAGE_HEIGHT, IMAGE_WIDTH = 900, 600
IN_CHANNELS = 1  # Single 2D variable zeta

def load_data():
    file_paths = sorted([os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith(".pth")])
    dataset = []
    for path in file_paths:
        data = torch.load(path)  
        #print(data.shape)
        data = data.squeeze(0).permute(2, 0, 1)  
        dataset.append(data)
    return torch.cat(dataset, dim=0) 

def main():
    #Prepare dataset + load model + optimizer + loss
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = load_data()
    data = data.unsqueeze(1)  # Add channel dimension -> (batch_size, 1, 900, 600)
    train_loader = torch.utils.data.DataLoader(data, batch_size=BATCH_SIZE, shuffle=True)

    model = MyViT2D(in_channels=IN_CHANNELS, embed_dim=EMBED_DIM, image_height=IMAGE_HEIGHT, image_width=IMAGE_WIDTH, patch_size=30)
    model = model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

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
    torch.save(model.state_dict(), "2d_vit.pth")
    print("Training complete.")

if __name__ == "__main__":
    main()