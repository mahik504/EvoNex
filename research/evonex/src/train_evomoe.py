import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import TESSDataset
from model_evonex import EvoMoE_Model
import os
import time

def train_evomoe():
    print("======================================================")
    print("EvoNex: Real TESS Data Training Pipeline Initialization")
    print("======================================================")
    
    tic_targets = [
        "25155310",  # WASP-126 (Known Exoplanet)
        "164726113", # TOI-125 (Known Exoplanet)
        "281541555", # Normal/Background
        "141914082"  # Normal/Background
    ]
    labels = [1, 1, 0, 0]
    
    print("\n[1/4] Initializing High-Speed HDF5 Dataloader...")
    cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "cache")
    dataset = TESSDataset(tic_ids=tic_targets, labels=labels, cache_dir=cache_dir)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    print("\n[2/4] Initializing EvoMoE Architecture...")
    model = EvoMoE_Model(num_classes=2) # Binary classification for this demo
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print("\n[3/4] Commencing Training Loop...")
    num_epochs = 5
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        start_time = time.time()
        for batch_idx, (lc_tensor, tic_tensor, label_tensor) in enumerate(dataloader):
            optimizer.zero_grad()
            
            logits, weights = model(lc_tensor, tic_tensor)
            
            loss = criterion(logits, label_tensor)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        epoch_time = time.time() - start_time
        avg_loss = running_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {avg_loss:.4f} - Time: {epoch_time:.2f}s")
        
    print("\n[4/4] Saving Trained Model Weights...")
    weights_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "weights")
    os.makedirs(weights_dir, exist_ok=True)
    
    weights_path = os.path.join(weights_dir, "evomoe_weights.pth")
    torch.save(model.state_dict(), weights_path)
    print(f"✅ Success: Model weights saved to {weights_path}")
    print("EvoNex Research Pipeline is fully trained and ready for API integration.")

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    train_evomoe()
