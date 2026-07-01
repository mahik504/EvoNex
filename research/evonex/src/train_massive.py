"""
EvoNex Massive Scale Training Pipeline (Phase 2)
Optimized for AWS A100 / Google Cloud TPU environments.
Designed to process 10,000+ TESS light curves using HDF5 data streaming.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import h5py
import os
import time
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from model_evonex import EvoMoE_Model

class MassiveTESSDataset(Dataset):
    """
    Streams data from a massive HDF5 file instead of loading all 10,000+ stars into RAM.
    Expected HDF5 structure:
    /lightcurves: (N, 2000) tensor of flux
    /tic_features: (N, 13) tensor of stellar parameters
    /labels: (N,) tensor of binary classification (1=exoplanet, 0=false positive)
    """
    def __init__(self, h5_path):
        self.h5_path = h5_path
        
        with h5py.File(self.h5_path, 'r') as f:
            self.length = f['labels'].shape[0]
            
    def __len__(self):
        return self.length
        
    def __getitem__(self, idx):
        with h5py.File(self.h5_path, 'r') as f:
            lc = torch.tensor(f['lightcurves'][idx], dtype=torch.float32)
            tic = torch.tensor(f['tic_features'][idx], dtype=torch.float32)
            label = torch.tensor(f['labels'][idx], dtype=torch.long)
            
        return lc, tic, label

def train_massive(h5_train_path, h5_test_path, batch_size=256, epochs=50):
    print(f"==================================================")
    print(f"🚀 EvoNex V2 Scale Training Pipeline Initiated")
    print(f"==================================================")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Hardware Accelerator: {device}")
    
    train_dataset = MassiveTESSDataset(h5_train_path)
    test_dataset = MassiveTESSDataset(h5_test_path)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    
    print(f"Training Samples: {len(train_dataset)} | Testing Samples: {len(test_dataset)}")
    
    model = EvoMoE_Model(num_classes=2).to(device)
    
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.2, 0.8]).to(device))
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_f1 = 0.0
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        start_time = time.time()
        
        for batch_idx, (lc, tic, labels) in enumerate(train_loader):
            lc, tic, labels = lc.to(device), tic.to(device), labels.to(device)
            
            optimizer.zero_grad()
            logits, gating_weights = model(lc, tic)
            loss = criterion(logits, labels)
            
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            epoch_loss += loss.item()
            
            if batch_idx % 50 == 0:
                print(f"Epoch {epoch+1}/{epochs} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")
                
        scheduler.step()
        
        model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for lc, tic, labels in test_loader:
                lc, tic = lc.to(device), tic.to(device)
                logits, _ = model(lc, tic)
                
                probs = torch.nn.functional.softmax(logits, dim=1)[:, 1]
                preds = torch.argmax(logits, dim=1)
                
                all_probs.extend(probs.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())
                
        precision = precision_score(all_labels, all_preds, zero_division=0)
        recall = recall_score(all_labels, all_preds, zero_division=0)
        f1 = f1_score(all_labels, all_preds, zero_division=0)
        try:
            roc_auc = roc_auc_score(all_labels, all_probs)
        except ValueError:
            roc_auc = 0.0 # Handles edge case if test set has only 1 class
            
        epoch_time = time.time() - start_time
        
        print(f"\n--- Epoch {epoch+1} Results ({epoch_time:.1f}s) ---")
        print(f"Train Loss: {epoch_loss/len(train_loader):.4f}")
        print(f"Val Precision: {precision:.4f} | Recall: {recall:.4f}")
        print(f"Val F1-Score:  {f1:.4f} | ROC-AUC: {roc_auc:.4f}")
        print(f"-----------------------------------------\n")
        
        if f1 > best_f1:
            best_f1 = f1
            save_path = os.path.join(os.path.dirname(__file__), "..", "weights", "evomoe_massive_v2.pth")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(f"🌟 New State-of-the-Art Model Saved! (F1: {f1:.4f})")

if __name__ == "__main__":
    print("WARNING: This script requires downloading the 25GB MAST HDF5 dataset.")
    print("To execute on Google Colab / AWS A100:")
    print("1. Run `python prepare_mast_hdf5.py` (Script to pull 10,000 FITS files)")
    print("2. Run this script pointing to the generated datasets.")
    
