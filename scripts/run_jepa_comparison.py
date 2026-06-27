import sys
import os
import json
import math
import random
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import multiprocessing
from torch.utils.data import DataLoader, TensorDataset

# Force usage of all CPU cores to parallelize torch operations
torch.set_num_threads(multiprocessing.cpu_count())
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from phy_l_jepa.architectures import MuellerPatchEncoder
from phy_l_jepa.hybrid_physics_jepa import HybridRetentionPhysicsJEPA
from phy_l_jepa.physics_features import FrozenLNPIVAEReference

DATA_DIR = PROJECT_ROOT / "data" / "GIGADATASET_COLAB_NPZ"
OUTPUT_DIR = Path("benchmark_results")
OUTPUT_DIR.mkdir(exist_ok=True)

# Configurations
SEEDS = list(range(10))
JEPA_EPOCHS = 100
MLP_EPOCHS = 100
BATCH_SIZE = 1024
DEVICE = torch.device("cpu")

def load_npz(path, cohort):
    data = np.load(path, allow_pickle=False)
    index = np.flatnonzero(
        (data["source"] == 0) & (data["cohort"] == cohort) & (data["label"] >= 0)
    )
    return {
        key: data[key][index] for key in ("X", "M", "pos", "wvl", "label", "specimen")
    }

def specimen_metrics(labels, probabilities, specimens):
    frame = pd.DataFrame({
        "label": labels, "probability": probabilities, "specimen": specimens,
    }).groupby("specimen", as_index=False).agg(
        label=("label", "first"), probability=("probability", "mean")
    )
    y = frame["label"].to_numpy().astype(int)
    p = frame["probability"].to_numpy()
    pred = p >= 0.5
    tp = np.sum((pred == 1) & (y == 1))
    fn = np.sum((pred == 0) & (y == 1))
    tn = np.sum((pred == 0) & (y == 0))
    fp = np.sum((pred == 1) & (y == 0))
    accuracy = np.mean(pred == y)
    
    # Mann-Whitney AUC
    positive = p[y == 1]
    negative = p[y == 0]
    if len(positive) > 0 and len(negative) > 0:
        auc = np.mean((positive[:, None] > negative[None, :]) + 0.5 * (positive[:, None] == negative[None, :]))
    else:
        auc = 0.5
    return {"accuracy": accuracy, "auc": auc}

class DownstreamMLP(nn.Module):
    def __init__(self, in_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 32), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(32, 16), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(16, 1)
        )
    def forward(self, x): 
        return self.net(x).squeeze(-1)

def run_experiment(model_type, seed, train_data, test_data):
    """
    model_type: "MaskedJEPA" (retention_weight=0.0) or "PhyL_JEPA" (retention_weight=1.0)
    """
    # Fix seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

    print(f"\n--- Running {model_type} | Seed {seed} ---")
    
    # Setup data
    # X shape: [B, 16, 5, 5]
    train_x = torch.from_numpy(train_data["X"].astype(np.float32))
    train_y = train_data["label"].astype(np.int64)
    test_x = torch.from_numpy(test_data["X"].astype(np.float32))
    test_y = test_data["label"].astype(np.int64)
    
    train_loader = DataLoader(TensorDataset(train_x), batch_size=BATCH_SIZE, shuffle=True)
    
    # 1. Initialize JEPA
    encoder = MuellerPatchEncoder(
        in_channels=16, patch_size=1, embed_dim=128, depth=4, num_heads=4, dropout=0.1, image_size=5
    )
    
    pivae_model_path = PROJECT_ROOT / "models" / "pivae_classic_lipschitz.pth"
    reference = FrozenLNPIVAEReference(model_path=str(pivae_model_path), patch_size=1)
    
    retention_weight = 1.0 if model_type == "PhyL_JEPA" else 0.0
    
    jepa = HybridRetentionPhysicsJEPA(
        encoder=encoder,
        reference=reference,
        predictor_depth=2,
        num_heads=4,
        mask_ratio=0.5,
        retention_weight=retention_weight,
        encoder_retention_weight=retention_weight,
        variance_weight=1.0,
        covariance_weight=0.01,
    ).to(DEVICE)
    
    params_to_optimize = (
        list(jepa.context_encoder.parameters()) +
        list(jepa.jepa.predictor.parameters()) +
        [jepa.jepa.mask_token] +
        list(jepa.retention_head.parameters()) +
        list(jepa.encoder_retention_head.parameters())
    )
    optimizer = torch.optim.AdamW(params_to_optimize, lr=1e-3, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=JEPA_EPOCHS)
    
    # 2. Pretrain JEPA
    jepa.train()
    start_time = time.time()
    for ep in range(JEPA_EPOCHS):
        for (batch_x,) in train_loader:
            batch_x = batch_x.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            res = jepa(batch_x, batch_x)
            loss = res["loss"].mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params_to_optimize, 1.0)
            optimizer.step()
            jepa.update_target()
        scheduler.step()
    print(f"JEPA pretraining finished in {time.time() - start_time:.1f}s")
    
    # 3. Extract Representations
    jepa.eval()
    def extract_features(x_tensor):
        feats = []
        with torch.no_grad():
            dl = DataLoader(TensorDataset(x_tensor), batch_size=BATCH_SIZE, shuffle=False)
            for (bx,) in dl:
                # Use context_encoder.represent to get [B, D] averaged over patches
                rep = jepa.context_encoder.represent(bx.to(DEVICE))
                feats.append(rep.cpu())
        return torch.cat(feats, dim=0).numpy()
        
    train_feats = extract_features(train_x)
    test_feats = extract_features(test_x)
    
    # Normalize features
    mean = train_feats.mean(axis=0, keepdims=True)
    std = train_feats.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    
    train_feats = (train_feats - mean) / std
    test_feats = (test_feats - mean) / std
    
    # 4. Train Downstream MLP
    mlp = DownstreamMLP(in_dim=128).to(DEVICE)
    mlp_opt = torch.optim.AdamW(mlp.parameters(), lr=2e-3, weight_decay=1e-4)
    
    ty = torch.from_numpy(train_y.astype(np.float32)).to(DEVICE)
    tx = torch.from_numpy(train_feats).to(DEVICE)
    
    positives = max(int(ty.sum().item()), 1)
    negatives = max(len(ty) - positives, 1)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(negatives / positives).to(DEVICE))
    
    mlp_loader = DataLoader(TensorDataset(tx, ty), batch_size=4096, shuffle=True)
    
    mlp.train()
    for ep in range(MLP_EPOCHS):
        for bx, by in mlp_loader:
            mlp_opt.zero_grad(set_to_none=True)
            loss = criterion(mlp(bx), by)
            loss.backward()
            mlp_opt.step()
            
    # 5. Evaluate
    mlp.eval()
    with torch.no_grad():
        test_tx = torch.from_numpy(test_feats).to(DEVICE)
        probs = torch.sigmoid(mlp(test_tx)).cpu().numpy()
        
    metrics = specimen_metrics(test_y, probs, test_data["specimen"])
    print(f"Metrics: {metrics}")
    return metrics

def main():
    print("Loading data...")
    train_data = load_npz(DATA_DIR / "train_cohort3_polamb_nonzero.npz", 3)
    test_data = load_npz(DATA_DIR / "test_cohort1_polamb_wvl0.npz", 1)
    
    # 1. Setup architectures and load data tensors for extraction
    train_x = torch.from_numpy(train_data["X"].astype(np.float32))
    train_y = train_data["label"].astype(np.int64)
    test_x = torch.from_numpy(test_data["X"].astype(np.float32))
    test_y = test_data["label"].astype(np.int64)
    
    # We will pretrain JEPA once (on seed 0) for each model type, then extract representations
    extracted_features = {}
    features_cache_path = OUTPUT_DIR / "extracted_features.npz"
    
    if features_cache_path.exists():
        print(f"Loading extracted features from cache {features_cache_path}...")
        cache = np.load(features_cache_path)
        for model_type in ["MaskedJEPA", "PhyL_JEPA"]:
            extracted_features[model_type] = (cache[f"{model_type}_train"], cache[f"{model_type}_test"])
    else:
        for model_type in ["MaskedJEPA", "PhyL_JEPA"]:
            print(f"\n=== Pretraining {model_type} on Seed 0 ===")
            # Fix seed for JEPA pretraining
            random.seed(0)
            np.random.seed(0)
            torch.manual_seed(0)
            if torch.cuda.is_available(): torch.cuda.manual_seed_all(0)
            
            train_loader = DataLoader(TensorDataset(train_x), batch_size=BATCH_SIZE, shuffle=True)
            
            encoder = MuellerPatchEncoder(
                in_channels=16, patch_size=1, embed_dim=64, depth=2, num_heads=2, dropout=0.1, image_size=5
            )
            pivae_model_path = PROJECT_ROOT / "models" / "pivae_classic_lipschitz.pth"
            reference = FrozenLNPIVAEReference(model_path=str(pivae_model_path), patch_size=1)
            
            retention_weight = 1.0 if model_type == "PhyL_JEPA" else 0.0
            
            jepa = HybridRetentionPhysicsJEPA(
                encoder=encoder,
                reference=reference,
                predictor_depth=2,
                num_heads=4,
                mask_ratio=0.5,
                retention_weight=retention_weight,
                encoder_retention_weight=retention_weight,
                variance_weight=1.0,
                covariance_weight=0.01,
            ).to(DEVICE)
            
            params_to_optimize = (
                list(jepa.context_encoder.parameters()) +
                list(jepa.jepa.predictor.parameters()) +
                [jepa.jepa.mask_token] +
                list(jepa.retention_head.parameters()) +
                list(jepa.encoder_retention_head.parameters())
            )
            optimizer = torch.optim.AdamW(params_to_optimize, lr=1e-3, weight_decay=0.05)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=JEPA_EPOCHS)
            
            # Pretrain
            jepa.train()
            start_time = time.time()
            for ep in range(JEPA_EPOCHS):
                for (batch_x,) in train_loader:
                    batch_x = batch_x.to(DEVICE)
                    optimizer.zero_grad(set_to_none=True)
                    res = jepa(batch_x, batch_x)
                    loss = res["loss"].mean()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(params_to_optimize, 1.0)
                    optimizer.step()
                    jepa.update_target()
                scheduler.step()
            print(f"JEPA pretraining finished in {time.time() - start_time:.1f}s")
            
            # Extract features
            jepa.eval()
            def extract_features(x_tensor):
                feats = []
                with torch.no_grad():
                    dl = DataLoader(TensorDataset(x_tensor), batch_size=BATCH_SIZE, shuffle=False)
                    for (bx,) in dl:
                        rep = jepa.context_encoder.represent(bx.to(DEVICE))
                        feats.append(rep.cpu())
                return torch.cat(feats, dim=0).numpy()
                
            train_feats = extract_features(train_x)
            test_feats = extract_features(test_x)
            
            # Normalize features
            mean = train_feats.mean(axis=0, keepdims=True)
            std = train_feats.std(axis=0, keepdims=True)
            std[std < 1e-6] = 1.0
            train_feats = (train_feats - mean) / std
            test_feats = (test_feats - mean) / std
            
            extracted_features[model_type] = (train_feats, test_feats)
        
        # Save to disk cache
        np.savez(
            features_cache_path,
            MaskedJEPA_train=extracted_features["MaskedJEPA"][0],
            MaskedJEPA_test=extracted_features["MaskedJEPA"][1],
            PhyL_JEPA_train=extracted_features["PhyL_JEPA"][0],
            PhyL_JEPA_test=extracted_features["PhyL_JEPA"][1]
        )
        print(f"Saved extracted features cache to {features_cache_path}")
        
    # 2. Evaluate MLP classification over 10 seeds under different label percentages
    LABEL_PCTS = [0.01, 0.02, 0.05, 0.10, 0.25, 0.50, 1.0]
    results = []
    print("\n=== Training Downstream MLP on 10 Seeds with Varying Label Percentages ===")
    
    # Pre-convert labels to numpy for easy indexing
    y_npy = train_y.to_numpy() if hasattr(train_y, "to_numpy") else train_y
    y_npy = y_npy.astype(np.float32)
    
    for pct in LABEL_PCTS:
        print(f"\n--- Testing with {pct*100:.1f}% of train labels ---")
        for seed in SEEDS:
            # Consistent indexing for subsampling based on seed
            np.random.seed(seed)
            n_total = len(y_npy)
            n_sub = max(int(n_total * pct), 2)
            indices = np.random.choice(n_total, size=n_sub, replace=False)
            
            # Ensure we have at least one positive and one negative sample if possible
            if len(np.unique(y_npy[indices])) < 2:
                # Fallback: force at least one of each class
                pos_idx = np.where(y_npy == 1.0)[0]
                neg_idx = np.where(y_npy == 0.0)[0]
                indices[0] = np.random.choice(pos_idx)
                indices[1] = np.random.choice(neg_idx)
                
            for model_type in ["MaskedJEPA", "PhyL_JEPA"]:
                train_feats, test_feats = extracted_features[model_type]
                
                # Set seed for MLP training
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
                
                mlp = DownstreamMLP(in_dim=64).to(DEVICE)
                mlp_opt = torch.optim.AdamW(mlp.parameters(), lr=2e-3, weight_decay=1e-4)
                
                sub_y = y_npy[indices]
                sub_x = train_feats[indices]
                
                ty = torch.from_numpy(sub_y).to(DEVICE)
                tx = torch.from_numpy(sub_x).to(DEVICE)
                
                positives = max(int(ty.sum().item()), 1)
                negatives = max(len(ty) - positives, 1)
                criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(negatives / positives).to(DEVICE))
                
                mlp_loader = DataLoader(TensorDataset(tx, ty), batch_size=min(4096, len(tx)), shuffle=True)
                
                mlp.train()
                for ep in range(MLP_EPOCHS):
                    for bx, by in mlp_loader:
                        mlp_opt.zero_grad(set_to_none=True)
                        loss = criterion(mlp(bx), by)
                        loss.backward()
                        mlp_opt.step()
                        
                mlp.eval()
                with torch.no_grad():
                    test_tx = torch.from_numpy(test_feats).to(DEVICE)
                    probs = torch.sigmoid(mlp(test_tx)).cpu().numpy()
                    
                metrics = specimen_metrics(test_y, probs, test_data["specimen"])
                
                results.append({
                    "model": model_type,
                    "seed": seed,
                    "label_pct": pct,
                    **metrics
                })
                pd.DataFrame(results).to_csv(OUTPUT_DIR / "benchmark_results_raw.csv", index=False)
                
    # Summary
    df = pd.DataFrame(results)
    summary = df.groupby(["model", "label_pct"]).agg({
        "accuracy": ["mean", "std"],
        "auc": ["mean", "std"]
    }).reset_index()
    
    summary.columns = ['_'.join(col).strip('_') for col in summary.columns.values]
    summary.to_csv(OUTPUT_DIR / "benchmark_summary.csv", index=False)
    
    print("\n=== FINAL BENCHMARK SUMMARY ===")
    print(summary.to_string(index=False))

if __name__ == "__main__":
    main()

