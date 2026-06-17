# -*- coding: utf-8 -*-
"""
Training utilities.
Supports both segmented and standard training modes.
Fix: test set uses y_hat_init for regime assignment to avoid label leakage.
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor


DEFAULT_CONFIG = {
    'epochs': 300,
    'lr': 0.005,
    'weight_decay': 1e-4,
    'hidden_dim': 64,
    'dropout': 0.3,
    'heads': 2,
    'threshold': 6.0,
}


def train_single_model(model, x, y, edge_index, train_idx, device,
                       epochs=300, lr=0.005, weight_decay=1e-4,
                       is_fusion=False, edge_index_ekg=None):
    model = model.to(device)
    x = x.to(device)
    y = y.to(device)
    edge_index = edge_index.to(device)
    if edge_index_ekg is not None:
        edge_index_ekg = edge_index_ekg.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        if is_fusion:
            pred, _ = model(x, edge_index, edge_index_ekg)
        else:
            pred = model(x, edge_index)

        loss = criterion(pred[train_idx], y[train_idx])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    return model


def evaluate_model(model, x, y, edge_index, test_idx, device, scaler_y,
                   is_fusion=False, edge_index_ekg=None):
    model.eval()
    x = x.to(device)
    y = y.to(device)
    edge_index = edge_index.to(device)
    if edge_index_ekg is not None:
        edge_index_ekg = edge_index_ekg.to(device)

    with torch.no_grad():
        if is_fusion:
            pred, attn_weights = model(x, edge_index, edge_index_ekg)
        else:
            pred = model(x, edge_index)
            attn_weights = None

    pred_np = pred[test_idx].cpu().numpy()
    y_np = y[test_idx].cpu().numpy()

    if scaler_y is not None:
        pred_np = scaler_y.inverse_transform(pred_np.reshape(-1, 1)).flatten()
        y_np = scaler_y.inverse_transform(y_np.reshape(-1, 1)).flatten()

    r2 = r2_score(y_np, pred_np)
    mae = mean_absolute_error(y_np, pred_np)
    rmse = np.sqrt(mean_squared_error(y_np, pred_np))

    return {
        'r2': r2,
        'mae': mae,
        'rmse': rmse,
        'predictions': pred_np,
        'targets': y_np,
        'attention_weights': attn_weights
    }
