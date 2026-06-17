# -*- coding: utf-8 -*-
"""
Inference Latency Measurement Script (v2)
Supplements Table 3 with Inference Latency and Throughput metrics.

Measurement protocol:
- model.eval() + torch.no_grad()
- CUDA event timing (torch.cuda.Event), accuracy ~0.5us
- Excludes data loading / preprocessing / backward / optimizer
- warmup=10, repeat=100
- hidden_dim=64, num_layers=2, dropout=0.1 (matches Table 3 experiments)
- k-NN=5 (matches main experiments)

Output units:
- Absolute inference time: us (microseconds)
- Per-node latency: us/node and ns/node
- Throughput: nodes/s
"""
import os
import sys
import csv
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv
from datetime import datetime

HIDDEN_DIM = 64
DROPOUT = 0.1
NUM_HEADS_CA = 4
K_NEIGHBORS = 5
WARMUP = 10
REPEAT = 100

CASES = [
    ("CaseA_ForestBiomass", 4000, 12),
    ("CaseB_CaliforniaHousing", 20640, 8),
    ("CaseC_USCountyPoverty", 2811, 9),
    ("CaseD_EuroSAT", 27000, 512),
]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = SCRIPT_DIR


class TKGModel(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim=1, dropout=0.1):
        super().__init__()
        self.conv1 = GATConv(in_dim, hidden_dim, heads=1, concat=False)
        self.conv2 = GATConv(hidden_dim, hidden_dim, heads=1, concat=False)
        self.fc = nn.Linear(hidden_dim, out_dim)
        self.dropout = dropout

    def forward(self, x, edge_index):
        h = F.relu(self.conv1(x, edge_index))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = F.relu(self.conv2(h, edge_index))
        h = F.dropout(h, p=self.dropout, training=self.training)
        return self.fc(h)


class EKGModel(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim=1, dropout=0.1):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, out_dim)
        self.dropout = dropout

    def forward(self, x, edge_index):
        h = F.relu(self.conv1(x, edge_index))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = F.relu(self.conv2(h, edge_index))
        h = F.dropout(h, p=self.dropout, training=self.training)
        return self.fc(h)


class LinearFusionModel(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim=1, dropout=0.1, alpha=0.5):
        super().__init__()
        self.tkg_conv1 = GATConv(in_dim, hidden_dim, heads=1, concat=False)
        self.tkg_conv2 = GATConv(hidden_dim, hidden_dim, heads=1, concat=False)
        self.ekg_conv1 = GCNConv(in_dim, hidden_dim)
        self.ekg_conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, out_dim)
        self.dropout = dropout
        self.alpha = alpha

    def forward(self, x, tkg_edge_index, ekg_edge_index):
        h_tkg = F.relu(self.tkg_conv1(x, tkg_edge_index))
        h_tkg = F.dropout(h_tkg, p=self.dropout, training=self.training)
        h_tkg = F.relu(self.tkg_conv2(h_tkg, tkg_edge_index))
        h_tkg = F.dropout(h_tkg, p=self.dropout, training=self.training)

        h_ekg = F.relu(self.ekg_conv1(x, ekg_edge_index))
        h_ekg = F.dropout(h_ekg, p=self.dropout, training=self.training)
        h_ekg = F.relu(self.ekg_conv2(h_ekg, ekg_edge_index))
        h_ekg = F.dropout(h_ekg, p=self.dropout, training=self.training)

        h_fused = self.alpha * h_tkg + (1 - self.alpha) * h_ekg
        return self.fc(h_fused)


class CrossAttentionFusionModel(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim=1, dropout=0.1, num_heads=4):
        super().__init__()
        self.tkg_conv1 = GATConv(in_dim, hidden_dim, heads=1, concat=False)
        self.tkg_conv2 = GATConv(hidden_dim, hidden_dim, heads=1, concat=False)
        self.ekg_conv1 = GCNConv(in_dim, hidden_dim)
        self.ekg_conv2 = GCNConv(hidden_dim, hidden_dim)

        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        assert hidden_dim % num_heads == 0

        self.W_q = nn.Linear(hidden_dim, hidden_dim)
        self.W_k = nn.Linear(hidden_dim, hidden_dim)
        self.W_v = nn.Linear(hidden_dim, hidden_dim)

        self.fusion_mlp = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim)
        )
        self.dropout = dropout

    def cross_attention(self, query_emb, key_value_emb):
        N = query_emb.size(0)
        Q = self.W_q(query_emb).view(N, self.num_heads, self.head_dim)
        K = self.W_k(key_value_emb).view(N, self.num_heads, self.head_dim)
        V = self.W_v(key_value_emb).view(N, self.num_heads, self.head_dim)

        scores = torch.sum(Q * K, dim=-1) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(scores.unsqueeze(-1), dim=1)
        attended = (attn_weights * V).view(N, -1)
        return attended

    def forward(self, x, tkg_edge_index, ekg_edge_index):
        h_tkg = F.relu(self.tkg_conv1(x, tkg_edge_index))
        h_tkg = F.dropout(h_tkg, p=self.dropout, training=self.training)
        h_tkg = F.relu(self.tkg_conv2(h_tkg, tkg_edge_index))
        h_tkg = F.dropout(h_tkg, p=self.dropout, training=self.training)

        h_ekg = F.relu(self.ekg_conv1(x, ekg_edge_index))
        h_ekg = F.dropout(h_ekg, p=self.dropout, training=self.training)
        h_ekg = F.relu(self.ekg_conv2(h_ekg, ekg_edge_index))
        h_ekg = F.dropout(h_ekg, p=self.dropout, training=self.training)

        h_attended = self.cross_attention(h_tkg, h_ekg)
        h_concat = torch.cat([h_tkg, h_ekg, h_attended], dim=-1)
        return self.fusion_mlp(h_concat)


def build_synthetic_edges(num_nodes, k=5, seed=42):
    """Build synthetic k-NN edge index."""
    rng = np.random.RandomState(seed)
    src, dst = [], []
    for i in range(num_nodes):
        neighbors = rng.choice(num_nodes, size=k, replace=False)
        for j in neighbors:
            if j != i:
                src.append(i)
                dst.append(j)
    return torch.tensor([src, dst], dtype=torch.long)


def measure_inference(model, forward_fn, device, warmup=WARMUP, repeat=REPEAT):
    """
    Measure inference latency and return a list of times in milliseconds
    using CUDA event timing.
    """
    model.eval()

    with torch.no_grad():
        for _ in range(warmup):
            forward_fn()
            if device.type == 'cuda':
                torch.cuda.synchronize()

        times_ms = []
        for _ in range(repeat):
            if device.type == 'cuda':
                torch.cuda.synchronize()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                forward_fn()
                end.record()
                torch.cuda.synchronize()
                times_ms.append(start.elapsed_time(end))  # ms
            else:
                import time
                t0 = time.perf_counter()
                forward_fn()
                times_ms.append((time.perf_counter() - t0) * 1000.0)

    return times_ms


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gpu_name = torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'
    print(f"Device: {device}")
    print(f"GPU: {gpu_name}")
    print(f"Hidden dim: {HIDDEN_DIM}, Dropout: {DROPOUT}, k-NN: {K_NEIGHBORS}")
    print(f"Warmup: {WARMUP}, Repeat: {REPEAT}")
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    all_raw_rows = []
    all_summary_rows = []

    for case_name, num_nodes, in_dim in CASES:
        print(f"{'='*70}")
        print(f"  {case_name}: N={num_nodes}, d={in_dim}")
        print(f"{'='*70}")

        edge_tkg = build_synthetic_edges(num_nodes, K_NEIGHBORS, seed=42).to(device)
        edge_ekg = build_synthetic_edges(num_nodes, K_NEIGHBORS, seed=123).to(device)
        edge_single = build_synthetic_edges(num_nodes, K_NEIGHBORS, seed=42).to(device)
        x = torch.randn(num_nodes, in_dim, device=device)

        models_config = [
            ("TKG", TKGModel(in_dim, HIDDEN_DIM, dropout=DROPOUT).to(device),
             lambda m, x=x, e=edge_single: m(x, e)),
            ("EKG", EKGModel(in_dim, HIDDEN_DIM, dropout=DROPOUT).to(device),
             lambda m, x=x, e=edge_single: m(x, e)),
            ("LinearFusion", LinearFusionModel(in_dim, HIDDEN_DIM, dropout=DROPOUT).to(device),
             lambda m, x=x, et=edge_tkg, ee=edge_ekg: m(x, et, ee)),
            ("CrossAttention", CrossAttentionFusionModel(in_dim, HIDDEN_DIM, dropout=DROPOUT,
                                                         num_heads=NUM_HEADS_CA).to(device),
             lambda m, x=x, et=edge_tkg, ee=edge_ekg: m(x, et, ee)),
        ]

        for model_name, model, fwd_fn in models_config:
            fn = lambda m=model: fwd_fn(m)
            times_ms = measure_inference(model, fn, device)

            times_us = [t * 1000.0 for t in times_ms]

            for i, (t_ms, t_us) in enumerate(zip(times_ms, times_us)):
                latency_us_per_node = t_us / num_nodes
                latency_ns_per_node = latency_us_per_node * 1000.0
                throughput = num_nodes / (t_ms / 1000.0)

                all_raw_rows.append({
                    "model": model_name,
                    "case": case_name,
                    "num_nodes": num_nodes,
                    "repeat": i,
                    "inference_time_ms": round(t_ms, 6),
                    "inference_time_us": round(t_us, 3),
                    "latency_us_per_node": round(latency_us_per_node, 6),
                    "latency_ns_per_node": round(latency_us_per_node * 1000, 3),
                    "throughput_nodes_per_sec": round(throughput, 2),
                    "device": str(device),
                    "gpu_name": gpu_name,
                })

            mean_us = np.mean(times_us)
            std_us = np.std(times_us)
            mean_us_per_node = np.mean([t / num_nodes for t in times_us])
            std_us_per_node = np.std([t / num_nodes for t in times_us])
            mean_ns_per_node = mean_us_per_node * 1000
            std_ns_per_node = std_us_per_node * 1000
            mean_tp = np.mean([num_nodes / (t / 1000.0) for t in times_ms])
            std_tp = np.std([num_nodes / (t / 1000.0) for t in times_ms])

            all_summary_rows.append({
                "model": model_name,
                "case": case_name,
                "num_nodes": num_nodes,
                "mean_inference_us": round(mean_us, 3),
                "std_inference_us": round(std_us, 3),
                "mean_latency_us_per_node": round(mean_us_per_node, 6),
                "std_latency_us_per_node": round(std_us_per_node, 6),
                "mean_latency_ns_per_node": round(mean_ns_per_node, 3),
                "std_latency_ns_per_node": round(std_ns_per_node, 3),
                "mean_throughput_nodes_per_sec": round(mean_tp, 2),
                "std_throughput_nodes_per_sec": round(std_tp, 2),
            })

            print(f"  {model_name:15s} | graph={mean_us:8.1f}+/-{std_us:5.1f} us | "
                  f"per_node={mean_ns_per_node:7.1f}+/-{std_ns_per_node:5.1f} ns | "
                  f"throughput={mean_tp/1e6:5.2f}M nodes/s")

        print()

    print("Computing Single-source averages (TKG+EKG)...")
    for case_name, num_nodes, in_dim in CASES:
        tkg = [r for r in all_summary_rows if r["model"] == "TKG" and r["case"] == case_name][0]
        ekg = [r for r in all_summary_rows if r["model"] == "EKG" and r["case"] == case_name][0]

        mean_graph = (tkg["mean_inference_us"] + ekg["mean_inference_us"]) / 2
        std_graph = math.sqrt(tkg["std_inference_us"]**2 + ekg["std_inference_us"]**2) / 2
        mean_ns = (tkg["mean_latency_ns_per_node"] + ekg["mean_latency_ns_per_node"]) / 2
        std_ns = math.sqrt(tkg["std_latency_ns_per_node"]**2 + ekg["std_latency_ns_per_node"]**2) / 2
        mean_tp = (tkg["mean_throughput_nodes_per_sec"] + ekg["mean_throughput_nodes_per_sec"]) / 2
        std_tp = math.sqrt(tkg["std_throughput_nodes_per_sec"]**2 + ekg["std_throughput_nodes_per_sec"]**2) / 2

        all_summary_rows.append({
            "model": "SingleSource_avg",
            "case": case_name,
            "num_nodes": num_nodes,
            "mean_inference_us": round(mean_graph, 3),
            "std_inference_us": round(std_graph, 3),
            "mean_latency_us_per_node": round(mean_ns / 1000, 6),
            "std_latency_us_per_node": round(std_ns / 1000, 6),
            "mean_latency_ns_per_node": round(mean_ns, 3),
            "std_latency_ns_per_node": round(std_ns, 3),
            "mean_throughput_nodes_per_sec": round(mean_tp, 2),
            "std_throughput_nodes_per_sec": round(std_tp, 2),
        })

    raw_fields = [
        "model", "case", "num_nodes", "repeat",
        "inference_time_ms", "inference_time_us",
        "latency_us_per_node", "latency_ns_per_node",
        "throughput_nodes_per_sec", "device", "gpu_name"
    ]
    raw_path = os.path.join(OUTPUT_DIR, "inference_cost_table5_raw.csv")
    with open(raw_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=raw_fields)
        w.writeheader()
        for row in all_raw_rows:
            w.writerow({k: row.get(k, "") for k in raw_fields})
    print(f"\nRaw CSV: {raw_path}")

    summary_fields = [
        "model", "case", "num_nodes",
        "mean_inference_us", "std_inference_us",
        "mean_latency_us_per_node", "std_latency_us_per_node",
        "mean_latency_ns_per_node", "std_latency_ns_per_node",
        "mean_throughput_nodes_per_sec", "std_throughput_nodes_per_sec"
    ]
    summary_path = os.path.join(OUTPUT_DIR, "inference_cost_table5_summary.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=summary_fields)
        w.writeheader()
        for row in all_summary_rows:
            w.writerow({k: row.get(k, "") for k in summary_fields})
    print(f"Summary CSV: {summary_path}")

    print(f"\n{'='*100}")
    print(f"  INFERENCE COST SUMMARY — GPU: {gpu_name}")
    print(f"{'='*100}")
    print(f"{'Model':<18} {'Case':<26} {'N':>6} {'Graph (us)':>18} {'Per-node (ns)':>18} {'Throughput':>18}")
    print("-" * 100)
    for row in all_summary_rows:
        g = f"{row['mean_inference_us']:.1f}+/-{row['std_inference_us']:.1f}"
        n = f"{row['mean_latency_ns_per_node']:.1f}+/-{row['std_latency_ns_per_node']:.1f}"
        t = f"{row['mean_throughput_nodes_per_sec']/1e6:.2f}M"
        print(f"{row['model']:<18} {row['case']:<26} {row['num_nodes']:>6} {g:>18} {n:>18} {t:>18}")

    print(f"\nEnd: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
