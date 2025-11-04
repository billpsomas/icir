"""
Utility functions for icir feature extraction and retrieval.
"""

import numpy as np
import torch
import csv


# =============================================================================
# Device Setup
# =============================================================================

def setup_device(gpu_id):
    """
    Setup CUDA device or fallback to CPU.
    
    Args:
        gpu_id: GPU device ID
    
    Returns:
        torch.device object
    """
    if torch.cuda.is_available() and gpu_id < torch.cuda.device_count():
        device = torch.device(f"cuda:{gpu_id}")
        print(f"Using GPU: {gpu_id}")
    else:
        print(f"GPU {gpu_id} not available. Using CPU instead.")
        device = torch.device("cpu")
    return device

# =============================================================================
# Evaluation Metrics
# =============================================================================

def compute_ap(ranks, nres):
    """
    Compute Average Precision for a single query.
    
    Args:
        ranks: Array of ranks where correct matches occur
        nres: Total number of relevant results
    
    Returns:
        Average precision score
    """
    num_ranks = len(ranks)
    ap = 0.0
    recall_step = 1.0 / (nres + 1e-5)
    
    for j in range(num_ranks):
        rank = ranks[j]
        # Precision at rank j
        precision_0 = 1.0 if rank == 0 else float(j) / rank
        precision_1 = float(j + 1) / (rank + 1)
        # Trapezoidal rule
        ap += (precision_0 + precision_1) * recall_step / 2.0
    
    return ap


def compute_map(correct):
    """
    Compute mean Average Precision across multiple queries.
    
    Args:
        correct: Binary matrix (num_queries, num_database) where 1 indicates correct match
    
    Returns:
        Tuple of (mAP score as percentage, list of per-query APs)
    """
    num_queries = correct.shape[0]
    ap_list = []
    
    for i in range(num_queries):
        # Find positions of correct matches
        positive_positions = np.where(correct[i] != 0)[0]
        ap = compute_ap(positive_positions, len(positive_positions))
        ap_list.append(ap)
    
    mAP = np.mean(ap_list)
    return np.around(mAP * 100, decimals=2), ap_list