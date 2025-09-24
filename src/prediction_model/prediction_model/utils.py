import torch
import numpy as np

NUM_JOINTS = 34
INPUT_SEQ_LEN = 30
OUTPUT_SEQ_LEN = 45

def get_adjacency_matrix():
    """
    Returns a skeletal adjacency matrix for a 34-joint model.
    This is a placeholder and should be adapted to your specific skeleton topology.
    An identity matrix is added for self-connections.
    """
    # Placeholder for a common 34-joint skeleton (e.g., from NVIDIA Isaac)
    # This matrix defines which joints are connected by limbs.
    # The actual connections depend on the joint mapping from your perception system.
    # Example connections (not a complete, accurate list):
    edges = [
        (0, 1), (1, 2), (2, 3), # Spine
        (2, 4), (4, 5), (5, 6), # Left Arm
        (2, 7), (7, 8), (8, 9), # Right Arm
        (0, 10), (10, 11), (11, 12), # Left Leg
        (0, 13), (13, 14), (14, 15) # Right Leg
    ]
    # Add more connections as per your skeleton model up to 33.

    adj = np.zeros((NUM_JOINTS, NUM_JOINTS))
    for i, j in edges:
        if i < NUM_JOINTS and j < NUM_JOINTS:
            adj[i, j] = 1
            adj[j, i] = 1
    
    # Add self-loops
    adj += np.eye(NUM_JOINTS)
    return torch.from_numpy(adj).float()

def normalize_adjacency(A):
    """Normalize the adjacency matrix."""
    D = torch.pow(A.sum(1).float(), -0.5)
    D = torch.diag(D)
    return D.mm(A).mm(D)