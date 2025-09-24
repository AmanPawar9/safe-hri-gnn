import torch
import torch.nn as nn
import numpy as np
from torch_geometric.nn import GCNConv
from .utils import get_adjacency_matrix, normalize_adjacency

class STGCNBlock(nn.Module):
    """Spatio-Temporal Graph Convolutional Block"""
    def __init__(self, in_features, out_features, num_nodes):
        super(STGCNBlock, self).__init__()
        self.gcn = GCNConv(in_channels=in_features, out_channels=out_features)
        self.gru = nn.GRU(input_size=out_features * num_nodes,
                          hidden_size=out_features * num_nodes,
                          batch_first=True)
        self.ln = nn.LayerNorm(out_features * num_nodes)

    def forward(self, x, edge_index, h=None):
        # x shape: (batch, seq_len, num_nodes, in_features)
        batch, seq_len, num_nodes, _ = x.shape
        
        # --- Spatial Convolution ---
        gcn_out = []
        for t in range(seq_len):
            frame = x[:, t, :, :] # (batch, num_nodes, in_features)
            frame_gcn = self.gcn(frame, edge_index)
            frame_gcn = torch.relu(frame_gcn)
            gcn_out.append(frame_gcn)
        x_gcn = torch.stack(gcn_out, dim=1) # (batch, seq_len, num_nodes, out_features)

        # --- Temporal Modeling ---
        # Flatten nodes and features for GRU
        x_flat = x_gcn.view(batch, seq_len, -1) # (batch, seq_len, num_nodes * out_features)
        x_gru, h_out = self.gru(x_flat, h) # x_gru: (batch, seq_len, hidden_size)
        
        # Apply layer normalization and reshape back
        x_gru_norm = self.ln(x_gru)
        x_out = x_gru_norm.view(batch, seq_len, num_nodes, -1) # (batch, seq_len, num_nodes, out_features)
        
        return x_out, h_out


class STGNNModel(torch.nn.Module):
    """Spatio-Temporal Graph Neural Network for Human Motion Prediction."""
    def __init__(self, num_nodes=34, in_features=3, out_features=3,
                 input_seq_len=30, output_seq_len=45):
        super(STGNNModel, self).__init__()
        self.num_nodes = num_nodes
        self.output_seq_len = output_seq_len
        
        # --- Adjacency Matrix ---
        adj = get_adjacency_matrix()
        self.edge_index = (adj > 0).nonzero().t().contiguous()
        self.edge_index = self.edge_index.long() # Ensure it's LongTensor
        
        # --- Model Layers ---
        self.input_embed = nn.Linear(in_features, 128)
        
        self.st_block1 = STGCNBlock(128, 64, num_nodes)
        self.st_block2 = STGCNBlock(64, 64, num_nodes)
        self.st_block3 = STGCNBlock(64, 128, num_nodes)
        
        self.output_layer = nn.Linear(128, out_features)
        self.decoder_fc = nn.Linear(128 * num_nodes, output_seq_len * num_nodes * out_features)
        

    def forward(self, x: torch.Tensor, adjacency_matrix: torch.Tensor = None) -> torch.Tensor:
        """
        x: Tensor of shape (batch, seq_in, nodes, feats)
        Returns: Tensor of shape (batch, seq_out, nodes, 3)
        """
        batch_size = x.shape[0]
        device = x.device
        
        # Move edge_index to the correct device
        self.edge_index = self.edge_index.to(device)
        
        # 1. Input Embedding
        x = self.input_embed(x)
        x = torch.relu(x)

        # 2. Spatio-Temporal Blocks (Encoder)
        x, h1 = self.st_block1(x, self.edge_index)
        x, h2 = self.st_block2(x, self.edge_index, h1)
        x, h3 = self.st_block3(x, self.edge_index, h2)
        
        # 3. Decoder
        # Use the last hidden state from the final GRU as the context vector
        last_hidden_state = h3.squeeze(0) # Shape: (batch, num_nodes * 128)
        
        # 4. Output Layer to predict the full future sequence
        out = self.decoder_fc(last_hidden_state)
        out = out.view(batch_size, self.output_seq_len, self.num_nodes, -1)
        
        return out

def train_model(model: STGNNModel, train_loader: torch.utils.data.DataLoader, optimizer, loss_fn) -> float:
    """Trains the model for one epoch and returns the average loss."""
    model.train()
    total_loss = 0
    for batch in train_loader:
        inputs, targets = batch
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def predict_future_poses(model: STGNNModel, pose_history: list) -> list:
    """
    Takes a history of poses and returns a predicted future sequence.
    pose_history: List of 30 frames, where each frame is a list of 34 joint (x, y, z) tuples.
    Returns: A list of 45 predicted frames.
    """
    model.eval()
    with torch.no_grad():
        # 1. Convert input to tensor
        input_array = np.array(pose_history).astype(np.float32) # Shape: (30, 34, 3)
        input_tensor = torch.from_numpy(input_array).unsqueeze(0) # Add batch dim -> (1, 30, 34, 3)

        # 2. Run inference
        predicted_tensor = model(input_tensor) # Shape: (1, 45, 34, 3)
        
        # 3. Convert output back to list of lists of tuples
        predicted_array = predicted_tensor.squeeze(0).cpu().numpy() # Shape: (45, 34, 3)
        
        output_list = []
        for frame in predicted_array:
            frame_tuples = [tuple(joint) for joint in frame]
            output_list.append(frame_tuples)
            
        return output_list