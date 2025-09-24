import torch
import torch.optim as optim
import torch.nn as nn
import argparse
from prediction_model.model import STGNNModel, train_model
from prediction_model.utils import INPUT_SEQ_LEN, OUTPUT_SEQ_LEN, NUM_JOINTS

# This is a placeholder for a real data loader.
class MockDataLoader:
    def __init__(self, batch_size=32):
        self.batch_size = batch_size

    def __iter__(self):
        for _ in range(100): # Simulate 100 batches per epoch
            inputs = torch.randn(self.batch_size, INPUT_SEQ_LEN, NUM_JOINTS, 3)
            targets = torch.randn(self.batch_size, OUTPUT_SEQ_LEN, NUM_JOINTS, 3)
            yield inputs, targets

    def __len__(self):
        return 100

def main():
    parser = argparse.ArgumentParser(description="Train ST-GNN model.")
    parser.add_argument('--data_path', type=str, required=True, help="Path to processed HDF5 data.")
    parser.add_argument('--epochs', type=int, default=50, help="Number of training epochs.")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate.")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size.")
    parser.add_argument('--output_path', type=str, default="stgnn_model.pth", help="Path to save the trained model.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Model, Optimizer, Loss ---
    model = STGNNModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.L1Loss() # Mean Absolute Error (MAE) as per PRD

    # --- Data Loading ---
    # Replace MockDataLoader with your actual HDF5 data loader
    print(f"Loading data from {args.data_path}...")
    train_loader = MockDataLoader(batch_size=args.batch_size)
    print("Data loading complete.")

    # --- Training Loop ---
    print("Starting training...")
    for epoch in range(args.epochs):
        avg_loss = train_model(model, train_loader, optimizer, loss_fn)
        print(f"Epoch {epoch+1}/{args.epochs}, Average Loss: {avg_loss:.4f}")

    # --- Save Model ---
    torch.save(model.state_dict(), args.output_path)
    print(f"Training complete. Model saved to {args.output_path}")

if __name__ == '__main__':
    main()