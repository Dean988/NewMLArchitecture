import torch
import torch.nn as nn
import torch.optim as optim
from src.model import LiquidMambaKAN
import time
import sys

def train_demo():
    print("="*60)
    print("LiquidMamba-KAN: Training Demonstration")
    print("="*60)
    
    # Hyperparameters
    BATCH_SIZE = 32
    SEQ_LEN = 128
    INPUT_DIM = 64
    D_MODEL = 128
    NUM_LAYERS = 4
    OUTPUT_DIM = 10
    EPOCHS = 5
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Model Initialization
    print(f"\nInitializing LiquidMamba-KAN Model...")
    model = LiquidMambaKAN(INPUT_DIM, D_MODEL, NUM_LAYERS, OUTPUT_DIM).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {param_count:,}")
    
    # Dummy Data
    print("Generating synthetic sequences (simulating complex temporal data)...")
    inputs = torch.randn(BATCH_SIZE * 10, SEQ_LEN, INPUT_DIM).to(device)
    targets = torch.randint(0, OUTPUT_DIM, (BATCH_SIZE * 10,)).to(device)
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Training Loop
    print("\nStarting Training Loop...")
    model.train()
    
    for epoch in range(EPOCHS):
        total_loss = 0
        start_time = time.time()
        
        # Simulate batches
        steps = len(inputs) // BATCH_SIZE
        
        for i in range(steps):
            batch_x = inputs[i*BATCH_SIZE : (i+1)*BATCH_SIZE]
            batch_y = targets[i*BATCH_SIZE : (i+1)*BATCH_SIZE]
            
            optimizer.zero_grad()
            
            # Forward pass
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Simple progress simulation
            if (i+1) % 5 == 0:
                sys.stdout.write(f"\r[Epoch {epoch+1}/{EPOCHS}] Step [{i+1}/{steps}] Loss: {loss.item():.4f}")
                sys.stdout.flush()
                
        avg_loss = total_loss / steps
        duration = time.time() - start_time
        print(f"\nExample Epoch {epoch+1} completed in {duration:.2f}s | Avg Loss: {avg_loss:.4f}")
        
    print("\n" + "="*60)
    print("Training Complete. Model is ready for inference.")
    print("="*60)

if __name__ == "__main__":
    train_demo()
