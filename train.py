import torch
from torch.cuda.amp import GradScaler, autocast
import os
import time
from data import Wikitext103Loader
from model import AlgebraicTransformerLM
# --- Hyperparameters ---
# Model Params (small, for an RTX 5090 this can be much larger)
VOCAB_SIZE = -1 # Will be set by loader
D_MODEL = 512
N_HEAD = 8
N_LAYERS = 6
D_FFN = 4 * D_MODEL
BLOCK_SIZE = 128     # Context window
DROPOUT = 0.1

# Training Params
N_EPOCHS = 10
BATCH_SIZE = 32
LR = 1e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DATA_DIR = "./data" # Directory to store Wikitext
CHECKPOINT_DIR = "./checkpoints"

# --- Helper Functions ---

def train(model, loader, optimizer, scaler, epoch):
    model.train()
    total_loss = 0
    start_time = time.time()
    
    for i, (xb, yb) in enumerate(loader):
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        
        # Forward pass with mixed precision
        with autocast():
            logits, loss = model(xb, yb)
        
        # Backward pass
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        
        # Print progress
        if i % 2000 == 0 and i > 0:
            elapsed = time.time() - start_time
            print(f"  Epoch {epoch+1} | Batch {i:5d}/{len(loader)} | Loss: {loss.item():.4f} | {elapsed:.2f}s")
    
    return total_loss / len(loader)

def validate(model, loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            with autocast():
                logits, loss = model(xb, yb)
            total_loss += loss.item()
    return total_loss / len(loader)

# --- Main Execution ---

if __name__ == "__main__":
    print(f"--- 1. Initializing Data Loader ---")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    loader = Wikitext103Loader(
        data_dir=DATA_DIR, 
        block_size=BLOCK_SIZE, 
        batch_size=BATCH_SIZE,
        min_freq=100 
    )
    train_loader, val_loader, VOCAB_SIZE = loader.load()
    
    print(f"\n--- 2. Initializing Model ---")
    model = AlgebraicTransformerLM(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        n_head=N_HEAD,
        n_layers=N_LAYERS,
        d_ffn=D_FFN,
        block_size=BLOCK_SIZE
    )
    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scaler = GradScaler()
    
    print(f"Model: {N_LAYERS} layers, {D_MODEL} d_model, {N_HEAD} heads, {BLOCK_SIZE} block_size")
    print(f"Params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    print(f"Device: {DEVICE}")

    print("\n--- 3. Starting Training ---")
    for epoch in range(N_EPOCHS):
        epoch_start_time = time.time()
        
        # Train
        train_loss = train(model, train_loader, optimizer, scaler, epoch)
        
        # Validate
        val_loss = validate(model, val_loader)
        
        # Calculate Perplexity (PPL)
        train_ppl = torch.exp(torch.tensor(train_loss))
        val_ppl = torch.exp(torch.tensor(val_loss))
        
        # Log results
        elapsed = time.time() - epoch_start_time
        print("-" * 60)
        print(f"Epoch {epoch+1:2d}/{N_EPOCHS} Summary:")
        print(f"  Time: {elapsed:.2f}s")
        print(f"  Train Loss: {train_loss:.4f} | Train PPL: {train_ppl:7.2f}")
        print(f"  Val Loss:   {val_loss:.4f} | Val PPL:   {val_ppl:7.2f}")
        
        # Save model checkpoint
        save_path = os.path.join(CHECKPOINT_DIR, f"algtr_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), save_path)
        print(f"  Checkpoint saved to {save_path}")
        print("-" * 60)

    print("--- 4. Training Complete ---")
