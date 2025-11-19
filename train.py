import torch
import torch.nn as nn
import os
import argparse
import pandas as pd
from tqdm import tqdm
from wiktext_loader import WikitextLoader

# Import both architectures
from algebraic_model import AlgebraicTransformerLM 
from gpt_model import GPT, GPTConfig # Direct NanoGPT imports

# --- CONFIGURATIONS ---
# 'power' is only used by the Algebraic model
CONFIGS = {
    'small': {
        'd_model': 384, 'n_head': 6, 'n_layers': 6, 'batch_size': 64, 
        'lr': 1e-3, 'dropout': 0.05, 'power': 32
    },
    'medium': {
        'd_model': 768, 'n_head': 12, 'n_layers': 12, 'batch_size': 64, 
        'lr': 6e-4, 'dropout': 0.1, 'power': 32
    },
    'large': {
        'd_model': 1024, 'n_head': 16, 'n_layers': 24, 'batch_size': 64, 
        'lr': 3e-4, 'dropout': 0.1, 'power': 32
    }
}

def evaluate(model, val_loader, device):
    """
    Runs validation on the FULL validation set.
    Returns: Average Loss, Average Next-Token Accuracy
    """
    model.eval()
    total_loss = 0
    total_correct = 0
    total_tokens = 0
    count = 0
    
    val_x, val_y = val_loader[0], val_loader[1]
    limit = len(val_x)
    
    with torch.no_grad():
        for i in range(limit):
            xb = val_x[i].to(device)
            yb = val_y[i].to(device)
            
            # NanoGPT and AlgebraicLM both support this signature
            logits, loss = model(xb, yb)
            total_loss += loss.item()
            
            if hasattr(model, 'final_softmax'):
                probs = model.final_softmax(logits)
            else:
                probs = torch.nn.functional.softmax(logits, dim=-1)
                
            preds = probs.argmax(dim=-1)
            total_correct += (preds == yb).sum().item()
            total_tokens += yb.numel()
            count += 1

    model.train()
    return total_loss / count, total_correct / total_tokens

def main():
    parser = argparse.ArgumentParser()
    # Renamed choice to 'nanogpt'
    parser.add_argument('--model_type', type=str, required=True, choices=['nanogpt', 'algebraic'])
    parser.add_argument('--size', type=str, default='small', choices=['small', 'medium', 'large'])
    args = parser.parse_args()
    
    cfg = CONFIGS[args.size]
    
    D_MODEL = cfg['d_model']
    N_HEAD = cfg['n_head']
    N_LAYERS = cfg['n_layers']
    BATCH_SIZE = cfg['batch_size']
    DROPOUT = cfg['dropout']
    POWER = cfg.get('power', 32)
    
    LR = cfg['lr'] 
    
    D_FFN = 4 * D_MODEL
    BLOCK_SIZE = 128
    WEIGHT_DECAY = 0.1
    EPOCHS = 1
    GRAD_CLIP = 1.0
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    RUN_NAME = f"{args.model_type}_{args.size}"
    LOG_FILE = f"{RUN_NAME}.csv"
    CHECKPOINT_DIR = f"checkpoints/{RUN_NAME}"
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    print(f"--- Training {args.model_type.upper()} Transformer ({args.size}) ---")
    print(f"Config: {D_MODEL}d / {N_LAYERS}L / {N_HEAD}h | LR: {LR}")
    print(f"Training for exactly {EPOCHS} epoch.")
    
    print("Loading Data...")
    loader = WikitextLoader(BLOCK_SIZE, BATCH_SIZE)
    train_data = loader.get_loader('train')
    val_data = loader.get_loader('validation')
    
    VOCAB_SIZE = train_data[2]
    train_x, train_y = train_data[0], train_data[1]
    
    print(f"Vocab Size: {VOCAB_SIZE}")
    print(f"Training Batches: {len(train_x)}")
    
    # 2. Initialize Model
    if args.model_type == 'algebraic':
        model = AlgebraicTransformerLM(
            vocab_size=VOCAB_SIZE, d_model=D_MODEL, n_head=N_HEAD,
            n_layers=N_LAYERS, d_ffn=D_FFN, block_size=BLOCK_SIZE, dropout=DROPOUT,
            power=POWER
        ).to(DEVICE)
    else:
        # Initialize NanoGPT via Config
        # NanoGPT expects Bias=True by default for GPT2 reproduction
        nanogpt_config = GPTConfig(
            block_size=BLOCK_SIZE,
            vocab_size=VOCAB_SIZE,
            n_layer=N_LAYERS,
            n_head=N_HEAD,
            n_embd=D_MODEL,
            dropout=DROPOUT,
            bias=True 
        )
        model = GPT(nanogpt_config).to(DEVICE)
    
    print(f"Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    total_steps = len(train_x) * EPOCHS
    
    warmup_steps = int(0.1 * total_steps)
    scheduler_warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps)
    scheduler_decay = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[scheduler_warmup, scheduler_decay], milestones=[warmup_steps])
    
    if not os.path.exists(LOG_FILE):
        pd.DataFrame(columns=['step', 'train_loss', 'val_loss', 'val_acc', 'lr']).to_csv(LOG_FILE, index=False)
    
    model.train()
    step = 0
    
    for epoch in range(EPOCHS):
        print(f"--- Epoch {epoch+1}/{EPOCHS} ---")
        progress_bar = tqdm(range(len(train_x)))
        
        for i in progress_bar:
            xb, yb = train_x[i].to(DEVICE), train_y[i].to(DEVICE)
            
            _, loss = model(xb, yb)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            scheduler.step()
            
            step += 1
            progress_bar.set_description(f"Train Loss: {loss.item():.4f}")
            
            if step % 100 == 0:
                val_loss, val_acc = evaluate(model, val_data, DEVICE)
                current_lr = scheduler.get_last_lr()[0]
                tqdm.write(f"Step {step} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | LR: {current_lr:.2e}")
                
                log_entry = pd.DataFrame([[step, loss.item(), val_loss, val_acc, current_lr]], 
                                       columns=['step', 'train_loss', 'val_loss', 'val_acc', 'lr'])
                log_entry.to_csv(LOG_FILE, mode='a', header=False, index=False)
                torch.save(model.state_dict(), f"{CHECKPOINT_DIR}/latest_model.pt")
            
            if step % 1000 == 0:
                torch.save(model.state_dict(), f"{CHECKPOINT_DIR}/step_{step}.pt")

    print(f"Training Complete. Log saved to {LOG_FILE}")

if __name__ == "__main__":
    main()
