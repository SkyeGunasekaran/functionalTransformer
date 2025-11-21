import torch
import os
import argparse
import csv
import time
import random
import math
import numpy as np
from tqdm import tqdm
from data.wikitext_loader import get_dataloaders
from models.fast_algebraic import FastAlgebraicTransformerLM
from models.nanogpt import GPT, GPTConfig 

# -----------------------------------------------------------------------------
# 1. Scientific Configuration
# -----------------------------------------------------------------------------
CONFIGS = {
    # Target: ~50M Params
    # Breakdown: 26M (Embeddings) + 25M (Layers) = ~51M Total
    'small':  {
        'd_model': 512, 
        'n_head': 8,  
        'n_layers': 8,   
        'batch_size': 8, 
        'accum_steps': 8, 
        'lr': 8e-4,      
        'dropout': 0.1
    },

    # Target: ~85M Params
    # Breakdown: 39M (Embeddings) + 42M (Layers) = ~81M Total
    'medium': {
        'd_model': 768, 
        'n_head': 12, 
        'n_layers': 6,   
        'batch_size': 8,  
        'accum_steps': 8, 
        'lr': 7e-4,    
        'dropout': 0.1
    },

    # Target: ~120M Params (Standard GPT-2 Small)
    # Breakdown: 39M (Embeddings) + 85M (Layers) = ~124M Total
    'large':  {
        'd_model': 768, 
        'n_head': 12, 
        'n_layers': 12, 
        'batch_size': 8,  
        'accum_steps': 8, 
        'lr': 6e-4,   
        'dropout': 0.1
    }
}

# -----------------------------------------------------------------------------
# 2. Power Scheduler (Curriculum Learning)
# -----------------------------------------------------------------------------
def update_model_power(model, progress, start_power=2.0, max_power=8.0):
    """
    Applies the annealing schedule to the Algebraic Attention.
    Schedule: Linear Ramp (0% -> 60%) -> Constant Max (60% -> 100%)
    """
    anneal_cutoff = 0.60 # Anneal over the first 60% of training
    
    if progress < anneal_cutoff:
        # Linear Ramp: Start -> Target
        pct = progress / anneal_cutoff
        new_power = start_power + (max_power - start_power) * pct
    else:
        # Constant Max Phase (The "Sharp" Phase)
        new_power = max_power
    
    # In-place update of registered buffers
    if hasattr(model, 'update_power'):
        model.update_power(new_power)
    
    return new_power

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.manual_seed_all(seed)
        # For strict scientific reproducibility (might slow down slightly)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_device():
    if torch.cuda.is_available(): return 'cuda'
    if torch.backends.mps.is_available(): return 'mps'
    return 'cpu'

@torch.no_grad()
def evaluate(model, val_loader, device, ctx):
    model.eval()
    total_loss = 0
    count = 0
    
    for xb, yb in val_loader:
        xb, yb = xb.to(device), yb.to(device)
        with ctx:
            _, loss = model(xb, yb)
        total_loss += loss.item()
        count += 1
        
    avg_loss = total_loss / count if count > 0 else 0
    model.train()
    return avg_loss

# -----------------------------------------------------------------------------
# 3. Main Loop
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, required=True, choices=['nanogpt', 'algebraic'])
    parser.add_argument('--size', type=str, default='small', choices=['small', 'medium', 'large'])
    parser.add_argument('--seed', type=int, default=1337)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=1) 
    parser.add_argument('--block_size', type=int, default=1024)
    args = parser.parse_args()

    set_seed(args.seed)
    DEVICE = get_device()
    # High precision for accumulation stability
    torch.set_float32_matmul_precision('high') 
    
    cfg = CONFIGS[args.size]
    
    # Run Config
    RUN_NAME = f"{args.model_type}_{args.size}_s{args.seed}"
    OUT_DIR = f"results/{RUN_NAME}"
    os.makedirs(OUT_DIR, exist_ok=True)
    
    # CSV Logger Setup
    csv_path = os.path.join(OUT_DIR, 'metrics.csv')
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    # Header
    csv_writer.writerow(['step', 'epoch', 'train_loss', 'val_loss', 'val_ppl', 'power', 'lr'])
    
    print(f"\n=== STARTING EXPERIMENT: {RUN_NAME} ===")
    print(f"Config: {cfg}")
    
    # Data
    print("Loading WikiText-103...")
    train_loader, val_loader, VOCAB_SIZE = get_dataloaders(
        cfg['batch_size'], args.block_size, num_workers=args.workers
    )
    
    # Calculation of steps
    steps_per_epoch = len(train_loader) // cfg['accum_steps']
    total_steps = steps_per_epoch * args.epochs
    print(f"Vocab: {VOCAB_SIZE} | Total Optimization Steps: {total_steps}")

    # Model
    if args.model_type == 'algebraic':
        model = FastAlgebraicTransformerLM(
            vocab_size=VOCAB_SIZE, 
            d_model=cfg['d_model'], 
            n_head=cfg['n_head'],
            n_layers=cfg['n_layers'], 
            d_ffn=4*cfg['d_model'], 
            block_size=args.block_size, 
            dropout=cfg['dropout'],
            power=2.0, # Initial Power
        )
    else:
        nanogpt_config = GPTConfig(
            block_size=args.block_size, vocab_size=VOCAB_SIZE, n_layer=cfg['n_layers'],
            n_head=cfg['n_head'], n_embd=cfg['d_model'], dropout=cfg['dropout'], bias=True
        )
        model = GPT(nanogpt_config)

    model.to(DEVICE)
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=0.1)
    
    # Scheduler (Warmup + Cosine Decay)
    warmup_steps = int(total_steps * 0.10) # 10% LR warmup
    scheduler1 = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=warmup_steps)
    scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(total_steps - warmup_steps))
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[scheduler1, scheduler2], milestones=[warmup_steps])

    # Mixed Precision
    pt_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    ctx = torch.amp.autocast(device_type=DEVICE, dtype=pt_dtype)
    scaler = torch.amp.GradScaler('cuda', enabled=(pt_dtype == torch.float16))

    # State Tracking
    model.train()
    train_loss_accum = 0.0
    log_step_accum = 0 
    global_step = 0
    current_power = 2.0
    start_time = time.time()

    # --- TRAINING LOOP ---
    optimizer.zero_grad(set_to_none=True)
    
    # Flatten loader for easier step tracking
    train_iter = iter(train_loader)
    
    # Progress bar based on optimization steps (not batches)
    pbar = tqdm(range(total_steps), desc="Training")
    
    for step in pbar:

        if args.model_type == 'algebraic':
            progress = step / total_steps
            current_power = update_model_power(model, progress)
        # Gradient Accumulation Loop
        for micro_step in range(cfg['accum_steps']):
            try:
                xb, yb = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader) # Restart epoch if needed
                xb, yb = next(train_iter)
            
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            
            with ctx:
                _, loss = model(xb, yb)
                # Scale loss for accumulation
                loss = loss / cfg['accum_steps']
            
            train_loss_accum += loss.item()
            scaler.scale(loss).backward()

        # Optimizer Step
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        
        global_step += 1
        log_step_accum += 1
        
        # --- PERIODIC EVALUATION (Every 100 steps) ---
        if global_step % 100 == 0 or global_step == total_steps:
            # Calculate metrics - FIX: Average over the steps since last log
            avg_train_loss = train_loss_accum / log_step_accum
            train_loss_accum = 0.0 # Reset accumulator
            log_step_accum = 0     # Reset step count
            
            val_loss = evaluate(model, val_loader, DEVICE, ctx)
            val_ppl = math.exp(val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            
            # Log to CSV
            csv_writer.writerow([global_step, 1, avg_train_loss, val_loss, val_ppl, current_power, current_lr])
            csv_file.flush() # Ensure data is written in case of crash
            
            # Update TQDM
            pbar.set_postfix({
                'Val Loss': f"{val_loss:.3f}",
                'PPL': f"{val_ppl:.1f}",
                'Power': f"{current_power:.1f}" if args.model_type == 'algebraic' else "N/A"
            })
            
            # Save intermediate checkpoint
            torch.save(model.state_dict(), os.path.join(OUT_DIR, 'latest_ckpt.pt'))

    # --- FINISH ---
    print("\nTraining Complete.")
    csv_file.close()
    
    # Save Final Model
    torch.save(model.state_dict(), os.path.join(OUT_DIR, 'final_model.pt'))
    print(f"Saved final model to {os.path.join(OUT_DIR, 'final_model.pt')}")

if __name__ == "__main__":
    main()
