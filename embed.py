import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List
import json
import random
from tqdm import tqdm

# --- Configuration Management ---
@dataclass
class TrainingConfig:
    """Centralized configuration for training hyperparameters."""
    glove_data_file: str = "glove_data.pt"
    final_embedding_file: str = "functional_embeddings_v1.pt"
    checkpoint_dir: str = "checkpoints"
    
    # Model architecture
    embedding_dim: int = 128
    num_samples: int = 100
    
    # Training hyperparameters
    regularization_strength: float = 0.001
    num_epochs: int = 15
    learning_rate: float = 0.001
    batch_size: int = 1024
    gradient_clip_val: float = 1.0
    
    # Dataset parameters
    steps_per_epoch: int = 10000
    validation_split: float = 0.1
    validation_steps: int = 1000
    
    # Early stopping
    early_stopping_patience: int = 3
    
    # System settings
    num_workers: int = 4
    pin_memory: bool = True
    seed: int = 42
    
    @property
    def t_min(self) -> float:
        return -(self.embedding_dim / 2.0)
    
    @property
    def t_max(self) -> float:
        return (self.embedding_dim / 2.0)

def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility across libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For deterministic behavior (may slightly impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# --- Model Definition ---
class FunctionalEmbeddingModel(nn.Module):
    """B-spline based functional embedding model."""
    
    def __init__(self, vocab_size: int, embedding_dim: int, num_samples: int, t_min: float, t_max: float):
        super().__init__()
        self.embedding_layer = nn.Embedding(vocab_size, embedding_dim)
        self.register_buffer('t', torch.linspace(t_min, t_max, num_samples), persistent=True)
        self.embedding_dim = embedding_dim
        self.num_samples = num_samples
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize embedding weights with small random values."""
        nn.init.normal_(self.embedding_layer.weight, mean=0.0, std=0.1)

    def _synthesize_function(self, params: torch.Tensor) -> torch.Tensor:
        """Generate functional embeddings from control points via linear interpolation."""
        control_points = params.unsqueeze(1)  # [batch, 1, embedding_dim]
        function = F.interpolate(
            control_points,
            size=self.num_samples,
            mode='linear',
            align_corners=True
        )
        return function.squeeze(1)  # [batch, num_samples]

    def forward(self, word_indices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        params = self.embedding_layer(word_indices)
        functions = self._synthesize_function(params)
        return functions, params

def functional_distance(f1: torch.Tensor, f2: torch.Tensor) -> torch.Tensor:
    """Compute L2 distance between functional embeddings with sigmoid activation."""
    l2_squared = torch.mean((f1 - f2) ** 2, dim=1)
    return torch.sigmoid(l2_squared)

# --- Dataset Classes ---
class GloVeDistanceDataset(Dataset):
    """Efficient dataset for pre-computed word pair distances."""
    
    def __init__(self, glove_vectors: torch.Tensor, pair_indices: torch.Tensor):
        """
        Args:
            glove_vectors: Normalized GloVe vectors [vocab_size, embedding_dim]
            pair_indices: Word index pairs [num_pairs, 2]
        """
        self.glove_vectors = glove_vectors
        self.pair_indices = pair_indices
        
        # Pre-compute target distances for speed
        vec1 = glove_vectors[pair_indices[:, 0]]
        vec2 = glove_vectors[pair_indices[:, 1]]
        cos_sim = torch.sum(vec1 * vec2, dim=1)
        self.target_distances = torch.clamp(1.0 - cos_sim, 0.0, 1.0)

    def __len__(self) -> int:
        return len(self.pair_indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        idx1, idx2 = self.pair_indices[idx]
        return idx1, idx2, self.target_distances[idx]

def generate_pairs(vocab_size: int, num_pairs: int, seed: int) -> torch.Tensor:
    """Generate random word pairs with reproducibility."""
    rng = np.random.RandomState(seed)
    pairs = rng.randint(0, vocab_size, size=(num_pairs, 2))
    return torch.tensor(pairs, dtype=torch.long)

# --- Loss Computation ---
def compute_loss(model: nn.Module,
                 idx1: torch.Tensor,
                 idx2: torch.Tensor,
                 target_dist: torch.Tensor,
                 reg_strength: float) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute total loss with detailed component tracking."""
    funcs1, params1 = model(idx1)
    funcs2, params2 = model(idx2)
    
    # Regression loss
    pred_dist = functional_distance(funcs1, funcs2)
    regression_loss = F.mse_loss(pred_dist, target_dist)
    
    # Smoothness regularization (encourages smooth functions)
    reg_loss_1 = torch.mean((params1[:, 1:] - params1[:, :-1]) ** 2)
    reg_loss_2 = torch.mean((params2[:, 1:] - params2[:, :-1]) ** 2)
    reg_loss = reg_strength * (reg_loss_1 + reg_loss_2)
    
    total_loss = regression_loss + reg_loss
    
    return total_loss, {
        "total": total_loss.item(),
        "regression": regression_loss.item(),
        "regularization": reg_loss.item()
    }

# --- Training and Validation Functions ---
def validate(model: nn.Module,
             val_loader: DataLoader,
             device: torch.device,
             config: TrainingConfig) -> Dict[str, float]:
    """Run validation loop and return averaged metrics."""
    model.eval()
    metrics_sum = {"total": 0.0, "regression": 0.0, "regularization": 0.0}
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="  Validating", leave=False, position=1):
            idx1, idx2, target_dist = [x.to(device, non_blocking=True) for x in batch]
            _, batch_metrics = compute_loss(model, idx1, idx2, target_dist, config.regularization_strength)
            
            for key in metrics_sum:
                metrics_sum[key] += batch_metrics[key]
    
    return {f"val_{k}": v / len(val_loader) for k, v in metrics_sum.items()}

def train_epoch(model: nn.Module,
                train_loader: DataLoader,
                optimizer: optim.Optimizer,
                device: torch.device,
                config: TrainingConfig,
                epoch: int) -> Dict[str, float]:
    """Train for one epoch with progress tracking."""
    model.train()
    metrics_sum = {"total": 0.0, "regression": 0.0, "regularization": 0.0}
    
    pbar = tqdm(train_loader, 
                desc=f"  Training Epoch {epoch+1}/{config.num_epochs}", 
                leave=False, 
                position=0)
    
    for batch in pbar:
        idx1, idx2, target_dist = [x.to(device, non_blocking=True) for x in batch]
        
        optimizer.zero_grad()
        total_loss, batch_metrics = compute_loss(model, idx1, idx2, target_dist, config.regularization_strength)
        
        total_loss.backward()
        
        # Gradient clipping for stability
        if config.gradient_clip_val > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_val)
        
        optimizer.step()
        
        # Accumulate metrics
        for key in metrics_sum:
            metrics_sum[key] += batch_metrics[key]
        
        # Update progress bar
        pbar.set_postfix({
            "Loss": f"{batch_metrics['total']:.4f}",
            "Reg": f"{batch_metrics['regularization']:.4f}"
        })
    
    return {f"train_{k}": v / len(train_loader) for k, v in metrics_sum.items()}

# --- Main Training Function ---
def train_model(config: TrainingConfig,
                glove_data: Dict,
                device: torch.device) -> nn.Module:
    """Full training pipeline with validation and checkpointing."""
    print("\n" + "="*50)
    print("Functional Embedding Training")
    print("="*50)
    print(f"Device: {device}")
    print(f"Embedding dimension: {config.embedding_dim}")
    print(f"Vocabulary size: {glove_data['vectors'].shape[0]}")
    print(f"Training pairs/epoch: {config.steps_per_epoch * config.batch_size}")
    print(f"Validation pairs: {config.validation_steps * config.batch_size}")
    print("="*50 + "\n")
    
    vocab_size = glove_data['vectors'].shape[0]
    glove_vectors = F.normalize(glove_data['vectors'], p=2, dim=1)
    
    # Split vocabulary into train/validation sets
    val_vocab_size = int(config.validation_split * vocab_size)
    train_vocab_size = vocab_size - val_vocab_size
    
    # Create vocabulary split
    indices = torch.randperm(vocab_size, generator=torch.Generator().manual_seed(config.seed))
    train_indices = indices[:train_vocab_size]
    val_indices = indices[train_vocab_size:]
    
    # Create pair generators (different seeds for train/val)
    train_pairs = generate_pairs(
        train_vocab_size, 
        config.steps_per_epoch * config.batch_size, 
        seed=config.seed
    )
    # Map back to original indices
    train_pairs = train_indices[train_pairs]
    
    val_pairs = generate_pairs(
        val_vocab_size,
        config.validation_steps * config.batch_size,
        seed=config.seed + 1
    )
    # Map back to original indices
    val_pairs = val_indices[val_pairs]
    
    # Generate all pairs first
    all_train_pairs = generate_pairs(vocab_size, 
                                    config.steps_per_epoch * config.batch_size, 
                                    seed=config.seed)

    # Split into train/val pair sets
    train_size = int(0.9 * len(all_train_pairs))
    train_pairs = all_train_pairs[:train_size]
    val_pairs = all_train_pairs[train_size:]

    train_dataset = GloVeDistanceDataset(glove_vectors, train_pairs)
    val_dataset = GloVeDistanceDataset(glove_vectors, val_pairs)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,  # Shuffle the order of pre-generated pairs
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=config.num_workers > 0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=config.num_workers > 0
    )
    
    # Initialize model
    model = FunctionalEmbeddingModel(
        vocab_size=vocab_size,
        embedding_dim=config.embedding_dim,
        num_samples=config.num_samples,
        t_min=config.t_min,
        t_max=config.t_max
    ).to(device)
    
    # Setup optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2
    )
    
    # Create checkpoint directory
    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Training loop with early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    metrics_history: List[Dict[str, float]] = []
    
    # Main epoch progress bar
    epoch_pbar = tqdm(
        range(config.num_epochs), 
        desc="Epoch Progress", 
        position=0,
        bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}"
    )
    
    for epoch in epoch_pbar:
        # Training
        train_metrics = train_epoch(model, train_loader, optimizer, device, config, epoch)
        
        # Validation
        val_metrics = validate(model, val_loader, device, config)
        
        # Record metrics
        epoch_metrics = {**train_metrics, **val_metrics}
        metrics_history.append(epoch_metrics)
        
        # Learning rate scheduling
        scheduler.step(val_metrics["val_total"])
        
        # Checkpointing
        is_best = val_metrics["val_total"] < best_val_loss
        if is_best:
            best_val_loss = val_metrics["val_total"]
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'metrics_history': metrics_history
            }, checkpoint_dir / "best_model.pt")
        else:
            patience_counter += 1
        
        # Save latest checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics_history': metrics_history
        }, checkpoint_dir / "latest_checkpoint.pt")
        
        # Update main progress bar
        epoch_pbar.set_postfix({
            "Train": f"{train_metrics['train_total']:.4f}",
            "Val": f"{val_metrics['val_total']:.4f}",
            "Best": f"{best_val_loss:.4f}",
            "Patience": f"{patience_counter}/{config.early_stopping_patience}"
        })
        
        # Log detailed metrics
        print(f"\nEpoch {epoch+1}/{config.num_epochs} Summary:")
        print(f"  Train Loss: {train_metrics['train_total']:.6f} "
              f"(Regression: {train_metrics['train_regression']:.6f}, "
              f"Reg: {train_metrics['train_regularization']:.6f})")
        print(f"  Val Loss: {val_metrics['val_total']:.6f} "
              f"(Regression: {val_metrics['val_regression']:.6f}, "
              f"Reg: {val_metrics['val_regularization']:.6f})")
        print(f"  Best Val Loss: {best_val_loss:.6f}")
        print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Early stopping check
        if patience_counter >= config.early_stopping_patience:
            print(f"\n⚠️  Early stopping triggered at epoch {epoch+1}")
            break
    
    # Load best model
    print("\nLoading best model for final evaluation...")
    checkpoint = torch.load(checkpoint_dir / "best_model.pt", map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Save metrics history
    with open(checkpoint_dir / "metrics_history.json", "w") as f:
        json.dump(metrics_history, f, indent=2)
    
    print(f"\n✅ Training complete! Best validation loss: {best_val_loss:.6f}")
    return model

# --- Save/Load Utilities ---
def save_final_embeddings(model: nn.Module,
                         word_to_idx: Dict[str, int],
                         config: TrainingConfig) -> None:
    """Save trained embeddings with metadata."""
    print(f"\n💾 Saving final embeddings to {config.final_embedding_file}...")
    
    output_path = Path(config.final_embedding_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.save({
        'word_to_idx': word_to_idx,
        'model_state_dict': model.state_dict(),
        'embedding_layer_state_dict': model.embedding_layer.state_dict(),
        'config': config,
    }, output_path)
    
    print("✅ Save complete!")

def load_glove_data(file_path: str) -> Optional[Dict]:
    """Load GloVe data with robust error handling."""
    path = Path(file_path)
    if not path.exists():
        print(f"❌ Error: {file_path} not found.")
        print("Please run `python glove_loader_full.py` first")
        return None
    
    try:
        data = torch.load(path, map_location='cpu')
        print(f"📥 Loaded GloVe data: {data['vectors'].shape[0]} words, {data['vectors'].shape[1]} dimensions")
        return data
    except Exception as e:
        print(f"❌ Error loading GloVe data: {e}")
        return None

# --- Main Execution ---
def main():
    """Main entry point with configuration and seeding."""
    config = TrainingConfig()
    
    # Set reproducibility seed
    set_seed(config.seed)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")
    
    # Load data
    glove_data = load_glove_data(config.glove_data_file)
    if glove_data is None:
        return
    
    # Train model
    trained_model = train_model(config, glove_data, device)
    
    # Save results
    save_final_embeddings(trained_model, glove_data['word_to_idx'], config)

if __name__ == "__main__":
    main()
