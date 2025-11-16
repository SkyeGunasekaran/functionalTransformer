import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import math
import time
from pathlib import Path
from dataclasses import dataclass, field
from datasets import load_dataset
from tqdm import tqdm
from torch.amp import GradScaler, autocast
import logging
from typing import Optional, Dict, Any
import json
from collections import defaultdict
import numpy as np
import random 
from torch.nn.attention import SDPBackend, sdpa_kernel 

# --- 1. Configuration & Logging ---
@dataclass
class TrainingConfig:
    EMBEDDING_DIM: int = 128
    NUM_SAMPLES: int = 100
    T_MIN: float = -64.0
    T_MAX: float = 64.0
    REGULARIZATION_STRENGTH: float = 0.001
    NUM_EPOCHS: int = 15
    LEARNING_RATE: float = 0.001
    BATCH_SIZE: int = 1024
    STEPS_PER_EPOCH: int = 10000

@dataclass
class Config:
    EMBEDDING_FILE: str = "functional_embeddings_v1.pt"
    CHECKPOINT_DIR: str = "checkpoints_phase2"
    
    # Dataset
    DATASET_NAME: str = 'wikitext'              # ADD THIS
    DATASET_CONFIG: str = 'wikitext-2-v1'        # ADD THIS

    # Paths
    EMBEDDING_FILE: str = "functional_embeddings_v1.pt"
    CHECKPOINT_DIR: str = "checkpoints_phase2"
    
    # Model
    D_MODEL: int = 128
    NHEAD: int = 4
    NUM_LAYERS: int = 6
    DIM_FEEDFORWARD: int = 512
    DROPOUT: float = 0.1
    SEQUENCE_LENGTH: int = 256
    
    # Training
    BATCH_SIZE: int = 32
    LEARNING_RATE: float = 0.0005
    NUM_EPOCHS: int = 30
    GRADIENT_ACCUMULATION_STEPS: int = 4  # Simulate larger batch size
    MAX_GRAD_NORM: float = 1.0
    WARMUP_STEPS: int = 1000
    WEIGHT_DECAY: float = 0.01
    
    # System
    NUM_WORKERS: int = 4
    PIN_MEMORY: bool = True
    SEED: int = 42
    MIXED_PRECISION: bool = True
    
    # Validation & Generation
    VALIDATION_EVERY_N_STEPS: int = 500
    EVAL_SAMPLES: int = 100
    TEMPERATURE: float = 0.8
    TOP_K: int = 50
    TOP_P: float = 0.95
    MAX_GEN_LEN: int = 50
    
    # Derived paths
    checkpoint_dir: Path = field(init=False)
    best_model_path: Path = field(init=False)
    latest_checkpoint_path: Path = field(init=False)
    metrics_path: Path = field(init=False)
    
    def __post_init__(self):
        self.checkpoint_dir = Path(self.CHECKPOINT_DIR)
        self.best_model_path = self.checkpoint_dir / "best_model.pt"
        self.latest_checkpoint_path = self.checkpoint_dir / "latest.pt"
        self.metrics_path = self.checkpoint_dir / "metrics.json"

def setup_logging():
    """Configure logging for research reproducibility."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        handlers=[
            logging.FileHandler('training.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# --- 2. Vocabulary & Embeddings ---
class Vocabulary:
    """Manages vocabulary built *exclusively* from training data."""
    
    def __init__(self, embedding_file: str, d_model: int, min_freq: int = 2):
        self.embedding_file = embedding_file
        self.d_model = d_model
        self.min_freq = min_freq
        
        # Special tokens
        self.PAD_TOKEN = "[PAD]"
        self.BOS_TOKEN = "[BOS]"
        self.EOS_TOKEN = "[EOS]"
        self.UNK_TOKEN = "[UNK]"
        
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.pad_idx = None
        self.unk_idx = None
        
    def build_from_embeddings(self) -> nn.Embedding:
        """Load embeddings and build vocabulary from pretrained weights."""
        logger.info(f"Loading embeddings from {self.embedding_file}")
        saved_data = torch.load(self.embedding_file, map_location='cpu', weights_only=False)
        
        # Start with special tokens
        special_tokens = [self.PAD_TOKEN, self.BOS_TOKEN, self.EOS_TOKEN, self.UNK_TOKEN]
        for token in special_tokens:
            if token not in self.word_to_idx:
                self.word_to_idx[token] = len(self.word_to_idx)
        
        # Load pretrained words
        pretrained_word_to_idx = saved_data['word_to_idx']
        embedding_state_dict = saved_data['embedding_layer_state_dict']
        pretrained_weights = embedding_state_dict['weight']
        
        if pretrained_weights.shape[1] != self.d_model:
            raise ValueError(f"Dimension mismatch: {pretrained_weights.shape[1]} vs {self.d_model}")
        
        logger.info(f"Loaded {len(pretrained_word_to_idx)} pretrained words")
        
        # Add pretrained words to vocab
        for word, idx in pretrained_word_to_idx.items():
            if word not in self.word_to_idx:
                self.word_to_idx[word] = len(self.word_to_idx)
        
        # Build reverse mapping
        self.idx_to_word = {i: w for w, i in self.word_to_idx.items()}
        self.pad_idx = self.word_to_idx[self.PAD_TOKEN]
        self.unk_idx = self.word_to_idx[self.UNK_TOKEN]
        
        # Create embedding layer
        vocab_size = len(self.word_to_idx)
        embedding_layer = nn.Embedding(vocab_size, self.d_model, padding_idx=self.pad_idx)
        
        # Initialize with pretrained weights (freeze by default)
        embedding_layer.weight.requires_grad = False
        embedding_layer.weight.data[:pretrained_weights.shape[0]] = pretrained_weights
        
        # Initialize special tokens
        for token in special_tokens:
            idx = self.word_to_idx[token]
            if idx >= pretrained_weights.shape[0]:
                nn.init.normal_(embedding_layer.weight.data[idx], mean=0, std=0.02)
        
        logger.info(f"Vocabulary built: {vocab_size} tokens (including {len(special_tokens)} special)")
        return embedding_layer
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # batch_first=True
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape is (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
# --- 3. Dataset with Training-Only Vocabulary ---
class TextDataset(Dataset):
    """
    Tokenizes text using a *fixed* vocabulary built from training data only.
    Validation words not in vocab map to UNK.
    """
    
    def __init__(self, hf_dataset, vocab: Vocabulary, seq_len: int):
        self.vocab = vocab
        self.seq_len = seq_len
        
        # Tokenize efficiently
        tokens = []
        for text in tqdm(hf_dataset['text'], desc="Tokenizing"):
            if text.strip():
                words = text.strip().lower().split()
                indices = [vocab.word_to_idx.get(w, vocab.unk_idx) for w in words]
                indices.append(vocab.word_to_idx[vocab.EOS_TOKEN])
                tokens.extend(indices)
        
        self.tokens = torch.tensor(tokens, dtype=torch.long)
        logger.info(f"Tokenized {len(self.tokens)} tokens")
    
    def __len__(self):
        return len(self.tokens) // (self.seq_len + 1)
    
    def __getitem__(self, idx):
        start = idx * (self.seq_len + 1)
        chunk = self.tokens[start:start + self.seq_len + 1]
        
        src = chunk[:self.seq_len]
        tgt = chunk[1:self.seq_len + 1]
        
        return src, tgt

# --- 4. Model (with improvements) ---
class FunctionalDecoderModel(nn.Module):
    def __init__(self, embedding_layer: nn.Embedding, config: Config):
        super().__init__()
        self.config = config
        self.pad_idx = embedding_layer.padding_idx
        self.use_flash = (
            hasattr(torch.nn.functional, 'scaled_dot_product_attention') 
            and torch.cuda.is_available()
        )
        if self.use_flash:
            # Check for Ampere+ GPU (Compute Capability 8.0+) for Flash V2
            if torch.cuda.get_device_capability() >= (8, 0):
                logger.info("Flash Attention enabled (SDPA backend: Flash V2)")
            else:
                logger.info("Flash Attention enabled (SDPA backend: Memory Efficient)")

        self.embedding_layer = embedding_layer
        self.pos_encoder = PositionalEncoding(
            config.D_MODEL, config.DROPOUT, config.SEQUENCE_LENGTH * 2
        )
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.D_MODEL,
            nhead=config.NHEAD,
            dim_feedforward=config.DIM_FEEDFORWARD,
            dropout=config.DROPOUT,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=config.NUM_LAYERS
        )

        
        self.lm_head = nn.Linear(config.D_MODEL, embedding_layer.num_embeddings)
        self._init_weights()
    def generate_causal_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        """Create causal mask for autoregressive decoding."""
        # This creates a mask where True values are positions that WILL be masked
        mask = torch.triu(torch.ones(sz, sz, device=device, dtype=torch.bool), diagonal=1)
        return mask
    def _init_weights(self):
        """Initialize decoder weights properly for GPT-style training."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
    def forward(self, src: torch.Tensor) -> torch.Tensor:
        
        # Generate the causal mask 
        causal_mask = self.generate_causal_mask(src.size(1), src.device)
        
        # Generate the padding mask 
        padding_mask = (src == self.pad_idx)
        
        src_embed = self.embedding_layer(src)
        src_embed = self.pos_encoder(src_embed)
        
        output = self.transformer_encoder(
            src_embed,
            mask=causal_mask,          
            src_key_padding_mask=padding_mask,
            is_causal=True            
        )
        
        return self.lm_head(output)
# --- 5. Training Utilities ---
class MetricsTracker:
    """Track and save training metrics."""
    
    def __init__(self, save_path: Path):
        self.save_path = save_path
        self.history = []
    
    def update(self, epoch: int, step: int, metrics: Dict[str, float]):
        self.history.append({
            'epoch': epoch,
            'step': step,
            **metrics
        })
        self.save()
    
    def save(self):
        with open(self.save_path, 'w') as f:
            json.dump(self.history, f, indent=2)

def get_lr_scheduler(optimizer: optim.Optimizer, config: Config, steps_per_epoch: int):
    """Linear warmup + cosine decay scheduler."""
    total_steps = config.NUM_EPOCHS * steps_per_epoch
    
    def lr_lambda(step: int):
        if step < config.WARMUP_STEPS:
            return step / config.WARMUP_STEPS
        progress = (step - config.WARMUP_STEPS) / (total_steps - config.WARMUP_STEPS)
        return 0.5 * (1 + math.cos(math.pi * progress))
    
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# --- 6. Training Loop ---
def train_step(model: nn.Module, 
               batch: tuple, 
               criterion: nn.Module, 
               scaler: GradScaler, 
               optimizer: optim.Optimizer, 
               device: torch.device,
               config: Config) -> float:
    """Single training step with gradient accumulation."""
    src, tgt = [x.to(device, non_blocking=True) for x in batch]
    
    # Select SDPA backend (Flash Attention 2 for Ampere+)
    if device.type == 'cuda' and torch.cuda.get_device_capability() >= (8, 0): 
        backend = SDPBackend.FLASH_ATTENTION
    else:
        # Use memory-efficient for older GPUs or CPU
        backend = SDPBackend.EFFICIENT_ATTENTION

    # --- WRAP autocast with sdpa_kernel ---
    with sdpa_kernel(backend), autocast(device_type=device.type, enabled=config.MIXED_PRECISION):
        logits = model(src)
        loss = criterion(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1))
        loss = loss / config.GRADIENT_ACCUMULATION_STEPS
    
    scaler.scale(loss).backward()
    
    return loss.item() * config.GRADIENT_ACCUMULATION_STEPS

def validate(model: nn.Module, 
             dataloader: DataLoader, 
             criterion: nn.Module, 
             device: torch.device,
             config: Config) -> float:
    """Full validation pass."""
    
    # Select SDPA backend (Flash Attention 2 for Ampere+)
    if device.type == 'cuda' and torch.cuda.get_device_capability() >= (8, 0): 
        backend = SDPBackend.FLASH_ATTENTION
    else:
        # Use memory-efficient for older GPUs or CPU
        backend = SDPBackend.EFFICIENT_ATTENTION
        
    model.eval()
    total_loss = 0
    
    # --- WRAP torch.no_grad() with sdpa_kernel ---
    with torch.no_grad(), sdpa_kernel(backend):
        for src, tgt in tqdm(dataloader, desc="Validating", leave=False):
            src, tgt = src.to(device), tgt.to(device) 
            
            with autocast(device_type=device.type, enabled=config.MIXED_PRECISION):
                logits = model(src)
                loss = criterion(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1))
            
            total_loss += loss.item()
    
    return total_loss / len(dataloader)
def train(model: nn.Module,
          train_loader: DataLoader,
          val_loader: DataLoader,
          optimizer: optim.Optimizer,
          criterion: nn.Module,
          scaler: GradScaler,
          scheduler: optim.lr_scheduler,
          config: Config,
          device: torch.device,
          metrics_tracker: MetricsTracker):
    """Main training loop with early stopping and best model saving."""
    
    best_val_ppl = float('inf')
    patience_counter = 0
    global_step = 0
    
    model.train()
    
    for epoch in range(config.NUM_EPOCHS):
        epoch_start = time.time()
        model.train()
        
        train_losses = []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS}")
        
        for step, batch in enumerate(pbar):
            # Training step
            loss = train_step(model, batch, criterion, scaler, optimizer, device, config)
            train_losses.append(loss)
            
            # Gradient accumulation
            if (step + 1) % config.GRADIENT_ACCUMULATION_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.MAX_GRAD_NORM)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                
                # Validation check
                if global_step % config.VALIDATION_EVERY_N_STEPS == 0:
                    val_loss = validate(model, val_loader, criterion, device, config)
                    val_ppl = math.exp(val_loss)
                    
                    metrics_tracker.update(epoch, global_step, {
                        'train_loss': sum(train_losses[-100:]) / min(100, len(train_losses)),
                        'val_loss': val_loss,
                        'val_ppl': val_ppl,
                        'lr': scheduler.get_last_lr()[0]
                    })
                    
                    # Early stopping & checkpointing
                    if val_ppl < best_val_ppl:
                        best_val_ppl = val_ppl
                        patience_counter = 0
                        save_checkpoint(model, optimizer, scaler, epoch, global_step, val_loss, config, is_best=True)
                    else:
                        patience_counter += 1
                    
                    logger.info(f"Step {global_step} | Val PPL: {val_ppl:.2f} | Best: {best_val_ppl:.2f} | Patience: {patience_counter}")
                    
                    if patience_counter >= 3:  # Early stopping
                        logger.info("Early stopping triggered")
                        return
            
            # Update progress bar
            if step % 10 == 0:
                pbar.set_postfix({
                    'loss': f'{loss:.3f}',
                    'ppl': f'{math.exp(loss):.1f}',
                    'lr': f'{scheduler.get_last_lr()[0]:.6f}'
                })
        
        # End of epoch validation
        val_loss = validate(model, val_loader, criterion, device, config)
        val_ppl = math.exp(val_loss)
        avg_train_loss = sum(train_losses) / len(train_losses)
        
        logger.info(
            f"Epoch {epoch+1} | Train PPL: {math.exp(avg_train_loss):.2f} | "
            f"Val PPL: {val_ppl:.2f} | Time: {time.time() - epoch_start:.1f}s"
        )
        
        save_checkpoint(model, optimizer, scaler, epoch, global_step, val_loss, config)

# --- 7. Checkpointing ---
def save_checkpoint(model: nn.Module,
                    optimizer: optim.Optimizer,
                    scaler: GradScaler,
                    epoch: int,
                    step: int,
                    loss: float,
                    config: Config,
                    is_best: bool = False):
    """Save model checkpoint with full state."""
    checkpoint = {
        'epoch': epoch,
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'loss': loss,
        'config': config,
    }
    
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, config.latest_checkpoint_path)
    
    if is_best:
        torch.save(checkpoint, config.best_model_path)
        logger.info(f"Saved best model to {config.best_model_path}")

def load_checkpoint(model: nn.Module,
                    optimizer: optim.Optimizer,
                    scaler: GradScaler,
                    config: Config,
                    device: torch.device) -> tuple:
    """Load checkpoint if exists."""
    if not config.latest_checkpoint_path.exists():
        logger.info("No checkpoint found, starting from scratch")
        return 0, 0
    
    logger.info(f"Loading checkpoint from {config.latest_checkpoint_path}")
    checkpoint = torch.load(config.latest_checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scaler.load_state_dict(checkpoint['scaler_state_dict'])
    
    start_epoch = checkpoint['epoch']
    start_step = checkpoint['step'] + 1
    
    logger.info(f"Resumed from epoch {start_epoch}, step {start_step}")
    return start_epoch, start_step

# --- 8. Generation with Advanced Sampling ---
@torch.no_grad()
@torch.no_grad()
def generate_text(model: nn.Module,
                  prompt: str,
                  vocab: Vocabulary,
                  device: torch.device,
                  config: Config,
                  max_len: Optional[int] = None) -> str:
    """Generate text with top-k and nucleus sampling."""
    model.eval()
    max_len = max_len or config.MAX_GEN_LEN
    
    # Tokenize prompt
    words = prompt.lower().split()
    tokens = [vocab.word_to_idx.get(w, vocab.unk_idx) for w in words]
    
    generated = tokens.copy()
    
    # Select SDPA backend (Flash Attention 2 for Ampere+)
    if device.type == 'cuda' and torch.cuda.get_device_capability() >= (8, 0): 
        backend = SDPBackend.FLASH_ATTENTION
    else:
        # Use memory-efficient for older GPUs or CPU
        backend = SDPBackend.EFFICIENT_ATTENTION

    # --- WRAP the generation loop with sdpa_kernel ---
    with sdpa_kernel(backend):
        for _ in range(max_len):
            src = torch.tensor([generated[-config.SEQUENCE_LENGTH:]], device=device)
            
            with autocast(device_type=device.type, enabled=config.MIXED_PRECISION):
                logits = model(src)
                next_token_logits = logits[0, -1, :] / config.TEMPERATURE
            
            # Top-k filtering
            if config.TOP_K > 0:
                top_k_logits, top_k_indices = torch.topk(next_token_logits, config.TOP_K)
                next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                next_token_logits[top_k_indices] = top_k_logits
            
            # Top-p (nucleus) filtering
            if config.TOP_P < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                sorted_indices_to_remove = cumulative_probs > config.TOP_P
                sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                sorted_indices_to_remove[0] = False
                
                indices_to_remove = sorted_indices_to_remove.scatter(0, sorted_indices, sorted_indices_to_remove)
                next_token_logits = next_token_logits.masked_fill(indices_to_remove, float('-inf'))
            
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            
            if next_token == vocab.word_to_idx[vocab.EOS_TOKEN]:
                break
            
            generated.append(next_token)
    
    return " ".join([vocab.idx_to_word[i] for i in generated])
# --- 9. Main ---
def main():
    config = Config()
    set_seed(config.SEED)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Starting training on {device}")
    
    # Load vocabulary and embeddings
    vocab = Vocabulary(config.EMBEDDING_FILE, config.D_MODEL)
    embedding_layer = vocab.build_from_embeddings()
    
    # Load datasets (using same vocab for both)
    logger.info("Loading datasets")
    raw_train = load_dataset(config.DATASET_NAME, config.DATASET_CONFIG, split='train')
    raw_val = load_dataset(config.DATASET_NAME, config.DATASET_CONFIG, split='validation')
    
    train_dataset = TextDataset(raw_train, vocab, config.SEQUENCE_LENGTH)
    val_dataset = TextDataset(raw_val, vocab, config.SEQUENCE_LENGTH)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        persistent_workers=config.NUM_WORKERS > 0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        persistent_workers=config.NUM_WORKERS > 0
    )
    
    # Initialize model
    model = FunctionalDecoderModel(embedding_layer, config).to(device)
    
    # Optimizer with weight decay (excluding LayerNorm and biases)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters() 
                      if not any(nd in n for nd in no_decay)],
            'weight_decay': config.WEIGHT_DECAY
        },
        {
            'params': [p for n, p in model.named_parameters() 
                      if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0
        }
    ]
    optimizer = optim.AdamW(optimizer_grouped_parameters, lr=config.LEARNING_RATE)
    
    # Learning rate scheduling
    scheduler = get_lr_scheduler(optimizer, config, len(train_loader))
    
    # Mixed precision scaler
    scaler = GradScaler(enabled=config.MIXED_PRECISION)
    
    # Metrics tracking
    metrics_tracker = MetricsTracker(config.metrics_path)
    
    # Load checkpoint
    start_epoch, start_step = load_checkpoint(model, optimizer, scaler, config, device)
    
    # Train
    train(model, train_loader, val_loader, optimizer, 
          nn.CrossEntropyLoss(ignore_index=vocab.pad_idx), 
          scaler, scheduler, config, device, metrics_tracker)
    
    # Final generation
    logger.info("Running generation examples")
    model.load_state_dict(torch.load(config.best_model_path, map_location=device)['model_state_dict'])
    
    prompts = [
        "the dog",
        "in a shocking turn of events",
        "the president announced",
        "functional embeddings are"
    ]
    
    for prompt in prompts:
        generated = generate_text(model, prompt, vocab, device, config)
        logger.info(f"\nPrompt: '{prompt}'\nGenerated: '{generated}'")

if __name__ == "__main__":
    main()
