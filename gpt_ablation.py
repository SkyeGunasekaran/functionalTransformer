import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from dataclasses import dataclass
from algebraic_model import RationalSoftmax, AlgebraicNorm, AlgebraicMSELoss
# =====================================================================
# 1. MODIFIED NANOGPT COMPONENTS
# ==============================================================================

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True
    # NEW: Experiment Control
    # 'baseline': Standard NanoGPT
    # 'exp_a': Algebraic Norm only
    # 'exp_b': Algebraic Attention (Rational Softmax) only
    # 'exp_c': Algebraic Loss only
    ablation_mode: str = 'baseline' 

class LayerNorm(nn.Module):
    """ Standard LayerNorm """
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class AblationCausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.ablation_mode = config.ablation_mode
        
        # Logic for Experiment B (Attention)
        if self.ablation_mode == 'exp_b':
            self.rational_softmax = RationalSoftmax(power=4)
            self.score_gain = nn.Parameter(torch.tensor(4.0)) # Critical for rational curves
            self.flash = False # Rational Softmax cannot use Flash Attention kernels
        else:
            self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
            if not self.flash:
                self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                            .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) 
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) 
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) 

        # --- EXPERIMENT B: ALGEBRAIC ATTENTION ---
        if self.ablation_mode == 'exp_b':
            # Manual implementation with Rational Softmax
            scores = (q @ k.transpose(-2, -1)) 
            
            # Normalize scores to fit rational curve (as per your original model logic)
            scores_mean = scores.mean(dim=-1, keepdim=True)
            scores = scores - scores_mean
            scores_mad = scores.abs().mean(dim=-1, keepdim=True) + 1e-6
            scores = (scores / scores_mad) * self.score_gain
            
            # Causal Masking (Manual)
            mask_bias = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
            scores = scores.masked_fill(mask_bias == 0, -1000.0)
            
            att = self.rational_softmax(scores)
            att = self.attn_dropout(att)
            y = att @ v 
            
        # --- BASELINE / EXP A / EXP C (Standard Attention) ---
        else:
            if self.flash:
                y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
            else:
                att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
                att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
                att = F.softmax(att, dim=-1)
                att = self.attn_dropout(att)
                y = att @ v 
                
        y = y.transpose(1, 2).contiguous().view(B, T, C) 
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x) # Note: Keeping GELU for ablations to isolate specific components
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class AblationBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ablation_mode = config.ablation_mode
        
        # --- EXPERIMENT A: ALGEBRAIC NORM ---
        if self.ablation_mode == 'exp_a':
            self.ln_1 = AlgebraicNorm(config.n_embd)
            self.ln_2 = AlgebraicNorm(config.n_embd)
        else:
            self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
            self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
            
        self.attn = AblationCausalSelfAttention(config)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

# ==============================================================================
# 2. THE ABLATION GPT MODEL
# ==============================================================================

class AblationGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([AblationBlock(config) for _ in range(config.n_layer)]),
        ))
        
        # Final Norm Selection
        if config.ablation_mode == 'exp_a':
            self.transformer.ln_f = AlgebraicNorm(config.n_embd)
        else:
            self.transformer.ln_f = LayerNorm(config.n_embd, bias=config.bias)
            
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight 
        
        # Loss function for Experiment C
        if config.ablation_mode == 'exp_c':
            self.criterion = AlgebraicMSELoss()
            
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))
        
        print(f"Model Initialized. Mode: {config.ablation_mode.upper()}. Params: {self.get_num_params()/1e6:.2f}M")

    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            
            # --- EXPERIMENT C: ALGEBRAIC LOSS ---
            if self.config.ablation_mode == 'exp_c':
                # To strictly test the LOSS (and not the Rational Softmax), 
                # we use Standard Softmax -> Probabilities -> Algebraic MSE.
                # This isolates the gradient dynamics of the MSE formulation.
                probs = F.softmax(logits, dim=-1)
                loss = self.criterion(probs.view(-1, probs.size(-1)), targets.view(-1))
            else:
                # Baseline, Exp A, Exp B use standard Cross Entropy
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss
