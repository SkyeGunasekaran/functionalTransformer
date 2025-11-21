import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# -----------------------------------------------------------------------------
# 1. STRICTLY TYPED JIT KERNELS (The Speedup Sauce)
# -----------------------------------------------------------------------------

@torch.jit.script
def fused_algebraic_norm(x: torch.Tensor, gain: torch.Tensor, bias: torch.Tensor, eps: float) -> torch.Tensor:
    mu = x.mean(dim=-1, keepdim=True)
    x_centered = x - mu
    l1_norm = x_centered.abs().mean(dim=-1, keepdim=True)
    return (gain * (x_centered / (l1_norm + eps))) + bias

@torch.jit.script
def fused_rational_softmax(x: torch.Tensor, power: torch.Tensor, eps: float) -> torch.Tensor:
    # 1. Algebraic Sigmoid
    # We use sign * x to avoid creating a new abs() tensor if memory is tight, 
    # but x.abs() is generally optimized enough.
    s = x / (x.abs() + 1.0)
    p_base = (s + 1.0) * 0.5 
    
    # 2. Dynamic Power (Annealing friendly)
    # Because 'power' is a 0-dim tensor, JIT handles this efficiently without recompiling
    unnorm_probs = torch.pow(p_base, power)
        
    # 3. Normalization
    sum_probs = unnorm_probs.sum(dim=-1, keepdim=True)
    return unnorm_probs / (sum_probs + eps)

@torch.jit.script
def fused_swish(x: torch.Tensor) -> torch.Tensor:
    return x * (((x / (x.abs() + 1.0)) + 1.0) * 0.5)

@torch.jit.script
def compute_chunk_scores(Q_chunk: torch.Tensor, K: torch.Tensor, 
                            alibi_slopes: torch.Tensor, 
                            k_idx: torch.Tensor, q_idx_start: int,
                            static_scale: float) -> torch.Tensor:
    # (B, H, Chunk, T)
    scores = torch.matmul(Q_chunk, K.transpose(-2, -1))
    scores = (scores * static_scale) * 16.0
    
    # Scalar generation (No tensor overhead)
    chunk_len = scores.size(-2)
    q_idx = torch.arange(q_idx_start, q_idx_start + chunk_len, device=scores.device, dtype=torch.float32)
    
    # Implicit Distance Matrix
    # (1, 1, 1, T) - (1, 1, Chunk, 1)
    distance = k_idx[None, None, None, :] - q_idx[None, None, :, None]
    
    # Apply ALiBi In-Place (Saves Memory)
    scores.addcmul_(alibi_slopes, distance.abs(), value=-1.0)
    
    # Masking: Use scalar -10000.0
    scores = torch.where(distance > 0, -10000.0, scores)
    return scores

# -----------------------------------------------------------------------------
# 2. OPTIMIZED MODULES
# -----------------------------------------------------------------------------

class AlgebraicNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.gain = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))
    
    def forward(self, x):
        return fused_algebraic_norm(x, self.gain, self.bias, self.eps)

class RationalSoftmax(nn.Module):
    def __init__(self, power=2.0, eps=1e-6): 
        super().__init__()
        # FIX: Register as buffer so it works with the JIT kernel expecting a Tensor
        self.register_buffer('power_tensor', torch.tensor(float(power)))
        self.eps = eps

    def forward(self, x):
        return fused_rational_softmax(x, self.power_tensor, self.eps)

class AlgebraicSwish(nn.Module):
    def forward(self, x):
        return fused_swish(x)

class AlgebraicAttention(nn.Module):
    def __init__(self, d_model, n_head, dropout=0.1, power=2.0, query_chunk_size=512):
        super().__init__()
        self.n_head = n_head
        self.d_head = d_model // n_head
        self.query_chunk_size = query_chunk_size
        
        # Annealing Buffer
        self.register_buffer('power_tensor', torch.tensor(float(power)))
        
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.static_scale = 1.0 / (self.d_head ** 0.5)
        self.dropout = nn.Dropout(dropout)
        self.softmax_eps = 1e-6

        # ALiBi Setup
        closest_power_of_2 = 2 ** math.floor(math.log2(n_head))
        base = 2 ** (-(2 ** -(math.log2(closest_power_of_2) - 3)))
        powers = torch.arange(1, 1 + closest_power_of_2)
        slopes = torch.pow(base, powers)
        if closest_power_of_2 != n_head:
            extra_base = 2 ** (-(2 ** -(math.log2(2 * closest_power_of_2) - 3)))
            num_rem = min(closest_power_of_2, n_head - closest_power_of_2)
            slopes = torch.cat([slopes, torch.pow(extra_base, torch.arange(1, 1 + 2 * num_rem, 2))])
        self.register_buffer("alibi_slopes", (slopes * 4.0).view(1, n_head, 1, 1))

    def forward(self, x):
        B, T, C = x.shape
        # Pre-allocate K index to avoid recreation in loop
        k_idx = torch.arange(T, device=x.device, dtype=torch.float32)

        Q = self.q_proj(x).reshape(B, T, self.n_head, self.d_head).transpose(1, 2)
        K = self.k_proj(x).reshape(B, T, self.n_head, self.d_head).transpose(1, 2)
        V = self.v_proj(x).reshape(B, T, self.n_head, self.d_head).transpose(1, 2)

        output_chunks = []
        current_power = self.power_tensor # Local reference

        for i in range(0, T, self.query_chunk_size):
            end = min(i + self.query_chunk_size, T)
            Q_chunk = Q[:, :, i:end, :]
            
            # JIT Kernel 1: Scores & Masking
            scores = compute_chunk_scores(Q_chunk, K, self.alibi_slopes, k_idx, i, self.static_scale)
            
            # JIT Kernel 2: Softmax (Dynamic Power)
            probs = fused_rational_softmax(scores, current_power, self.softmax_eps)
            
            attn_weights = self.dropout(probs)
            output_chunks.append(torch.matmul(attn_weights, V))
            
        out = torch.cat(output_chunks, dim=2)
        out = out.transpose(1, 2).reshape(B, T, C)
        return self.out_proj(out)

class FastAlgebraicTransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model, n_head, n_layers, d_ffn, block_size, dropout=0.1, power=2.0):
        super().__init__()
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, d_model)
        
        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                'attn': AlgebraicAttention(d_model, n_head, dropout, power, query_chunk_size=128),
                'ffn': nn.Sequential(
                    nn.Linear(d_model, d_ffn),
                    AlgebraicSwish(),
                    nn.Linear(d_ffn, d_model),
                    nn.Dropout(dropout)
                ),
                'norm1': AlgebraicNorm(d_model), 'norm2': AlgebraicNorm(d_model)
            }) for _ in range(n_layers)
        ])
        
        self.final_norm = AlgebraicNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)
        # FIX: Use the class that now correctly has a power buffer
        self.final_softmax = RationalSoftmax(power=power)
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None: torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def update_power(self, new_power):
        """
        Updates the power parameter for all attention heads and the final softmax.
        Call this in your training loop.
        """
        # Update Attention blocks
        for block in self.blocks:
            block['attn'].power_tensor.fill_(new_power)
        # Update Final Softmax
        self.final_softmax.power_tensor.fill_(new_power)

    def forward(self, idx, targets=None):
        x = self.token_embedding_table(idx)
        for block in self.blocks:
            x = x + block['attn'](block['norm1'](x))
            x = x + block['ffn'](block['norm2'](x))
        x = self.final_norm(x)
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            # Optim: View logits directly
            logits_flat = logits.view(-1, logits.size(-1))
            targets_flat = targets.view(-1)
            
            # Use the Final Softmax (which is now annealed!)
            probs = self.final_softmax(logits_flat)
            
            # Gather optimized
            p_target = probs.gather(dim=-1, index=targets_flat.unsqueeze(-1)).squeeze(-1)
            p_target = torch.clamp(p_target, min=1e-9, max=1.0)
            loss = -torch.log(p_target).mean()
            
        return logits, loss
