import torch
import torch.nn as nn
import torch.nn.functional as F
import math
# ----------------------------------------------------------------------
# 1. Algebraic Norm (Rational L1 Projection)
# ----------------------------------------------------------------------
class AlgebraicNorm(nn.Module):
    """
    Projects data onto the L1 'Diamond'.
    Formula: y = g * (x / (mean(|x|) + eps))
    Properties: Purely rational, sparsity-inducing, no square roots.
    """
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.gain = nn.Parameter(torch.ones(d_model))
    
    def forward(self, x):
        # Mean Absolute Deviation (L1 proxy)
        l1_norm = x.abs().mean(dim=-1, keepdim=True)
        return self.gain * (x / (l1_norm + self.eps))

# ----------------------------------------------------------------------
# 2. Activations & Rational Softmax
# ----------------------------------------------------------------------
class AlgebraicReLU(nn.Module):
    def forward(self, x):
        # Piecewise linear (Identity for x>0, Zero constant for x<=0)
        # This is fundamentally algebraic (Order 1 Spline).
        return torch.where(x > 0, x, torch.tensor(0.0, device=x.device, dtype=x.dtype))

class RationalSoftmax(nn.Module):
    """
    Uses Domain Compression & Polynomial Sharpening.
    1. Squash (-inf, inf) -> (-1, 1) via x / (1+|x|)
    2. Shift to (0, 1)
    3. Sharpen via power k to mimic 'winner-takes-all'
    """
    def __init__(self, power=4, eps=1e-6):
        super().__init__()
        self.power = power 
        self.eps = eps

    def forward(self, x):
        # 1. Domain Compression (Algebraic Sigmoid)
        # No hard clamping needed; asymptotes handle infinity.
        # s in (-1, 1)
        s = x / (x.abs() + 1.0)
        
        # 2. Shift to Probability Base (0, 1)
        p_base = (s + 1.0) / 2.0
        
        # 3. Polynomial Sharpening (Degree Elevation)
        unnorm_probs = p_base.pow(self.power)
        
        # 4. Normalization
        return unnorm_probs / (unnorm_probs.sum(dim=-1, keepdim=True) + self.eps)

# ----------------------------------------------------------------------
# 3. Algebraic Loss (Geometric Distance)
# ----------------------------------------------------------------------
class AlgebraicMSELoss(nn.Module):
    """
    Measures Euclidean distance on the probability simplex.
    Optimized to avoid creating dense One-Hot tensors for large vocabularies.
    L = sum((p - y)^2) = sum(p^2) - 2*p_target + 1
    """
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        
    def forward(self, probs, targets):
        # probs: [Batch*Seq, Vocab]
        # targets: [Batch*Seq]
        
        # 1. Sum of squared probabilities (The "Energy" of the prediction)
        sum_sq_probs = probs.pow(2).sum(dim=-1)
        
        # 2. Probability assigned to the correct target
        # We use gather to avoid creating a massive One-Hot tensor
        p_target = probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
        
        # 3. Algebraic Expansion of (p - one_hot)^2
        loss = sum_sq_probs - 2*p_target + 1.0
        
        return loss.mean()

# ----------------------------------------------------------------------
# 4. Algebraic Attention
# ----------------------------------------------------------------------
class AlgebraicAttention(nn.Module):
    def __init__(self, d_model, n_head, dropout=0.1, power=4):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.d_head = d_model // n_head
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        # We remove head_scale logic in favor of strict Score Normalization
        # This learnable gain controls exactly how "sharp" the attention is.
        # We init at 4.0 because x=4.0 is where Rational Sigmoid hits ~0.8 (strong signal)
        self.score_gain = nn.Parameter(torch.tensor(4.0))
        
        self.rational_softmax = RationalSoftmax(power)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, causal_mask=None):
        B, T, C = x.shape
        Q = self.q_proj(x).view(B, T, self.n_head, self.d_head).transpose(1, 2)
        K = self.k_proj(x).view(B, T, self.n_head, self.d_head).transpose(1, 2)
        V = self.v_proj(x).view(B, T, self.n_head, self.d_head).transpose(1, 2)
        
        # 1. Raw Expansion
        scores = torch.matmul(Q, K.transpose(-2, -1))
        
        # 2. ALGEBRAIC SCORE NORMALIZATION (The Fix)
        # We force the scores to fit into the rational curve's sensitive region.
        
        # A. Zero Centering (Remove shift)
        scores_mean = scores.mean(dim=-1, keepdim=True)
        scores = scores - scores_mean
        
        # B. L1 Variance Normalization (Rational "Standard Deviation")
        # mad = Mean Absolute Deviation
        scores_mad = scores.abs().mean(dim=-1, keepdim=True) + 1e-6
        scores = scores / scores_mad
        
        # C. Controlled Scaling
        # Now scores are effectively [-1, 1] (on average). 
        # We multiply by score_gain to stretch them to [-4, 4] or whatever the model needs.
        scores = scores * self.score_gain
        
        if causal_mask is not None:
            scores = scores.masked_fill(causal_mask, -1000.0)
        
        # Contraction
        attn_weights = self.rational_softmax(scores)
        attn_weights = self.dropout(attn_weights)
        out = torch.matmul(attn_weights, V)
        return self.out_proj(out.transpose(1, 2).reshape(B, T, C))
# 5. Transformer Block (Pre-Norm)
# ----------------------------------------------------------------------
class AlgebraicTransformerBlock(nn.Module):
    def __init__(self, d_model, n_head, d_ffn, dropout=0.1, power=4):
        super().__init__()
        self.attn = AlgebraicAttention(d_model, n_head, dropout, power)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ffn),
            AlgebraicReLU(), 
            nn.Linear(d_ffn, d_model),
            nn.Dropout(dropout)
        )
        self.norm1 = AlgebraicNorm(d_model)
        self.norm2 = AlgebraicNorm(d_model)
        
    def forward(self, x, causal_mask=None):
        x = x + self.attn(self.norm1(x), causal_mask=causal_mask)
        x = x + self.ffn(self.norm2(x))
        return x

# ----------------------------------------------------------------------
# 6. The Model
# ----------------------------------------------------------------------
class AlgebraicTransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model, n_head, n_layers, d_ffn, block_size, dropout=0.1, power=4):
        super().__init__()
        self.block_size = block_size
        self.vocab_size = vocab_size
        
        self.token_embedding_table = nn.Embedding(vocab_size, d_model)
        self.position_embedding_table = nn.Embedding(block_size, d_model)
        self.blocks = nn.ModuleList([
            AlgebraicTransformerBlock(d_model, n_head, d_ffn, dropout, power) 
            for _ in range(n_layers)
        ])
        
        self.final_norm = AlgebraicNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)
        
        # Causal Mask
        causal_mask = torch.triu(torch.ones(block_size, block_size), diagonal=1).bool()
        self.register_buffer('causal_mask', causal_mask.view(1, 1, block_size, block_size))
        
        # Purely Algebraic Loss
        self.criterion = AlgebraicMSELoss(num_classes=vocab_size)

        # final softmax for inference/loss
        self.final_softmax = RationalSoftmax(power=4)
    def forward(self, idx, targets=None):
        B, T = idx.shape
        
        # 1. Embedding (Lookup is technically algebraic selection)
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        mask = self.causal_mask[:, :, :T, :T]
        
        # 2. Process Blocks
        for block in self.blocks:
            x = block(x, causal_mask=mask)
            
        # 3. Final Projection
        x = self.final_norm(x)
        logits = self.lm_head(x) # [B, T, Vocab]
        
        loss = None
        if targets is not None:
            # Important: The output of lm_head is unbounded "Energy".
            # We must project it to Probability Space before calculating 
            # Geometric distance, otherwise MSE is meaningless.
            
            # We use the same Rational Softmax logic here:
            probs = self.final_softmax(logits)
            
            flat_probs = probs.reshape(-1, self.vocab_size)
            flat_targets = targets.reshape(-1)
            
            loss = self.criterion(flat_probs, flat_targets)
            
        return logits, loss
    def generate(self, idx, max_new_tokens):
        # Simple generation loop
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = self.final_softmax(logits)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
