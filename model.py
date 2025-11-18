import torch 
import torch.nn as nn
import torch.nn.functional as F


class AlgebraicLayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.a = nn.Parameter(torch.tensor([1.0, 0.5, 0.1]))
        self.b_raw = nn.Parameter(torch.tensor([1.0, 0.3, 0.05]))
    
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        variance = x.var(dim=-1, keepdim=True, unbiased=False)
        z = variance + self.eps
        b = F.softplus(self.b_raw, beta=10)
        p = self.a[0] + self.a[1] * z + self.a[2] * z**2
        q = b[0] + b[1] * z + b[2] * z**2
        inv_sqrt = p / q
        x_normalized = (x - mean) * inv_sqrt
        return x_normalized * self.gamma + self.beta

class AlgebraicAttention(nn.Module):
    def __init__(self, d_model, n_head, max_relative_positions=32):
        super().__init__()
        self.n_head = n_head
        self.d_head = d_model // n_head
        self.scale = self.d_head ** -0.5
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.num_buckets = 2 * max_relative_positions - 1
        self.max_relative_positions = max_relative_positions
        self.relative_position_bias = nn.Embedding(self.num_buckets, self.n_head)

    def _compute_t5_buckets(self, relative_positions):
        buckets = relative_positions.clamp(
            -self.max_relative_positions + 1, 
            self.max_relative_positions - 1
        )
        buckets = buckets + (self.max_relative_positions - 1)
        return buckets

    def forward(self, x, casual_mask=None): # Added mask for LM
        B, T, C = x.shape
        
        Q = self.q_proj(x).view(B, T, self.n_head, self.d_head).transpose(1, 2)
        K = self.k_proj(x).view(B, T, self.n_head, self.d_head).transpose(1, 2)
        V = self.v_proj(x).view(B, T, self.n_head, self.d_head).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        q_pos = torch.arange(T, device=x.device).unsqueeze(1)
        k_pos = torch.arange(T, device=x.device).unsqueeze(0)
        relative_positions = k_pos - q_pos
        
        buckets = self._compute_t5_buckets(relative_positions)
        bias = self.relative_position_bias(buckets).permute(2, 0, 1)
        
        scores = scores + bias.unsqueeze(0)
        
        # Apply casual mask for language modeling
        if casual_mask is not None:
            scores = scores.masked_fill(casual_mask == 0, float('-inf'))
                    
        if casual_mask is not None:
             scores = scores.masked_fill(casual_mask == 0, 0) # Zero-out masked scores

        attn_weights = F.relu(scores) + 1e-6
        
        # Re-apply mask to weights in case of numerical issues
        if casual_mask is not None:
            attn_weights = attn_weights.masked_fill(casual_mask == 0, 0)
            
        attn_weights = attn_weights / (attn_weights.sum(dim=-1, keepdim=True) + 1e-6)
        
        out = torch.matmul(attn_weights, V)
        return self.out_proj(out.transpose(1, 2).reshape(B, T, C))

class AlgebraicTransformerBlock(nn.Module):
    def __init__(self, d_model, n_head, d_ffn, dropout=0.1):
        super().__init__()
        self.norm1 = AlgebraicLayerNorm(d_model)
        # This T5-bias implementation's max_relative_positions must be >= BLOCK_SIZE/2
        self.attn = AlgebraicAttention(d_model, n_head, max_relative_positions=128) 
        self.norm2 = AlgebraicLayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ffn),
            nn.ReLU(),
            nn.Linear(d_ffn, d_model),
            nn.Dropout(dropout)
        )
        self.residual_scale = nn.Parameter(torch.tensor(0.5))
        
    def forward(self, x, casual_mask=None):
        scale = torch.clamp(self.residual_scale, 0.2, 1.0)
        x = x + scale * self.attn(self.norm1(x), casual_mask=casual_mask)
        x = x + scale * self.ffn(self.norm2(x))
        return x

class AlgebraicTransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model, n_head, n_layers, d_ffn, block_size):
        super().__init__()
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, d_model)
        
        self.blocks = nn.ModuleList(
            [AlgebraicTransformerBlock(d_model, n_head, d_ffn) for _ in range(n_layers)]
        )
        self.ln_f = AlgebraicLayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)
        
        # Create the casual mask once
        mask = (1 - torch.tril(torch.ones(1, 1, block_size, block_size))).bool()
        self.register_buffer('casual_mask', mask)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        
        # Ensure input is not longer than block_size
        idx = idx[:, -self.block_size:]
        T = idx.shape[1]
        
        tok_emb = self.token_embedding_table(idx) # [B, T, C]
        
        # Get the correct-sized mask
        mask = self.casual_mask[:, :, :T, :T]
        
        x = tok_emb
        for block in self.blocks:
            x = block(x, casual_mask=mask)
            
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            # Align targets
            targets = targets[:, -T:]
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            
        return logits, loss
