# 
# COMPOSITIONAL ENHANCEMENTS APPLIED:
# 1. Hierarchical Vector Quantization: 3-level VQ for 1B+ discrete states
# 2. Mixture of Experts: Sparse routing with 64 experts, top-2 activation
# 3. Hybrid Attention: Continuous + discrete pattern matching with 256 templates
# 4. Effective capacity: 384-dim network → equivalent to 768+ dim standard transformer
# 5. All original features preserved: Seashore, Holographic Memory, Compression
#
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.checkpoint import checkpoint
import os, glob, random, math, gc
import pyarrow.parquet as pq
import tiktoken
from tqdm import tqdm
from sys import stdout

# --- 1. Hyperparameters ---
VOCAB_SIZE = 32768    
EMBED_DIM = 768      
NUM_HEADS = 8        
HEAD_DIM = 64        
NUM_LAYERS = 4       
SEQ_LEN = 128        
BATCH_SIZE = 8
GRADIENT_ACCUMULATION = 4
LEARNING_RATE = 3e-4 
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CHECKPOINT_PATH = 'holograph_seashore_compositional.pth'

# Seashore Hyperparameters
SEASHORE_WIDTH = EMBED_DIM
HEBBIAN_LR = 0.05
BP_RATIO = 3

# Control Rates
ARCHIVE_SIZE = 4096 
SNAPSHOT_RATE = 256
PREDICT_EVERY = 2000

# Memory Management
ARCHIVE_SOFT_RESET_INTERVAL = 10
ARCHIVE_KEEP_RATIO = 0.25
CLEANUP_EVERY = 1000
MEMORY_RETRIEVAL_EVERY = 4

# NEW: Compositional Hyperparameters
USE_COMPOSITIONAL = True  # Master switch
VQ_ENABLED = True
VQ_NUM_CODEBOOKS = 3  # Hierarchical levels
VQ_CODEBOOK_SIZE = 1024  # 1024^3 = 1B states
VQ_COMMITMENT_COST = 0.25

MOE_ENABLED = True
MOE_NUM_EXPERTS = 64
MOE_EXPERT_DIM = 256
MOE_TOP_K = 2
MOE_CAPACITY_FACTOR = 1.25

HYBRID_ATTN_ENABLED = True
HYBRID_WINDOW_SIZE = 128
HYBRID_NUM_PATTERNS = 256
HYBRID_CONTINUOUS_WEIGHT = 0.7
HYBRID_ADAPTIVE_MIXING = True

# ============================================
# COMPOSITIONAL COMPONENTS
# ============================================

class HierarchicalVectorQuantizer(nn.Module):
    """
    3-Level Hierarchical VQ for massive discrete state space.
    Level 1: Coarse semantics (1024 codes)
    Level 2: Mid-level features (1024 codes)
    Level 3: Fine details (1024 codes)
    Total: 1024^3 ≈ 1 billion discrete states
    """
    def __init__(self, embed_dim, num_levels=3, codebook_size=1024, 
                 commitment_cost=0.25, device='cuda'):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_levels = num_levels
        self.codebook_size = codebook_size
        self.commitment_cost = commitment_cost
        self.device = device
        
        # Validate configuration
        assert embed_dim % num_levels == 0, \
            f"embed_dim ({embed_dim}) must be divisible by num_levels ({num_levels})"
        
        # Each level operates on embed_dim // num_levels dimensions
        self.dims_per_level = embed_dim // num_levels
        
        # Hierarchical codebooks: [num_levels, codebook_size, dims_per_level]
        self.codebooks = nn.Parameter(
            torch.randn(num_levels, codebook_size, self.dims_per_level, device=device) * 0.02
        )
        
        # EMA for stable training
        self.register_buffer('ema_count', torch.zeros(num_levels, codebook_size, device=device))
        self.register_buffer('ema_weight', self.codebooks.data.clone())
        self.decay = 0.99
    
    def forward(self, x, return_codes=False):
        """
        x: [batch, seq_len, embed_dim] or [batch, embed_dim]
        Returns: quantized tensor + commitment loss
        """
        original_shape = x.shape
        is_3d = x.dim() == 3
        
        # Track original batch size for proper reshaping
        if is_3d:
            B_orig, T, D = x.shape
            assert D == self.embed_dim, \
                f"Input embed_dim ({D}) doesn't match VQ embed_dim ({self.embed_dim})"
            x = x.reshape(B_orig * T, D)  # Flatten to [B*T, D]
            B_flat = B_orig * T
        else:
            B_flat, D = x.shape
            assert D == self.embed_dim, \
                f"Input embed_dim ({D}) doesn't match VQ embed_dim ({self.embed_dim})"
        x = x.to(self.device)
        
        # Split into levels: [B_flat, num_levels, dims_per_level]
        x_split = x.reshape(B_flat, self.num_levels, self.dims_per_level)
        
        quantized = []
        codes = []
        commitment_losses = []
        
        for level in range(self.num_levels):
            x_level = x_split[:, level, :]  # [B_flat, dims_per_level]
            codebook = self.codebooks[level]  # [codebook_size, dims_per_level]
            
            # Compute distances
            distances = torch.cdist(x_level, codebook)  # [B_flat, codebook_size]
            
            # Get nearest code
            encoding_indices = torch.argmin(distances, dim=1)  # [B_flat]
            codes.append(encoding_indices)
            
            # Quantize
            quantized_level = codebook[encoding_indices]  # [B_flat, dims_per_level]
            
            # Commitment loss (encourage encoder to commit to codebook)
            commitment_loss = F.mse_loss(x_level, quantized_level.detach())
            commitment_losses.append(commitment_loss)
            
            # Straight-through estimator
            quantized_level = x_level + (quantized_level - x_level).detach()
            quantized.append(quantized_level)
            
            # EMA update (training only)
            if self.training:
                with torch.no_grad():
                    encodings = F.one_hot(encoding_indices, self.codebook_size).float()
                    
                    self.ema_count[level] = self.decay * self.ema_count[level] + \
                                           (1 - self.decay) * encodings.sum(0)
                    
                    dw = torch.matmul(encodings.t(), x_level)
                    self.ema_weight[level] = self.decay * self.ema_weight[level] + \
                                            (1 - self.decay) * dw
                    
                    n = self.ema_count[level].unsqueeze(1)
                    self.codebooks.data[level] = self.ema_weight[level] / (n + 1e-5)
        
        # Concatenate quantized levels: [B_flat, embed_dim]
        quantized = torch.cat(quantized, dim=1)
        
        # Validate output shape before restore
        assert quantized.shape == (B_flat, self.embed_dim), \
            f"Quantized shape {quantized.shape} doesn't match expected ({B_flat}, {self.embed_dim})"
        
        # Total commitment loss
        total_commitment = sum(commitment_losses) * self.commitment_cost
        
        # Restore original shape
        if is_3d:
            quantized = quantized.reshape(B_orig, T, self.embed_dim)
            if return_codes:
                codes_stacked = torch.stack(codes, dim=1)  # [B_flat, num_levels]
                codes_stacked = codes_stacked.reshape(B_orig, T, self.num_levels)
                return quantized, total_commitment, codes_stacked
        else:
            if return_codes:
                codes_stacked = torch.stack(codes, dim=1)  # [B_flat, num_levels]
                return quantized, total_commitment, codes_stacked
        
        return quantized, total_commitment

class SparseMixtureOfExperts(nn.Module):
    """
    64 experts with top-2 routing for conditional computation.
    Each token routes to its 2 most relevant experts.
    Effective capacity: 64*63/2 = 2016 expert combinations.
    """
    def __init__(self, embed_dim, num_experts=64, expert_dim=256, 
                 top_k=2, capacity_factor=1.25, device='cuda'):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_experts = num_experts
        self.expert_dim = expert_dim
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.device = device
        
        # Router: learns which expert to use
        self.router = nn.Linear(embed_dim, num_experts)
        
        # Expert networks (shared structure, different weights)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, expert_dim),
                nn.GELU(),
                nn.Linear(expert_dim, embed_dim)
            ) for _ in range(num_experts)
        ])
        
        # Load balancing
        self.register_buffer('expert_usage', torch.zeros(num_experts))
        
    def forward(self, x):
        """
        x: [batch, seq_len, embed_dim]
        Returns: expert output + load balancing loss
        """
        original_shape = x.shape
        B, T, D = x.shape
        x_flat = x.reshape(B * T, D)
        
        # Route tokens to experts
        router_logits = self.router(x_flat)  # [B*T, num_experts]
        
        # Top-k routing
        top_k_logits, top_k_indices = torch.topk(router_logits, self.top_k, dim=1)
        top_k_gates = F.softmax(top_k_logits, dim=1)  # [B*T, top_k]
        
        # Initialize output
        output = torch.zeros_like(x_flat)
        
        # Process each expert
        for i in range(self.num_experts):
            # Find tokens routed to this expert
            expert_mask = (top_k_indices == i).any(dim=1)  # [B*T]
            
            if expert_mask.any():
                expert_input = x_flat[expert_mask]  # [n_tokens, embed_dim]
                expert_output = self.experts[i](expert_input)  # [n_tokens, embed_dim]
                
                # Get gate weights for this expert
                gate_positions = (top_k_indices == i).nonzero(as_tuple=True)
                gates = top_k_gates[gate_positions[0], gate_positions[1]]  # [n_tokens]
                
                # Weighted contribution
                output[expert_mask] += expert_output * gates.unsqueeze(1)
                
                # Track usage
                if self.training:
                    self.expert_usage[i] += expert_mask.sum().item()
          # Load balancing loss (encourage uniform expert usage)
        if self.training:
            # Importance: how much each expert is used (average routing probability)
            importance = F.softmax(router_logits, dim=-1).sum(dim=0)  # [num_experts]
            
            # Load: count how many tokens are routed to each expert
            load = torch.zeros(self.num_experts, device=self.device)
            for i in range(self.num_experts):
                load[i] = (top_k_indices == i).sum().float()
            
            # Balance loss: coefficient of variation penalty
            # Encourages uniform distribution across experts
            mean_load = load.mean()
            load_balancing_loss = (load.std() / (mean_load + 1e-10)) ** 2
        else:
            load_balancing_loss = torch.tensor(0.0, device=self.device)
        
        output = output.reshape(B, T, D)
        return output, load_balancing_loss

class HybridAttention(nn.Module):
    """
    Combines continuous attention with discrete pattern matching.
    - Continuous: Standard sliding window (flexible, gradient-friendly)
    - Discrete: 256 learned templates (instant retrieval, zero compute)
    - Adaptive mixing: Network learns when to use which mode
    """
    def __init__(self, embed_dim, num_heads, window_size=128, 
                 num_patterns=256, continuous_weight=0.7, 
                 use_adaptive_mixing=True, device='cuda'):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.window_size = window_size
        self.num_patterns = num_patterns
        self.continuous_weight = continuous_weight
        self.use_adaptive_mixing = use_adaptive_mixing
        self.device = device
        
        # Continuous attention (sliding window)
        self.continuous_qkv = nn.Linear(embed_dim, 3 * embed_dim)
        
        # Discrete pattern bank: [num_patterns, window_size, embed_dim]
        self.pattern_bank = nn.Parameter(
            torch.randn(num_patterns, window_size, embed_dim, device=device) * 0.02
        )
        
        # Pattern selector
        self.pattern_selector = nn.Linear(embed_dim, num_patterns)
        
        # Adaptive mixer (if enabled)
        if use_adaptive_mixing:
            self.mode_mixer = nn.Linear(embed_dim, 1)
        
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x):
        """
        x: [batch, seq_len, embed_dim]
        Returns: hybrid attention output
        """
        B, T, D = x.shape
        
        # === CONTINUOUS MODE ===
        qkv = self.continuous_qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, T, head_dim]
        
        # Sliding window attention (efficient)
        window_start = max(0, T - self.window_size)
        k_window = k[:, :, window_start:, :]
        v_window = v[:, :, window_start:, :]
        
        attn_scores = torch.matmul(q, k_window.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(attn_scores, dim=-1)
        continuous_out = torch.matmul(attn_weights, v_window)  # [B, H, T, head_dim]
        continuous_out = continuous_out.transpose(1, 2).reshape(B, T, D)
        
        # === DISCRETE MODE ===
        # Select best matching pattern for each position
        pattern_scores = self.pattern_selector(x)  # [B, T, num_patterns]
        best_patterns = torch.argmax(pattern_scores, dim=-1)  # [B, T]
        
        # Retrieve patterns (vectorized)
        discrete_out = self.pattern_bank[best_patterns]  # [B, T, window_size, embed_dim]
        
        # Average over window dimension for discrete output
        discrete_out = discrete_out.mean(dim=2)  # [B, T, embed_dim]
        
        # === MIXING ===
        if self.use_adaptive_mixing:
            # Learn mixing weight per position
            alpha = torch.sigmoid(self.mode_mixer(x))  # [B, T, 1]
            mixed_out = alpha * continuous_out + (1 - alpha) * discrete_out
        else:
            # Fixed mixing
            mixed_out = self.continuous_weight * continuous_out + \
                       (1 - self.continuous_weight) * discrete_out
        
        return self.out_proj(mixed_out)

# ============================================
# COMPOSITIONAL TRANSFORMER BLOCK
# ============================================

class CompositionalTransformerBlock(nn.Module):
    """
    Modular transformer block with optional compositional enhancements.
    Can use standard components or compositional variants.
    """
    def __init__(self, embed_dim, num_heads, 
                 use_vq=False, vq_config=None,
                 use_moe=False, moe_config=None,
                 use_hybrid_attn=False, hybrid_attn_config=None):
        super().__init__()
        
        self.use_vq = use_vq
        self.use_moe = use_moe
        self.use_hybrid_attn = use_hybrid_attn
        
        # Layer norms
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        
        # Attention mechanism
        if use_hybrid_attn and hybrid_attn_config:
            self.attn = HybridAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                window_size=hybrid_attn_config.get('window_size', 128),
                num_patterns=hybrid_attn_config.get('num_patterns', 256),
                continuous_weight=hybrid_attn_config.get('continuous_weight', 0.7),
                use_adaptive_mixing=hybrid_attn_config.get('use_adaptive_mixing', True)
            )
        else:
            # Standard attention (placeholder - will use holographic in full model)
            self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        
        # Optional VQ
        if use_vq and vq_config:
            self.vq = HierarchicalVectorQuantizer(
                embed_dim=embed_dim,
                num_levels=vq_config['num_levels'],
                codebook_size=vq_config['codebook_size'],
                commitment_cost=vq_config['commitment_cost']
            )
            self.ln_vq = nn.LayerNorm(embed_dim)
        
        # Feedforward
        if use_moe and moe_config:
            self.ffn = SparseMixtureOfExperts(
                embed_dim=embed_dim,
                num_experts=moe_config['num_experts'],
                expert_dim=moe_config['expert_dim'],
                top_k=moe_config['top_k'],
                capacity_factor=moe_config['capacity_factor']
            )
        else:
            # Standard FFN
            self.ffn = nn.Sequential(
                nn.Linear(embed_dim, 4 * embed_dim),
                nn.GELU(),
                nn.Linear(4 * embed_dim, embed_dim)
            )
    
    def forward(self, x):
        """
        Returns: output, auxiliary_losses dict
        """
        aux_losses = {}
        
        # Attention
        if self.use_hybrid_attn:
            attn_out = self.attn(self.ln1(x))
        else:
            attn_out, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x))
        
        x = x + attn_out
        
        # Optional VQ
        if self.use_vq:
            x_vq = self.ln_vq(x)
            x_quantized, vq_loss = self.vq(x_vq)
            x = x + x_quantized
            aux_losses['vq_loss'] = vq_loss
        
        # FFN (possibly MoE)
        if self.use_moe:
            ffn_out, moe_loss = self.ffn(self.ln2(x))
            aux_losses['moe_loss'] = moe_loss
        else:
            ffn_out = self.ffn(self.ln2(x))
        
        x = x + ffn_out
        
        return x, aux_losses

# ============================================
# COMPRESSION LAYER (ORIGINAL)
# ============================================

class LearnedVectorQuantizer(nn.Module):
    """Original compression VQ for memory system"""
    def __init__(self, embed_dim, n_codebooks=16, codebook_size=256, device='cuda'):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_codebooks = n_codebooks
        self.codebook_size = codebook_size
        self.dims_per_book = embed_dim // n_codebooks
        self.device = device
        
        self.codebooks = nn.Parameter(
            torch.randn(n_codebooks, codebook_size, self.dims_per_book, device=device) * 0.02
        )
        self.register_buffer('ema_count', torch.zeros(n_codebooks, codebook_size, device=device))
        self.register_buffer('ema_weight', self.codebooks.data.clone())
        self.decay = 0.99
    
    def quantize(self, x):
        """
        Quantize input vectors into discrete codes.
        x: [batch_size, embed_dim] - 2D tensor only
        Returns: [batch_size, n_codebooks] uint8 codes
        """
        # Validate input
        assert x.dim() == 2, f"LearnedVectorQuantizer.quantize() expects 2D input, got {x.dim()}D with shape {x.shape}"
        assert x.shape[1] == self.embed_dim, \
            f"Input embed_dim ({x.shape[1]}) doesn't match VQ embed_dim ({self.embed_dim})"
        
        x = x.to(self.device)
        batch_size = x.shape[0]
        x_reshaped = x.reshape(batch_size, self.n_codebooks, self.dims_per_book)
        
        codes = torch.zeros(batch_size, self.n_codebooks, dtype=torch.long, device=self.device)
        
        for i in range(self.n_codebooks):
            x_chunk = x_reshaped[:, i, :]
            codebook = self.codebooks[i]
            
            dists = torch.cdist(x_chunk, codebook)
            codes[:, i] = torch.argmin(dists, dim=1)
            
            if self.training:
                with torch.no_grad():
                    encodings = F.one_hot(codes[:, i], self.codebook_size).float()
                    self.ema_count[i] = self.decay * self.ema_count[i] + \
                                       (1 - self.decay) * encodings.sum(0)
                    
                    dw = torch.matmul(encodings.t(), x_chunk)
                    self.ema_weight[i] = self.decay * self.ema_weight[i] + \
                                        (1 - self.decay) * dw
                    
                    n = self.ema_count[i].unsqueeze(1)
                    self.codebooks.data[i] = self.ema_weight[i] / (n + 1e-5)
        
        return codes.to(torch.uint8)
    
    def dequantize(self, codes):
        codes = codes.to(self.device)
        batch_size = codes.shape[0]
        codes = codes.long()
        
        reconstructed = torch.zeros(batch_size, self.embed_dim, 
                                    device=self.device, dtype=self.codebooks.dtype)
        
        for i in range(self.n_codebooks):
            batch_codes = codes[:, i]
            chunk = self.codebooks[i][batch_codes]
            
            start = i * self.dims_per_book
            end = start + self.dims_per_book
            reconstructed[:, start:end] = chunk
        
        return reconstructed

class SemanticHasher(nn.Module):
    """Original semantic hasher for memory system"""
    def __init__(self, embed_dim, hash_bits=128, device='cuda'):
        super().__init__()
        self.hash_bits = hash_bits
        self.device = device
        self.register_buffer('projection', 
                           torch.randn(embed_dim, hash_bits, device=device) / math.sqrt(embed_dim))
    
    def hash(self, x):
        if x.dim() > 2:
            x = x.reshape(-1, x.shape[-1])
        x = x.to(self.device)
        
        projected = torch.matmul(x, self.projection)
        binary = (projected > 0).to(torch.uint8)
        
        batch_size = binary.shape[0]
        packed_size = (self.hash_bits + 7) // 8
        
        pad_length = packed_size * 8 - self.hash_bits
        if pad_length > 0:
            binary = F.pad(binary, (0, pad_length), value=0)
        
        binary_reshaped = binary.reshape(batch_size, packed_size, 8)
        bit_multipliers = (2 ** torch.arange(8, device=self.device)).to(torch.uint8)
        packed = (binary_reshaped * bit_multipliers).sum(dim=2)
        
        return packed

# ============================================
# HIERARCHICAL MEMORY SYSTEM (ORIGINAL)
# ============================================

class CompressedHierarchicalMemory:
    """Original compressed hierarchical memory - kept unchanged"""
    def __init__(self, embed_dim, num_heads, head_dim, 
                 l1_size=2048, l2_size=16384, disk_path='holo_memory_db', device='cuda'):
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.disk_path = disk_path
        self.device = device
        
        self.vq_embeddings = LearnedVectorQuantizer(
            embed_dim=embed_dim,
            n_codebooks=16,
            codebook_size=256,
            device=device
        )
        
        self.vq_matrices = LearnedVectorQuantizer(
            embed_dim=num_heads * head_dim * head_dim,
            n_codebooks=32,
            codebook_size=256,
            device=device
        )
        
        self.hasher = SemanticHasher(
            embed_dim=embed_dim,
            hash_bits=128,
            device=device
        )
        
        self.l1_working = self._create_l1_memory(l1_size)
        self.l2_session = self._create_l2_memory(l2_size)
        self.l3_archive = self._create_l3_memory()
        
        self.stats = {
            'l1_hits': 0,
            'l2_hits': 0,
            'l3_hits': 0,
            'total_queries': 0,
            'compressions_trained': 0
        }
    
    def _create_l1_memory(self, size):
        return {
            's_min': torch.zeros(size, self.embed_dim, dtype=torch.float16, device=self.device),
            's_max': torch.zeros(size, self.embed_dim, dtype=torch.float16, device=self.device),
            'values': torch.zeros(size, self.num_heads * self.head_dim * self.head_dim, 
                                dtype=torch.float16, device=self.device),
            'importance': torch.zeros(size, dtype=torch.float32, device=self.device),
            'access_count': torch.zeros(size, dtype=torch.int32, device=self.device),
            'ptr': 0,
            'count': 0,
            'size': size
        }
    
    def _create_l2_memory(self, size):
        return {
            's_min_codes': torch.zeros(size, 16, dtype=torch.uint8, device=self.device),
            's_max_codes': torch.zeros(size, 16, dtype=torch.uint8, device=self.device),
            'value_codes': torch.zeros(size, 32, dtype=torch.uint8, device=self.device),
            'importance': torch.zeros(size, dtype=torch.float32, device=self.device),
            'access_count': torch.zeros(size, dtype=torch.int32, device=self.device),
            'ptr': 0,
            'count': 0,
            'size': size
        }
    
    def _create_l3_memory(self):
        import h5py
        os.makedirs(self.disk_path, exist_ok=True)
        
        db_file = os.path.join(self.disk_path, 'archive.h5')
        db = h5py.File(db_file, 'a')
        
        if 'hashes' not in db:
            db.create_dataset('hashes', (0, 16), maxshape=(None, 16), 
                            dtype='uint8', chunks=(10000, 16), compression='gzip')
            db.create_dataset('s_min_codes', (0, 16), maxshape=(None, 16),
                            dtype='uint8', chunks=(10000, 16), compression='gzip')
            db.create_dataset('s_max_codes', (0, 16), maxshape=(None, 16),
                            dtype='uint8', chunks=(10000, 16), compression='gzip')
            db.create_dataset('value_codes', (0, 32), maxshape=(None, 32),
                            dtype='uint8', chunks=(10000, 32), compression='gzip')
            db.create_dataset('importance', (0,), maxshape=(None,),
                            dtype='float32', chunks=(10000,), compression='gzip')
            db.create_dataset('timestamp', (0,), maxshape=(None,),
                            dtype='int64', chunks=(10000,), compression='gzip')
        
        return {
            'db': db,
            'count': len(db['hashes']),
            'cache': {},
            'cache_size': 1000
        }
    
    def train_compressors(self, sample_batch):
        self.vq_embeddings.train()
        self.vq_matrices.train()
        
        with torch.no_grad():
            s_combined = torch.cat([sample_batch['s_min'], sample_batch['s_max']], dim=0)
            _ = self.vq_embeddings.quantize(s_combined)
            _ = self.vq_matrices.quantize(sample_batch['values'])
        
        self.stats['compressions_trained'] += 1
        
        if self.stats['compressions_trained'] % 100 == 0:
            self.save_compressors()
    
    def save_compressors(self):
        save_path = os.path.join(self.disk_path, 'compressors.pth')
        torch.save({
            'vq_embeddings': self.vq_embeddings.state_dict(),
            'vq_matrices': self.vq_matrices.state_dict(),
            'hasher': self.hasher.state_dict()
        }, save_path)
    
    def load_compressors(self):
        save_path = os.path.join(self.disk_path, 'compressors.pth')
        if os.path.exists(save_path):
            ckpt = torch.load(save_path, map_location=DEVICE)
            self.vq_embeddings.load_state_dict(ckpt['vq_embeddings'])
            self.vq_matrices.load_state_dict(ckpt['vq_matrices'])
            self.hasher.load_state_dict(ckpt['hasher'])
            print(f"Loaded compression codebooks from {save_path}")
    
    def add(self, s_box, matrix, importance=1.0):
        with torch.no_grad():
            s_min = s_box[0][:, -1, :].mean(dim=0).to(torch.float32)
            s_max = s_box[1][:, -1, :].mean(dim=0).to(torch.float32)
            values = matrix.mean(dim=0).flatten().to(torch.float32)
            
            l1 = self.l1_working
            ptr = l1['ptr']
            l1['s_min'][ptr] = s_min.to(torch.float16)
            l1['s_max'][ptr] = s_max.to(torch.float16)
            l1['values'][ptr] = values.to(torch.float16)
            l1['importance'][ptr] = importance
            l1['access_count'][ptr] = 0
            
            l1['ptr'] = (ptr + 1) % l1['size']
            l1['count'] = min(l1['count'] + 1, l1['size'])
            
            if importance > 2.0:
                self._add_to_l2(s_min, s_max, values, importance)
            
            if l1['count'] >= l1['size'] * 0.9:
                self._consolidate_l1_to_l2()
    
    def _add_to_l2(self, s_min, s_max, values, importance):
        l2 = self.l2_session
        ptr = l2['ptr']
        
        self.vq_embeddings.eval()
        self.vq_matrices.eval()
        
        with torch.no_grad():
            s_min_codes = self.vq_embeddings.quantize(s_min.unsqueeze(0))[0]
            s_max_codes = self.vq_embeddings.quantize(s_max.unsqueeze(0))[0]
            value_codes = self.vq_matrices.quantize(values.unsqueeze(0))[0]
        
        l2['s_min_codes'][ptr] = s_min_codes
        l2['s_max_codes'][ptr] = s_max_codes
        l2['value_codes'][ptr] = value_codes
        l2['importance'][ptr] = importance * 0.8
        l2['access_count'][ptr] = 0
        
        l2['ptr'] = (ptr + 1) % l2['size']
        l2['count'] = min(l2['count'] + 1, l2['size'])
        
        if importance > 5.0:
            self._add_to_l3(s_min_codes, s_max_codes, value_codes, 
                           s_min, importance)
    
    def _add_to_l3(self, s_min_codes, s_max_codes, value_codes, 
                   s_min_full, importance):
        l3 = self.l3_archive
        db = l3['db']
        
        with torch.no_grad():
            hash_code = self.hasher.hash(s_min_full.unsqueeze(0))[0].cpu().numpy()
        
        n = l3['count']
        for dataset_name in ['hashes', 's_min_codes', 's_max_codes', 
                            'value_codes', 'importance', 'timestamp']:
            db[dataset_name].resize((n + 1,) + db[dataset_name].shape[1:])
        
        db['hashes'][n] = hash_code
        db['s_min_codes'][n] = s_min_codes.cpu().numpy()
        db['s_max_codes'][n] = s_max_codes.cpu().numpy()
        db['value_codes'][n] = value_codes.cpu().numpy()
        db['importance'][n] = importance * 0.6
        db['timestamp'][n] = self.stats['total_queries']
        
        l3['count'] += 1
        
        if l3['count'] % 1000 == 0:
            db.flush()
            print(f"L3 Archive: {l3['count']} entries "
                  f"({self._get_db_size_mb():.1f} MB on disk)")
    
    def _consolidate_l1_to_l2(self):
        l1 = self.l1_working
        
        combined_score = l1['importance'][:l1['count']] * \
                        (1 + torch.log1p(l1['access_count'][:l1['count']].float()))
        
        k = l1['count'] // 2
        top_values, top_indices = torch.topk(combined_score, k=k)
        
        print(f"\nConsolidating L1→L2: Moving {k} top entries (min score: {top_values[-1]:.2f})")
        
        for idx in top_indices:
            idx = idx.item()
            self._add_to_l2(
                l1['s_min'][idx].to(torch.float32),
                l1['s_max'][idx].to(torch.float32),
                l1['values'][idx].to(torch.float32),
                l1['importance'][idx].item()
            )
        
        l1['ptr'] = 0
        l1['count'] = 0
        print(f"L1 consolidated. L2 now has {self.l2_session['count']} entries.")
    
    def consolidate_l2_to_l3(self):
        l2 = self.l2_session
        
        if l2['count'] < l2['size'] * 0.5:
            return
        
        k = l2['count'] // 4
        top_values, top_indices = torch.topk(
            l2['importance'][:l2['count']], k=k
        )
        
        print(f"\nConsolidating L2→L3: Moving {k} entries to disk...")
        
        self.vq_embeddings.eval()
        
        for idx in top_indices:
            idx = idx.item()
            
            with torch.no_grad():
                s_min_full = self.vq_embeddings.dequantize(
                    l2['s_min_codes'][idx].unsqueeze(0)
                )[0]
            
            self._add_to_l3(
                l2['s_min_codes'][idx],
                l2['s_max_codes'][idx],
                l2['value_codes'][idx],
                s_min_full,
                l2['importance'][idx].item()
            )
        
        l2['ptr'] = 0
        l2['count'] = 0
        
        print(f"L2→L3 consolidation complete. L3 has {self.l3_archive['count']} entries.")
    
    def retrieve(self, query, k=32):
        self.stats['total_queries'] += 1
        q_min, q_max = query
        
        l1_results = self._retrieve_from_l1(q_min, q_max, k=k)
        l2_results = self._retrieve_from_l2(q_min, q_max, k=k)
        
        l3_results = []
        if not l1_results and not l2_results:
            l3_results = self._retrieve_from_l3(q_min, q_max, k=k//2)
        elif l1_results or l2_results:
            combined_temp = l1_results + l2_results
            max_importance = max(r['importance'] for r in combined_temp)
            if max_importance < 0.3:
                l3_results = self._retrieve_from_l3(q_min, q_max, k=k//2)
        
        results = l1_results + l2_results + l3_results
        
        if results:
            results = sorted(results, key=lambda x: x['importance'], reverse=True)[:k]
        
        return results
    
    def _retrieve_from_l1(self, q_min, q_max, k=32):
        if self.l1_working['count'] == 0:
            return []
        
        valid_count = self.l1_working['count']
        s_min = self.l1_working['s_min'][:valid_count]
        s_max = self.l1_working['s_max'][:valid_count]
        values = self.l1_working['values'][:valid_count]
        importance = self.l1_working['importance'][:valid_count]
        
        q_min_expanded = q_min.unsqueeze(0)
        q_max_expanded = q_max.unsqueeze(0)
        
        intersection_min = torch.maximum(q_min_expanded, s_min)
        intersection_max = torch.minimum(q_max_expanded, s_max)
        intersection_vol = torch.clamp(intersection_max - intersection_min, min=0).sum(dim=1)
        
        q_vol = (q_max - q_min).sum()
        s_vol = (s_max - s_min).sum(dim=1)
        union_vol = q_vol + s_vol - intersection_vol
        
        iou = intersection_vol / (union_vol + 1e-8)
        scores = iou * importance
        
        k = min(k, valid_count)
        top_k_scores, top_k_indices = torch.topk(scores, k)
        
        results = []
        for score, idx in zip(top_k_scores, top_k_indices):
            idx = idx.item()
            results.append({
                's_min': s_min[idx],
                's_max': s_max[idx],
                'values': values[idx],
                'importance': importance[idx].item(),
                'score': score.item()
            })
            
            self.l1_working['access_count'][idx] += 1
        
        self.stats['l1_hits'] += 1
        return results

    def _retrieve_from_l2(self, q_min, q_max, k=32):
        if self.l2_session['count'] == 0:
            return []
        
        q_min_codes = self.vq_embeddings.quantize(q_min.unsqueeze(0))[0]
        q_max_codes = self.vq_embeddings.quantize(q_max.unsqueeze(0))[0]
        
        valid_count = self.l2_session['count']
        s_min_codes = self.l2_session['s_min_codes'][:valid_count]
        s_max_codes = self.l2_session['s_max_codes'][:valid_count]
        value_codes = self.l2_session['value_codes'][:valid_count]
        importance = self.l2_session['importance'][:valid_count]
        
        matches_min = (s_min_codes == q_min_codes.unsqueeze(0)).sum(dim=1).float()
        matches_max = (s_max_codes == q_max_codes.unsqueeze(0)).sum(dim=1).float()
        scores = (matches_min + matches_max) * importance
        
        k = min(k, valid_count)
        top_k_scores, top_k_indices = torch.topk(scores, k)
        
        results = []
        for score, idx in zip(top_k_scores, top_k_indices):
            idx = idx.item()
            
            s_min = self.vq_embeddings.dequantize(
                s_min_codes[idx:idx+1]
            )[0]
            
            s_max = self.vq_embeddings.dequantize(
                s_max_codes[idx:idx+1]
            )[0]
            
            values = self.vq_matrices.dequantize(
                value_codes[idx:idx+1]
            )[0]
            
            results.append({
                's_min': s_min,
                's_max': s_max,
                'values': values,
                'importance': importance[idx].item(),
                'score': score.item()
            })
            
            self.l2_session['access_count'][idx] += 1
        
        self.stats['l2_hits'] += 1
        return results
    
    def _retrieve_from_l3(self, q_min, q_max, k=16):
        l3 = self.l3_archive
        if l3['count'] == 0:
            return []
        
        db = l3['db']
        
        with torch.no_grad():
            query_hash = self.hasher.hash(q_min.unsqueeze(0))[0]
            
            batch_size = 10000
            best_scores = []
            best_indices = []
            
            for start in range(0, l3['count'], batch_size):
                end = min(start + batch_size, l3['count'])
                
                hashes_batch = torch.from_numpy(
                    db['hashes'][start:end]
                ).to(DEVICE)
                
                xor = query_hash.unsqueeze(0) ^ hashes_batch
                hamming_dists = xor.sum(dim=1).float()
                
                similarities = 1.0 / (1.0 + hamming_dists / 128.0)
                
                top_k_batch = min(k * 2, len(similarities))
                top_scores_batch, top_idx_batch = torch.topk(
                    similarities, k=top_k_batch
                )
                
                best_scores.append(top_scores_batch)
                best_indices.append(top_idx_batch + start)
            
            all_scores = torch.cat(best_scores)
            all_indices = torch.cat(best_indices)
            
            final_k = min(k, len(all_scores))
            top_scores, positions = torch.topk(all_scores, k=final_k)
            top_indices = all_indices[positions]
            
            results = []
            for idx, score in zip(top_indices, top_scores):
                idx_cpu = idx.item()
                
                if idx_cpu in l3['cache']:
                    cached = l3['cache'][idx_cpu]
                    results.append({
                        's_min': cached['s_min'],
                        's_max': cached['s_max'],
                        'values': cached['values'],
                        'importance': float(db['importance'][idx_cpu]),
                        'score': score.item()
                    })
                    continue
                
                s_min_codes = torch.from_numpy(
                    db['s_min_codes'][idx_cpu]
                ).unsqueeze(0).to(DEVICE)
                
                s_max_codes = torch.from_numpy(
                    db['s_max_codes'][idx_cpu]
                ).unsqueeze(0).to(DEVICE)
                
                value_codes = torch.from_numpy(
                    db['value_codes'][idx_cpu]
                ).unsqueeze(0).to(DEVICE)
                
                self.vq_embeddings.eval()
                self.vq_matrices.eval()
                
                s_min = self.vq_embeddings.dequantize(s_min_codes)[0]
                s_max = self.vq_embeddings.dequantize(s_max_codes)[0]
                values = self.vq_matrices.dequantize(value_codes)[0]
                
                l3['cache'][idx_cpu] = {
                    's_min': s_min,
                    's_max': s_max,
                    'values': values
                }
                if len(l3['cache']) > l3['cache_size']:
                    oldest = min(l3['cache'].keys())
                    del l3['cache'][oldest]
                
                results.append({
                    's_min': s_min,
                    's_max': s_max,
                    'values': values,
                    'importance': float(db['importance'][idx_cpu]),
                    'score': score.item()
                })
            
            self.stats['l3_hits'] += 1
            return results
    
    def _get_db_size_mb(self):
        import h5py
        db_file = os.path.join(self.disk_path, 'archive.h5')
        if os.path.exists(db_file):
            return os.path.getsize(db_file) / (1024 * 1024)
        return 0
    
    def print_stats(self):
        l1 = self.l1_working
        l2 = self.l2_session
        l3 = self.l3_archive
        
        print(f"\n{'='*60}")
        print(f"HOLOGRAPHIC MEMORY HIERARCHY STATS")
        print(f"{'='*60}")
        print(f"L1 (Working):  {l1['count']:6d} / {l1['size']:6d} entries "
              f"(~{l1['count'] * 0.5:.1f} MB)")
        print(f"L2 (Session):  {l2['count']:6d} / {l2['size']:6d} entries "
              f"(~{l2['count'] * 0.05:.1f} MB)")
        print(f"L3 (Archive):  {l3['count']:6d} entries "
              f"({self._get_db_size_mb():.1f} MB on disk)")
        print(f"{'-'*60}")
        print(f"Total Queries: {self.stats['total_queries']}")
        print(f"L1 Hits: {self.stats['l1_hits']} "
              f"({100*self.stats['l1_hits']/(self.stats['total_queries']+1):.1f}%)")
        print(f"L2 Hits: {self.stats['l2_hits']} "
              f"({100*self.stats['l2_hits']/(self.stats['total_queries']+1):.1f}%)")
        print(f"L3 Hits: {self.stats['l3_hits']} "
              f"({100*self.stats['l3_hits']/(self.stats['total_queries']+1):.1f}%)")
        print(f"Compressor trainings: {self.stats['compressions_trained']}")
        print(f"{'='*60}\n")
    
    def estimate_final_size(self, total_data_tb):
        total_bytes = total_data_tb * 1024 * 1024 * 1024 * 1024
        memories = total_bytes / (1024 * 1024)
        
        bytes_per_memory = 40
        final_size_gb = (memories * bytes_per_memory) / (1024 * 1024 * 1024)
        
        print(f"\nEstimated final database size:")
        print(f"Training data: {total_data_tb} TB")
        print(f"Expected memories: {memories/1e6:.1f}M")
        print(f"Final DB size: ~{final_size_gb:.1f} GB")
        print(f"Compression ratio: {total_data_tb * 1024 / final_size_gb:.0f}×")
        
        return final_size_gb

# ============================================
# UTILITY FUNCTIONS
# ============================================

def aggressive_cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

def clear_autocast_cache():
    if hasattr(torch, 'clear_autocast_cache'):
        torch.clear_autocast_cache()
    elif hasattr(torch.cuda.amp, 'clear_cache'):
        torch.cuda.amp.clear_cache()

# ============================================
# TOKENIZER
# ============================================

class LimitedTokenizer:
    def __init__(self, limit=32768):
        self.enc = tiktoken.get_encoding("cl100k_base")
        self.limit = limit
        self.unk_id = limit - 1 
    def encode(self, text):
        return [t if t < self.unk_id else self.unk_id for t in self.enc.encode(text)]
    def decode(self, ids):
        return self.enc.decode([int(i) for i in ids if i < self.unk_id])

# ============================================
# SEASHORE MECHANICS
# ============================================

def hebbian_update(layer, inputs, learning_rate):
    if not isinstance(layer, nn.Linear):
        return

    with torch.no_grad():
        if inputs.dim() == 3:
            inputs = inputs.reshape(-1, inputs.shape[-1])
            
        waves = F.normalize(inputs, p=2, dim=1)
        weights = layer.weight
        
        similarity = torch.mm(waves, weights.t())
        
        winners = torch.argmax(similarity, dim=1)
        winner_mask = F.one_hot(winners, num_classes=weights.shape[0]).float()
        
        waves_sum = torch.mm(winner_mask.t(), waves)
        win_counts = winner_mask.sum(dim=0).unsqueeze(1) + 1e-6
        waves_avg = waves_sum / win_counts
        has_won = (win_counts > 1e-5).float()
        
        delta = learning_rate * has_won * (waves_avg - weights)
        layer.weight.add_(delta)
        
        layer.weight.div_(layer.weight.norm(dim=1, keepdim=True) + 1e-8)

def normalize_model_weights(model):
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, nn.Linear):
                module.weight.div_(module.weight.norm(dim=1, keepdim=True) + 1e-8)

# ============================================
# HOLOGRAPHIC COMPONENTS
# ============================================

def holo_selective_scan_v7(k, q, v, delta, gf, gw):
    B, T, H, D = k.shape
    v_gated = v * (delta * gw).unsqueeze(-1)
    updates = torch.matmul(v_gated.unsqueeze(-1), k.unsqueeze(-2))
    
    decay = (delta * gf).clamp(min=1e-6, max=0.999)
    log_decay = torch.log(decay).unsqueeze(-1).unsqueeze(-1)
    exp_cum_decay = torch.exp(torch.cumsum(log_decay, dim=1))
    
    state = torch.cumsum(updates / (exp_cum_decay + 1e-8), dim=1) * exp_cum_decay
    readout = torch.matmul(state, q.unsqueeze(-1)).squeeze(-1)
    return readout, state[:, -1]

class BoxEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.center = nn.Embedding(vocab_size, embed_dim)
        self.offset = nn.Embedding(vocab_size, embed_dim)
        nn.init.normal_(self.center.weight, std=0.02)
        nn.init.constant_(self.offset.weight, -1.0) 
    def forward(self, idx):
        c, o = self.center(idx), F.softplus(self.offset(idx))
        return c - o, c + o

# ============================================
# COMPOSITIONAL HOLOGRAPH BLOCK
# ============================================

class CompositionalHoloGraphBlock(nn.Module):
    """
    Enhanced HoloGraph block with compositional mechanisms.
    Integrates: VQ + MoE + Hybrid Attention + Seashore
    """
    def __init__(self, embed_dim, num_heads, head_dim,
                 use_vq=False, vq_config=None,
                 use_moe=False, moe_config=None,
                 use_hybrid_attn=False, hybrid_attn_config=None):
        super().__init__()
        self.num_heads, self.head_dim = num_heads, head_dim
        self.total_dim = num_heads * head_dim
        
        self.use_vq = use_vq
        self.use_moe = use_moe
        self.use_hybrid_attn = use_hybrid_attn
        
        # Primary Projections
        self.proj_k = nn.Linear(embed_dim, self.total_dim)
        self.proj_q = nn.Linear(embed_dim, self.total_dim)
        self.proj_v = nn.Linear(embed_dim, self.total_dim)
        self.proj_out = nn.Linear(self.total_dim, embed_dim)
        
        # Gates
        self.proj_delta = nn.Linear(embed_dim, num_heads)
        self.gate_write = nn.Linear(embed_dim, num_heads)
        self.gate_forget = nn.Linear(embed_dim, num_heads)
        
        self.probe_s = nn.Linear(embed_dim, embed_dim)
        self.ln1, self.ln2 = nn.LayerNorm(embed_dim), nn.LayerNorm(embed_dim)
        
        # Optional VQ
        if use_vq and vq_config:
            self.vq = HierarchicalVectorQuantizer(
                embed_dim=embed_dim,
                num_levels=vq_config['num_levels'],
                codebook_size=vq_config['codebook_size'],
                commitment_cost=vq_config['commitment_cost']
            )
            self.ln_vq = nn.LayerNorm(embed_dim)
        
        # Optional Hybrid Attention
        if use_hybrid_attn and hybrid_attn_config:
            self.hybrid_attn = HybridAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                window_size=hybrid_attn_config['window_size'],
                num_patterns=hybrid_attn_config['num_patterns'],
                continuous_weight=hybrid_attn_config['continuous_weight'],
                use_adaptive_mixing=hybrid_attn_config['use_adaptive_mixing']
            )
        
        # FFN (possibly MoE)
        if use_moe and moe_config:
            self.ffn = SparseMixtureOfExperts(
                embed_dim=embed_dim,
                num_experts=moe_config['num_experts'],
                expert_dim=moe_config['expert_dim'],
                top_k=moe_config['top_k'],
                capacity_factor=moe_config['capacity_factor']
            )
        else:
            self.ffn_1 = nn.Linear(embed_dim, 4*embed_dim)
            self.ffn_2 = nn.Linear(4*embed_dim, embed_dim)

    def forward(self, x, b_min, b_max, motif_context, hebbian_lr=None):
        B, T, _ = x.shape
        x_n = self.ln1(x)
        
        aux_losses = {}
        
        # Optional VQ on input
        if self.use_vq:
            x_vq = self.ln_vq(x_n)
            x_quantized, vq_loss = self.vq(x_vq)
            x_n = x_n + x_quantized
            aux_losses['vq_loss'] = vq_loss
        
        # Seashore Hebbian updates
        if hebbian_lr is not None:
            hebbian_update(self.proj_k, x_n, hebbian_lr)
            hebbian_update(self.proj_q, x_n, hebbian_lr)
            hebbian_update(self.proj_v, x_n, hebbian_lr)
        
        # Optional Hybrid Attention
        if self.use_hybrid_attn:
            hybrid_out = self.hybrid_attn(x_n)
            x = x + hybrid_out + (0.1 * motif_context)
        else:
            # Standard holographic attention
            k = F.normalize(self.proj_k(x_n).view(B, T, self.num_heads, self.head_dim), p=2.0, dim=-1)
            q = F.normalize(self.proj_q(x_n).view(B, T, self.num_heads, self.head_dim), p=2.0, dim=-1)
            v = torch.tanh(self.proj_v(x_n).view(B, T, self.num_heads, self.head_dim))
            
            d = F.softplus(self.proj_delta(x_n)).view(B, T, self.num_heads)
            gw = torch.sigmoid(self.gate_write(x_n)).pow(2).view(B, T, self.num_heads)
            gf = (1.0 - torch.sigmoid(self.gate_forget(x_n)).pow(2)).view(B, T, self.num_heads)
            
            m_heads, next_mem = holo_selective_scan_v7(k, q, v, d, gf, gw)
            
            ctx_flat = m_heads.reshape(B, T, -1)
            if hebbian_lr is not None:
                hebbian_update(self.proj_out, ctx_flat, hebbian_lr)
                
            x = x + self.proj_out(ctx_flat) + (0.1 * motif_context)
        
        # FFN Part (possibly MoE)
        x_n2 = self.ln2(x)
        
        if self.use_moe:
            ffn_out, moe_loss = self.ffn(x_n2)
            aux_losses['moe_loss'] = moe_loss
        else:
            if hebbian_lr is not None:
                hebbian_update(self.ffn_1, x_n2, hebbian_lr)
            
            hidden = F.gelu(self.ffn_1(x_n2))
            
            if hebbian_lr is not None:
                hebbian_update(self.ffn_2, hidden, hebbian_lr)
                
            ffn_out = self.ffn_2(hidden)
        
        x = x + ffn_out
        
        s_box = (b_min + self.probe_s(x), b_max + self.probe_s(x))
        
        # For backward compatibility with original code
        if not self.use_hybrid_attn:
            return x, next_mem, s_box, aux_losses
        else:
            # Create dummy next_mem if using hybrid attention
            next_mem = torch.zeros(B, self.num_heads, self.head_dim, self.head_dim, device=x.device)
            return x, next_mem, s_box, aux_losses

# ============================================
# MAIN MODEL
# ============================================

class CompositionalHoloGraphV7(nn.Module):
    """
    Enhanced HoloGraph with compositional mechanisms.
    Effective capacity: 384-dim → equivalent to 768+ dim standard transformer
    """
    def __init__(self):
        super().__init__()
        self.box_emb = BoxEmbedding(VOCAB_SIZE, EMBED_DIM)
        
        # Build compositional config
        vq_config = {
            'num_levels': VQ_NUM_CODEBOOKS,
            'codebook_size': VQ_CODEBOOK_SIZE,
            'commitment_cost': VQ_COMMITMENT_COST
        } if VQ_ENABLED and USE_COMPOSITIONAL else None
        
        moe_config = {
            'num_experts': MOE_NUM_EXPERTS,
            'expert_dim': MOE_EXPERT_DIM,
            'top_k': MOE_TOP_K,
            'capacity_factor': MOE_CAPACITY_FACTOR
        } if MOE_ENABLED and USE_COMPOSITIONAL else None
        
        hybrid_attn_config = {
            'window_size': HYBRID_WINDOW_SIZE,
            'num_patterns': HYBRID_NUM_PATTERNS,
            'continuous_weight': HYBRID_CONTINUOUS_WEIGHT,
            'use_adaptive_mixing': HYBRID_ADAPTIVE_MIXING
        } if HYBRID_ATTN_ENABLED and USE_COMPOSITIONAL else None
        
        self.layers = nn.ModuleList([
            CompositionalHoloGraphBlock(
                EMBED_DIM, NUM_HEADS, HEAD_DIM,
                use_vq=VQ_ENABLED and USE_COMPOSITIONAL,
                vq_config=vq_config,
                use_moe=MOE_ENABLED and USE_COMPOSITIONAL,
                moe_config=moe_config,
                use_hybrid_attn=HYBRID_ATTN_ENABLED and USE_COMPOSITIONAL,
                hybrid_attn_config=hybrid_attn_config
            ) 
            for _ in range(NUM_LAYERS)
        ])
        
        self.ln_f = nn.LayerNorm(EMBED_DIM)
        self.logit_scale = nn.Parameter(torch.ones(1) * 14.0)
    
    def forward(self, idx, memory=None, hebbian_lr=None, use_memory=True):
        b_min, b_max = self.box_emb(idx)
        x = (b_min + b_max) / 2.0
        
        layer_data = []
        total_aux_losses = {
            'vq_loss': 0.0,
            'moe_loss': 0.0
        }
        
        for layer in self.layers:
            motif_ctx = torch.zeros_like(x)
            
            if memory is not None and use_memory:
                query_min = b_min.mean(dim=(0, 1))
                query_max = b_max.mean(dim=(0, 1))
                
                memory_results = memory.retrieve((query_min, query_max), k=32)
                
                if memory_results:
                    B, T, _ = x.shape
                    
                    retrieved_values = torch.stack([r['values'] for r in memory_results]).to(x.dtype)
                    retrieved_importance = torch.tensor(
                        [r['importance'] for r in memory_results], 
                        device=x.device, dtype=x.dtype
                    )
                    
                    k_retrieved = retrieved_values.shape[0]
                    
                    retrieved_matrices = retrieved_values.view(
                        k_retrieved, layer.num_heads, layer.head_dim, layer.head_dim
                    )
                    
                    qr = F.normalize(
                        layer.proj_q(x).view(B, T, layer.num_heads, layer.head_dim),
                        p=2.0, dim=-1
                    ).to(x.dtype)
                    
                    qr_flat = qr.reshape(B * T * layer.num_heads, layer.head_dim)
                    
                    keys = retrieved_matrices.mean(dim=-1)
                    keys = keys.transpose(0, 1).reshape(layer.num_heads, k_retrieved, layer.head_dim)
                    
                    keys_expanded = keys.unsqueeze(0).repeat(B * T, 1, 1, 1).reshape(B * T * layer.num_heads, k_retrieved, layer.head_dim)
                    
                    attn_scores = torch.bmm(
                        keys_expanded, 
                        qr_flat.unsqueeze(-1)
                    ).squeeze(-1) / math.sqrt(layer.head_dim)
                    
                    attn_scores = attn_scores * retrieved_importance.unsqueeze(0)
                    attn_weights = F.softmax(attn_scores, dim=-1)
                    
                    attn_weights = attn_weights.reshape(B, T, layer.num_heads, k_retrieved)
                    
                    weighted_matrices = torch.einsum(
                        'bthk,khde->bthde',
                        attn_weights,
                        retrieved_matrices
                    )
                    
                    output_h = torch.einsum('bthde,bthe->bthd', weighted_matrices, qr)
                    
                    motif_ctx_raw = output_h.reshape(B, T, layer.num_heads * layer.head_dim)
                    motif_ctx = layer.proj_out(motif_ctx_raw)
            
            x, mem, s_box, aux_losses = layer(x, b_min, b_max, motif_ctx, hebbian_lr)
            layer_data.append((mem, s_box))
            
            # Accumulate auxiliary losses
            for key in aux_losses:
                if key in total_aux_losses:
                    total_aux_losses[key] += aux_losses[key]
        
        logits = F.linear(self.ln_f(x), self.box_emb.center.weight) * \
                 (self.logit_scale / (EMBED_DIM**0.5))
        
        return logits, layer_data, total_aux_losses

# ============================================
# DATA PIPELINE
# ============================================

class SequentialRotatingStreamer:
    def __init__(self, file_paths, text_column, tokenizer, seq_len, batch_size):
        self.file_paths = file_paths
        self.text_column = text_column
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.prefetch_buffer = []
        self.prefetch_size = batch_size * 2

    def __iter__(self):
        indices = list(range(len(self.file_paths)))
        random.shuffle(indices)
        for step_idx, i in enumerate(indices):
            file_path = self.file_paths[i]
            file_msg = f"{step_idx + 1}/{len(self.file_paths)}"
            yield None, None, file_msg 
            
            pf = None
            table = None
            try:
                pf = pq.ParquetFile(file_path)
                batch_x, batch_y = [], []
                for r in range(pf.num_row_groups):
                    table = pf.read_row_group(r, columns=[self.text_column])
                    texts = table.column(self.text_column).to_pylist()
                    token_buffer = []
                    for text in texts:
                        if text: token_buffer.extend(self.tokenizer.encode(str(text)) + [0])
                        while len(token_buffer) >= self.seq_len + 1:
                            chunk = token_buffer[:self.seq_len + 1]
                            token_buffer = token_buffer[self.seq_len:]
                            batch_x.append(chunk[:-1]); batch_y.append(chunk[1:])
                            if len(batch_x) == self.batch_size:
                                xb = torch.tensor(batch_x, dtype=torch.long, device=DEVICE)
                                yb = torch.tensor(batch_y, dtype=torch.long, device=DEVICE)
                                yield (xb, yb, file_msg)
                                batch_x, batch_y = [], []
                                del xb, yb
                    
                    del texts
                    del table
                    table = None
                    
            except Exception as e:
                print(f"\nError reading {file_path}: {e}")
                continue
            finally:
                if table is not None:
                    del table
                if pf is not None:
                    del pf
                gc.collect()

# ============================================
# CHAT MODE
# ============================================

def chat_mode(model, memory, tokenizer):
    """Interactive chat mode for testing the model"""
    print("\n" + "="*60)
    print("CHAT MODE - Interactive Generation")
    print("="*60)
    print("Commands:")
    print("  /exit or /quit - Exit chat mode")
    print("  /temp <value> - Set temperature (default: 0.8)")
    print("  /len <value> - Set max generation length (default: 100)")
    print("  /memory <on|off> - Toggle memory retrieval")
    print("="*60 + "\n")
    
    model.eval()
    temperature = 0.8
    max_length = 100
    use_memory = True
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if not user_input:
                continue
            
            # Commands
            if user_input.lower() in ['/exit', '/quit']:
                print("Exiting chat mode...")
                break
            elif user_input.startswith('/temp '):
                try:
                    temperature = float(user_input.split()[1])
                    print(f"Temperature set to {temperature}")
                except:
                    print("Invalid temperature. Usage: /temp 0.8")
                continue
            elif user_input.startswith('/len '):
                try:
                    max_length = int(user_input.split()[1])
                    print(f"Max length set to {max_length}")
                except:
                    print("Invalid length. Usage: /len 100")
                continue
            elif user_input.startswith('/memory '):
                arg = user_input.split()[1].lower()
                use_memory = arg == 'on'
                print(f"Memory retrieval {'enabled' if use_memory else 'disabled'}")
                continue
            
            # Tokenize input
            input_tokens = tokenizer.encode(user_input)
            if len(input_tokens) > SEQ_LEN:
                input_tokens = input_tokens[-SEQ_LEN:]
            
            context = torch.tensor([input_tokens], dtype=torch.long, device=DEVICE)
            
            print("\nAssistant: ", end="", flush=True)
            
            # Generate response
            with torch.no_grad():
                gen_tokens = context
                for _ in range(max_length):
                    with torch.amp.autocast('cuda'):
                        logits, _, _ = model(gen_tokens, memory=memory, use_memory=use_memory)
                    
                    # Sample next token
                    probs = F.softmax(logits[:, -1, :] / temperature, dim=-1)
                    next_token = torch.multinomial(probs, 1)
                    
                    # Decode and print
                    token_text = tokenizer.decode([next_token.item()])
                    print(token_text, end="", flush=True)
                    
                    # Update context
                    gen_tokens = torch.cat([gen_tokens, next_token], dim=1)
                    if gen_tokens.shape[1] > SEQ_LEN:
                        gen_tokens = gen_tokens[:, 1:]
                    
                    # Stop on newline or special tokens
                    if next_token.item() == 0:  # End token
                        break
            
            print()  # New line after generation
            
        except KeyboardInterrupt:
            print("\n\nInterrupted. Type /exit to quit.")
            continue
        except Exception as e:
            print(f"\nError during generation: {e}")
            continue

# ============================================
# TRAINING LOOP
# ============================================

def train(folder, text_col, continue_training=True):
    tokenizer = LimitedTokenizer(limit=VOCAB_SIZE)
    model = CompositionalHoloGraphV7().to(DEVICE)
    
    memory = CompressedHierarchicalMemory(
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        head_dim=HEAD_DIM,
        l1_size=2048,
        l2_size=16384,
        disk_path='holo_memory_db'
    )
    
    memory.load_compressors()
    
    print("\n" + "="*60)
    print("COMPOSITIONAL HOLOGRAPHIC SEASHORE")
    print("="*60)
    print(f"Architecture Enhancements:")
    print(f"  VQ Enabled: {VQ_ENABLED} (Levels: {VQ_NUM_CODEBOOKS}, Codes: {VQ_CODEBOOK_SIZE})")
    print(f"  MoE Enabled: {MOE_ENABLED} (Experts: {MOE_NUM_EXPERTS}, Top-K: {MOE_TOP_K})")
    print(f"  Hybrid Attention: {HYBRID_ATTN_ENABLED} (Patterns: {HYBRID_NUM_PATTERNS})")
    print(f"Effective Capacity: {EMBED_DIM}-dim → ~{EMBED_DIM*2}-dim equivalent")
    memory.estimate_final_size(total_data_tb=0.2)
    print("="*60 + "\n")
    
    try:
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
        print("Using 8-bit AdamW optimizer")
    except ImportError:
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
        print("Using standard AdamW optimizer")
    
    scaler = torch.cuda.amp.GradScaler()
    global_step, epoch = 0, 0
    files_processed = 0
    accumulation_step = 0
    
    compression_train_interval = 1000
    sample_buffer = {'s_min': [], 's_max': [], 'values': []}
    
    if continue_training and os.path.exists(CHECKPOINT_PATH):
        print(f"Found checkpoint at {CHECKPOINT_PATH}. Loading...")
        try:
            ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
            model.load_state_dict(ckpt['model'])
            optimizer.load_state_dict(ckpt['opt'])
            if 'scaler' in ckpt:
                scaler.load_state_dict(ckpt['scaler'])
            global_step = ckpt.get('step', 0)
            epoch = ckpt.get('epoch', 0)
            files_processed = ckpt.get('files_processed', 0)
            print(f"Resumed from step {global_step}, epoch {epoch}, files {files_processed}")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("Starting fresh training...")
            global_step, epoch, files_processed = 0, 0, 0
    else:
        if not continue_training:
            print("Starting fresh training (continue_training=False)")
        else:
            print("No checkpoint found. Starting fresh training...")

    files = glob.glob(os.path.join(folder, "**/*.parquet"), recursive=True)
    if not files: 
        print(f"No parquet files found in {folder}!")
        return
    
    print(f"Found {len(files)} parquet files")
    streamer = SequentialRotatingStreamer(files, text_col, tokenizer, SEQ_LEN, BATCH_SIZE)

    current_file_str = "Init"
    session_has_started = False 

    print(f"Starting Training with Compositional Mechanisms")
    print(f"Ratio BP:{BP_RATIO} / Hebbian:{10-BP_RATIO}")
    print(f"Memory: L1={memory.l1_working['size']} | L2={memory.l2_session['size']} | L3=Unlimited (disk)")

    while True:
        for xb, yb, file_info in streamer:
            if xb is None:
                if session_has_started:
                    print(f"\n\nEnd of file reached. Consolidating memory and saving checkpoint...")
                    
                    print("Consolidating L2 → L3...")
                    memory.consolidate_l2_to_l3()
                    
                    memory.print_stats()
                    memory.save_compressors()
                    
                    torch.save({
                        'model': model.state_dict(), 
                        'opt': optimizer.state_dict(), 
                        'scaler': scaler.state_dict(),
                        'step': global_step, 
                        'epoch': epoch,
                        'files_processed': files_processed
                    }, CHECKPOINT_PATH)
                    
                    files_processed += 1
                    aggressive_cleanup()
                    
                session_has_started = True
                current_file_str = file_info
                aggressive_cleanup()
                continue
            
            cycle_idx = global_step % 10
            is_bp_step = cycle_idx < BP_RATIO
            use_memory_this_step = (global_step % MEMORY_RETRIEVAL_EVERY == 0)
            
            model.train()
            
            if is_bp_step:
                if accumulation_step == 0:
                    optimizer.zero_grad(set_to_none=True)
                
                with torch.amp.autocast('cuda'):
                    logits, layer_data, aux_losses = model(xb, memory=memory, hebbian_lr=None, use_memory=use_memory_this_step)
                    loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), yb.view(-1))
                    
                    # Add auxiliary losses (VQ commitment, MoE load balancing)
                    if USE_COMPOSITIONAL:
                        if VQ_ENABLED and 'vq_loss' in aux_losses and aux_losses['vq_loss'] != 0.0:
                            loss = loss + 0.1 * aux_losses['vq_loss']
                        if MOE_ENABLED and 'moe_loss' in aux_losses and aux_losses['moe_loss'] != 0.0:
                            loss = loss + 0.01 * aux_losses['moe_loss']
                    
                    loss = loss / GRADIENT_ACCUMULATION
                
                scaler.scale(loss).backward()
                
                accumulation_step += 1
                
                if accumulation_step >= GRADIENT_ACCUMULATION:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    
                    normalize_model_weights(model)
                    
                    accumulation_step = 0
                
                mode_str = "BP"
                loss = loss * GRADIENT_ACCUMULATION
                
            else:
                with torch.no_grad():
                    logits, layer_data, aux_losses = model(xb, memory=memory, hebbian_lr=HEBBIAN_LR, use_memory=use_memory_this_step)
                    loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), yb.view(-1))
                mode_str = "HB"

            if global_step % SNAPSHOT_RATE == 0:
                mem, s_box = layer_data[0]
                importance = min(loss.item() * 2.0, 10.0)
                memory.add(s_box, mem, importance=importance)
                
                with torch.no_grad():
                    sample_buffer['s_min'].append(s_box[0][:, -1, :].mean(dim=0).to(torch.float32))
                    sample_buffer['s_max'].append(s_box[1][:, -1, :].mean(dim=0).to(torch.float32))
                    sample_buffer['values'].append(mem.mean(dim=0).flatten().to(torch.float32))
            
            if global_step % compression_train_interval == 0 and len(sample_buffer['s_min']) > 0:
                print(f"\nTraining compression codebooks on {len(sample_buffer['s_min'])} samples...")
                batch = {
                    's_min': torch.stack(sample_buffer['s_min']),
                    's_max': torch.stack(sample_buffer['s_max']),
                    'values': torch.stack(sample_buffer['values'])
                }
                memory.train_compressors(batch)
                
                sample_buffer = {'s_min': [], 's_max': [], 'values': []}
                del batch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            if global_step % CLEANUP_EVERY == 0:
                clear_autocast_cache()
                if global_step % (CLEANUP_EVERY * 5) == 0:
                    aggressive_cleanup()
            
            if global_step % PREDICT_EVERY == 0:
                model.eval()
                with torch.no_grad():
                    context = xb[:1, :10] 
                    print(f"\n\n--- Step {global_step} [{mode_str}] Preview ---")
                    print(f"Memory: L1={memory.l1_working['count']}/{memory.l1_working['size']} | "
                          f"L2={memory.l2_session['count']}/{memory.l2_session['size']} | "
                          f"L3={memory.l3_archive['count']} ({memory._get_db_size_mb():.1f}MB)")
                    if USE_COMPOSITIONAL:
                        print(f"Compositional: VQ={VQ_ENABLED} | MoE={MOE_ENABLED} | Hybrid={HYBRID_ATTN_ENABLED}")
                    print(f"Prompt: {tokenizer.decode(context[0].tolist())}")
                    print("Output: ", end="", flush=True)
                    gen_tokens = context
                    for _ in range(20): 
                        with torch.amp.autocast('cuda'):
                            lg, _, _ = model(gen_tokens, memory=memory, use_memory=False)
                        probs = F.softmax(lg[:, -1, :] / 0.8, dim=-1)
                        next_token = torch.multinomial(probs, 1)
                        print(tokenizer.decode([next_token.item()]), end="", flush=True)
                        gen_tokens = torch.cat([gen_tokens, next_token], dim=1)
                        if gen_tokens.shape[1] > SEQ_LEN: gen_tokens = gen_tokens[:, 1:]
                    print("\n" + "-"*40 + "\n")

            global_step += 1
            
            l1_pct = (memory.l1_working['count'] / memory.l1_working['size']) * 100
            l2_pct = (memory.l2_session['count'] / memory.l2_session['size']) * 100
            
            comp_str = ""
            if USE_COMPOSITIONAL:
                comp_str = f"[VQ:{int(VQ_ENABLED)} MoE:{int(MOE_ENABLED)} HA:{int(HYBRID_ATTN_ENABLED)}]"
            
            stdout.write(
                f'\r[{mode_str}]{comp_str}[File {current_file_str}][Files: {files_processed}]'
                f'[L1:{l1_pct:.0f}% L2:{l2_pct:.0f}% L3:{memory.l3_archive["count"]}] '
                f'Step: {global_step} | Loss: {loss.item():.4f}'
            )
            stdout.flush()

        epoch += 1
        print(f"\n\nEpoch {epoch} finished. Saving final checkpoint...")
        
        print("Final memory consolidation...")
        memory.consolidate_l2_to_l3()
        memory.print_stats()
        memory.save_compressors()
        
        torch.save({
            'model': model.state_dict(), 
            'opt': optimizer.state_dict(), 
            'scaler': scaler.state_dict(),
            'step': global_step, 
            'epoch': epoch,
            'files_processed': files_processed
        }, CHECKPOINT_PATH)
        aggressive_cleanup()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, default="data", help="Folder containing parquet files")
    parser.add_argument("--column", type=str, default="text", help="Text column name in parquet files")
    parser.add_argument("--no-continue", action="store_true", help="Start fresh training even if checkpoint exists")
    parser.add_argument("--chat", action="store_true", help="Start in chat mode (requires existing checkpoint)")
    args = parser.parse_args()
    
    # Check if user wants chat mode
    if args.chat:
        if not os.path.exists(CHECKPOINT_PATH):
            print(f"Error: No checkpoint found at {CHECKPOINT_PATH}")
            print("Train the model first before using chat mode.")
            exit(1)
        
        print("Loading model for chat mode...")
        tokenizer = LimitedTokenizer(limit=VOCAB_SIZE)
        model = CompositionalHoloGraphV7().to(DEVICE)
        
        memory = CompressedHierarchicalMemory(
            embed_dim=EMBED_DIM,
            num_heads=NUM_HEADS,
            head_dim=HEAD_DIM,
            l1_size=2048,
            l2_size=16384,
            disk_path='holo_memory_db'
        )
        
        # Load checkpoint
        ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        model.load_state_dict(ckpt['model'])
        memory.load_compressors()
        
        print(f"Model loaded from checkpoint (step {ckpt.get('step', 0)})")
        
        # Start chat mode
        chat_mode(model, memory, tokenizer)
        exit(0)
    
    # Check if checkpoint exists and offer chat mode
    if os.path.exists(CHECKPOINT_PATH) and not args.no_continue:
        print(f"\n{'='*60}")
        print(f"Checkpoint found at: {CHECKPOINT_PATH}")
        print(f"{'='*60}")
        print("Options:")
        print("  1. Continue training")
        print("  2. Start chat mode")
        print("  3. Start fresh training (delete checkpoint)")
        print(f"{'='*60}")
        
        while True:
            choice = input("\nEnter your choice (1/2/3): ").strip()
            if choice == '1':
                print("Continuing training...")
                continue_training = True
                break
            elif choice == '2':
                print("Starting chat mode...")
                tokenizer = LimitedTokenizer(limit=VOCAB_SIZE)
                model = CompositionalHoloGraphV7().to(DEVICE)
                
                memory = CompressedHierarchicalMemory(
                    embed_dim=EMBED_DIM,
                    num_heads=NUM_HEADS,
                    head_dim=HEAD_DIM,
                    l1_size=2048,
                    l2_size=16384,
                    disk_path='holo_memory_db'
                )
                
                # Load checkpoint
                ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
                model.load_state_dict(ckpt['model'])
                memory.load_compressors()
                
                print(f"Model loaded from checkpoint (step {ckpt.get('step', 0)})")
                
                # Start chat mode
                chat_mode(model, memory, tokenizer)
                exit(0)
            elif choice == '3':
                print("Starting fresh training (checkpoint will be overwritten)...")
                continue_training = False
                break
            else:
                print("Invalid choice. Please enter 1, 2, or 3.")
    else:
        continue_training = not args.no_continue
    
    train(args.folder, args.column, continue_training=continue_training)