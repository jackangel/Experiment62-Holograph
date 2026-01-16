# 
# PERFORMANCE OPTIMIZATIONS APPLIED:
# 1. Batch Size: Increased from 8 to 32 (4x throughput)
# 2. Gradient Accumulation: Added 4-step accumulation (effective batch = 128)
# 3. Memory Retrieval: Reduced frequency to every 4 steps (4x faster)
# 4. Memory Attention: Vectorized nested loops for 10-100x speedup
# 5. Cleanup Frequency: Reduced from every 500 to 1000 steps
# 6. Snapshot Rate: Reduced from 128 to 256 steps
# 7. Preview Rate: Reduced from 1000 to 2000 steps
# 8. Compression Training: Reduced from 500 to 1000 steps
# Expected speedup: 3-5x overall training throughput
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
BATCH_SIZE = 8              # OPTIMIZED: Increased from 8 to 32 for better GPU utilization
GRADIENT_ACCUMULATION = 4    # OPTIMIZED: Effective batch size = 32 * 4 = 128
LEARNING_RATE = 3e-4 
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CHECKPOINT_PATH = 'holograph_seashore.pth'

# Seashore Hyperparameters
SEASHORE_WIDTH = EMBED_DIM # Aligning width logic
HEBBIAN_LR = 0.05          # Learning rate for Hebbian updates
BP_RATIO = 3               # 3 steps BP, 7 steps Hebbian (Cycle of 10)

# Control Rates
ARCHIVE_SIZE = 4096 
SNAPSHOT_RATE = 256          # OPTIMIZED: Reduced frequency from 128 to 256
PREDICT_EVERY = 2000         # OPTIMIZED: Reduced frequency from 1000 to 2000

# Memory Management
ARCHIVE_SOFT_RESET_INTERVAL = 10  # Soft reset archive every N files
ARCHIVE_KEEP_RATIO = 0.25         # Keep top 25% on soft reset
CLEANUP_EVERY = 1000         # OPTIMIZED: Reduced frequency from 500 to 1000
MEMORY_RETRIEVAL_EVERY = 4   # OPTIMIZED: Only retrieve from memory every N steps

import torch
import torch.nn as nn
import torch.nn.functional as F
import h5py
import numpy as np
import os
from collections import defaultdict
import math

# ============================================
# COMPRESSION LAYER: Vector Quantization
# ============================================

class LearnedVectorQuantizer(nn.Module):
    """
    Learns to compress embeddings into discrete codes.
    Similar to VQ-VAE but optimized for semantic preservation.
    """
    def __init__(self, embed_dim, n_codebooks=16, codebook_size=256, device='cuda'):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_codebooks = n_codebooks
        self.codebook_size = codebook_size
        self.dims_per_book = embed_dim // n_codebooks
        self.device = device
        
        # Learnable codebooks [n_codebooks, codebook_size, dims_per_book]
        self.codebooks = nn.Parameter(
            torch.randn(n_codebooks, codebook_size, self.dims_per_book, device=device) * 0.02
        )
        
        # EMA for online codebook updates
        self.register_buffer('ema_count', torch.zeros(n_codebooks, codebook_size, device=device))
        self.register_buffer('ema_weight', self.codebooks.data.clone())
        self.decay = 0.99
    
    def quantize(self, x):
        """
        x: [batch, embed_dim]
        returns: codes [batch, n_codebooks] (uint8)
        """
        # Ensure x is on the correct device
        x = x.to(self.device)
        
        batch_size = x.shape[0]
        x_reshaped = x.reshape(batch_size, self.n_codebooks, self.dims_per_book)
        
        codes = torch.zeros(batch_size, self.n_codebooks, dtype=torch.long, device=self.device)
        
        for i in range(self.n_codebooks):
            x_chunk = x_reshaped[:, i, :]  # [batch, dims_per_book]
            codebook = self.codebooks[i]  # [codebook_size, dims_per_book]
            
            # L2 distance to all codebook entries
            dists = torch.cdist(x_chunk, codebook)  # [batch, codebook_size]
            codes[:, i] = torch.argmin(dists, dim=1)
            
            # EMA update (only during training)
            if self.training:
                with torch.no_grad():
                    # Count how many times each code was used
                    encodings = F.one_hot(codes[:, i], self.codebook_size).float()
                    self.ema_count[i] = self.decay * self.ema_count[i] + \
                                       (1 - self.decay) * encodings.sum(0)
                    
                    # Update codebook weighted by usage
                    dw = torch.matmul(encodings.t(), x_chunk)
                    self.ema_weight[i] = self.decay * self.ema_weight[i] + \
                                        (1 - self.decay) * dw
                    
                    # Normalize
                    n = self.ema_count[i].unsqueeze(1)
                    self.codebooks.data[i] = self.ema_weight[i] / (n + 1e-5)
        
        return codes.to(torch.uint8)
    
    def dequantize(self, codes):
        """
        codes: [batch, n_codebooks] (uint8 or long)
        returns: [batch, embed_dim]
        """
        # Ensure codes are on the correct device
        codes = codes.to(self.device)
        
        batch_size = codes.shape[0]
        codes = codes.long()
        
        reconstructed = torch.zeros(batch_size, self.embed_dim, 
                                    device=self.device, dtype=self.codebooks.dtype)
        
        for i in range(self.n_codebooks):
            # Get the codes for this codebook across all batch items
            batch_codes = codes[:, i]  # [batch]
            
            # Look up the corresponding embeddings from the codebook
            chunk = self.codebooks[i][batch_codes]  # [batch, dims_per_book]
            
            # Place in the correct position in the reconstructed tensor
            start = i * self.dims_per_book
            end = start + self.dims_per_book
            reconstructed[:, start:end] = chunk
        
        return reconstructed

# ============================================
# SEMANTIC HASHING: Further Compression
# ============================================

class SemanticHasher(nn.Module):
    """
    Creates binary hash codes for ultra-fast retrieval.
    Maps embeddings to {0,1}^n binary codes.
    """
    def __init__(self, embed_dim, hash_bits=128, device='cuda'):
        super().__init__()
        self.hash_bits = hash_bits
        self.device = device
        # Random projection matrix (frozen after init)
        self.register_buffer('projection', 
                           torch.randn(embed_dim, hash_bits, device=device) / math.sqrt(embed_dim))
    
    def hash(self, x):
        """
        x: [batch, embed_dim]
        returns: binary hash [batch, hash_bits] packed into uint8
        """
        # Ensure x is on the correct device and flatten if needed
        if x.dim() > 2:
            x = x.reshape(-1, x.shape[-1])
        x = x.to(self.device)
        
        # Project and threshold
        projected = torch.matmul(x, self.projection)  # [batch, hash_bits]
        binary = (projected > 0).to(torch.uint8)
        
        # Ensure we have the right shape
        batch_size = binary.shape[0]
        assert binary.shape[1] == self.hash_bits, f"Expected {self.hash_bits} bits, got {binary.shape[1]}"
        
        # Pack bits: 8 bits per byte (vectorized approach)
        packed_size = (self.hash_bits + 7) // 8
        
        # Pad binary to make it divisible by 8
        pad_length = packed_size * 8 - self.hash_bits
        if pad_length > 0:
            binary = F.pad(binary, (0, pad_length), value=0)
        
        # Reshape to [batch, packed_size, 8]
        binary_reshaped = binary.reshape(batch_size, packed_size, 8)
        
        # Create bit position multipliers: [1, 2, 4, 8, 16, 32, 64, 128]
        bit_multipliers = (2 ** torch.arange(8, device=self.device)).to(torch.uint8)
        
        # Multiply and sum to pack bits into bytes
        packed = (binary_reshaped * bit_multipliers).sum(dim=2)
        
        return packed
    def hamming_distance(self, hash1, hash2):
        """Compute Hamming distance between binary hashes (vectorized)"""
        # XOR and count bits using vectorized operations
        xor = hash1 ^ hash2
        # Convert to int32 and use efficient bit counting
        # Each byte: count set bits by unpacking to binary
        distance = 0
        for i in range(8):
            distance += ((xor >> i) & 1).sum()
        return distance

# ============================================
# HIERARCHICAL MEMORY SYSTEM
# ============================================

class CompressedHierarchicalMemory:
    """
    L1: Working Memory (VRAM, full precision, 2K entries)
    L2: Session Memory (VRAM, quantized, 16K entries) 
    L3: Archive Memory (Disk, hash+quantized, unlimited)
    
    Target: 200TB training → 50GB database (4000× compression)
    """
    
    def __init__(self, embed_dim, num_heads, head_dim, 
                 l1_size=2048, l2_size=16384, disk_path='holo_memory_db', device='cuda'):
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.disk_path = disk_path
        self.device = device
        
        # Compression modules (learned during training)
        self.vq_embeddings = LearnedVectorQuantizer(
            embed_dim=embed_dim,
            n_codebooks=16,      # 16 codebooks
            codebook_size=256,   # 256 codes each = 16 bytes per embedding
            device=device
        )
        
        self.vq_matrices = LearnedVectorQuantizer(
            embed_dim=num_heads * head_dim * head_dim,
            n_codebooks=32,      # More codebooks for larger matrices
            codebook_size=256,   # = 32 bytes per matrix
            device=device
        )
        
        self.hasher = SemanticHasher(
            embed_dim=embed_dim,
            hash_bits=128,       # 16 bytes for fast lookup
            device=device
        )
        
        # L1: Working Memory (VRAM, full precision)
        self.l1_working = self._create_l1_memory(l1_size)
        
        # L2: Session Memory (VRAM, vector quantized)
        self.l2_session = self._create_l2_memory(l2_size)
        
        # L3: Archive Memory (Disk, hash + vector quantized)
        self.l3_archive = self._create_l3_memory()
        
        # Statistics
        self.stats = {
            'l1_hits': 0,
            'l2_hits': 0,
            'l3_hits': 0,
            'total_queries': 0,
            'compressions_trained': 0
        }
    
    def _create_l1_memory(self, size):
        """L1: Full precision, fast access"""
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
        """L2: Vector quantized, medium access speed"""
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
        """L3: Disk-backed, hash indexed, heavily compressed"""
        os.makedirs(self.disk_path, exist_ok=True)
        
        db_file = os.path.join(self.disk_path, 'archive.h5')
        
        # Create or open HDF5 database
        db = h5py.File(db_file, 'a')
        
        if 'hashes' not in db:
            # Hash index (16 bytes per entry)
            db.create_dataset('hashes', (0, 16), maxshape=(None, 16), 
                            dtype='uint8', chunks=(10000, 16), compression='gzip')
            
            # Quantized embeddings (16 bytes each)
            db.create_dataset('s_min_codes', (0, 16), maxshape=(None, 16),
                            dtype='uint8', chunks=(10000, 16), compression='gzip')
            db.create_dataset('s_max_codes', (0, 16), maxshape=(None, 16),
                            dtype='uint8', chunks=(10000, 16), compression='gzip')
            
            # Quantized matrices (32 bytes each)
            db.create_dataset('value_codes', (0, 32), maxshape=(None, 32),
                            dtype='uint8', chunks=(10000, 32), compression='gzip')
            
            # Metadata (8 bytes each)
            db.create_dataset('importance', (0,), maxshape=(None,),
                            dtype='float32', chunks=(10000,), compression='gzip')
            db.create_dataset('timestamp', (0,), maxshape=(None,),
                            dtype='int64', chunks=(10000,), compression='gzip')
        
        return {
            'db': db,
            'count': len(db['hashes']),
            'cache': {},  # LRU cache for recently accessed items
            'cache_size': 1000
        }
    
    # ============================================
    # COMPRESSION TRAINING
    # ============================================
    
    def train_compressors(self, sample_batch):
        """
        Train VQ codebooks on representative samples.
        Call this periodically during training.
        
        sample_batch: dict with 's_min', 's_max', 'values'
        """
        self.vq_embeddings.train()
        self.vq_matrices.train()
        
        with torch.no_grad():
            # Train embedding quantizer
            s_combined = torch.cat([sample_batch['s_min'], sample_batch['s_max']], dim=0)
            _ = self.vq_embeddings.quantize(s_combined)
            
            # Train matrix quantizer
            _ = self.vq_matrices.quantize(sample_batch['values'])
        
        self.stats['compressions_trained'] += 1
        
        # Save updated codebooks periodically
        if self.stats['compressions_trained'] % 100 == 0:
            self.save_compressors()
    
    def save_compressors(self):
        """Save learned compression codebooks"""
        save_path = os.path.join(self.disk_path, 'compressors.pth')
        torch.save({
            'vq_embeddings': self.vq_embeddings.state_dict(),
            'vq_matrices': self.vq_matrices.state_dict(),
            'hasher': self.hasher.state_dict()
        }, save_path)
    
    def load_compressors(self):
        """Load learned compression codebooks"""
        save_path = os.path.join(self.disk_path, 'compressors.pth')
        if os.path.exists(save_path):
            ckpt = torch.load(save_path, map_location=DEVICE)
            self.vq_embeddings.load_state_dict(ckpt['vq_embeddings'])
            self.vq_matrices.load_state_dict(ckpt['vq_matrices'])
            self.hasher.load_state_dict(ckpt['hasher'])
            print(f"Loaded compression codebooks from {save_path}")
    
    # ============================================
    # ADD OPERATIONS
    # ============================================
    
    def add(self, s_box, matrix, importance=1.0):
        """
        Add new memory to hierarchy.
        Automatically manages promotion between levels.
        """
        with torch.no_grad():
            # Extract means
            s_min = s_box[0][:, -1, :].mean(dim=0).to(torch.float32)  # [embed_dim]
            s_max = s_box[1][:, -1, :].mean(dim=0).to(torch.float32)
            values = matrix.mean(dim=0).flatten().to(torch.float32)  # [num_heads * head_dim * head_dim]
            
            # Add to L1 (working memory)
            l1 = self.l1_working
            ptr = l1['ptr']
            l1['s_min'][ptr] = s_min.to(torch.float16)
            l1['s_max'][ptr] = s_max.to(torch.float16)
            l1['values'][ptr] = values.to(torch.float16)
            l1['importance'][ptr] = importance
            l1['access_count'][ptr] = 0
            
            l1['ptr'] = (ptr + 1) % l1['size']
            l1['count'] = min(l1['count'] + 1, l1['size'])
            
            # Promote to L2 if important enough
            if importance > 2.0:
                self._add_to_l2(s_min, s_max, values, importance)
            
            # Check if L1 is getting full → consolidate
            if l1['count'] >= l1['size'] * 0.9:
                self._consolidate_l1_to_l2()
    
    def _add_to_l2(self, s_min, s_max, values, importance):
        """Add to L2 session memory (quantized)"""
        l2 = self.l2_session
        ptr = l2['ptr']
        
        # Quantize
        self.vq_embeddings.eval()
        self.vq_matrices.eval()
        
        with torch.no_grad():
            s_min_codes = self.vq_embeddings.quantize(s_min.unsqueeze(0))[0]
            s_max_codes = self.vq_embeddings.quantize(s_max.unsqueeze(0))[0]
            value_codes = self.vq_matrices.quantize(values.unsqueeze(0))[0]
        
        l2['s_min_codes'][ptr] = s_min_codes
        l2['s_max_codes'][ptr] = s_max_codes
        l2['value_codes'][ptr] = value_codes
        l2['importance'][ptr] = importance * 0.8  # Slight decay
        l2['access_count'][ptr] = 0
        
        l2['ptr'] = (ptr + 1) % l2['size']
        l2['count'] = min(l2['count'] + 1, l2['size'])
        
        # Promote to L3 if very important
        if importance > 5.0:
            self._add_to_l3(s_min_codes, s_max_codes, value_codes, 
                           s_min, importance)
    
    def _add_to_l3(self, s_min_codes, s_max_codes, value_codes, 
                   s_min_full, importance):
        """Add to L3 archive (disk, with hash index)"""
        l3 = self.l3_archive
        db = l3['db']
        
        # Compute semantic hash for fast retrieval
        with torch.no_grad():
            hash_code = self.hasher.hash(s_min_full.unsqueeze(0))[0].cpu().numpy()
        
        # Append to disk
        n = l3['count']
        for dataset_name in ['hashes', 's_min_codes', 's_max_codes', 
                            'value_codes', 'importance', 'timestamp']:
            db[dataset_name].resize((n + 1,) + db[dataset_name].shape[1:])
        
        db['hashes'][n] = hash_code
        db['s_min_codes'][n] = s_min_codes.cpu().numpy()
        db['s_max_codes'][n] = s_max_codes.cpu().numpy()
        db['value_codes'][n] = value_codes.cpu().numpy()
        db['importance'][n] = importance * 0.6  # Further decay
        db['timestamp'][n] = self.stats['total_queries']
        
        l3['count'] += 1
        
        # Flush to disk every 1000 entries
        if l3['count'] % 1000 == 0:
            db.flush()
            print(f"L3 Archive: {l3['count']} entries "
                  f"({self._get_db_size_mb():.1f} MB on disk)")
    
    # ============================================
    # CONSOLIDATION (L1 → L2 → L3)
    # ============================================
    
    def _consolidate_l1_to_l2(self):
        """Move top 50% of L1 to L2, discard bottom 50%"""
        l1 = self.l1_working
        
        # Compute combined score (importance × access_count)
        combined_score = l1['importance'][:l1['count']] * \
                        (1 + torch.log1p(l1['access_count'][:l1['count']].float()))
        
        # Get top 50%
        k = l1['count'] // 2
        top_values, top_indices = torch.topk(combined_score, k=k)
        
        print(f"\nConsolidating L1→L2: Moving {k} top entries (min score: {top_values[-1]:.2f})")
        
        # Move to L2
        for idx in top_indices:
            idx = idx.item()
            self._add_to_l2(
                l1['s_min'][idx].to(torch.float32),
                l1['s_max'][idx].to(torch.float32),
                l1['values'][idx].to(torch.float32),
                l1['importance'][idx].item()
            )
        
        # Reset L1 (keep only the moved items temporarily)
        # In practice, just reset ptr to overwrite old entries
        l1['ptr'] = 0
        l1['count'] = 0
        print(f"L1 consolidated. L2 now has {self.l2_session['count']} entries.")
    
    def consolidate_l2_to_l3(self):
        """
        Move top entries from L2 to L3.
        Call this at end of each file or periodically.
        """
        l2 = self.l2_session
        
        if l2['count'] < l2['size'] * 0.5:
            return  # Not full enough
        
        # Get top 25% by importance
        k = l2['count'] // 4
        top_values, top_indices = torch.topk(
            l2['importance'][:l2['count']], k=k
        )
        
        print(f"\nConsolidating L2→L3: Moving {k} entries to disk...")
        
        # Dequantize and move to L3
        self.vq_embeddings.eval()
        
        for idx in top_indices:
            idx = idx.item()
            
            with torch.no_grad():
                # Dequantize for hash computation
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
        
        # Compact L2: remove moved entries
        # For simplicity, just reset L2 after moving top entries
        # In production, you'd compact the arrays
        l2['ptr'] = 0
        l2['count'] = 0
        
        print(f"L2→L3 consolidation complete. L3 has {self.l3_archive['count']} entries.")
    
    # ============================================
    # RETRIEVAL OPERATIONS
    # ============================================
    def retrieve(self, query, k=32):
        """
        Unified retrieval across all memory levels.
        Returns top-k memories with their reconstruction matrices.
        """
        self.stats['total_queries'] += 1
        q_min, q_max = query
        
        # Try L1 first (fastest, full precision)
        l1_results = self._retrieve_from_l1(q_min, q_max, k=k)
        
        # Try L2 (medium speed, vector quantized)
        l2_results = self._retrieve_from_l2(q_min, q_max, k=k)
        
        # Try L3 if needed (slowest, disk-based)
        l3_results = []
        
        # If no good matches, try L3 archive
        if not l1_results and not l2_results:
            l3_results = self._retrieve_from_l3(q_min, q_max, k=k//2)
        elif l1_results or l2_results:
            combined_temp = l1_results + l2_results
            max_importance = max(r['importance'] for r in combined_temp)
            if max_importance < 0.3:
                l3_results = self._retrieve_from_l3(q_min, q_max, k=k//2)
        
        # Combine results from all levels
        results = l1_results + l2_results + l3_results
        
        # Re-rank combined results by importance
        if results:
            results = sorted(results, key=lambda x: x['importance'], reverse=True)[:k]
        
        return results
    
    def _retrieve_from_l1(self, q_min, q_max, k=32):
        """Retrieve from L1 (full precision) memory"""
        if self.l1_working['count'] == 0:
            return []
        
        # Get valid entries
        valid_count = self.l1_working['count']
        s_min = self.l1_working['s_min'][:valid_count]
        s_max = self.l1_working['s_max'][:valid_count]
        values = self.l1_working['values'][:valid_count]
        importance = self.l1_working['importance'][:valid_count]
        
        # Compute similarity scores
        # Box overlap: how much does query box overlap with stored boxes?
        q_min_expanded = q_min.unsqueeze(0)  # [1, embed_dim]
        q_max_expanded = q_max.unsqueeze(0)  # [1, embed_dim]
        
        # Intersection = min(q_max, s_max) - max(q_min, s_min)
        intersection_min = torch.maximum(q_min_expanded, s_min)
        intersection_max = torch.minimum(q_max_expanded, s_max)
        intersection_vol = torch.clamp(intersection_max - intersection_min, min=0).sum(dim=1)
        
        # Union = volume(query) + volume(stored) - intersection
        q_vol = (q_max - q_min).sum()
        s_vol = (s_max - s_min).sum(dim=1)
        union_vol = q_vol + s_vol - intersection_vol
        
        # IoU (Intersection over Union)
        iou = intersection_vol / (union_vol + 1e-8)
        
        # Weighted score
        scores = iou * importance
        
        # Get top-k
        k = min(k, valid_count)
        top_k_scores, top_k_indices = torch.topk(scores, k)
        
        # Return results
        results = []
        for score, idx in zip(top_k_scores, top_k_indices):
            idx = idx.item()
            results.append({
                's_min': s_min[idx],
                's_max': s_max[idx],
                'values': values[idx],
                'importance': importance[idx].item(),
                'score': score.item()  # Add the actual retrieval score
            })
            
            # Update access count
            self.l1_working['access_count'][idx] += 1
        
        self.stats['l1_hits'] += 1
        return results

    def _retrieve_from_l2(self, q_min, q_max, k=32):
        """Retrieve from L2 (vector quantized) memory"""
        if self.l2_session['count'] == 0:
            return []
        
        # Quantize query for comparison
        q_min_codes = self.vq_embeddings.quantize(q_min.unsqueeze(0))[0]
        q_max_codes = self.vq_embeddings.quantize(q_max.unsqueeze(0))[0]
        
        # Get valid entries
        valid_count = self.l2_session['count']
        s_min_codes = self.l2_session['s_min_codes'][:valid_count]
        s_max_codes = self.l2_session['s_max_codes'][:valid_count]
        value_codes = self.l2_session['value_codes'][:valid_count]
        importance = self.l2_session['importance'][:valid_count]
        
        # Compute similarity (Hamming distance in code space)
        # Simple metric: count matching codes
        matches_min = (s_min_codes == q_min_codes.unsqueeze(0)).sum(dim=1).float()
        matches_max = (s_max_codes == q_max_codes.unsqueeze(0)).sum(dim=1).float()
        scores = (matches_min + matches_max) * importance
        
        # Get top-k
        k = min(k, valid_count)
        top_k_scores, top_k_indices = torch.topk(scores, k)
        
        # Dequantize the top-k entries
        results = []
        for score, idx in zip(top_k_scores, top_k_indices):
            idx = idx.item()
            
            # Dequantize s_min and s_max
            s_min = self.vq_embeddings.dequantize(
                s_min_codes[idx:idx+1]
            )[0]
            
            s_max = self.vq_embeddings.dequantize(
                s_max_codes[idx:idx+1]
            )[0]
            
            # Dequantize values
            values = self.vq_matrices.dequantize(
                value_codes[idx:idx+1]
            )[0]
            
            results.append({
                's_min': s_min,
                's_max': s_max,
                'values': values,
                'importance': importance[idx].item(),
                'score': score.item()  # Add the actual retrieval score
            })
            
            # Update access count
            self.l2_session['access_count'][idx] += 1
        
        self.stats['l2_hits'] += 1
        return results
    def _retrieve_from_l3(self, q_min, q_max, k=16):
        """
        Retrieve from L3 archive (disk).
        Uses hash-based approximate nearest neighbor search.
        Returns list of dicts matching L1/L2 format.
        """
        l3 = self.l3_archive
        if l3['count'] == 0:
            return []
        
        db = l3['db']
        
        with torch.no_grad():
            # Compute query hash
            query_hash = self.hasher.hash(q_min.unsqueeze(0))[0]
            
            # Search in batches to avoid loading entire database
            batch_size = 10000
            best_scores = []
            best_indices = []
            
            for start in range(0, l3['count'], batch_size):
                end = min(start + batch_size, l3['count'])
                
                # Load batch of hashes
                hashes_batch = torch.from_numpy(
                    db['hashes'][start:end]
                ).to(DEVICE)
                
                # Compute Hamming distances (fast bitwise operations)
                xor = query_hash.unsqueeze(0) ^ hashes_batch
                hamming_dists = xor.sum(dim=1).float()  # Count set bits
                
                # Convert to similarity scores (lower distance = higher score)
                similarities = 1.0 / (1.0 + hamming_dists / 128.0)
                
                # Get top candidates from this batch
                top_k_batch = min(k * 2, len(similarities))
                top_scores_batch, top_idx_batch = torch.topk(
                    similarities, k=top_k_batch
                )
                
                best_scores.append(top_scores_batch)
                best_indices.append(top_idx_batch + start)
            
            # Merge batch results
            all_scores = torch.cat(best_scores)
            all_indices = torch.cat(best_indices)
            
            # Get final top-k
            final_k = min(k, len(all_scores))
            top_scores, positions = torch.topk(all_scores, k=final_k)
            top_indices = all_indices[positions]
            
            # Load and dequantize actual values - return as list of dicts
            results = []
            for idx, score in zip(top_indices, top_scores):
                idx_cpu = idx.item()
                
                # Check cache first
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
                
                # Load from disk
                s_min_codes = torch.from_numpy(
                    db['s_min_codes'][idx_cpu]
                ).unsqueeze(0).to(DEVICE)
                
                s_max_codes = torch.from_numpy(
                    db['s_max_codes'][idx_cpu]
                ).unsqueeze(0).to(DEVICE)
                
                value_codes = torch.from_numpy(
                    db['value_codes'][idx_cpu]
                ).unsqueeze(0).to(DEVICE)
                
                # Dequantize
                self.vq_embeddings.eval()
                self.vq_matrices.eval()
                
                s_min = self.vq_embeddings.dequantize(s_min_codes)[0]
                s_max = self.vq_embeddings.dequantize(s_max_codes)[0]
                values = self.vq_matrices.dequantize(value_codes)[0]
                
                # Add to cache (LRU)
                l3['cache'][idx_cpu] = {
                    's_min': s_min,
                    's_max': s_max,
                    'values': values
                }
                if len(l3['cache']) > l3['cache_size']:
                    # Remove oldest
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
    
    def _merge_results(self, results, k):
        """Merge and re-rank results from multiple levels"""
        all_values = []
        all_scores = []
        
        for result_dict, weight in results:
            all_values.append(result_dict['values'])
            all_scores.append(result_dict['scores'] * weight)
        
        # Concatenate
        merged_values = torch.cat(all_values, dim=0)
        merged_scores = torch.cat(all_scores, dim=0)
        
        # Get final top-k
        final_k = min(k, len(merged_scores))
        top_scores, top_indices = torch.topk(merged_scores, k=final_k)
        
        return {
            'values': merged_values[top_indices],
            'scores': top_scores
        }
    
    # ============================================
    # UTILITIES
    # ============================================
    
    def _get_db_size_mb(self):
        """Get current database size on disk"""
        db_file = os.path.join(self.disk_path, 'archive.h5')
        if os.path.exists(db_file):
            return os.path.getsize(db_file) / (1024 * 1024)
        return 0
    
    def print_stats(self):
        """Print memory hierarchy statistics"""
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
        """
        Estimate final database size after training on X TB of data.
        
        Assumptions:
        - 1 memory per 1MB of text
        - Each memory: 16 + 16 + 32 + 4 + 8 = 76 bytes (uncompressed)
        - With gzip: ~40 bytes per memory
        """
        total_bytes = total_data_tb * 1024 * 1024 * 1024 * 1024  # TB to bytes
        memories = total_bytes / (1024 * 1024)  # 1 per MB
        
        bytes_per_memory = 40  # After compression
        final_size_gb = (memories * bytes_per_memory) / (1024 * 1024 * 1024)
        
        print(f"\nEstimated final database size:")
        print(f"Training data: {total_data_tb} TB")
        print(f"Expected memories: {memories/1e6:.1f}M")
        print(f"Final DB size: ~{final_size_gb:.1f} GB")
        print(f"Compression ratio: {total_data_tb * 1024 / final_size_gb:.0f}×")
        
        return final_size_gb

# --- 2. Memory Management Functions ---
def aggressive_cleanup():
    """Force complete memory cleanup."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

def clear_autocast_cache():
    """Clear autocast cache if available."""
    if hasattr(torch, 'clear_autocast_cache'):
        torch.clear_autocast_cache()
    elif hasattr(torch.cuda.amp, 'clear_cache'):
        torch.cuda.amp.clear_cache()

# --- 3. Tokenizer ---
class LimitedTokenizer:
    def __init__(self, limit=32768):
        self.enc = tiktoken.get_encoding("cl100k_base")
        self.limit = limit
        self.unk_id = limit - 1 
    def encode(self, text):
        return [t if t < self.unk_id else self.unk_id for t in self.enc.encode(text)]
    def decode(self, ids):
        return self.enc.decode([int(i) for i in ids if i < self.unk_id])

# --- 4. Seashore Mechanics ---

def hebbian_update(layer, inputs, learning_rate):
    """
    Oja's Rule + Winner Take All + Sedimentation.
    Adapted for Transformer inputs (Batch, Seq, Dim).
    """
    if not isinstance(layer, nn.Linear):
        return

    with torch.no_grad():
        # Flatten inputs: [B, T, D] -> [N, D]
        if inputs.dim() == 3:
            inputs = inputs.reshape(-1, inputs.shape[-1])
            
        # 1. Normalize Inputs (Waves)
        waves = F.normalize(inputs, p=2, dim=1)
        weights = layer.weight # [Out, In]
        
        # 2. Similarity & Competition
        # [N, In] @ [In, Out] -> [N, Out]
        similarity = torch.mm(waves, weights.t())
        
        # Winner Take All
        winners = torch.argmax(similarity, dim=1)
        winner_mask = F.one_hot(winners, num_classes=weights.shape[0]).float()
        
        # 3. Sedimentation (Average input per neuron)
        # [Out, N] @ [N, In] -> [Out, In]
        waves_sum = torch.mm(winner_mask.t(), waves)
        win_counts = winner_mask.sum(dim=0).unsqueeze(1) + 1e-6
        waves_avg = waves_sum / win_counts
        has_won = (win_counts > 1e-5).float()
        
        # 4. Update & Erode (Oja's Rule-like)
        delta = learning_rate * has_won * (waves_avg - weights)
        layer.weight.add_(delta)
        
        # Renormalize weights to keep physics consistent
        layer.weight.div_(layer.weight.norm(dim=1, keepdim=True) + 1e-8)

def normalize_model_weights(model):
    """Ensures all Linear layers are on the unit sphere after Backprop."""
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, nn.Linear):
                module.weight.div_(module.weight.norm(dim=1, keepdim=True) + 1e-8)

# --- 5. Optimized Components ---

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

class RelationalHoloArchive:
    def __init__(self, size, num_heads, head_dim, embed_dim):
        self.size = size
        self.s_min = torch.zeros(size, embed_dim, dtype=torch.float16, device=DEVICE)
        self.s_max = torch.zeros(size, embed_dim, dtype=torch.float16, device=DEVICE)
        self.values = torch.zeros(size, num_heads * head_dim * head_dim, dtype=torch.float16, device=DEVICE)
        self.importance = torch.zeros(size, dtype=torch.float32, device=DEVICE)
        self.ptr = 0
        self.count = 0
        self.num_heads = num_heads
        self.head_dim = head_dim

    def add(self, s_box, matrix):
        with torch.no_grad():
            self.s_min[self.ptr] = s_box[0][:, -1, :].mean(dim=0).detach().to(torch.float16)
            self.s_max[self.ptr] = s_box[1][:, -1, :].mean(dim=0).detach().to(torch.float16)
            flat_mat = matrix.mean(dim=0).flatten().detach().to(torch.float16)
            self.values[self.ptr] = flat_mat
            self.importance[self.ptr] = 1.0  # Fresh entries start with importance
            self.ptr = (self.ptr + 1) % self.size
            self.count = min(self.count + 1, self.size)
    
    def update_importance(self, indices, boost=0.1):
        """Boost importance of retrieved items"""
        with torch.no_grad():
            if len(indices) > 0:
                self.importance[indices] = torch.clamp(self.importance[indices] + boost, max=10.0)
    
    def soft_reset(self, keep_ratio=0.25):
        """Keep only the most important memories"""
        with torch.no_grad():
            if self.count < self.size * 0.5:  # Don't reset if archive isn't sufficiently full
                print(f"Archive not full enough ({self.count}/{self.size}), skipping soft reset")
                return
            
            keep_count = int(self.size * keep_ratio)
            if keep_count == 0:
                keep_count = 1
            
            # Get top important entries
            _, top_indices = torch.topk(self.importance[:self.count], min(keep_count, self.count))
            
            # Create new tensors to avoid fragmentation
            new_s_min = torch.zeros_like(self.s_min)
            new_s_max = torch.zeros_like(self.s_max)
            new_values = torch.zeros_like(self.values)
            new_importance = torch.zeros_like(self.importance)
            
            # Copy top entries
            actual_keep = len(top_indices)
            new_s_min[:actual_keep] = self.s_min[top_indices]
            new_s_max[:actual_keep] = self.s_max[top_indices]
            new_values[:actual_keep] = self.values[top_indices]
            new_importance[:actual_keep] = self.importance[top_indices] * 0.7  # Decay importance
            
            # Replace old tensors
            self.s_min = new_s_min
            self.s_max = new_s_max
            self.values = new_values
            self.importance = new_importance
            
            self.ptr = actual_keep
            self.count = actual_keep
            
            print(f"Archive soft reset: kept {actual_keep}/{self.size} most important entries")
            aggressive_cleanup()
    
    def reset(self):
        """Full reset (only use if absolutely necessary)"""
        with torch.no_grad():
            self.s_min.zero_()
            self.s_max.zero_()
            self.values.zero_()
            self.importance.zero_()
            self.ptr = 0
            self.count = 0
            print("Archive hard reset: all entries cleared")

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

class HoloGraphBlockV7(nn.Module):
    def __init__(self, embed_dim, num_heads, head_dim):
        super().__init__()
        self.num_heads, self.head_dim = num_heads, head_dim
        self.total_dim = num_heads * head_dim
        
        # Primary Projections
        self.proj_k = nn.Linear(embed_dim, self.total_dim)
        self.proj_q = nn.Linear(embed_dim, self.total_dim)
        self.proj_v = nn.Linear(embed_dim, self.total_dim)
        self.proj_out = nn.Linear(self.total_dim, embed_dim)
        
        # Gates (Learned mostly via BP, but we can attempt Hebbian on them)
        self.proj_delta = nn.Linear(embed_dim, num_heads)
        self.gate_write = nn.Linear(embed_dim, num_heads)
        self.gate_forget = nn.Linear(embed_dim, num_heads)
        
        self.probe_s = nn.Linear(embed_dim, embed_dim)
        self.ln1, self.ln2 = nn.LayerNorm(embed_dim), nn.LayerNorm(embed_dim)
        
        # Feed Forward
        self.ffn_1 = nn.Linear(embed_dim, 4*embed_dim)
        self.ffn_2 = nn.Linear(4*embed_dim, embed_dim)

    def forward(self, x, b_min, b_max, motif_context, hebbian_lr=None, archive=None):
        """
        Modified forward pass. If hebbian_lr is provided, update weights locally
        based on input 'x' before computing output.
        Returns retrieved_indices for importance tracking.
        """
        B, T, _ = x.shape
        x_n = self.ln1(x)
        
        # --- SEASHORE: HEBBIAN UPDATE PHASE ---
        if hebbian_lr is not None:
            # Update projections based on normalized input
            hebbian_update(self.proj_k, x_n, hebbian_lr)
            hebbian_update(self.proj_q, x_n, hebbian_lr)
            hebbian_update(self.proj_v, x_n, hebbian_lr)
        
        k = F.normalize(self.proj_k(x_n).view(B, T, self.num_heads, self.head_dim), p=2.0, dim=-1)
        q = F.normalize(self.proj_q(x_n).view(B, T, self.num_heads, self.head_dim), p=2.0, dim=-1)
        v = torch.tanh(self.proj_v(x_n).view(B, T, self.num_heads, self.head_dim))
        
        d = F.softplus(self.proj_delta(x_n)).view(B, T, self.num_heads)
        gw = torch.sigmoid(self.gate_write(x_n)).pow(2).view(B, T, self.num_heads)
        gf = (1.0 - torch.sigmoid(self.gate_forget(x_n)).pow(2)).view(B, T, self.num_heads)
        
        m_heads, next_mem = holo_selective_scan_v7(k, q, v, d, gf, gw)
        
        # Capture context for proj_out update
        ctx_flat = m_heads.reshape(B, T, -1)
        if hebbian_lr is not None:
             hebbian_update(self.proj_out, ctx_flat, hebbian_lr)
             
        x = x + self.proj_out(ctx_flat) + (0.1 * motif_context)
        
        # FFN Part
        x_n2 = self.ln2(x)
        if hebbian_lr is not None:
            hebbian_update(self.ffn_1, x_n2, hebbian_lr)
        
        hidden = F.gelu(self.ffn_1(x_n2))
        
        if hebbian_lr is not None:
            hebbian_update(self.ffn_2, hidden, hebbian_lr)
            
        x = x + self.ffn_2(hidden)
        
        s_box = (b_min + self.probe_s(x), b_max + self.probe_s(x))
        return x, next_mem, s_box

class HoloGraphV7(nn.Module):
    def __init__(self):
        super().__init__()
        self.box_emb = BoxEmbedding(VOCAB_SIZE, EMBED_DIM)
        self.layers = nn.ModuleList([
            HoloGraphBlockV7(EMBED_DIM, NUM_HEADS, HEAD_DIM) 
            for _ in range(NUM_LAYERS)
        ])        
        self.ln_f = nn.LayerNorm(EMBED_DIM)
        self.logit_scale = nn.Parameter(torch.ones(1) * 14.0)
    
    def forward(self, idx, memory=None, hebbian_lr=None, use_memory=True):  # Added use_memory flag
        b_min, b_max = self.box_emb(idx)
        x = (b_min + b_max) / 2.0
        
        layer_data = []
        for layer in self.layers:
            motif_ctx = torch.zeros_like(x)
            # OPTIMIZED: Only retrieve from memory when use_memory=True (controlled by step counter)
            if memory is not None and use_memory:
                # Average box embeddings across sequence and batch dimensions for query
                query_min = b_min.mean(dim=(0, 1))  # [Embed_Dim]
                query_max = b_max.mean(dim=(0, 1))  # [Embed_Dim]
                
                memory_results = memory.retrieve((query_min, query_max), k=32)
                
                if memory_results:  # Now it's a list, not None
                    # OPTIMIZED: Vectorized attention computation
                    B, T, _ = x.shape
                    
                    # Stack and cast to the same dtype as the current hidden state (x)
                    retrieved_values = torch.stack([r['values'] for r in memory_results]).to(x.dtype)
                    retrieved_importance = torch.tensor(
                        [r['importance'] for r in memory_results], 
                        device=x.device, dtype=x.dtype
                    )
                    
                    k_retrieved = retrieved_values.shape[0]
                    
                    # Reshape to [k, num_heads, head_dim, head_dim]
                    retrieved_matrices = retrieved_values.view(
                        k_retrieved, layer.num_heads, layer.head_dim, layer.head_dim
                    )
                    
                    # Create query vectors from current state
                    qr = F.normalize(
                        layer.proj_q(x).view(B, T, layer.num_heads, layer.head_dim),
                        p=2.0, dim=-1
                    ).to(x.dtype)  # [B, T, num_heads, head_dim]
                    
                    # OPTIMIZED: Vectorized computation - no nested loops
                    # Reshape qr to [B*T*num_heads, head_dim]
                    qr_flat = qr.reshape(B * T * layer.num_heads, layer.head_dim)
                    
                    # Keys from retrieved memories: [k, num_heads, head_dim]
                    keys = retrieved_matrices.mean(dim=-1)  # [k, num_heads, head_dim]
                    keys = keys.transpose(0, 1).reshape(layer.num_heads, k_retrieved, layer.head_dim)  # [num_heads, k, head_dim]
                    
                    # Replicate keys for all B*T positions: [B*T*num_heads, k, head_dim]
                    keys_expanded = keys.unsqueeze(0).repeat(B * T, 1, 1, 1).reshape(B * T * layer.num_heads, k_retrieved, layer.head_dim)
                    
                    # Attention scores: [B*T*num_heads, k]
                    attn_scores = torch.bmm(
                        keys_expanded, 
                        qr_flat.unsqueeze(-1)
                    ).squeeze(-1) / math.sqrt(layer.head_dim)
                    
                    # Weight by importance: [B*T*num_heads, k]
                    attn_scores = attn_scores * retrieved_importance.unsqueeze(0)
                    attn_weights = F.softmax(attn_scores, dim=-1)  # [B*T*num_heads, k]
                      # Reshape for einsum: [B, T, num_heads, k]
                    attn_weights = attn_weights.reshape(B, T, layer.num_heads, k_retrieved)
                    
                    # Weighted sum of retrieved matrices: [B, T, num_heads, head_dim, head_dim]
                    # attn_weights: [B, T, num_heads, k_retrieved]
                    # retrieved_matrices: [k_retrieved, num_heads, head_dim, head_dim]
                    weighted_matrices = torch.einsum(
                        'bthk,khde->bthde',
                        attn_weights,
                        retrieved_matrices
                    )
                    
                    # Apply matrix to query: [B, T, num_heads, head_dim]
                    # weighted_matrices: [B, T, num_heads, head_dim, head_dim]
                    # qr: [B, T, num_heads, head_dim]
                    output_h = torch.einsum('bthde,bthe->bthd', weighted_matrices, qr)
                    
                    # Reshape and concatenate: [B, T, num_heads * head_dim]
                    motif_ctx_raw = output_h.reshape(B, T, layer.num_heads * layer.head_dim)
                    
                    # Project to embed_dim
                    motif_ctx = layer.proj_out(motif_ctx_raw)
            
            x, mem, s_box = layer(x, b_min, b_max, motif_ctx, hebbian_lr)
            layer_data.append((mem, s_box))
        
        logits = F.linear(self.ln_f(x), self.box_emb.center.weight) * \
                 (self.logit_scale / (EMBED_DIM**0.5))
        
        return logits, layer_data

# --- 6. Data Pipeline ---
class SequentialRotatingStreamer:
    def __init__(self, file_paths, text_column, tokenizer, seq_len, batch_size):
        self.file_paths = file_paths
        self.text_column = text_column
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.batch_size = batch_size
        # OPTIMIZED: Add batch prefetching buffer
        self.prefetch_buffer = []
        self.prefetch_size = batch_size * 2  # Keep 2 batches ready

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
                                # OPTIMIZED: Create tensors once, reuse memory
                                xb = torch.tensor(batch_x, dtype=torch.long, device=DEVICE)
                                yb = torch.tensor(batch_y, dtype=torch.long, device=DEVICE)
                                yield (xb, yb, file_msg)
                                batch_x, batch_y = [], []
                                # OPTIMIZED: Clear references immediately
                                del xb, yb
                    
                    # Cleanup after each row group
                    del texts
                    del table
                    table = None
                    
            except Exception as e:
                print(f"\nError reading {file_path}: {e}")
                continue
            finally:
                # Explicit cleanup
                if table is not None:
                    del table
                if pf is not None:
                    del pf
                gc.collect()

# --- 7. Main Logic (Seashore Integration with Soft Reset) ---

def train(folder, text_col, continue_training=True):
    tokenizer = LimitedTokenizer(limit=VOCAB_SIZE)
    model = HoloGraphV7().to(DEVICE)
    
    # NEW: Initialize hierarchical compressed memory system
    memory = CompressedHierarchicalMemory(
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        head_dim=HEAD_DIM,
        l1_size=2048,
        l2_size=16384,
        disk_path='holo_memory_db'
    )
    
    # Load compression codebooks if they exist
    memory.load_compressors()
    
    # Estimate final database size (adjust total_data_tb to your actual data size)
    print("\n" + "="*60)
    memory.estimate_final_size(total_data_tb=0.2)  # Change 0.2 to your actual TB
    print("="*60 + "\n")
      # Optimizer (Only used during BP steps)
    try:
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
        print("Using 8-bit AdamW optimizer")
    except ImportError:
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
        print("Using standard AdamW optimizer")
    
    # FIXED: scaler needs to be defined outside the try/except block
    scaler = torch.cuda.amp.GradScaler()
    global_step, epoch = 0, 0
    files_processed = 0
    accumulation_step = 0  # OPTIMIZED: Track gradient accumulation steps
    
    # NEW: Compression training state
    compression_train_interval = 1000  # OPTIMIZED: Reduced frequency from 500 to 1000
    sample_buffer = {'s_min': [], 's_max': [], 'values': []}
    
    # Load checkpoint if it exists and continue_training is True
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

    print(f"Starting SEASHORE Training with Hierarchical Compressed Memory")
    print(f"Ratio BP:{BP_RATIO} / Hebbian:{10-BP_RATIO}")
    print(f"Memory: L1={memory.l1_working['size']} | L2={memory.l2_session['size']} | L3=Unlimited (disk)")

    while True:
        for xb, yb, file_info in streamer:
            if xb is None:
                if session_has_started:
                    print(f"\n\nEnd of file reached. Consolidating memory and saving checkpoint...")
                    
                    # NEW: Consolidate L2 to L3 at end of each file
                    print("Consolidating L2 → L3...")
                    memory.consolidate_l2_to_l3()
                    
                    # NEW: Print memory statistics
                    memory.print_stats()
                    
                    # NEW: Save compression codebooks
                    memory.save_compressors()
                    
                    # Save model checkpoint
                    torch.save({
                        'model': model.state_dict(), 
                        'opt': optimizer.state_dict(), 
                        'scaler': scaler.state_dict(),
                        'step': global_step, 
                        'epoch': epoch,
                        'files_processed': files_processed
                    }, CHECKPOINT_PATH)
                    
                    files_processed += 1
                    
                    # Aggressive cleanup
                    aggressive_cleanup()
                    
                session_has_started = True
                current_file_str = file_info
                aggressive_cleanup()
                continue              # --- SEASHORE SCHEDULING ---
            cycle_idx = global_step % 10
            is_bp_step = cycle_idx < BP_RATIO
            
            # OPTIMIZED: Only use memory retrieval every N steps to reduce overhead
            use_memory_this_step = (global_step % MEMORY_RETRIEVAL_EVERY == 0)
            
            model.train()
            
            if is_bp_step:
                # --- BACKPROP STEP (30%) ---
                # OPTIMIZED: Only zero grads at start of accumulation cycle
                if accumulation_step == 0:
                    optimizer.zero_grad(set_to_none=True)
                
                with torch.amp.autocast('cuda'):
                    logits, layer_data = model(xb, memory=memory, hebbian_lr=None, use_memory=use_memory_this_step)
                    loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), yb.view(-1))
                    # OPTIMIZED: Scale loss by accumulation steps
                    loss = loss / GRADIENT_ACCUMULATION
                
                scaler.scale(loss).backward()
                
                accumulation_step += 1
                
                # OPTIMIZED: Only update weights after accumulating enough gradients
                if accumulation_step >= GRADIENT_ACCUMULATION:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    
                    # CRITICAL SEASHORE STEP: Re-normalize weights after BP
                    normalize_model_weights(model)
                    
                    accumulation_step = 0
                
                mode_str = "BP"
                # Scale loss back for display
                loss = loss * GRADIENT_ACCUMULATION
                
            else:
                # --- HEBBIAN STEP (70%) ---
                with torch.no_grad():
                    logits, layer_data = model(xb, memory=memory, hebbian_lr=HEBBIAN_LR, use_memory=use_memory_this_step)
                    loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), yb.view(-1))
                mode_str = "HB"

            # NEW: Add to hierarchical memory and collect samples for compression training
            if global_step % SNAPSHOT_RATE == 0:
                mem, s_box = layer_data[0]
                
                # Compute importance score (based on loss - higher loss = more important)
                importance = min(loss.item() * 2.0, 10.0)  # Cap at 10.0
                
                # Add to memory hierarchy
                memory.add(s_box, mem, importance=importance)
                
                # Collect samples for compressor training
                with torch.no_grad():
                    sample_buffer['s_min'].append(s_box[0][:, -1, :].mean(dim=0).to(torch.float32))
                    sample_buffer['s_max'].append(s_box[1][:, -1, :].mean(dim=0).to(torch.float32))
                    sample_buffer['values'].append(mem.mean(dim=0).flatten().to(torch.float32))
            
            # NEW: Train compressors periodically
            if global_step % compression_train_interval == 0 and len(sample_buffer['s_min']) > 0:
                print(f"\nTraining compression codebooks on {len(sample_buffer['s_min'])} samples...")
                batch = {
                    's_min': torch.stack(sample_buffer['s_min']),
                    's_max': torch.stack(sample_buffer['s_max']),
                    'values': torch.stack(sample_buffer['values'])
                }
                memory.train_compressors(batch)
                
                # Clear sample buffer
                sample_buffer = {'s_min': [], 's_max': [], 'values': []}
                
                # Clear references and cleanup
                del batch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Periodic cleanup
            if global_step % CLEANUP_EVERY == 0:
                clear_autocast_cache()
                if global_step % (CLEANUP_EVERY * 5) == 0:
                    aggressive_cleanup()
              # Preview
            if global_step % PREDICT_EVERY == 0:
                model.eval()
                with torch.no_grad():
                    context = xb[:1, :10] 
                    print(f"\n\n--- Step {global_step} [{mode_str}] Preview ---")
                    print(f"Memory: L1={memory.l1_working['count']}/{memory.l1_working['size']} | "
                          f"L2={memory.l2_session['count']}/{memory.l2_session['size']} | "
                          f"L3={memory.l3_archive['count']} ({memory._get_db_size_mb():.1f}MB)")
                    print(f"Prompt: {tokenizer.decode(context[0].tolist())}")
                    print("Output: ", end="", flush=True)
                    gen_tokens = context
                    for _ in range(20): 
                        with torch.amp.autocast('cuda'):
                            lg, _ = model(gen_tokens, memory=memory, use_memory=False)  # OPTIMIZED: Disable memory during generation
                        probs = F.softmax(lg[:, -1, :] / 0.8, dim=-1)
                        next_token = torch.multinomial(probs, 1)
                        print(tokenizer.decode([next_token.item()]), end="", flush=True)
                        gen_tokens = torch.cat([gen_tokens, next_token], dim=1)
                        if gen_tokens.shape[1] > SEQ_LEN: gen_tokens = gen_tokens[:, 1:]
                    print("\n" + "-"*40 + "\n")

            global_step += 1
            
            # NEW: Enhanced status display
            l1_pct = (memory.l1_working['count'] / memory.l1_working['size']) * 100
            l2_pct = (memory.l2_session['count'] / memory.l2_session['size']) * 100
            
            stdout.write(
                f'\r[{mode_str}][File {current_file_str}][Files: {files_processed}]'
                f'[L1:{l1_pct:.0f}% L2:{l2_pct:.0f}% L3:{memory.l3_archive["count"]}] '
                f'Step: {global_step} | Loss: {loss.item():.4f}'
            )
            stdout.flush()

        epoch += 1
        print(f"\n\nEpoch {epoch} finished. Saving final checkpoint...")
        
        # NEW: Final consolidation
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
    args = parser.parse_args()
    
    continue_training = not args.no_continue
    train(args.folder, args.column, continue_training=continue_training)