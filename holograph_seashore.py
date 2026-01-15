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
LEARNING_RATE = 3e-4 
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CHECKPOINT_PATH = 'holograph_seashore.pth'

# Seashore Hyperparameters
SEASHORE_WIDTH = EMBED_DIM # Aligning width logic
HEBBIAN_LR = 0.05          # Learning rate for Hebbian updates
BP_RATIO = 3               # 3 steps BP, 7 steps Hebbian (Cycle of 10)

# Control Rates
ARCHIVE_SIZE = 4096 
SNAPSHOT_RATE = 128   
PREDICT_EVERY = 1000  

# Memory Management
ARCHIVE_SOFT_RESET_INTERVAL = 10  # Soft reset archive every N files
ARCHIVE_KEEP_RATIO = 0.25         # Keep top 25% on soft reset
CLEANUP_EVERY = 500               # Aggressive cleanup every N steps

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
        self.layers = nn.ModuleList([HoloGraphBlockV7(EMBED_DIM, NUM_HEADS, HEAD_DIM) for _ in range(NUM_LAYERS)])
        self.ln_f = nn.LayerNorm(EMBED_DIM)
        self.logit_scale = nn.Parameter(torch.ones(1) * 14.0)
        
    def forward(self, idx, archive=None, hebbian_lr=None):
        b_min, b_max = self.box_emb(idx)
        x = (b_min + b_max) / 2.0 
        B = x.shape[0]

        layer_data = []
        for layer in self.layers:
            motif_ctx = torch.zeros_like(x)
            retrieved_indices = []
            
            # --- Retrieval with Importance Tracking ---
            if archive and archive.count > 20:
                with torch.no_grad():
                    s_min, s_max = b_min + layer.probe_s(x), b_max + layer.probe_s(x)
                    q_min, q_max = s_min.mean(dim=1, keepdim=True), s_max.mean(dim=1, keepdim=True)
                    curr_s_min = archive.s_min[:archive.count].unsqueeze(0).to(x.dtype)
                    curr_s_max = archive.s_max[:archive.count].unsqueeze(0).to(x.dtype)
                    i_min = torch.max(q_min, curr_s_min); i_max = torch.min(q_max, curr_s_max)
                    scores = torch.mean(torch.log(F.softplus(i_max - i_min) + 1e-6), dim=-1)
                    weights = F.softmax(scores * 2.0, dim=-1)
                    
                    # Track which memories were retrieved
                    top_k = min(32, archive.count)
                    _, top_indices = torch.topk(weights[0], top_k)
                    retrieved_indices = top_indices[top_indices < archive.count].cpu().tolist()
                    
                    vals = archive.values[:archive.count].to(x.dtype)
                    fused = torch.matmul(weights, vals).view(B, layer.num_heads, layer.head_dim, layer.head_dim)
                
                qr = F.normalize(layer.proj_q(x).view(B, x.shape[1], layer.num_heads, layer.head_dim), p=2.0, dim=-1)
                motif_ctx = layer.proj_out(torch.matmul(fused.unsqueeze(1), qr.unsqueeze(-1)).squeeze(-1).reshape(B, x.shape[1], -1))
                
                # Update importance of retrieved items
                if len(retrieved_indices) > 0 and archive is not None:
                    archive.update_importance(torch.tensor(retrieved_indices, device=DEVICE), boost=0.1)

            # REMOVED CHECKPOINTING - Direct call always
            x, mem, s_box = layer(x, b_min, b_max, motif_ctx, hebbian_lr, archive)
            
            layer_data.append((mem, s_box))
            
        logits = F.linear(self.ln_f(x), self.box_emb.center.weight) * (self.logit_scale / (EMBED_DIM**0.5))
        return logits, layer_data

# --- 6. Data Pipeline ---
class SequentialRotatingStreamer:
    def __init__(self, file_paths, text_column, tokenizer, seq_len, batch_size):
        self.file_paths = file_paths
        self.text_column = text_column
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.batch_size = batch_size

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
                                yield (torch.tensor(batch_x, dtype=torch.long, device=DEVICE),
                                       torch.tensor(batch_y, dtype=torch.long, device=DEVICE),
                                       file_msg)
                                batch_x, batch_y = [], []
                    
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
    archive = RelationalHoloArchive(ARCHIVE_SIZE, NUM_HEADS, HEAD_DIM, EMBED_DIM)
    
    # Optimizer (Only used during BP steps)
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

    print(f"Starting SEASHORE Training. Ratio BP:{BP_RATIO} / Hebbian:{10-BP_RATIO}")
    print(f"Archive soft reset every {ARCHIVE_SOFT_RESET_INTERVAL} files (keeping top {ARCHIVE_KEEP_RATIO*100}%)")

    while True:
        for xb, yb, file_info in streamer:
            if xb is None:
                if session_has_started:
                    print(f"\nEnd of file reached. Saving checkpoint...")
                    torch.save({
                        'model': model.state_dict(), 
                        'opt': optimizer.state_dict(), 
                        'scaler': scaler.state_dict(),
                        'step': global_step, 
                        'epoch': epoch,
                        'files_processed': files_processed
                    }, CHECKPOINT_PATH)
                    
                    files_processed += 1
                    
                    # Soft reset archive periodically to prevent fragmentation
                    if files_processed % ARCHIVE_SOFT_RESET_INTERVAL == 0:
                        print(f"Initiating archive soft reset after {files_processed} files...")
                        archive.soft_reset(keep_ratio=ARCHIVE_KEEP_RATIO)
                    
                    # Aggressive cleanup
                    aggressive_cleanup()
                    
                session_has_started = True
                current_file_str = file_info
                aggressive_cleanup()
                continue 
            
            # --- SEASHORE SCHEDULING ---
            cycle_idx = global_step % 10
            is_bp_step = cycle_idx < BP_RATIO
            
            model.train()
            
            if is_bp_step:
                # --- BACKPROP STEP (30%) ---
                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast():
                    logits, layer_data = model(xb, archive=archive, hebbian_lr=None)
                    loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), yb.view(-1))
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                
                # CRITICAL SEASHORE STEP: Re-normalize weights after BP
                normalize_model_weights(model)
                mode_str = "BP"
                
            else:
                # --- HEBBIAN STEP (70%) ---
                # No gradients, local updates inside the model forward pass
                with torch.no_grad():
                    logits, layer_data = model(xb, archive=archive, hebbian_lr=HEBBIAN_LR)
                    loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), yb.view(-1))
                mode_str = "HB"

            # Archive Snapshot
            if global_step % SNAPSHOT_RATE == 0:
                mem, s_box = layer_data[0]
                archive.add(s_box, mem)
            
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
                    print(f"\n\n--- Step {global_step} [{mode_str}] Preview (Archive: {archive.count}/{ARCHIVE_SIZE}) ---")
                    print(f"Prompt: {tokenizer.decode(context[0].tolist())}")
                    print("Output: ", end="", flush=True)
                    gen_tokens = context
                    for _ in range(20): 
                        with torch.cuda.amp.autocast():
                            lg, _ = model(gen_tokens, archive=archive)
                        probs = F.softmax(lg[:, -1, :] / 0.8, dim=-1)
                        next_token = torch.multinomial(probs, 1)
                        print(tokenizer.decode([next_token.item()]), end="", flush=True)
                        gen_tokens = torch.cat([gen_tokens, next_token], dim=1)
                        if gen_tokens.shape[1] > SEQ_LEN: gen_tokens = gen_tokens[:, 1:]
                    print("\n" + "-"*40 + "\n")

            global_step += 1
            stdout.write(f'\r[{mode_str}][File {current_file_str}][Files: {files_processed}][Arch: {archive.count}/{ARCHIVE_SIZE}] Step: {global_step} | Loss: {loss.item():.4f}')
            stdout.flush()

        epoch += 1
        print(f"\n\nEpoch {epoch} finished. Saving final checkpoint...")
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