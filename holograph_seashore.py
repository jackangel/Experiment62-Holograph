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
BATCH_SIZE = 16
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

# --- 2. Tokenizer ---
class LimitedTokenizer:
    def __init__(self, limit=32768):
        self.enc = tiktoken.get_encoding("cl100k_base")
        self.limit = limit
        self.unk_id = limit - 1 
    def encode(self, text):
        return [t if t < self.unk_id else self.unk_id for t in self.enc.encode(text)]
    def decode(self, ids):
        return self.enc.decode([int(i) for i in ids if i < self.unk_id])

# --- 3. Seashore Mechanics ---

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

# --- 4. Optimized Components ---

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
        self.ptr = 0; self.count = 0
        self.num_heads = num_heads
        self.head_dim = head_dim

    def add(self, s_box, matrix):
        with torch.no_grad():
            self.s_min[self.ptr] = s_box[0][:, -1, :].mean(dim=0).detach().to(torch.float16)
            self.s_max[self.ptr] = s_box[1][:, -1, :].mean(dim=0).detach().to(torch.float16)
            flat_mat = matrix.mean(dim=0).flatten().detach().to(torch.float16)
            self.values[self.ptr] = flat_mat
            self.ptr = (self.ptr + 1) % self.size
            self.count = min(self.count + 1, self.size)

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

    def forward(self, x, b_min, b_max, motif_context, hebbian_lr=None):
        """
        Modified forward pass. If hebbian_lr is provided, update weights locally
        based on input 'x' before computing output.
        """
        B, T, _ = x.shape
        x_n = self.ln1(x)
        
        # --- SEASHORE: HEBBIAN UPDATE PHASE ---
        if hebbian_lr is not None:
            # Update projections based on normalized input
            hebbian_update(self.proj_k, x_n, hebbian_lr)
            hebbian_update(self.proj_q, x_n, hebbian_lr)
            hebbian_update(self.proj_v, x_n, hebbian_lr)
            # FFN Updates (based on ln2 output later)
        
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
        # Embeddings are usually lookup tables, harder to apply Oja's rule directly
        # efficiently, so we stick to BP for embeddings/output head
        b_min, b_max = self.box_emb(idx)
        x = (b_min + b_max) / 2.0 
        B = x.shape[0]

        layer_data = []
        for layer in self.layers:
            motif_ctx = torch.zeros_like(x)
            
            # --- Retrieval ---
            if archive and archive.count > 20:
                with torch.no_grad():
                    s_min, s_max = b_min + layer.probe_s(x), b_max + layer.probe_s(x)
                    q_min, q_max = s_min.mean(dim=1, keepdim=True), s_max.mean(dim=1, keepdim=True)
                    curr_s_min = archive.s_min[:archive.count].unsqueeze(0).to(x.dtype)
                    curr_s_max = archive.s_max[:archive.count].unsqueeze(0).to(x.dtype)
                    i_min = torch.max(q_min, curr_s_min); i_max = torch.min(q_max, curr_s_max)
                    scores = torch.mean(torch.log(F.softplus(i_max - i_min) + 1e-6), dim=-1)
                    weights = F.softmax(scores * 2.0, dim=-1)
                    vals = archive.values[:archive.count].to(x.dtype)
                    fused = torch.matmul(weights, vals).view(B, layer.num_heads, layer.head_dim, layer.head_dim)
                
                qr = F.normalize(layer.proj_q(x).view(B, x.shape[1], layer.num_heads, layer.head_dim), p=2.0, dim=-1)
                motif_ctx = layer.proj_out(torch.matmul(fused.unsqueeze(1), qr.unsqueeze(-1)).squeeze(-1).reshape(B, x.shape[1], -1))

            # Pass hebbian_lr down to layer
            if self.training and hebbian_lr is None:
                x, mem, s_box = checkpoint(layer, x, b_min, b_max, motif_ctx, None, use_reentrant=True)
            else:
                # Direct call during Hebbian phase or Inference
                x, mem, s_box = layer(x, b_min, b_max, motif_ctx, hebbian_lr)
            
            layer_data.append((mem, s_box))
            
        logits = F.linear(self.ln_f(x), self.box_emb.center.weight) * (self.logit_scale / (EMBED_DIM**0.5))
        return logits, layer_data

# --- 5. Data Pipeline ---
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
            except Exception as e:
                print(f"Error reading {file_path}: {e}"); continue

# --- 6. Main Logic (Seashore Integration) ---

def train(folder, text_col):
    tokenizer = LimitedTokenizer(limit=VOCAB_SIZE)
    model = HoloGraphV7().to(DEVICE)
    archive = RelationalHoloArchive(ARCHIVE_SIZE, NUM_HEADS, HEAD_DIM, EMBED_DIM)
    
    # Optimizer (Only used during BP steps)
    try:
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    except ImportError:
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

    scaler = torch.cuda.amp.GradScaler()
    global_step, epoch = 0, 0
    
    if os.path.exists(CHECKPOINT_PATH):
        print("Loading checkpoint...")
        ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['opt'])
        global_step, epoch = ckpt.get('step', 0), ckpt.get('epoch', 0)

    files = glob.glob(os.path.join(folder, "**/*.parquet"), recursive=True)
    if not files: print(f"No parquet files found in {folder}!"); return
    streamer = SequentialRotatingStreamer(files, text_col, tokenizer, SEQ_LEN, BATCH_SIZE)

    current_file_str = "Init"
    session_has_started = False 

    print(f"Starting SEASHORE Training. Ratio BP:{BP_RATIO} / Hebbian:{10-BP_RATIO}")

    while True:
        for xb, yb, file_info in streamer:
            if xb is None:
                if session_has_started:
                    print(f"End of file reached. Saving...")
                    torch.save({'model': model.state_dict(), 'opt': optimizer.state_dict(), 'step': global_step, 'epoch': epoch}, CHECKPOINT_PATH)
                session_has_started = True
                current_file_str = file_info
                gc.collect(); torch.cuda.empty_cache()
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
                    # We run forward to generate "waves" and trigger hebbian_update internally
                    # Pass hebbian_lr to activate the logic inside HoloGraphBlockV7
                    logits, layer_data = model(xb, archive=archive, hebbian_lr=HEBBIAN_LR)
                    
                    # Calculate loss just for monitoring (not backward)
                    loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), yb.view(-1))
                mode_str = "HB"

            # Archive Snapshot
            if global_step % SNAPSHOT_RATE == 0:
                mem, s_box = layer_data[0]
                archive.add(s_box, mem)
            
            # Preview
            if global_step % PREDICT_EVERY == 0:
                model.eval()
                with torch.no_grad():
                    context = xb[:1, :10] 
                    print(f"\n\n--- Step {global_step} [{mode_str}] Preview ---")
                    print(f"Prompt: {tokenizer.decode(context[0].tolist())}")
                    print("Output: ", end="", flush=True)
                    gen_tokens = context
                    for _ in range(20): 
                        # Inference always uses normal forward (hebbian_lr=None)
                        with torch.cuda.amp.autocast():
                            lg, _ = model(gen_tokens, archive=archive)
                        probs = F.softmax(lg[:, -1, :] / 0.8, dim=-1)
                        next_token = torch.multinomial(probs, 1)
                        print(tokenizer.decode([next_token.item()]), end="", flush=True)
                        gen_tokens = torch.cat([gen_tokens, next_token], dim=1)
                        if gen_tokens.shape[1] > SEQ_LEN: gen_tokens = gen_tokens[:, 1:]
                    print("\n" + "-"*40 + "\n")

            global_step += 1
            stdout.write(f'\r[{mode_str}][File {current_file_str}] Step: {global_step} | Loss: {loss.item():.4f}')

        epoch += 1
        print(f"\nEpoch {epoch} finished. Saving...")
        torch.save({'model': model.state_dict(), 'opt': optimizer.state_dict(), 'step': global_step, 'epoch': epoch}, CHECKPOINT_PATH)
		
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, default="data")
    parser.add_argument("--column", type=str, default="text")
    args = parser.parse_args()
    train(args.folder, args.column)