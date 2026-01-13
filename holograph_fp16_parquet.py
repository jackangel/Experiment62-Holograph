import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.checkpoint import checkpoint
import os, glob, random, math
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
CHECKPOINT_PATH = 'holograph_v7_integrated.pth'

# Control Rates
ARCHIVE_SIZE = 4096 
SNAPSHOT_RATE = 128   
PREDICT_EVERY = 1000  

# --- 2. Tokenizer (Limited Tiktoken) ---
class LimitedTokenizer:
    def __init__(self, limit=32768):
        self.enc = tiktoken.get_encoding("cl100k_base")
        self.limit = limit
        self.unk_id = limit - 1 
    def encode(self, text):
        return [t if t < self.unk_id else self.unk_id for t in self.enc.encode(text)]
    def decode(self, ids):
        return self.enc.decode([int(i) for i in ids if i < self.unk_id])

# --- 3. Optimized V7 Components ---

def holo_selective_scan_v7(k, q, v, delta, gf, gw):
    B, T, H, D = k.shape
    # Gating and Updates
    v_gated = v * (delta * gw).unsqueeze(-1)
    updates = torch.matmul(v_gated.unsqueeze(-1), k.unsqueeze(-2))
    
    # Log-space stability
    decay = (delta * gf).clamp(min=1e-6, max=0.999)
    log_decay = torch.log(decay).unsqueeze(-1).unsqueeze(-1)
    exp_cum_decay = torch.exp(torch.cumsum(log_decay, dim=1))
    
    # Parallel Scan
    state = torch.cumsum(updates / (exp_cum_decay + 1e-8), dim=1) * exp_cum_decay
    readout = torch.matmul(state, q.unsqueeze(-1)).squeeze(-1)
    return readout, state[:, -1]

class RelationalHoloArchive:
    def __init__(self, size, num_heads, head_dim, embed_dim):
        self.size = size
        self.s_min = torch.zeros(size, embed_dim, dtype=torch.float16, device=DEVICE)
        self.s_max = torch.zeros(size, embed_dim, dtype=torch.float16, device=DEVICE)
        self.values = torch.zeros(size, num_heads, head_dim, head_dim, dtype=torch.float16, device=DEVICE)
        self.ptr = 0; self.count = 0

    def add(self, s_box, matrix):
        with torch.no_grad():
            self.s_min[self.ptr] = s_box[0][:, -1, :].mean(dim=0).detach().to(torch.float16)
            self.s_max[self.ptr] = s_box[1][:, -1, :].mean(dim=0).detach().to(torch.float16)
            self.values[self.ptr] = matrix.mean(dim=0).detach().to(torch.float16)
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
        self.proj_k = nn.Linear(embed_dim, self.total_dim)
        self.proj_q = nn.Linear(embed_dim, self.total_dim)
        self.proj_v = nn.Linear(embed_dim, self.total_dim)
        self.proj_out = nn.Linear(self.total_dim, embed_dim)
        self.proj_delta = nn.Linear(embed_dim, num_heads)
        self.gate_write = nn.Linear(embed_dim, num_heads)
        self.gate_forget = nn.Linear(embed_dim, num_heads)
        self.probe_s = nn.Linear(embed_dim, embed_dim)
        self.ln1, self.ln2 = nn.LayerNorm(embed_dim), nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(nn.Linear(embed_dim, 4*embed_dim), nn.GELU(), nn.Linear(4*embed_dim, embed_dim))

    def forward(self, x, b_min, b_max, motif_context):
        B, T, _ = x.shape
        x_n = self.ln1(x)
        k = F.normalize(self.proj_k(x_n).view(B, T, self.num_heads, self.head_dim), p=2.0, dim=-1)
        q = F.normalize(self.proj_q(x_n).view(B, T, self.num_heads, self.head_dim), p=2.0, dim=-1)
        v = torch.tanh(self.proj_v(x_n).view(B, T, self.num_heads, self.head_dim))
        
        d = F.softplus(self.proj_delta(x_n)).view(B, T, self.num_heads)
        gw = torch.sigmoid(self.gate_write(x_n)).pow(2).view(B, T, self.num_heads)
        gf = (1.0 - torch.sigmoid(self.gate_forget(x_n)).pow(2)).view(B, T, self.num_heads)
        
        m_heads, next_mem = holo_selective_scan_v7(k, q, v, d, gf, gw)
        x = x + self.proj_out(m_heads.reshape(B, T, -1)) + (0.1 * motif_context)
        x = x + self.ffn(self.ln2(x))
        s_box = (b_min + self.probe_s(x), b_max + self.probe_s(x))
        return x, next_mem, s_box

class HoloGraphV7(nn.Module):
    def __init__(self):
        super().__init__()
        self.box_emb = BoxEmbedding(VOCAB_SIZE, EMBED_DIM)
        self.layers = nn.ModuleList([HoloGraphBlockV7(EMBED_DIM, NUM_HEADS, HEAD_DIM) for _ in range(NUM_LAYERS)])
        self.ln_f = nn.LayerNorm(EMBED_DIM)
        self.logit_scale = nn.Parameter(torch.ones(1) * 14.0)
        
    def forward(self, idx, archive=None):
        b_min, b_max = self.box_emb(idx)
        x = (b_min + b_max) / 2.0 
        
        layer_data = []
        for layer in self.layers:
            # Relational Retrieval (Outside Checkpoint for stability)
            motif_ctx = torch.zeros_like(x)
            if archive and archive.count > 20:
                with torch.no_grad():
                    s_min, s_max = b_min + layer.probe_s(x), b_max + layer.probe_s(x)
                    q_min, q_max = s_min.mean(dim=1, keepdim=True), s_max.mean(dim=1, keepdim=True)
                    i_min = torch.max(q_min, archive.s_min[:archive.count].unsqueeze(0).to(x.dtype))
                    i_max = torch.min(q_max, archive.s_max[:archive.count].unsqueeze(0).to(x.dtype))
                    scores = torch.mean(torch.log(F.softplus(i_max - i_min) + 1e-6), dim=-1)
                    w = F.softmax(scores * 2.0, dim=-1).view(x.shape[0], 1, archive.count, 1, 1, 1)
                    fused = torch.sum(w * archive.values[:archive.count].to(x.dtype), dim=2)
                
                qr = F.normalize(layer.proj_q(x).view(x.shape[0], x.shape[1], layer.num_heads, layer.head_dim), p=2.0, dim=-1)
                motif_ctx = layer.proj_out(torch.matmul(fused, qr.unsqueeze(-1)).squeeze(-1).reshape(x.shape[0], x.shape[1], -1))

            if self.training:
                # Use Reentrant=True for Mixed Precision stability
                x, mem, s_box = checkpoint(layer, x, b_min, b_max, motif_ctx, use_reentrant=True)
            else:
                x, mem, s_box = layer(x, b_min, b_max, motif_ctx)
            layer_data.append((mem, s_box))
            
        logits = F.linear(self.ln_f(x), self.box_emb.center.weight) * (self.logit_scale / (EMBED_DIM**0.5))
        return logits, layer_data

# --- 4. Data Pipeline ---
class SequentialRotatingStreamer:
    def __init__(self, file_paths, text_column, tokenizer, seq_len, batch_size):
        self.file_paths = file_paths
        self.text_column = text_column
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.batch_size = batch_size

    def __iter__(self):
        random.shuffle(self.file_paths)
        for file_path in self.file_paths:
            yield None, None, file_path 
            try:
                pf = pq.ParquetFile(file_path)
                batch_x, batch_y = [], []
                for i in range(pf.num_row_groups):
                    table = pf.read_row_group(i, columns=[self.text_column])
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
                                       file_path)
                                batch_x, batch_y = [], []
            except Exception as e:
                print(f"Error reading {file_path}: {e}"); continue

# --- 5. Main Logic ---

def train(folder, text_col):
    tokenizer = LimitedTokenizer(limit=VOCAB_SIZE)
    model = HoloGraphV7().to(DEVICE)
    archive = RelationalHoloArchive(ARCHIVE_SIZE, NUM_HEADS, HEAD_DIM, EMBED_DIM)
    
    # 8-bit Optimizer setup
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
        model.load_state_dict(ckpt['model']); optimizer.load_state_dict(ckpt['opt'])
        global_step, epoch = ckpt.get('step', 0), ckpt.get('epoch', 0)

    files = glob.glob(os.path.join(folder, "**/*.parquet"), recursive=True)
    if not files: print(f"No parquet files found in {folder}!"); return
    streamer = SequentialRotatingStreamer(files, text_col, tokenizer, SEQ_LEN, BATCH_SIZE)

    while True:
        for xb, yb, current_file in streamer:
            if xb is None: continue # Skip file-boundary markers
            
            model.train()
            optimizer.zero_grad(set_to_none=True)
            
            with torch.cuda.amp.autocast():
                logits, layer_data = model(xb, archive=archive)
                loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), yb.view(-1))
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            
            # Archive Snapshot logic
            if global_step % SNAPSHOT_RATE == 0:
                mem, s_box = layer_data[0]
                archive.add(s_box, mem)
            
            # Periodic Prediction (Improved to show 20 tokens)
            if global_step % PREDICT_EVERY == 0:
                model.eval()
                with torch.no_grad():
                    # Take the first 10 tokens of the current batch as a prompt
                    context = xb[:1, :10] 
                    print(f"\n\n--- Step {global_step} Generation Preview ---")
                    print(f"Prompt: {tokenizer.decode(context[0].tolist())}")
                    print("Output: ", end="", flush=True)
                    
                    gen_tokens = context
                    for _ in range(20): # Predict 20 tokens instead of 1
                        with torch.cuda.amp.autocast():
                            lg, _ = model(gen_tokens, archive=archive)
                        
                        # Use sampling instead of argmax for better variety
                        probs = F.softmax(lg[:, -1, :] / 0.8, dim=-1)
                        next_token = torch.multinomial(probs, 1)
                        
                        print(tokenizer.decode([next_token.item()]), end="", flush=True)
                        gen_tokens = torch.cat([gen_tokens, next_token], dim=1)
                        if gen_tokens.shape[1] > SEQ_LEN: gen_tokens = gen_tokens[:, 1:]
                    print("\n" + "-"*40 + "\n")

            global_step += 1
            stdout.write(f'\rStep: {global_step} | Loss: {loss.item():.4f}')

        epoch += 1
        torch.save({'model': model.state_dict(), 'opt': optimizer.state_dict(), 'step': global_step, 'epoch': epoch}, CHECKPOINT_PATH)

def chat():
    tokenizer = LimitedTokenizer(limit=VOCAB_SIZE)
    model = HoloGraphV7().to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE)['model'])
    model.eval()
    archive = RelationalHoloArchive(ARCHIVE_SIZE, NUM_HEADS, HEAD_DIM, EMBED_DIM)
    
    print("\n--- HoloGraph V7 Chat Ready ---")
    while True:
        inp = input("\nYou: ")
        if inp.lower() in ['exit', 'quit']: break
        ids = tokenizer.encode(inp)
        tokens = torch.tensor(ids, dtype=torch.long, device=DEVICE).unsqueeze(0)
        
        print("Bot: ", end="", flush=True)
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                for _ in range(128):
                    logits, layer_data = model(tokens, archive=archive)
                    # Sampling with temperature
                    probs = F.softmax(logits[:, -1, :] / 0.8, dim=-1)
                    curr_token = torch.multinomial(probs, 1)
                    
                    word = tokenizer.decode([curr_token.item()])
                    print(word, end="", flush=True)
                    
                    tokens = torch.cat([tokens, curr_token], dim=1)
                    if tokens.shape[1] > SEQ_LEN: tokens = tokens[:, 1:]
                    if curr_token.item() == 0: break 
        print()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, default="data")
    parser.add_argument("--column", type=str, default="text")
    args = parser.parse_args()

    if os.path.exists(CHECKPOINT_PATH):
        mode = input("Checkpoint found. (C)hat or (T)rain? ").lower()
        if mode == 'c': chat()
        else: train(args.folder, args.column)
    else:
        train(args.folder, args.column)