# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "numpy",
#     "torch",
#     "tqdm",
#     "wandb",
#     "tiktoken"
# ]
# ///

import torch
import torch.nn as nn
from torch.nn import functional as F
import wandb
import tiktoken
from tqdm import tqdm

# Hyperparams
epochs = 10000
n_embd = 256
n_layers = 12
n_heads = 6
dropout = 0.3
learning_rate = 1e-4
block_size = 256
batch_size = 44
device = 'cuda' if torch.cuda.is_available() else 'cpun'

print(f"Using device: {device}")

# === Tokenization and Dataset ===
enc = tiktoken.get_encoding("gpt2")  # or cl100k_base
encode = lambda s: enc.encode(s, disallowed_special=())
decode = lambda t: enc.decode(t)
vocab_size = enc.n_vocab

with open('pg52882.txt') as f:
    text = f.read()

full_data = torch.tensor(encode(text), dtype=torch.long)
print(f"Tokens: {len(full_data)}, Vocab size: {vocab_size}")

data = full_data[int(len(full_data)*0.9):]
n = int(0.9 * len(data))
train_data = data[:n]
validation_data = data[n:]

def get_batch(split):
    data = train_data if split == 'train' else validation_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i+1:i + block_size + 1] for i in ix])
    return x, y

# === Transformer Model ===

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        return wei @ v

class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_heads)])
        self.proj = nn.Linear(n_heads * head_size, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.proj(torch.cat([h(x) for h in self.heads], dim=-1)))

class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        head_size = n_embd // n_heads
        self.sa_heads = MultiHeadAttention(n_heads, head_size)
        self.ffwd = FeedForward()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa_heads(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block() for _ in range(n_layers)], nn.LayerNorm(n_embd))
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = self.blocks(tok_emb + pos_emb)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits / 0.9, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# === Training Setup ===

m = BigramLanguageModel().to(device)
# m = torch.compile(m) 

optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate, weight_decay=1e-2)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

run = wandb.init(project="my-awesome-project", config={
    "learning_rate": learning_rate,
    "epochs": epochs,
    "dropout": dropout,
    "n_layers": n_layers,
    "n_embd": n_embd,
    "block_size": block_size,
    "tokenizer": "gpt2"
})

wandb.watch(m)

best_val_loss = float('inf')

def evaluate_val_loss(num_batches=5):
    m.eval()
    losses = []
    with torch.no_grad():
        for _ in range(num_batches):
            xb, yb = get_batch('validation')
            xb, yb = xb.to(device), yb.to(device)
            _, loss = m(xb, yb)
            losses.append(loss.item())
    m.train()
    return sum(losses) / len(losses)

# === Training Loop ===

m.train()
for step in (pbar := tqdm(range(epochs))):
    xb, yb = get_batch('train')
    xb, yb = xb.to(device), yb.to(device)

    logits, loss = m(xb, yb)
    pbar.set_description(f"loss: {loss.item():.4f}")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    scheduler.step()

    wandb.log({
        "loss": loss.item(),
        "step": step,
        "lr": scheduler.get_last_lr()[0]
    })

    if step % 100 == 0:
        val_loss_value = evaluate_val_loss()
        wandb.log({"val_loss": val_loss_value, "step": step})
        print(f"Validation loss: {val_loss_value:.4f}")

        if val_loss_value < best_val_loss:
            best_val_loss = val_loss_value
            torch.save(m.state_dict(), "best_model.pt")
            wandb.save("best_model.pt")
            print("✅ New best model saved.")

    if step % 500 == 0 and step > 0:
        torch.save(m.state_dict(), f"model_step{step}.pt")
        wandb.save(f"model_step{step}.pt")

print(f"Final training loss: {loss.item():.4f}")

# === Generate Sample ===

idx = torch.zeros((1, 1), dtype=torch.long, device=device)
with torch.no_grad():
    m.eval()
    generated = decode(m.generate(idx, max_new_tokens=500)[0].tolist())
    print("\nSample output:\n" + generated)