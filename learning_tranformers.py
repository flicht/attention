# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "numpy",
#     "torch",
#     "tqdm",
#     "wandb",
# ]
# ///


import torch
import torch.nn as nn
from torch.nn import functional as F
import wandb
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'
epochs=10000

print(f"Using device: {device}")

# Setup
with open('pg52882.txt') as f:
  text = f.read()

print(f"Number of characters: {len(text)}")
characters = sorted(list(set(text)))
vocab_size = len(characters)
print(f"Unique characters: {''.join(characters)}")

stoi = {j:i for i,j in enumerate(characters)}
itos = {i:j for i,j in enumerate(characters)}

encode = lambda x: [stoi[j] for j in x]
decode = lambda x: ''.join([itos[j] for j in x])

# Cut of the end of the data
full_data = torch.tensor(encode(text), dtype=torch.long)
end=0.9
full_data_length = len(full_data)
data = full_data[int(full_data_length*0.9):]

# Reserve some for validation
n = int(0.9*len(data))
train_data = data[:n]
validation_data = data[n:]

# Vars
n_embd = 256
n_layers = 12
n_heads = 6
dropout = 0.3
learning_rate = 1e-5

batch_size = 184
block_size = 256

def get_batch(split):
  data = train_data if split == 'train' else validation_data
  ix = torch.randint(len(data) - block_size, (batch_size, ))
  x = torch.stack([data[i:i + block_size] for i in ix])
  y = torch.stack ([data[i+1:i+block_size+1] for i in ix])
  return x, y

class Head(nn.Module):
  def __init__(self, head_size):
    super().__init__()
    self.key = nn.Linear(n_embd, head_size, bias=False)
    self.query = nn.Linear(n_embd, head_size, bias=False)
    self.value = nn.Linear(n_embd, head_size, bias=False)
    self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

  def forward(self, x):
    B, T, C = x.shape
    k = self.key(x) # (B, T, head_size)
    q = self.query(x) # (B, T, head_size)
    v = self.value(x) # (B, T, head_size)

    wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5
    wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
    wei = F.softmax(wei, dim=-1)

    out = wei @ v
    return out

class MultiHeadAttention(nn.Module):
  def __init__(self, n_heads, head_size):
    super().__init__()
    self.heads = nn.ModuleList([Head(head_size) for _ in range(n_heads)])
    self.proj = nn.Linear(n_heads * head_size, n_embd)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    out = torch.cat([h(x) for h in self.heads], dim=-1)
    out = self.proj(out)
    return self.dropout(out)  # (B, T, C)

class FeedForward(nn.Module):
  def __init__(self, n_embd):
    super().__init__()
    self.net = nn.Sequential(
        nn.Linear(n_embd, 4 * n_embd),
        nn.ReLU(),
        nn.Linear(4 * n_embd, n_embd),
        nn.Dropout(dropout)  # Add dropout for regularization
    )

  def forward(self, x):
    return self.net(x)

class Block(nn.Module):
  def __init__(self, n_embd, n_heads):
    super().__init__()
    head_size = n_embd // n_heads
    self.sa_heads = MultiHeadAttention(n_heads, head_size)
    self.ffwd = FeedForward(n_embd)
    self.ln1 = nn.LayerNorm(n_embd)
    self.ln2 = nn.LayerNorm(n_embd)

  def forward(self, x):
    x = self.sa_heads(self.ln1(x)) + x # skip connection
    x = self.ffwd(self.ln2(x)) + x # skip connection
    return x

class BigramLanguageModel(nn.Module):
  def __init__(self):
    super().__init__()
    self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
    self.position_embedding_table = nn.Embedding(block_size, n_embd)
    self.blocks = nn.Sequential(
        *[Block(n_embd, n_heads=n_heads) for _ in range(n_layers)],
        nn.LayerNorm(n_embd)  # final layer norm
    )
    self.lm_head = nn.Linear(n_embd, vocab_size)

  def forward(self, idx, targets=None):
    B, T = idx.shape

    tok_emb = self.token_embedding_table(idx) # (B, T, C)
    pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, C)
    x = tok_emb + pos_emb # (B, T, C)
    x=self.blocks(x) # (B, T, C)
    # (B, T, C) -> (B, T, vocab_size)
    logits = self.lm_head(x)

    if targets is None:
      loss = None
    else:

      B, T, C = logits.shape
      logits = logits.view(B*T, C)
      targets = targets.view(B*T)
      loss = F.cross_entropy(logits, targets)

    return logits, loss

  def generate(self, idx, max_new_tokens):
    for _ in range(max_new_tokens):
      idx_cond = idx[:, -block_size:] # (B, T)
      logits, loss = self(idx_cond)
      logits = logits[:, -1, :]
      probs = F.softmax(logits, dim=-1)
      idx_next = torch.multinomial(probs, num_samples=1)
      idx = torch.cat((idx, idx_next), dim=1)

    return idx

m = BigramLanguageModel().to(device)
m = torch.compile(m)


optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate, weight_decay=1e-2)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=epochs,       # number of steps to complete 1 cosine cycle
    eta_min=1e-6        # final learning rate after decay
)


run = wandb.init(
    project="my-awesome-project",    # Specify your project
    config={                         # Track hyperparameters and metadata
        "learning_rate": learning_rate,
        "epochs": epochs,
    },
)

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

    wandb.log({"loss": loss.item(), "step": step, "lr": scheduler.get_last_lr()[0]})

    if step % 100 == 0:
        print(f"Step {step}, loss: {loss.item():.4f}")
        with torch.no_grad():
            m.eval()
            xv, yv = get_batch('validation')
            xv, yv = xv.to(device), yv.to(device)

            val_logits, val_loss = m(xv, yv)
            wandb.log({"step": step, "val_loss": val_loss.item()})
            print(f"Validation loss: {val_loss.item():.4f}")

        m.train()  # <--- Important: switch back to train mode after eval
    
    if step % 500 == 0 and step > 0:
        val_loss_value = val_loss.item()
        checkpoint_path = f"model_val{val_loss_value:.4f}_step{step}.pt"
        torch.save(m.state_dict(), checkpoint_path)
        wandb.save(checkpoint_path)
        wandb.log({
            "checkpoint_saved_step": step,
            "checkpoint_val_loss": val_loss_value,
            "checkpoint_path": checkpoint_path
        })
        print(f"Saved checkpoint: {checkpoint_path}")

print(f"Final training loss: {loss.item():.4f}")



idx = torch.zeros((1, 1), dtype=torch.long, device=device)
with torch.no_grad():
  print(decode(m.generate(idx, max_new_tokens=500)[0].tolist()))


