from pathlib import Path
import math
import numpy as np
import torch
from torch.optim import AdamW
from tqdm import tqdm

from model import TinyLLM

DATA_BIN = Path("data/tokens/train_ids.bin")
CKPT_DIR = Path("checkpoints")
CKPT_DIR.mkdir(exist_ok=True)

# Config (CPU-friendly)
VOCAB_SIZE = 8000
CTX = 128
D_MODEL = 256
N_LAYERS = 4
N_HEADS = 4
D_FF = 1024
DROPOUT = 0.1

BATCH_SIZE = 4           # micro-batch
GRAD_ACCUM = 4           # batch efectivo = 64 secuencias
LR = 3e-4
WEIGHT_DECAY = 0.1
MAX_STEPS = 2000       # sube luego
WARMUP_STEPS = 100
CLIP = 1.0
DEVICE = "cpu"

def get_batch(data, ctx, batch_size, device):
    # ix = torch.randint(0, len(data) - ctx - 1, (batch_size,))
    # x = torch.stack([torch.from_numpy(data[i:i+ctx]) for i in ix]).to(device)
    # y = torch.stack([torch.from_numpy(data[i+1:i+ctx+1].copy()) for i in ix]).to(device)
    # return x.long(), y.long()
    max_start = len(data) - ctx - 1
    if max_start <= 0:
        raise ValueError(
            f"Not enough tokens ({len(data)}) for context size ({ctx})"
        )

    ix = torch.randint(0, max_start, (batch_size,))
    x = torch.stack([
        torch.from_numpy(data[i:i+ctx].copy())
        for i in ix
    ]).to(device)

    y = torch.stack([
        torch.from_numpy(data[i+1:i+ctx+1].copy())
        for i in ix
    ]).to(device)

    return x.long(), y.long()

def lr_schedule(step):
    if step < WARMUP_STEPS:
        return LR * step / WARMUP_STEPS
    
    progress = (step - WARMUP_STEPS) / max(1, (MAX_STEPS - WARMUP_STEPS))
    return 0.1 * LR + 0.9 * LR * 0.5 * (1.0 + math.cos(math.pi * progress))


def main():
    if not DATA_BIN.exists():
        raise SystemExit(f"Data file not found at {DATA_BIN}. Run prepare_tokens.py first.")
    
    data = np.fromfile(DATA_BIN, dtype=np.int32)
    print("DATA LEN:", len(data))
    print("CTX:", CTX)
    print("BATCH_SIZE:", BATCH_SIZE)
    assert len(data) > CTX + 1, "Dataset too small for chosen context"
    if len(data) < 1_000_000:
        print(f"Warning: small dataset ({len(data)} tokens). Training may overfit.")


    model = TinyLLM(VOCAB_SIZE, CTX, D_MODEL, N_LAYERS, N_HEADS, D_FF, DROPOUT).to(DEVICE)
    opt = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    model.train()
    opt.zero_grad(set_to_none=True)

    pbar = tqdm(range(MAX_STEPS), desc="Training")
    for step in pbar:
        lr = lr_schedule(step)
        for g in opt.param_groups:
            g["lr"] = lr

        loss_accum = 0.0
        for _ in range(GRAD_ACCUM):
            x, y = get_batch(data, CTX, BATCH_SIZE, DEVICE)
            _, loss = model(x, y)
            (loss / GRAD_ACCUM).backward()
            loss_accum += loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        opt.step()
        opt.zero_grad(set_to_none=True)

        pbar.set_postfix({"loss": f"{loss_accum/GRAD_ACCUM:.4f}", "lr": f"{lr:.2e}"})

        if (step + 1) % 500 == 0:
            ckpt = {
                "model": model.state_dict(),
                "step": step + 1,
                "config": dict(
                    vocab_size=VOCAB_SIZE, ctx=CTX, d_model=D_MODEL,
                    n_layers=N_LAYERS, n_heads=N_HEADS, d_ff=D_FF, dropout=DROPOUT
                )
            }
            torch.save(ckpt, CKPT_DIR / f"tinygpt_step_{step+1}.pt")
    
    torch.save(model.state_dict(), CKPT_DIR / "tinygpt_last.pt")
    print("Training complete. Final model saved to:", CKPT_DIR / "tinygpt_last.pt")

if __name__ == "__main__":
    main()