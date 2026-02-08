import torch
import sentencepiece as spm
from pathlib import Path
from model import TinyLLM

SPM_MODEL = Path("data/spm/spm_unigram_8k.model")
CKPT = Path("checkpoints/tinygpt_step_3500.pt")

D_MODEL = 256
N_LAYERS = 4
N_HEADS = 4
D_FF = 1024
DROPOUT = 0.1
DEVICE = "cpu"

@torch.no_grad()
def generate(model, idx, max_new_tokens=150, temperature=0.9, top_k=50):
    model.eval()
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.ctx:]   # ✅ usa el ctx del checkpoint
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / temperature

        if top_k is not None:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float("inf")

        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_id], dim=1)
    return idx

def main():
    sp = spm.SentencePieceProcessor()
    sp.load(str(SPM_MODEL))

    if not CKPT.exists():
        print("Checkpoint not found; sampling random weights will be nonsense.")
        return

    ckpt = torch.load(CKPT, map_location=DEVICE)

    cfg = ckpt.get("config")
    if cfg is None:
        raise SystemExit(
            "Checkpoint has no 'config'. Point CKPT to a step checkpoint like "
            "'checkpoints/tinyllm_step_1500.pt' that includes config."
        )

    # Instancia el modelo con la MISMA config del entrenamiento (incluye ctx correcto)
    model = TinyLLM(
        cfg["vocab_size"],
        cfg["ctx"],
        cfg["d_model"],
        cfg["n_layers"],
        cfg["n_heads"],
        cfg["d_ff"],
        cfg["dropout"],
    ).to(DEVICE)

    model.load_state_dict(ckpt["model"], strict=True)

    prompt = "Hola, today we will talk about"
    ids = sp.encode(prompt, out_type=int)
    idx = torch.tensor([ids], dtype=torch.long, device=DEVICE)

    out = generate(model, idx, max_new_tokens=150, temperature=0.7, top_k=30)
    text = sp.decode(out[0].tolist())
    print(text)

if __name__ == "__main__":
    main()