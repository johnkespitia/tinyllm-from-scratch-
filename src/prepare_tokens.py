from pathlib import Path
import numpy as np
import sentencepiece as spm
from tqdm import tqdm

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/tokens")
OUT_DIR.mkdir(parents=True, exist_ok=True)
SPM_MODEL = Path("data/spm/spm_unigram_8k.model")

def main():
    if not SPM_MODEL.exists():
        raise SystemExit(f"SPM model not found at {SPM_MODEL}. Run train_tokenizer.py first.")
    
    sp = spm.SentencePieceProcessor()
    sp.load(str(SPM_MODEL))

    all_ids = []
    files = sorted(RAW_DIR.glob("*.txt"))
    if not files:
        raise SystemExit("No .txt files found in data/raw")
    
    for f in tqdm(files, desc="Tokenizing files"):
        text = f.read_text(encoding="utf-8", errors="ignore")
        ids = sp.encode(text, out_type=int)
        all_ids.extend(ids + [sp.eos_id()])  
    
    arr = np.array(all_ids, dtype=np.int32)
    out_path = OUT_DIR / "train_ids.bin"
    arr.tofile(out_path)
    print(f"Tokenized data saved to {out_path} (total tokens: {len(all_ids)})")

if __name__ == "__main__":
    main()
