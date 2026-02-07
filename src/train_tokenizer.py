import sentencepiece as spm
from pathlib import Path

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/spm")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    inputs = sorted(RAW_DIR.glob("*.txt"))
    if not inputs:
        raise SystemExit("No .txt files found in data/raw")
    
    input_arg = ",".join(str(p) for p in inputs)
    model_prefix = str(OUT_DIR / "spm_unigram_8k")
    spm.SentencePieceTrainer.Train(
        input=input_arg,
        model_prefix=model_prefix,
        vocab_size=8000,
        model_type="unigram",
        character_coverage=1.0,   # mixto ES/EN ok
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        user_defined_symbols=[],
        train_extremely_large_corpus=False
    )

    print("Tokenizer saved to:", model_prefix + ".model")

if __name__ == "__main__":
    main()