"""Phase 6, step 1: download TinyStories, train a small BPE tokenizer, save token files.

    python mygpt/prepare.py                  # 100k stories (~20M tokens), good for CPU
    python mygpt/prepare.py --stories 500000 # bigger, for a GPU run

Output in mygpt/data/: tokenizer.json, train.bin, val.bin (uint16 token ids).
"""
import argparse
from pathlib import Path

import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

DATA = Path(__file__).resolve().parent / "data"
EOS = "<|end|>"


def stories(split: str, n: int):
    ds = load_dataset("roneneldan/TinyStories", split=split, streaming=True)
    for i, row in enumerate(ds):
        if i >= n:
            break
        yield row["text"].strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stories", type=int, default=100_000)
    ap.add_argument("--val-stories", type=int, default=2_000)
    ap.add_argument("--vocab", type=int, default=4096)  # small vocab keeps the embedding table small
    a = ap.parse_args()
    DATA.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {a.stories:,} training stories (streaming)...")
    train_text = list(stories("train", a.stories))
    val_text = list(stories("validation", a.val_stories))

    print(f"Training a {a.vocab}-token BPE tokenizer...")
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=a.vocab, special_tokens=[EOS],
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tok.train_from_iterator(train_text, trainer)
    tok.save(str(DATA / "tokenizer.json"))
    eos = tok.token_to_id(EOS)

    for name, texts in (("train", train_text), ("val", val_text)):
        ids = []
        for enc in tok.encode_batch(texts):
            ids.extend(enc.ids)
            ids.append(eos)
        arr = np.array(ids, dtype=np.uint16)
        arr.tofile(DATA / f"{name}.bin")
        print(f"{name}: {len(arr):,} tokens")
    print("Done. Next: python mygpt/train.py")


if __name__ == "__main__":
    main()
