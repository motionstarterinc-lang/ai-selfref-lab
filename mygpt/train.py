"""Phase 6, step 2: train your GPT.

    python mygpt/train.py                   # 'cpu' preset: ~1.3M params, ~1 h on a laptop CPU
    python mygpt/train.py --preset gpu      # ~12M params, for Colab / an NVIDIA GPU
    python mygpt/train.py --max-iters 200   # quick smoke test

Writes mygpt/out/ckpt.pt (best val loss) and mygpt/out/log.csv. Ctrl+C stops safely; the best checkpoint is kept.
"""
import argparse
import csv
import math
import time
from pathlib import Path

import numpy as np
import torch

from model import GPT, GPTConfig, save

HERE = Path(__file__).resolve().parent
PRESETS = {
    "cpu": dict(n_layer=4, n_head=4, n_embd=128, block_size=256, batch=32, lr=2e-3, max_iters=4000),
    "gpu": dict(n_layer=6, n_head=6, n_embd=384, block_size=256, batch=64, lr=1e-3, max_iters=6000),
}


def get_batch(data, block, batch, device):
    ix = np.random.randint(0, len(data) - block - 1, batch)
    x = torch.from_numpy(np.stack([data[i:i + block] for i in ix]).astype(np.int64))
    y = torch.from_numpy(np.stack([data[i + 1:i + 1 + block] for i in ix]).astype(np.int64))
    return x.to(device), y.to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=PRESETS, default="cpu")
    ap.add_argument("--max-iters", type=int)
    ap.add_argument("--eval-every", type=int, default=200)
    a = ap.parse_args()
    p = PRESETS[a.preset] | ({"max_iters": a.max_iters} if a.max_iters else {})

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(1337)
    np.random.seed(1337)
    train = np.memmap(HERE / "data" / "train.bin", dtype=np.uint16, mode="r")
    val = np.memmap(HERE / "data" / "val.bin", dtype=np.uint16, mode="r")
    from tokenizers import Tokenizer
    vocab = Tokenizer.from_file(str(HERE / "data" / "tokenizer.json")).get_vocab_size()

    cfg = GPTConfig(vocab_size=vocab, block_size=p["block_size"], n_layer=p["n_layer"], n_head=p["n_head"], n_embd=p["n_embd"])
    model = GPT(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=p["lr"], betas=(0.9, 0.95), weight_decay=0.1)
    warmup, total = 100, p["max_iters"]
    print(f"{model.n_params() / 1e6:.2f}M params · {device} · {len(train):,} train tokens · {total} steps")

    out = HERE / "out"
    out.mkdir(exist_ok=True)
    log = open(out / "log.csv", "w", newline="")
    w = csv.writer(log)
    w.writerow(["step", "train_loss", "val_loss", "lr", "elapsed_s"])

    @torch.no_grad()
    def evaluate():
        model.eval()
        losses = [model(*get_batch(val, cfg.block_size, p["batch"], device))[1].item() for _ in range(20)]
        model.train()
        return sum(losses) / len(losses)

    best, t0, step = float("inf"), time.time(), 0
    try:
        for step in range(total + 1):
            lr = p["lr"] * min(1, (step + 1) / warmup) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / total)))
            for g in opt.param_groups:
                g["lr"] = lr
            if step % a.eval_every == 0 or step == total:
                vl = evaluate()
                w.writerow([step, f"{loss.item():.4f}" if step else "", f"{vl:.4f}", f"{lr:.2e}", int(time.time() - t0)])
                log.flush()
                if vl < best:
                    best = vl
                    save(model, out / "ckpt.pt", step, vl)
                el = time.time() - t0
                eta = el / max(step, 1) * (total - step)
                print(f"step {step:>5}/{total}  val loss {vl:.3f}  best {best:.3f}  {el / 60:.1f} min elapsed  ~{eta / 60:.0f} min left")
            x, y = get_batch(train, cfg.block_size, p["batch"], device)
            _, loss = model(x, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    except KeyboardInterrupt:
        print("\nStopped early. Best checkpoint is saved.")
    log.close()
    print(f"Best val loss {best:.3f} -> {out / 'ckpt.pt'}. Try: python mygpt/sample.py")


if __name__ == "__main__":
    main()
