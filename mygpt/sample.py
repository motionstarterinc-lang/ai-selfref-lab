"""Phase 6, step 3: make your GPT write.   python mygpt/sample.py "Once upon a time" """
import sys
from pathlib import Path

import torch
from tokenizers import Tokenizer

from model import load

HERE = Path(__file__).resolve().parent
tok = Tokenizer.from_file(str(HERE / "data" / "tokenizer.json"))
model = load(HERE / "out" / "ckpt.pt")
prompt = sys.argv[1] if len(sys.argv) > 1 else "Once upon a time"
idx = torch.tensor([tok.encode(prompt).ids])
for i in range(3):
    out = model.generate(idx, 120, eos_id=tok.token_to_id("<|end|>"))
    print(f"--- sample {i + 1} ---\n{tok.decode(out[0].tolist())}\n")
