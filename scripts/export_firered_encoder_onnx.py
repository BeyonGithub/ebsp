import os
import sys
import time
import argparse
import logging
from typing import List

import torch
import torch.nn as nn


def _try_add_repo_to_syspath(repo: str = None):
    candidates: List[str] = []
    if repo:
        candidates.append(repo)
    # common layout
    candidates += [
        "asr/FireRedASR",
        "asr\\FireRedASR",
        "FireRedASR",
        "../asr/FireRedASR",
        "../../asr/FireRedASR",
    ]
    root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    for c in candidates:
        p = c
        if not os.path.isabs(p):
            p = os.path.abspath(os.path.join(root, p))
        if os.path.exists(p):
            if p not in sys.path:
                sys.path.insert(0, p)
            logging.info("[Export] repo added to sys.path: %s", p)
            return p
    raise RuntimeError("FireRedASR repo not found; try --repo")

def build_encoder_wrapper(aed):
    class EncoderWrapper(nn.Module):
        def __init__(self, enc: nn.Module):
            super().__init__()
            self.encoder = enc
        def forward(self, padded_input: torch.Tensor, input_lengths: torch.Tensor):
            # original AED encoder returns (enc_out, enc_len, enc_mask)
            enc_out, enc_len, _enc_mask = self.encoder(padded_input, input_lengths)
            return enc_out, enc_len
    return EncoderWrapper(aed.encoder)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--repo", default=None)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda", "cuda:0"])
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--feat_dim", type=int, default=80)
    ap.add_argument("--feat_hz", type=int, default=100)
    ap.add_argument("--sample_sec", type=float, default=20.0)
    ap.add_argument("--log", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(level=getattr(logging, args.log.upper(), logging.INFO),
                        format="%(asctime)s %(levelname)s %(message)s")

    _try_add_repo_to_syspath(args.repo)

    from fireredasr.models.fireredasr import FireRedAsr

    t0 = time.time()
    asr = FireRedAsr.from_pretrained("aed", args.model_dir)
    aed = asr.model
    logging.info("[Export] load FireRed AED done in %.2fs", time.time()-t0)

    wrapper = build_encoder_wrapper(aed).to(args.device)

    # Dummy inputs for tracing
    B = 1
    T = int(args.sample_sec * args.feat_hz)
    F = args.feat_dim
    x = torch.randn(B, T, F, dtype=torch.float32, device=args.device)
    ilen = torch.tensor([T], dtype=torch.int64, device=args.device)

    # Dynamic axes
    dynamic_axes = {
        "padded_input": {0: "B", 1: "T"},
        "input_lengths": {0: "B"},
        "enc_out": {0: "B", 1: "Tprime"},
        "enc_len": {0: "B"},
    }

    t1 = time.time()
    logging.info("[Export] exporting to ONNX: %s", args.out)
    torch.onnx.export(
        wrapper, (x, ilen), args.out,
        input_names=["padded_input", "input_lengths"],
        output_names=["enc_out", "enc_len"],
        dynamic_axes=dynamic_axes,
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
    )
    logging.info("[Export] done in %.2fs", time.time()-t1)
    logging.info("[Export] total time %.2fs", time.time()-t0)

if __name__ == "__main__":
    main()
