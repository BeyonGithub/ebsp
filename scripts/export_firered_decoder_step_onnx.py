# NOTE: Experimental skeleton for exporting a decoder *step* function to ONNX.
# FireRed AED's decoder uses beam search with autoregressive state; exporting full search
# is complex. A practical approach is to export a single-step function and loop in Python.
# This script inspects the model for a suitable step API and exits with guidance if not found.

import os
import sys
import time
import argparse
import logging

import torch
import torch.nn as nn


def _try_add_repo_to_syspath(repo: str = None):
    candidates = [repo] if repo else []
    candidates += ["asr/FireRedASR", "asr\\FireRedASR", "FireRedASR", "../asr/FireRedASR", "../../asr/FireRedASR"]
    root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    for c in candidates:
        if not c:
            continue
        p = c if os.path.isabs(c) else os.path.abspath(os.path.join(root, c))
        if os.path.exists(p):
            if p not in sys.path:
                sys.path.insert(0, p)
            logging.info("[ExportDec] repo added to sys.path: %s", p)
            return p
    raise RuntimeError("FireRedASR repo not found; try --repo")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--repo", default=None)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--log", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(level=getattr(logging, args.log.upper(), logging.INFO),
                        format="%(asctime)s %(levelname)s %(message)s")

    _try_add_repo_to_syspath(args.repo)
    from fireredasr.models.fireredasr import FireRedAsr

    asr = FireRedAsr.from_pretrained("aed", args.model_dir)
    aed = asr.model
    dec = getattr(aed, "decoder", None)
    if dec is None:
        raise RuntimeError("AED model has no .decoder attribute; cannot export")

    # Try to find a 'step' like function
    step = None
    for name in ["step", "forward_step", "decode_step", "inference_step"]:
        if hasattr(dec, name):
            step = getattr(dec, name)
            logging.info("[ExportDec] using decoder.%s as step function", name)
            break
    if step is None:
        raise RuntimeError("Decoder step API not found. Please implement a wrapper around your decoder to expose a (enc_out, enc_mask, prev_tokens) -> (logits, new_states) API and then export.")

    class StepWrapper(nn.Module):
        def __init__(self, dec: nn.Module):
            super().__init__()
            self.dec = dec
        def forward(self, enc_out, enc_mask, tokens):
            # This forward signature is a placeholder; you may need to adapt to the actual decoder.step signature.
            return self.dec.forward_step(enc_out, enc_mask, tokens)

    wrapper = StepWrapper(dec).eval()
    # Dummy shapes (B=1, T'=200, D=1280), tokens length 1
    enc_out = torch.randn(1, 200, 1280, dtype=torch.float32)
    enc_mask = torch.ones(1, 200, dtype=torch.bool)
    tokens = torch.ones(1, 1, dtype=torch.long)  # last token ids

    torch.onnx.export(
        wrapper, (enc_out, enc_mask, tokens), args.out,
        input_names=["enc_out", "enc_mask", "tokens"],
        output_names=["logits"],
        dynamic_axes={"enc_out": {0: "B", 1: "Tprime"}, "enc_mask": {0: "B", 1: "Tprime"}, "tokens": {0: "B", 1: "U"}, "logits": {0: "B", 1: "U"}},
        export_params=True, opset_version=args.opset, do_constant_folding=True,
    )
    logging.info("[ExportDec] exported to %s", args.out)


if __name__ == "__main__":
    main()
