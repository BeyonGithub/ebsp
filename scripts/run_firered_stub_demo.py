# -*- coding: utf-8 -*-
"""
演示如何使用 FireRedClientStub 一键处理长音频（使用假模型推理）。
"""
import os, json, argparse

import os, sys
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_ROOT = os.path.dirname(THIS_DIR)
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

from segmentation.inference.firered_client_stub import FireRedClientStub, run_on_wav

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", default="data/audio/demo.wav")
    ap.add_argument("--minutes", type=float, default=60.0)
    ap.add_argument("--devices", default="0,1")
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--out", default="data/sample/llm_stub_output.json")
    args = ap.parse_args()

    client = FireRedClientStub(devices=args.devices, batch_size=args.batch_size, workers=args.workers)
    out = run_on_wav(client, wav=args.wav, minutes=args.minutes, out_json=args.out)
    print("[OK] 输出:", args.out, " 段数:", len(out["segments"]), " 文本前80:", out["text"][:80])
