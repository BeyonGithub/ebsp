# -*- coding: utf-8 -*-
"""
scripts/run_firered_cli_onnx_demo.py
- 演示如何用 ONNX Encoder inproc 跑 CPU 推理（不改原 CLI）
"""
import argparse, logging, os, json
from segmentation.inference.firered_client_stub import FireRedClientStub, run_on_wav
from segmentation.inference.firered_client_inproc_onnx import FireRedInprocOnnxRunner

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True)
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--onnx_encoder", required=True)
    ap.add_argument("--minutes", type=int, default=None)
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--log", default="INFO")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    logging.basicConfig(level=getattr(logging, args.log.upper(), logging.INFO), format="%(asctime)s %(levelname)s %(message)s")
    # 用 stub 切片
    tmp_client = FireRedClientStub(seg_sec=None, overlap_sec=None)  # 会从 YAML 读取
    batches = tmp_client.build_batches(args.wav, minutes=args.minutes)

    # ONNX inproc runner
    runner = FireRedInprocOnnxRunner(asr_type="aed", model_dir=args.model_dir, onnx_encoder=args.onnx_encoder, device="cpu", batch=args.batch)
    infer_core = lambda items, dev=None: runner.infer_items(items)

    # 复用现有的 run_on_wav（会自动 stitch & refine_output）
    out = run_on_wav(tmp_client, wav=args.wav, minutes=args.minutes, out_json=args.out, infer_override=infer_core)
    print(json.dumps(out, ensure_ascii=False)[:2000])
    if args.out:
        print("[OK] saved:", args.out)
