# -*- coding: utf-8 -*-
"""
scripts/run_firered_inproc_demo.py
- 进程内推理 demo：一次加载模型，批量处理所有切片
- 默认 AED；如需 LLM，请确保模型目录包含 LLM 依赖（与 CLI 一致）
"""
import os, sys, argparse, json, time, logging

THIS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(THIS)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# 允许把 FireRed 仓库加入 sys.path（若你不是 pip 安装）
def _maybe_add_repo_to_syspath(repo_path: str):
    if repo_path and os.path.isdir(repo_path):
        if repo_path not in sys.path:
            sys.path.insert(0, repo_path)

from segmentation.inference.firered_client_stub import FireRedClientStub, run_on_wav
from segmentation.inference.firered_client_inproc import FireRedInprocRunner

def wrap_infer_with_timing(infer_fn, total_slices):
    done = {"n":0, "t0": time.time()}
    def _wrapped(items, dev):
        res = infer_fn(items)
        done["n"] += len(items)
        dt = time.time() - done["t0"]
        rate = done["n"]/dt if dt>0 else 0.0
        remain = max(0, total_slices - done["n"])
        eta = remain / rate if rate>0 else float("inf")
        logging.info("[Inproc] progress=%d/%d  elapsed=%.1fs  ETA=%.1fs  rate=%.2f seg/s",
                     done["n"], total_slices, dt, eta, rate)
        return res
    return _wrapped

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True)
    ap.add_argument("--minutes", type=float, default=None)
    ap.add_argument("--repo", default="asr/FireRedASR", help="FireRedASR 仓库路径，不是 pip 装的话需要加到 sys.path")
    ap.add_argument("--asr_type", choices=["aed","llm"], default="aed")
    ap.add_argument("--model_dir", required=True, help="FireRed 模型目录")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--beam_size", type=int, default=1)
    ap.add_argument("--decode_max_len", type=int, default=0)
    ap.add_argument("--seg_sec", type=float, default=60.0)
    ap.add_argument("--overlap_sec", type=float, default=0.0)
    ap.add_argument("--workers", type=int, default=0, help="切片阶段的并行；推理阶段为进程内 batch")
    ap.add_argument("--no_ffmpeg", action="store_true")
    ap.add_argument("--out", default="data/sample/firered_inproc_output.json")
    ap.add_argument("--log", default="INFO", choices=["DEBUG","INFO","WARNING","ERROR"])
    args = ap.parse_args()

    logging.basicConfig(level=getattr(logging, args.log),
                        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    _maybe_add_repo_to_syspath(args.repo)

    # 先预切片，拿到 slices
    tmp_client = FireRedClientStub(devices="0", workers=0, batch_size=1,
                                   no_ffmpeg=args.no_ffmpeg, use_threads=True, infer_fn=lambda a,b:[],
                                   seg_sec=args.seg_sec, overlap_sec=args.overlap_sec)
    batches = tmp_client.build_batches(args.wav, minutes=args.minutes)
    total_slices = sum(len(b) for b in batches)
    logging.info("Planned slices: %d (seg=%.1fs overlap=%.1fs)", total_slices, args.seg_sec, args.overlap_sec)

    # 创建进程内 Runner
    runner = FireRedInprocRunner(asr_type=args.asr_type, model_dir=args.model_dir,
                                 device=args.device, batch=args.batch,
                                 beam_size=args.beam_size, decode_max_len=args.decode_max_len)

    # 包装为 FireRedClientStub 的 infer_fn 接口
    infer_fn = wrap_infer_with_timing(lambda items: runner.infer_items(items), total_slices)

    client = FireRedClientStub(devices="0", workers=args.workers, batch_size=total_slices,
                               no_ffmpeg=args.no_ffmpeg, use_threads=True, infer_fn=lambda items, dev: infer_fn(items),
                               seg_sec=args.seg_sec, overlap_sec=args.overlap_sec)

    t0 = time.time()
    out = run_on_wav(client, wav=args.wav, minutes=args.minutes, out_json=args.out)
    logging.info("Done. Output=%s  Preview=%s  TotalElapsed=%.1fs", args.out, out["text"][:120], time.time()-t0)
    print(json.dumps({"ok": True, "out": args.out}, ensure_ascii=False))
