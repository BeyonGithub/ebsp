# -*- coding: utf-8 -*-
"""
Patched run_firered_cli_demo.py (ROOT->sys.path fix)
- 将工程 ROOT 目录加入 sys.path，修复 `ModuleNotFoundError: No module named 'segmentation'`
- 其余行为同上一版（GPU 优先、自动读 YAML、懒加载标点、独立 metrics、耗时日志）
"""
import os, sys, json, time, argparse, logging
from pathlib import Path

try:
    import yaml
except Exception:
    yaml = None

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent

# >>> NEW: ensure project root is importable so that `segmentation.*` can be found
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
    # NOTE: Logging not configured yet; use print as a safe early message
    print(f"[Path][Early] Added project ROOT to sys.path: {ROOT}", file=sys.stderr)

DEFAULT_FIRED_CLI_YAML = ROOT / "training" / "configs" / "firered_cli.yaml"
DEFAULT_POST_YAML = ROOT / "training" / "configs" / "postprocess.yaml"


def _load_yaml(p: Path, default: dict):
    if yaml is None:
        logging.warning("pyyaml not installed; using defaults for %s", p)
        return default
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or default
    return default


def _detect_device():
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _resolve_repo(repo_arg: str | None, cli_cfg: dict):
    # 1) explicit
    if repo_arg:
        cand = Path(repo_arg)
        if not cand.is_absolute():
            cand = (ROOT / repo_arg).resolve()
        if cand.exists():
            return cand
    # 2) yaml hints
    for key in ("repo", "repo_candidates"):
        if key in cli_cfg:
            v = cli_cfg[key]
            if isinstance(v, str):
                p = Path(v)
                if not p.is_absolute():
                    p = (ROOT / v).resolve()
                if p.exists():
                    return p
            elif isinstance(v, (list, tuple)):
                for it in v:
                    p = Path(it)
                    if not p.is_absolute():
                        p = (ROOT / it).resolve()
                    if p.exists():
                        return p
    # 3) heuristics
    for it in ["asr/FireRedASR", "asr\\FireRedASR", "FireRedASR"]:
        p = (ROOT / it).resolve()
        if p.exists():
            return p
    return None


def _ensure_sys_path(p: Path):
    p = str(p)
    if p not in sys.path:
        sys.path.insert(0, p)
        logging.info("[Path] Added to sys.path: %s", p)


def _bootstrap_punc(post_cfg: dict):
    t0 = time.perf_counter()
    try:
        # now that ROOT in sys.path, this import should succeed
        from segmentation.postprocess.punc_bootstrap import bootstrap_once
        bootstrap_once(post_cfg.get("punctuation", {}))
        dt = time.perf_counter() - t0
        logging.info("[PUNC] bootstrap ok in %.2fs", dt)
    except Exception as e:
        dt = time.perf_counter() - t0
        logging.warning("[PUNC] bootstrap failed in %.2fs: %s", dt, e)


def _build_stub(post_cfg: dict):
    from segmentation.inference.firered_client_stub import FireRedClientStub
    seg_sec = float(post_cfg.get("segmentation", {}).get("seg_sec", 20.0))
    overlap_sec = float(post_cfg.get("segmentation", {}).get("overlap_sec", 1.0))
    logging.info("[Seg] seg_sec=%.1f  overlap_sec=%.1f", seg_sec, overlap_sec)
    return FireRedClientStub(seg_sec=seg_sec, overlap_sec=overlap_sec)


def _load_runner(engine: str, asr_type: str, model_dir: str, device: str, batch: int, decode_max_len: int):
    decode_opts = {}
    if decode_max_len and int(decode_max_len) > 0:
        decode_opts["decode_max_len"] = int(decode_max_len)

    if engine == "onnx":
        from segmentation.inference.firered_client_inproc_onnx import FireRedInprocOnnxRunner
        runner = FireRedInprocOnnxRunner(asr_type=asr_type, model_dir=model_dir, device=device, batch=batch, decode_opts=decode_opts)
        infer = lambda items, dev=None: runner.infer_items(items)
        return infer, runner
    else:
        from segmentation.inference.firered_client_inproc import FireRedInprocRunner
        runner = FireRedInprocRunner(asr_type=asr_type, model_dir=model_dir, device=device, batch=batch, decode_opts=decode_opts)
        infer = lambda items, dev=None: runner.infer_items(items)
        return infer, runner


def _refine_and_write(out_path: Path, segments: list, post_cfg: dict):
    t0 = time.perf_counter()
    from segmentation.postprocess.refine_output import refine_output
    final_text, punc_detail, metrics = refine_output(segments, post_cfg)
    dt = time.perf_counter() - t0
    logging.info("[Refine] took %.2fs (len(segments)=%d)", dt, len(segments))

    main_obj = {
        "text": final_text,
        "punc_detail": punc_detail,
        "segments": segments,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(main_obj, f, ensure_ascii=False, indent=2)
    logging.info("[Write] main json -> %s", out_path)

    mpath = Path(str(out_path) + ".metrics.json")
    with open(mpath, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    logging.info("[Write] metrics -> %s", mpath)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True)
    ap.add_argument("--use", default="aed", choices=["aed", "llm"])
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--repo", default=None)
    ap.add_argument("--engine", default=None, choices=[None, "inproc", "onnx"], help="override engine; default: from YAML or inproc")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--decode_max_len", type=int, default=0)
    ap.add_argument("--minutes", type=float, default=None)
    ap.add_argument("--out", default="firered_cli_output.json")
    ap.add_argument("--log", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s"
    )

    # 再次确保 ROOT 在 sys.path（此时 logging 已建立）
    _ensure_sys_path(ROOT)

    dev = _detect_device()
    cli_cfg = _load_yaml(DEFAULT_FIRED_CLI_YAML, default={})
    post_cfg = _load_yaml(DEFAULT_POST_YAML, default={"segmentation": {"seg_sec": 20.0, "overlap_sec": 1.0},
                                                      "stitch": {"chars_per_sec": 3.5},
                                                      "punctuation": {"enabled": True}})
    engine_yaml = (cli_cfg.get("engine") or "inproc")
    engine = args.engine or engine_yaml or "inproc"
    logging.info("Start: use=%s engine(cli)=%s engine(yaml)=%s -> engine=%s minutes=%s device=%s",
                 args.use, args.engine, engine_yaml, engine, args.minutes, dev)

    _bootstrap_punc(post_cfg)

    repo = _resolve_repo(args.repo, cli_cfg) or Path("asr/FireRedASR")
    logging.info("[Repo] Resolved FireRedASR repo: %s (%.2fs)", str(repo), 0.0)
    _ensure_sys_path(repo)

    stub = _build_stub(post_cfg)
    infer_fn, runner = _load_runner(engine, args.use, args.model_dir, dev, args.batch, args.decode_max_len)

    t0 = time.perf_counter()
    batches = stub.build_batches(args.wav, minutes=args.minutes)
    t1 = time.perf_counter()
    total_slices = sum(len(b) for b in batches)
    logging.info("[Stub] Planned slices: %d (%.2fs)", total_slices, t1 - t0)

    segments = []
    for bi, batch in enumerate(batches):
        logging.info("[Infer] batch #%d size=%d", bi, len(batch))
        tt = time.perf_counter()
        outs = infer_fn(batch, None)
        dt = time.perf_counter() - tt
        logging.info("[Infer] batch #%d done in %.2fs", bi, dt)
        for it, hyp in zip(batch, outs):
            seg = {
                "start": float(it["start"]),
                "end": float(it["end"]),
                "text": hyp.get("text", ""),
                "cache_wav": it.get("cache_wav"),
            }
            segments.append(seg)

    _refine_and_write(Path(args.out), segments, post_cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
