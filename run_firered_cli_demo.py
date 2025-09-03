# -*- coding: utf-8 -*-
"""
scripts/run_firered_cli_demo.py  (Repo validate relaxed + diagnostics)
- 放宽 repo 校验：只要存在目录 <repo>/fireredasr/ 且其中有任意 .py 文件即可（兼容 PEP 420 namespace packages）
- 打印首个匹配到的有效 repo
"""
import os, sys, argparse, json, time, logging, importlib, importlib.util, glob

THIS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(THIS)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

def _yaml_or_none(path):
    try:
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception:
        return None

def _validate_repo(path: str) -> bool:
    if not path: return False
    p = os.path.expanduser(os.path.expandvars(path))
    # 允许相对 ROOT
    p_abs = os.path.abspath(os.path.join(ROOT, p)) if not os.path.isabs(p) else p
    pkg_dir = os.path.join(p_abs, "fireredasr")
    if not os.path.isdir(pkg_dir):
        return False
    # 允许 namespace 包：只要有任意 .py 文件即可
    has_py = any(name.lower().endswith(".py") for name in os.listdir(pkg_dir))
    has_init = os.path.exists(os.path.join(pkg_dir, "__init__.py"))
    return has_py or has_init

def _autodetect_repo(args_repo: str, yaml_path: str) -> str|None:
    cand = []
    if args_repo: cand.append(args_repo)
    cfg = _yaml_or_none(yaml_path)
    if cfg:
        if isinstance(cfg.get("repo"), str):
            cand.append(cfg["repo"])
        if isinstance(cfg.get("repo_candidates"), (list, tuple)):
            cand.extend([c for c in cfg["repo_candidates"] if isinstance(c, str)])
    env = os.environ.get("FIRERED_REPO")
    if env: cand.append(env)
    common = ["asr/FireRedASR", r"asr\FireRedASR", "FireRedASR", os.path.join("asr","FireRedASR")]
    cand.extend(common)
    # 绝对化与去重
    seen = set(); normalized = []
    for c in cand:
        if not c: continue
        c1 = os.path.expanduser(os.path.expandvars(c))
        if c1 not in seen:
            seen.add(c1); normalized.append(c1)
        if not os.path.isabs(c1):
            c2 = os.path.abspath(os.path.join(ROOT, c1))
            if c2 not in seen:
                seen.add(c2); normalized.append(c2)
    for p in normalized:
        if _validate_repo(p):
            # 返回绝对路径
            p_abs = os.path.abspath(os.path.join(ROOT, p)) if not os.path.isabs(p) else p
            logging.info("[Repo] Resolved FireRedASR repo: %s", p_abs)
            return p_abs
    return None

def _add_repo_to_syspath(repo_path: str):
    p = os.path.expanduser(os.path.expandvars(repo_path))
    if not os.path.isabs(p):
        p = os.path.abspath(os.path.join(ROOT, p))
    if p not in sys.path:
        sys.path.insert(0, p)
        logging.info("[Path] Added repo to sys.path: %s", p)

from segmentation.inference.firered_client_stub import FireRedClientStub, run_on_wav
from segmentation.inference.firered_client_impl import infer_llm_cli, infer_aed_cli, infer_aed_cli_batch

def _load_inproc_runner():
    try:
        mod = importlib.import_module("segmentation.inference.firered_client_inproc")
        return getattr(mod, "FireRedInprocRunner")
    except Exception as e:
        logging.warning("Standard import failed: %s", e)
        inproc_path = os.path.join(ROOT, "segmentation", "inference", "firered_client_inproc.py")
        if os.path.exists(inproc_path):
            spec = importlib.util.spec_from_file_location("segmentation.inference.firered_client_inproc", inproc_path)
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                return getattr(mod, "FireRedInprocRunner", None)
        return None

def wrap_infer_with_timing(infer_fn, total_slices):
    done = {"n":0, "t0": time.time()}
    def _wrapped(items, dev):
        res = infer_fn(items, dev) if dev is not None else infer_fn(items)
        done["n"] += len(items)
        dt = time.time() - done["t0"]
        rate = done["n"]/dt if dt>0 else 0.0
        remain = max(0, total_slices - done["n"])
        eta = remain / rate if rate>0 else float("inf")
        logging.info("[Engine] progress=%d/%d  elapsed=%.1fs  ETA=%.1fs  rate=%.2f seg/s",
                     done["n"], total_slices, dt, eta, rate)
        return res
    return _wrapped

def _infer_model_dir_from_yaml(config_path: str, use: str) -> str|None:
    cfg = _yaml_or_none(config_path)
    if not cfg: return None
    if use == "aed":
        for k in ("aed", "aed_batch"):
            if k in cfg and isinstance(cfg[k], dict):
                m = cfg[k].get("model")
                if m and os.path.isdir(m):
                    return m
    elif use == "llm":
        sec = cfg.get("llm", {})
        m = sec.get("model")
        if m and os.path.isdir(m):
            return m
    return None

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True)
    ap.add_argument("--minutes", type=float, default=None)
    ap.add_argument("--config", default="training/configs/firered_cli.yaml")
    ap.add_argument("--use", choices=["llm","aed"], default="aed")
    ap.add_argument("--engine", choices=["cli","batch","inproc"], default=None)
    ap.add_argument("--devices", default="0")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--no_ffmpeg", action="store_true")
    ap.add_argument("--out", default="data/sample/firered_cli_output.json")
    ap.add_argument("--log", default="INFO", choices=["DEBUG","INFO","WARNING","ERROR"])
    ap.add_argument("--seg_sec", type=float, default=60.0)
    ap.add_argument("--overlap_sec", type=float, default=0.0)
    ap.add_argument("--repo", default="")  # 允许留空，走自动探测
    ap.add_argument("--model_dir", default="")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--beam_size", type=int, default=1)
    ap.add_argument("--decode_max_len", type=int, default=0)
    ap.add_argument("--decode_min_len", type=int, default=0)
    ap.add_argument("--repetition_penalty", type=float, default=1.0)
    ap.add_argument("--llm_length_penalty", type=float, default=0.0)
    ap.add_argument("--temperature", type=float, default=1.0)

    args = ap.parse_args()
    logging.basicConfig(level=getattr(logging, args.log),
                        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    logging.info("Start: use=%s engine=%s workers=%d devices=%s minutes=%s",
                 args.use, args.engine, args.workers, args.devices, args.minutes)

    if args.engine is None:
        args.engine = "inproc"
        logging.info("Engine not specified, default -> inproc")

    tmp_client = FireRedClientStub(devices=args.devices, workers=0, batch_size=1,
                                   no_ffmpeg=args.no_ffmpeg, use_threads=True, infer_fn=lambda a,b:[],
                                   seg_sec=args.seg_sec, overlap_sec=args.overlap_sec)
    batches = tmp_client.build_batches(args.wav, minutes=args.minutes)
    total_slices = sum(len(b) for b in batches)
    logging.info("Planned slices: %d (seg=%.1fs overlap=%.1fs)", total_slices, args.seg_sec, args.overlap_sec)

    if args.engine == "inproc":
        repo = _autodetect_repo(args.repo, args.config)
        if not repo:
            cfg = _yaml_or_none(args.config) or {}
            cands = []
            if args.repo: cands.append(args.repo)
            if isinstance(cfg.get("repo"), str): cands.append(cfg["repo"])
            if isinstance(cfg.get("repo_candidates"), (list, tuple)): cands.extend(cfg["repo_candidates"])
            cands.extend(["asr/FireRedASR", r"asr\\FireRedASR", "FireRedASR", os.path.join("asr","FireRedASR")])
            raise RuntimeError("Auto-detect repo failed. Please set --repo or YAML repo/repo_candidates.\n"
                               f" Tried candidates (relative to ROOT={ROOT} also absolutized):\n  - " + "\n  - ".join(map(str, cands)))
        _add_repo_to_syspath(repo)

        FireRedInprocRunner = _load_inproc_runner()
        if FireRedInprocRunner is None:
            expected = os.path.join(ROOT, "segmentation", "inference", "firered_client_inproc.py")
            raise RuntimeError("Inproc runner not available. Expected path: %s (exists=%s)" % (expected, os.path.exists(expected)))

        if not args.model_dir:
            inferred = _infer_model_dir_from_yaml(args.config, args.use)
            if inferred:
                args.model_dir = inferred
                logging.info("Auto-inferred model_dir from YAML: %s", args.model_dir)
        if not args.model_dir:
            raise ValueError("--model_dir is required for inproc (或在 YAML aed/llm 段配置 model 路径)")

        runner = FireRedInprocRunner(
            asr_type=args.use, model_dir=args.model_dir, device=args.device,
            batch=args.batch, beam_size=args.beam_size, decode_max_len=args.decode_max_len,
            decode_min_len=args.decode_min_len, repetition_penalty=args.repetition_penalty,
            llm_length_penalty=args.llm_length_penalty, temperature=args.temperature
        )
        infer_core = lambda items, dev: runner.infer_items(items)
        client_batch_size = total_slices
    elif args.engine == "batch":
        if args.use == "llm":
            logging.warning("LLM batch not implemented; falling back to CLI per-slice.")
            infer_core = (lambda items, dev: infer_llm_cli(items, dev, args.config))
            client_batch_size = 1
        else:
            infer_core = (lambda items, dev: infer_aed_cli_batch(items, dev, args.config))
            client_batch_size = total_slices
    else:
        infer_core = (lambda items, dev: infer_llm_cli(items, dev, args.config)) if args.use=="llm" \
                  else (lambda items, dev: infer_aed_cli(items, dev, args.config))
        client_batch_size = 1

    infer_fn = wrap_infer_with_timing(infer_core, total_slices)
    client = FireRedClientStub(devices=args.devices, workers=args.workers, batch_size=client_batch_size,
                               no_ffmpeg=args.no_ffmpeg, use_threads=True, infer_fn=infer_fn,
                               seg_sec=args.seg_sec, overlap_sec=args.overlap_sec)

    t0 = time.time()
    out = run_on_wav(client, wav=args.wav, minutes=args.minutes, out_json=args.out)
    logging.info("Done. Output=%s  Preview=%s  TotalElapsed=%.1fs", args.out, out["text"][:120], time.time()-t0)
    print(json.dumps({"ok": True, "out": args.out}, ensure_ascii=False))
