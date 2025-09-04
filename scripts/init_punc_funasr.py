# -*- coding: utf-8 -*-
"""
scripts/init_punc_funasr.py  (MODEL-ID FIX)
- 兼容 FunASR >=1.1 / >=1.2 的不同 API
- 自动修正错误/过期的模型 ID：优先 alias "ct-punc"，再尝试官方仓 "damo/punc_ct-transformer_cn-en-common-vocab471067-large"
  以及旧版 "damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch"
- 支持本地离线目录（传入 --model_id 为本地路径即可）
"""
import argparse, os, sys, subprocess, json, yaml

CANDIDATE_MODELS = [
    # alias (FunASR 官方 README 推荐)
    "ct-punc",
    # 新版/多语模型（ModelScope damo 命名空间）
    "damo/punc_ct-transformer_cn-en-common-vocab471067-large",
    # 旧版中文模型
    "damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch",
]

def sh(cmd, check=True):
    print("[CMD]", cmd)
    r = subprocess.run(cmd, shell=True)
    if check and r.returncode != 0:
        raise SystemExit(r.returncode)

def ensure_pkgs():
    print("[INFO] Checking funasr/modelscope ...")
    try:
        import funasr  # noqa
        import modelscope  # noqa
        print("[OK] funasr & modelscope already installed.")
    except Exception:
        print("[INFO] Installing/Upgrading funasr & modelscope ...")
        sh(f"{sys.executable} -m pip install -U funasr modelscope")

def try_build(model_id: str, cache_dir: str|None, device: str):
    import importlib
    funasr = importlib.import_module("funasr")
    AutoModel = getattr(funasr, "AutoModel", None)
    if AutoModel is None:
        raise RuntimeError("funasr.AutoModel not found (please upgrade funasr).")
    kw = {"trust_remote_code": True}
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        kw["cache_dir"] = cache_dir
        os.environ.setdefault("MODELSCOPE_CACHE", cache_dir)
    # local dir shortcut
    if os.path.isdir(model_id):
        # 本地模型目录
        print(f"[TRY] Local dir: {model_id}")
        try:
            if hasattr(AutoModel, "from_pretrained"):
                return AutoModel.from_pretrained(model_id, device=device, **kw)
            else:
                return AutoModel(model=model_id, device=device, **kw)
        except TypeError:
            return AutoModel.from_pretrained(model_id, **kw) if hasattr(AutoModel, "from_pretrained") else AutoModel(model=model_id, **kw)
    # remote id / alias
    print(f"[TRY] Remote id: {model_id}")
    try:
        if hasattr(AutoModel, "from_pretrained"):
            return AutoModel.from_pretrained(model_id, device=device, **kw)
        else:
            return AutoModel(model=model_id, device=device, **kw)
    except TypeError:
        return AutoModel.from_pretrained(model_id, **kw) if hasattr(AutoModel, "from_pretrained") else AutoModel(model=model_id, **kw)

def build_with_fallback(user_model_id: str|None, cache_dir: str|None, device: str):
    errs = []
    tried = []
    if user_model_id:
        tried.append(user_model_id)
        try:
            return try_build(user_model_id, cache_dir, device), tried, errs
        except Exception as e:
            errs.append((user_model_id, str(e)))
    for mid in CANDIDATE_MODELS:
        tried.append(mid)
        try:
            return try_build(mid, cache_dir, device), tried, errs
        except Exception as e:
            errs.append((mid, str(e)))
    raise RuntimeError("All candidate punctuation models failed.\nTried:\n- " + "\n- ".join(tried) +
                       "\nLast error: " + (errs[-1][1] if errs else "N/A"))

def warmup(model):
    text = "这是一次标点恢复初始化测试，系统正在预热模型"
    try:
        out = model.generate(input=text)
    except TypeError:
        out = model.generate({"text": text})
    print("[WARMUP]", out)

def update_yaml(model_id: str, cache_dir: str|None, device: str, path="training/configs/postprocess.yaml"):
    import yaml
    cfg = {}
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            try: cfg = yaml.safe_load(f) or {}
            except Exception: cfg = {}
    p = cfg.setdefault("punctuation", {})
    p["enable"] = True
    p["prefer"] = "funasr"
    fn = p.setdefault("funasr", {})
    fn["model_id"] = model_id
    fn["cache_dir"] = cache_dir
    fn["device"] = device
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    print("[OK] postprocess.yaml updated:", path)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", default=None, help="可填 'ct-punc' 或 ModelScope 全名，留空则自动尝试候选列表")
    ap.add_argument("--cache_dir", default=None)
    ap.add_argument("--device", default="cuda:0" if (os.getenv("CUDA_VISIBLE_DEVICES") is not None) else "cpu")
    args = ap.parse_args()

    ensure_pkgs()
    model, tried, errs = build_with_fallback(args.model_id, args.cache_dir, args.device)
    print("[INFO] Resolved punctuation model:", tried[-1])
    warmup(model)
    update_yaml(tried[-1], args.cache_dir, args.device)
    print("[DONE] FunASR punctuation initialized.")
