# -*- coding: utf-8 -*-
"""
scripts/punc_bootstrap.py
在首次运行时自动初始化 FunASR 标点恢复模型（ct-punc），将模型预拉取到本地缓存。
- 读取传入的 punctuation 配置（model_id/cache_dir/device），也支持环境变量 FUNASR_CACHE / MODELSCOPE_CACHE
- 兼容 FunASR 的 AutoModel(...) / AutoModel.from_pretrained(...)
- 离线/网络受限时不会中断主流程，只记录 WARNING
"""
from __future__ import annotations
import os, sys, logging, tempfile

DEFAULT_MODEL = "damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch"

def _pick_cache_dir(cfg_cache: str|None) -> str:
    if cfg_cache and len(cfg_cache) > 0:
        return os.path.abspath(os.path.expanduser(cfg_cache))
    for env in ("FUNASR_CACHE", "MODELSCOPE_CACHE", "HF_HOME"):
        v = os.environ.get(env)
        if v:
            return os.path.abspath(os.path.expanduser(v))
    # fallback
    return os.path.join(os.path.expanduser("~"), ".cache", "funasr")

def _pick_device(dev: str|None) -> str:
    want = (dev or "auto").lower()
    if want == "cpu":
        return "cpu"
    if want.startswith("cuda"):
        return want
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda:0"
    except Exception:
        pass
    return "cpu"

def _stamp_path(cache_dir: str, model_id: str) -> str:
    safe = model_id.replace("/", "_")
    return os.path.join(cache_dir, f".punc_ready_{safe}.stamp")

def _instantiate_model(model_id: str, device: str, cache_dir: str):
    try:
        from funasr import AutoModel
    except Exception as e:
        logging.warning("[PUNC] funasr not installed or import failed: %s", e)
        return None
    # Try the two common constructors
    try:
        m = AutoModel(model=model_id, device=device, cache_dir=cache_dir, disable_update=True)
        return m
    except Exception as e1:
        try:
            # Older API
            m = AutoModel.from_pretrained(model=model_id, device=device, cache_dir=cache_dir)
            return m
        except Exception as e2:
            logging.warning("[PUNC] AutoModel init failed: %s | fallback: %s", e1, e2)
            return None

def ensure_punc_ready(punc_cfg: dict|None):
    if not punc_cfg:
        logging.info("[PUNC] no punctuation config; skip bootstrap.")
        return False
    if not punc_cfg.get("enable", False):
        logging.info("[PUNC] punctuation disabled; skip bootstrap.")
        return False
    backend = (punc_cfg.get("backend") or "").lower()
    if backend not in ("funasr_ct_punc", "funasr-ct-punc", "ct-punc"):
        logging.info("[PUNC] backend=%s not funasr ct-punc; skip bootstrap.", backend)
        return False
    model_id = punc_cfg.get("model_id") or DEFAULT_MODEL
    cache_dir = _pick_cache_dir(punc_cfg.get("cache_dir"))
    device = _pick_device(punc_cfg.get("device"))

    os.makedirs(cache_dir, exist_ok=True)
    stamp = _stamp_path(cache_dir, model_id)
    if os.path.exists(stamp):
        logging.info("[PUNC] already initialized (stamp exists): %s", stamp)
        return True

    logging.info("[PUNC] initializing model: %s  device=%s  cache=%s", model_id, device, cache_dir)
    m = _instantiate_model(model_id, device, cache_dir)
    if m is None:
        logging.warning("[PUNC] init skipped (funasr missing or model unreachable).")
        return False

    # quick warmup (optional)
    try:
        _ = m.generate(input="今天 天气 不错 我们 去 公园 散步", batch_size=1)
    except Exception as e:
        logging.info("[PUNC] warmup skipped: %s", e)

    try:
        with open(stamp, "w", encoding="utf-8") as f:
            f.write("ok")
        logging.info("[PUNC] initialized and stamped: %s", stamp)
    except Exception as e:
        logging.debug("[PUNC] write stamp failed: %s", e)

    return True
