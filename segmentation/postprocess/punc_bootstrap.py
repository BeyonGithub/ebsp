# -*- coding: utf-8 -*-
"""
FunASR 标点模型懒加载 + 熔断回退
- 第一次调用时下载/加载（可配置 cache_dir/device）
- 失败时回退为原文，并返回 used=False 的 metrics
- 统一入口：lazy_punc_predict(text, cfg, logger=None)
"""
import os, time, json
from pathlib import Path

_PUNC_CTX = {"ready": False, "model": None, "pipeline": None, "stamp": None}

def ensure_punc_bootstrap(cfg: dict):
    """
    预热模型（可选）。若已经 ready，则直接返回。
    """
    if _PUNC_CTX.get("ready"):
        return
    _init_from_cfg(cfg)

def _init_from_cfg(cfg: dict):
    punc = (cfg or {}).get("punctuation", {}) if cfg is not None else {}
    enabled = bool(punc.get("enabled", True))
    if not enabled:
        return
    model_id = punc.get("model_id", "damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch")
    device = punc.get("device", "cpu")
    cache_dir = punc.get("cache_dir", ".cache/funasr")
    stamp = Path(cache_dir) / (".punc_ready_" + model_id.replace("/", "_") + ".stamp")
    _PUNC_CTX["stamp"] = stamp

    # 已有 stamp 则视为 ready（懒加载仍可能再次构建 pipeline）
    if stamp.exists():
        _PUNC_CTX["ready"] = True
        return

    # 尝试加载一次，若失败不抛异常
    try:
        _load_funasr(model_id, device, cache_dir)
        stamp.parent.mkdir(parents=True, exist_ok=True)
        stamp.write_text("ok", encoding="utf-8")
        _PUNC_CTX["ready"] = True
    except Exception:
        # 忽略错误，保持懒加载路径
        pass

def _load_funasr(model_id, device, cache_dir):
    from funasr import AutoModel
    model = AutoModel(
        model=model_id,
        device=device,
        hub="ms",
        cache_dir=cache_dir,
        dtype="bf16" if device == "cuda" else "fp32",
        disable_update=True,
    )
    _PUNC_CTX["model"] = model
    _PUNC_CTX["pipeline"] = None  # AutoModel 直接 __call__ 推理

def lazy_punc_predict(text: str, cfg: dict, logger=None):
    """
    返回： (punct_text, detail_json, metrics_dict)
    - 若失败：返回原文、None、{used: False, ...}
    """
    t0 = time.time()
    metrics = {"used": False, "time_sec": 0.0, "len_in": len(text or ""), "len_out": 0}
    if not text:
        return text, None, metrics

    punc = (cfg or {}).get("punctuation", {}) if cfg is not None else {}
    if not punc.get("enabled", True):
        return text, None, metrics

    model_id = punc.get("model_id", "damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch")
    device = punc.get("device", "cpu")
    cache_dir = punc.get("cache_dir", ".cache/funasr")

    # 若尚未加载，尝试加载一次；失败则回退
    if _PUNC_CTX.get("model") is None:
        try:
            _load_funasr(model_id, device, cache_dir)
            # 写 stamp
            stamp = _PUNC_CTX.get("stamp")
            if stamp:
                stamp.parent.mkdir(parents=True, exist_ok=True)
                stamp.write_text("ok", encoding="utf-8")
        except Exception as e:
            if logger: logger.warning("[PUNC] lazy load failed: %s", e)
            metrics["time_sec"] = time.time() - t0
            return text, None, metrics

    # 推理
    try:
        model = _PUNC_CTX["model"]
        out = model(input=text, cache={}, language="zn", use_itn=False)
        # 兼容不同版本的输出：优先取 text，若无则回退 input
        if isinstance(out, dict):
            puncted = out.get("text", out.get("preds", text))
            detail = out
        elif isinstance(out, (list, tuple)) and out:
            # 某些版本返回 list[dict]
            o0 = out[0] if isinstance(out[0], dict) else {}
            puncted = o0.get("text", text)
            detail = o0
        else:
            puncted = text
            detail = None
        metrics.update({"used": True, "time_sec": time.time()-t0, "len_out": len(puncted or "")})
        return puncted, detail, metrics
    except Exception as e:
        if logger: logger.warning("[PUNC] predict failed: %s", e)
        metrics["time_sec"] = time.time() - t0
        return text, None, metrics
