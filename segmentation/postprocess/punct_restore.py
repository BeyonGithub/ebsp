# -*- coding: utf-8 -*-
"""
segmentation/postprocess/punct_restore.py (MODEL-ID FIX)
- 运行时自动尝试 "ct-punc" / "damo/punc_ct-transformer_cn-en-common-vocab471067-large"
  / "damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch"
- 本地路径亦可（若 YAML 提供的是本地目录）
"""
from __future__ import annotations
import importlib, logging, re, os

_PUNC = None
_PUNC_META = {}
CANDIDATE_MODELS = [
    "ct-punc",
    "damo/punc_ct-transformer_cn-en-common-vocab471067-large",
    "damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch",
]

def _build(model_id, cache_dir, device):
    funasr = importlib.import_module("funasr")
    AutoModel = getattr(funasr, "AutoModel", None)
    if AutoModel is None:
        raise RuntimeError("funasr.AutoModel not found")
    kw = {"trust_remote_code": True}
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        kw["cache_dir"] = cache_dir
        os.environ.setdefault("MODELSCOPE_CACHE", cache_dir)
    if hasattr(AutoModel, "from_pretrained"):
        try:
            return AutoModel.from_pretrained(model_id, device=device, **kw)
        except TypeError:
            return AutoModel.from_pretrained(model_id, **kw)
    else:
        try:
            return AutoModel(model=model_id, device=device, **kw)
        except TypeError:
            return AutoModel(model=model_id, **kw)

def init_funasr_punc(cache_dir: str|None=None, device: str="cpu", model_id: str|None=None):
    global _PUNC, _PUNC_META
    try:
        importlib.import_module("funasr")
    except Exception as e:
        logging.error("[PUNC] funasr not installed: %s", e)
        return None

    tried = []
    if model_id:
        tried.append(model_id)
        try:
            _PUNC = _build(model_id, cache_dir, device)
            _PUNC_META = {"model_id": model_id, "device": device, "cache_dir": cache_dir}
            logging.info("[PUNC] init ok: %s", model_id)
            return _PUNC
        except Exception as e:
            logging.warning("[PUNC] init %s failed: %s", model_id, e)

    for mid in CANDIDATE_MODELS:
        tried.append(mid)
        try:
            _PUNC = _build(mid, cache_dir, device)
            _PUNC_META = {"model_id": mid, "device": device, "cache_dir": cache_dir}
            logging.info("[PUNC] init ok: %s", mid)
            return _PUNC
        except Exception as e:
            logging.warning("[PUNC] init %s failed: %s", mid, e)
    logging.error("[PUNC] All candidates failed: %s", tried)
    return None

def _funasr_add_punct(text: str, cfg: dict|None=None) -> str|None:
    global _PUNC
    if _PUNC is None and cfg:
        fn_cfg = (cfg.get("punctuation") or {}).get("funasr") or {}
        model_id = fn_cfg.get("model_id", None)
        device = fn_cfg.get("device", "cpu")
        cache_dir = fn_cfg.get("cache_dir", None)
        init_funasr_punc(cache_dir=cache_dir, device=device, model_id=model_id)

    if _PUNC is None:
        return None
    try:
        r = _PUNC.generate(input=text)
    except TypeError:
        try:
            r = _PUNC.generate({"text": text})
        except Exception as e:
            logging.warning("[PUNC] FunASR generate failed: %s", e)
            return None
    if isinstance(r, list) and r and isinstance(r[0], dict) and "text" in r[0]:
        return r[0]["text"]
    if isinstance(r, dict) and "text" in r:
        return r["text"]
    return text

def simple_rule_punc(text: str, max_len: int = 30):
    if not text: return text
    t = re.sub(r"\s+", "", text)
    t = re.sub(r"(因此|但是|同时|此外|随后|然后|最后|首先|其次|另外|并且|而且|面对|针对)", r"，\1", t)
    t = re.sub(r"(一是|二是|三是|首先|其次)", r"\1，", t)
    out=[]; buf=""
    for ch in t:
        buf += ch
        if len(buf)>=max_len or ch in "吗吧呢呀啊！？””]】）":
            if buf and buf[-1] not in "。！？":
                buf += "。"
            out.append(buf); buf=""
    if buf:
        if buf[-1] not in "。！？":
            buf += "。"
        out.append(buf)
    return " ".join(out)

def restore(text: str, prefer="auto", cfg: dict|None=None) -> str:
    if not text:
        return text
    if prefer in ("auto","funasr"):
        r = _funasr_add_punct(text, cfg)
        if r: return r
    return simple_rule_punc(text)
