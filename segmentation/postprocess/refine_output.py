# -*- coding: utf-8 -*-
"""
后处理：重叠去重 + 整体标点恢复 + 指标
- 顶层 text 为“拼接后 + 整体标点”的最终结果
- 新增 punc_detail 保存原始标点 JSON
- 返回 metrics（由 CLI 弹出并另存）
"""
import time
from typing import List, Dict, Any

def _estimate_max_overlap_chars(cfg):
    seg = (cfg or {}).get("segmentation", {}) if cfg is not None else {}
    stitch = (cfg or {}).get("stitch", {})
    # 优先使用 stitch.max_overlap_chars，其次自动估计 overlap_sec * chars_per_sec
    if "max_overlap_chars" in stitch and stitch["max_overlap_chars"] is not None:
        try:
            return int(stitch["max_overlap_chars"])
        except Exception:
            pass
    overlap_sec = float(seg.get("overlap_sec", 1.0) or 1.0)
    chars_per_sec = float(stitch.get("chars_per_sec", 3.5) or 3.5)
    return int(max(1, round(overlap_sec * chars_per_sec)))

def _stitch_segments(segments: List[Dict[str, Any]], max_overlap_chars: int):
    """
    仅基于文本重叠去重；不对分片 text 做标点。
    segments: [{start,end,text,cache_wav?}, ...]
    """
    texts = [(s.get("text") or "") for s in segments]
    # 简单重叠去重：如果后一段前 N 个字与前一段末 N 个字重复，则去掉后面的前 N 个字
    dedup_chars = 0
    out = []
    prev = ""
    for i, t in enumerate(texts):
        if not prev:
            out.append(t)
            prev = t
            continue
        n = min(max_overlap_chars, len(prev), len(t))
        drop = 0
        if n > 0:
            head = t[:n]
            tail = prev[-n:]
            # 逐步寻找最大重叠
            for k in range(n, 0, -1):
                if tail[-k:] == head[:k]:
                    drop = k
                    break
        if drop > 0:
            out.append(t[drop:])
            dedup_chars += drop
            prev = prev + t[drop:]
        else:
            out.append(t)
            prev = prev + t
    stitched = "".join(out)
    return stitched, dedup_chars

def refine_output(segments: List[Dict[str, Any]], cfg: dict, logger=None) -> Dict[str, Any]:
    t0 = time.time()
    max_overlap_chars = _estimate_max_overlap_chars(cfg)
    if logger: logger.info("[Refine] max_overlap_chars=%d", max_overlap_chars)

    # 1) 拼接 + 去重
    t1 = time.time()
    stitched, removed = _stitch_segments(segments, max_overlap_chars=max_overlap_chars)
    t2 = time.time()
    if logger: logger.info("[Refine] stitch done: len=%d removed_chars=%d time=%.2fs", len(stitched), removed, t2-t1)

    # 2) 整体标点（懒加载 + 熔断）
    from segmentation.postprocess.punc_bootstrap import lazy_punc_predict
    puncted, punc_detail, punc_metrics = lazy_punc_predict(stitched, cfg, logger=logger)
    t3 = time.time()
    if logger: logger.info("[Refine] punc used=%s len=%d time=%.2fs", punc_metrics.get("used"), len(puncted or ""), t3-t2)

    # 3) 汇总
    metrics = {
        "stitch": {"removed_overlap_chars": int(removed), "time_sec": round(t2-t1, 3)},
        "punc": punc_metrics,
        "total_time_sec": round(t3 - t0, 3),
    }
    return {
        "text": puncted or stitched or "",
        "punc_detail": punc_detail,
        "segments": segments,
        "metrics": metrics,
    }
