# -*- coding: utf-8 -*-
from typing import List, Tuple, Dict, Any
import random
def _dur(x: Dict[str, Any]) -> float:
    return float(max(0.0, x.get("end",0.0) - x.get("start",0.0)))
def make_buckets(chunks: List[Dict[str,Any]], bucket_ranges: List[Tuple[float,float]]):
    buckets = { br: [] for br in bucket_ranges }
    other = []
    for ch in chunks:
        d = _dur(ch); placed=False
        for br in bucket_ranges:
            lo, hi = br
            if d >= lo and d < hi:
                buckets[br].append(ch); placed=True; break
        if not placed: other.append(ch)
    out = []
    for br in bucket_ranges:
        if buckets[br]: out.append({"bucket": br, "items": buckets[br]})
    if other: out.append({"bucket": ("other",""), "items": other})
    return out
def schedule_batches(buckets: List[Dict[str,Any]], batch_size: int = 1, shuffle: bool = False):
    batches = []
    for b in buckets:
        items = list(b["items"])
        if shuffle: random.shuffle(items)
        if batch_size <= 1:
            for it in items: batches.append({"bucket": b["bucket"], "items": [it]})
        else:
            for i in range(0, len(items), batch_size):
                batches.append({"bucket": b["bucket"], "items": items[i:i+batch_size]})
    return batches
