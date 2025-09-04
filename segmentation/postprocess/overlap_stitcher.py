# -*- coding: utf-8 -*-
from typing import List, Dict, Any
import re
def _normalize(s: str) -> str:
    return re.sub(r"\s+","", s.strip())
def _char_trigrams(s: str):
    s=_normalize(s)
    if len(s)<3: return set([s]) if s else set()
    return set(s[i:i+3] for i in range(len(s)-2))
def _jaccard(a: str, b: str) -> float:
    A,B=_char_trigrams(a),_char_trigrams(b)
    if not A or not B: return 0.0
    inter=len(A&B); uni=len(A|B); return float(inter/uni) if uni else 0.0
def _lev_ratio(a: str, b: str) -> float:
    la,lb=len(a),len(b)
    if la==0 and lb==0: return 1.0
    if la==0 or lb==0: return 0.0
    dp=list(range(lb+1))
    for i in range(1,la+1):
        prev=dp[0]; dp[0]=i
        for j in range(1,lb+1):
            cur=dp[j]; cost=0 if a[i-1]==b[j-1] else 1
            dp[j]=min(dp[j]+1, dp[j-1]+1, prev+cost); prev=cur
    dist=dp[lb]; mx=max(la,lb); return 1.0-float(dist)/mx
def _pick_dedup_offset(prev_tail: str, next_head: str, method="jaccard", thr=0.6):
    best_k,best_sim=0,0.0
    for k in range(0, min(len(next_head), 30)+1):
        cand = next_head[:k]
        if not cand: continue
        sim = _jaccard(prev_tail[-30:], cand) if method=="jaccard" else _lev_ratio(prev_tail[-30:], cand)
        if sim>best_sim: best_sim, best_k = sim, k
    cut = best_k if best_sim >= thr else 0
    return cut, best_sim
def _ensure_punc(s: str, pause_s: float):
    s=s.rstrip()
    if not s: return s
    last=s[-1]
    if last in "。！？!?，,、；;":
        if last in "。!！?" and pause_s <= 0.3: s = s[:-1]+"，"
        if last in "，,、" and pause_s >= 0.6: s = s[:-1]+"。"
        return s
    if pause_s >= 0.6: return s+"。"
    elif pause_s <= 0.3: return s+"，"
    else: return s
def stitch_transcripts(chunks: List[Dict[str,Any]], overlap_s: float = 1.2,
                       dedup_method: str = "jaccard", dedup_thr: float = 0.6,
                       apply_punc_fix: bool = True) -> Dict[str,Any]:
    if not chunks: return {"text":"", "segments":[]}
    out=[dict(chunks[0])]
    for i in range(1,len(chunks)):
        prev=out[-1]; cur=dict(chunks[i])
        pause=max(0.0, cur["start"]-prev["end"])
        tail=_normalize(prev.get("text",""))[-50:]; head=_normalize(cur.get("text",""))[:50]
        cut,sim=_pick_dedup_offset(tail,head,method=dedup_method,thr=dedup_thr)
        if cut>0: cur["text"]=cur["text"][cut:]
        if apply_punc_fix: prev["text"]=_ensure_punc(prev.get("text",""), pause)
        merged=(prev.get("text","").rstrip()+" "+cur.get("text","").lstrip()).strip()
        out[-1]["text"]=merged; out[-1]["end"]=cur["end"]
    full="".join(seg.get("text","") for seg in out)
    return {"text":full,"segments":out}
