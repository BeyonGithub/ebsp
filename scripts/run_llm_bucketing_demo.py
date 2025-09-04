#!/usr/bin/env python
# -*- coding: utf-8 -*-
import random
from segmentation.inference.length_bucket_scheduler import make_buckets, schedule_batches
from segmentation.postprocess.overlap_stitcher import stitch_transcripts

def fake_chunks(total_minutes=1.5, target_len=26.0, overlap=1.2):
    total_s = total_minutes*60
    t = 0.0; sid=1; chunks=[]
    while t < total_s:
        seg = target_len + random.uniform(-2.0, 2.0)
        st, ed = t, min(t+seg, total_s)
        chunks.append({"sid": sid, "wav":"data/audio/demo.wav", "start": round(st,2), "end": round(ed,2), "overlap_s": overlap})
        t = ed - overlap; sid += 1
    return chunks

def fake_infer(batch):
    out=[]
    for ch in batch:
        core = "本期节目带来森林防火与交通安全知识，" if (ch['sid']%2==1) else "我们关注应急演练与乡村建设进展，"
        txt = ("感谢收听，" if ch['sid']>1 else "") + core + ("感谢收听，" if ch['sid']<99 else "")
        out.append({"start": ch["start"], "end": ch["end"], "text": txt, "conf": 0.92})
    return out

def main():
    chunks = fake_chunks(total_minutes=1.5, target_len=26.0, overlap=1.2)
    buckets = make_buckets(chunks, bucket_ranges=[(22,24),(24,26),(26,28),(28,30)])
    batches = schedule_batches(buckets, batch_size=1, shuffle=False)
    results=[]
    for bt in batches:
        results.extend(fake_infer(bt["items"]))
    stitched = stitch_transcripts(results, overlap_s=1.2, dedup_method="jaccard", dedup_thr=0.6, apply_punc_fix=True)
    print("[OK] segments:", len(stitched["segments"]))
    print("[TEXT]", stitched["text"][:160] + ("..." if len(stitched["text"])>160 else ""))

if __name__ == "__main__":
    main()
