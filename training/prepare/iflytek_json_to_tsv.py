# -*- coding: utf-8 -*-
import os, json, argparse, glob
from collections import defaultdict
from typing import List, Dict, Any, Tuple

def safe_float(x, default=0.0):
    try: return float(x)
    except: return default

def parse_generic(js: Dict[str,Any]):
    words = js.get("words", [])
    utt_conf = safe_float(js.get("utt_conf", js.get("confidence", 0.9)))
    items = []
    for w in words:
        items.append({
            "text": w.get("text") or w.get("w") or "",
            "start": safe_float(w.get("start", w.get("bg", 0.0))),
            "end": safe_float(w.get("end", w.get("ed", 0.0))),
            "conf": safe_float(w.get("conf", w.get("sc", 90.0))) / (1.0 if "conf" in w else 100.0)
        })
    if not items and js.get("text"):
        items = [{"text": js["text"], "start": 0.0, "end": 0.0, "conf": utt_conf}]
    return items, utt_conf

def parse_iflytek(js: Dict[str,Any]):
    data = js.get("data") or js
    ws = data.get("ws", [])
    items = []
    for seg in ws:
        bg = safe_float(seg.get("bg", 0.0)) / (1000.0 if seg.get("bg", None) else 1.0)
        ed = safe_float(seg.get("ed", 0.0)) / (1000.0 if seg.get("ed", None) else 1.0)
        cws = seg.get("cw", [])
        if not cws: continue
        best = max(cws, key=lambda x: x.get("sc", 0.0))
        items.append({
            "text": best.get("w",""),
            "start": bg, "end": ed,
            "conf": safe_float(best.get("sc", 90.0)) / 100.0
        })
    utt_conf = sum([x["conf"] for x in items]) / max(1, len(items)) if items else 0.9
    return items, utt_conf

def read_json(path: str):
    try:
        return json.load(open(path, "r", encoding="utf-8"))
    except Exception:
        with open(path, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
        js = {"words":[]}; t=0.0
        for ln in lines:
            try:
                o=json.loads(ln); js["words"].append({
                    "text": o.get("text",""), "start": o.get("start", t), "end": o.get("end", t+1.0),
                    "conf": o.get("conf", 0.9)
                }); t=o.get("end", t+1.0)
            except: pass
        return js

def sentence_pack(words: List[Dict[str,Any]], max_gap=0.8, max_len=32.0):
    segs=[]; cur=[]
    for w in words:
        if not cur: cur=[w]; continue
        gap = w["start"] - cur[-1]["end"]
        dur = w["end"] - cur[0]["start"]
        if gap <= max_gap and dur <= max_len:
            cur.append(w)
        else:
            segs.append(cur); cur=[w]
    if cur: segs.append(cur)
    return segs

def detok_ch(words): return "".join(w["text"] for w in words)
def seg_conf(words): return sum(w["conf"] for w in words)/max(1,len(words))

def process_dir(json_dir, audio_root, out_tsv, out_meta, make_textgrid=False, domain="radio", region="NA", spk="unknown"):
    from pathlib import Path
    from textgrid_writer import write_textgrid

    count=0; kept=0; bad=0
    rows=[]
    json_files = sorted(glob.glob(os.path.join(json_dir, "**", "*.json"), recursive=True))
    for jf in json_files:
        js = read_json(jf)
        words, utt_conf = (parse_iflytek(js) if ("ws" in json.dumps(js) or "cw" in json.dumps(js)) else parse_generic(js))
        if not words: continue
        segs = sentence_pack(words, max_gap=0.8, max_len=32.0)
        base = os.path.splitext(os.path.basename(jf))[0]
        wav = None
        for ext in [".wav",".flac",".mp3"]:
            p = os.path.join(audio_root, base+ext)
            if os.path.exists(p): wav=p; break
        if wav is None: bad+=1; continue
        for si, seg in enumerate(segs):
            start = seg[0]["start"]; end = seg[-1]["end"]
            text = detok_ch(seg); weight = round(max(0.2, min(1.1, seg_conf(seg))), 3)
            rows.append([wav, f"{start:.2f}", f"{end:.2f}", text, f"{weight:.3f}", domain, region, spk])
            count+=1
        kept+=1
        if make_textgrid:
            tg_path = os.path.join(os.path.dirname(wav), base + ".TextGrid")
            intervals = [(seg[0]["start"], seg[-1]["end"], detok_ch(seg)) for seg in segs]
            try:
                Path(os.path.dirname(tg_path)).mkdir(parents=True, exist_ok=True)
                write_textgrid(tg_path, intervals)
            except Exception: pass

    with open(out_tsv, "w", encoding="utf-8") as f:
        f.write("wav\tstart\tend\ttext\tweight\tdomain\tregion\tspk\n")
        for r in rows: f.write("\t".join(map(str,r))+"\n")
    meta = {"files_processed": kept, "segments": count, "missing_audio": bad}
    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print("[OK] TSV:", out_tsv, "segments:", count, "files:", kept, "missing_audio:", bad)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--json_dir", required=True)
    ap.add_argument("--audio_root", required=True)
    ap.add_argument("--out_tsv", required=True)
    ap.add_argument("--out_meta", required=True)
    ap.add_argument("--make_textgrid", action="store_true")
    ap.add_argument("--domain", default="radio")
    ap.add_argument("--region", default="NA")
    ap.add_argument("--spk", default="unknown")
    args = ap.parse_args()
    process_dir(args.json_dir, args.audio_root, args.out_tsv, args.out_meta, args.make_textgrid, args.domain, args.region, args.spk)
