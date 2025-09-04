# -*- coding: utf-8 -*-
import argparse, os, json, csv
def mock_align(tsv, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    with open(tsv, "r", encoding="utf-8") as f:
        rd = csv.DictReader(f, delimiter="\t")
        for i,r in enumerate(rd):
            uid = f"utt_{i:06d}"
            nframes = max(10, int((float(r["end"])-float(r["start"])) * 100))
            toks = list(r["text"])[:80]
            steps = max(1, nframes // max(1,len(toks)))
            align = [{"t": t, "frame": j*steps} for j,t in enumerate(toks)]
            json.dump({"uid":uid,"align":align}, open(os.path.join(out_dir, uid+".json"),"w",encoding="utf-8"), ensure_ascii=False, indent=2)
    print("[OK] wrote mock alignments to", out_dir)
if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", required=True); ap.add_argument("--out_dir", required=True)
    args = ap.parse_args(); mock_align(args.tsv, args.out_dir)
