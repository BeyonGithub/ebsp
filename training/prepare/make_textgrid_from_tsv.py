# -*- coding: utf-8 -*-
import argparse, csv, os
from textgrid_writer import write_textgrid
def main(tsv, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    groups = {}
    with open(tsv, "r", encoding="utf-8") as f:
        rd = csv.DictReader(f, delimiter="\t")
        for r in rd:
            wav = r["wav"]; start=float(r["start"]); end=float(r["end"]); text=r["text"]
            key = os.path.splitext(os.path.basename(wav))[0]
            groups.setdefault(key, []).append((start, end, text))
    for k, ints in groups.items():
        ints.sort(key=lambda x: x[0])
        write_textgrid(os.path.join(out_dir, f"{k}.TextGrid"), ints)
    print("[OK] wrote", len(groups), "TextGrids to", out_dir)
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    main(args.tsv, args.out_dir)
