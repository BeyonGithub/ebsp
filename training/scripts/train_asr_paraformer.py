# -*- coding: utf-8 -*-
import argparse, csv
def dry_run(tsv_path, limit=5):
    n=0; total=0
    with open(tsv_path, "r", encoding="utf-8") as f:
        rd = csv.DictReader(f, delimiter="\t")
        for r in rd:
            total+=1
            if n<limit:
                print(f"[{n}] wav={r['wav']} {r['start']}-{r['end']} weight={r['weight']} text_len={len(r['text'])}")
                n+=1
    print(f"[OK] rows={total}, previewed={n}")
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", required=True)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()
    if args.dry_run:
        dry_run(args.tsv); return
    cmd = "python -m funasr.bin.train --config training/configs/paraformer/train.yaml --train_data {}".format(args.tsv)
    print("[INFO]", cmd)
if __name__=="__main__": main()
