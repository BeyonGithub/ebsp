# -*- coding: utf-8 -*-
import argparse, csv
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tsv", required=True); ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()
    if args.dry_run:
        with open(args.tsv, "r", encoding="utf-8") as f:
            rd = csv.DictReader(f, delimiter="\t")
            for i,r in enumerate(rd):
                if i>=5: break
                print(f"[{i}] wav={r['wav']} {r['start']}-{r['end']} text='{r['text'][:30]}'... weight={r['weight']}")
        return
    print("python -m funasr.bin.train --config training/configs/rnnt/train.yaml --train_data {}".format(args.tsv))
if __name__=="__main__": main()
