# -*- coding: utf-8 -*-
import argparse
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True)
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--hotwords", default=None)
    ap.add_argument("--lm_arpa", default=None)
    args = ap.parse_args()
    cmd = ["python","-m","funasr.bin.recog","--config",args.config,"--model_dir",args.model_dir,"--wav",args.wav]
    if args.hotwords: cmd += ["--hotword", args.hotwords]
    if args.lm_arpa: cmd += ["--lm", args.lm_arpa, "--lm_weight", "0.8"]
    print("[INFO] Decode command:"); print(" ".join(cmd))
if __name__=="__main__": main()
