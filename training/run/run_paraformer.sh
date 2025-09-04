#!/usr/bin/env bash
set -e
python training/prepare/iflytek_json_to_tsv.py --json_dir data/iflytek_json --audio_root data/audio --out_tsv data/train_pairs.tsv --out_meta data/train_meta.json --make_textgrid
python training/scripts/ctc_align_distill_stub.py --tsv data/train_pairs.tsv --out_dir training/align
python -m funasr.bin.train --config training/configs/paraformer/train.yaml --train_data data/train_pairs.tsv --output exp/paraformer
