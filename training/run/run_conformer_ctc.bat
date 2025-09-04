@echo off
python training\prepare\iflytek_json_to_tsv.py --json_dir data\iflytek_json --audio_root data\audio --out_tsv data\train_pairs.tsv --out_meta data\train_meta.json
python -m funasr.bin.train --config training\configs\conformer_ctc\train.yaml --train_data data\train_pairs.tsv --output exp\conformer_ctc
