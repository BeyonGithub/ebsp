
# Emergency Broadcast Segmentation — PoC Pack (v5 + Plus)

包含：
- 训练骨架（样本权重 / CTC 对齐蒸馏 stub / LM & 热词配置）
- 讯飞 JSON → TSV & TextGrid 清洗脚本
- 三套基座模板（Paraformer / Conformer-CTC / RNNT）
- **新增**：长度分桶调度器、重叠去重 + 标点修正模块，及可运行演示脚本

## 目录概览
training/
  prepare/iflytek_json_to_tsv.py
  prepare/textgrid_writer.py
  prepare/make_textgrid_from_tsv.py
  configs/paraformer/{train.yaml, decode.yaml}
  configs/conformer_ctc/{train.yaml, decode.yaml}
  configs/rnnt/{train.yaml, decode.yaml}
  configs/hotwords/hotwords.txt
  configs/lm/kenlm.arpa
  scripts/{train_asr_paraformer.py, train_asr_conformer_ctc.py, train_asr_rnnt.py, ctc_align_distill_stub.py, decode_with_lm_hotword.py}
  run/run_paraformer.sh|.bat
  run/run_conformer_ctc.sh|.bat
  run/run_rnnt.sh|.bat

segmentation/
  inference/length_bucket_scheduler.py          # 新增
  postprocess/overlap_stitcher.py               # 新增

scripts/run_llm_bucketing_demo.py               # 新增


## FireRed 客户端占位（切片缓存 + 并行调度）
- 代码：`segmentation/inference/firered_client_stub.py`
- 一键演示：
```bash
python -m segmentation.inference.firered_client_stub --wav data/audio/demo.wav --minutes 60 --devices 0,1 --out data/sample/llm_stub_output.json
# 或
python scripts/run_firered_stub_demo.py --wav data/audio/demo.wav --minutes 60
```
说明：默认使用“假模型”生成文本；接入真实 FireRed 时，实现/注入 `infer_fn(items, device_id)`，items 中包含 `cache_wav` 切片路径可直接读取。


### Windows/WinFix 建议
- 避免卡住：使用 **线程执行器 + 关闭 ffmpeg 切片** 或 **workers=0** 单线程。
- 示例：
```bat
python -m segmentation.inference.firered_client_stub ^
  --wav data\audio\demo.wav ^
  --minutes 20 ^
  --workers 0 ^
  --no_ffmpeg ^
  --batch_size 1 ^
  --out data\sample\llm_stub_output.json
```
或者启用线程并行：
```bat
python -m segmentation.inference.firered_client_stub ^
  --wav data\audio\demo.wav ^
  --minutes 20 ^
  --workers 8 ^
  --use_threads ^
  --no_ffmpeg ^
  --out data\sample\llm_stub_output.json
```


## 真实对接 FireRed（CLI 模板方式）
1) 打开 `training/configs/firered_cli.yaml`，把 `repo/model/llm/cmd_template` 按你的本地路径与脚本名修改。
2) 测试（LLM-L；Windows 推荐加 `--workers 0 --no_ffmpeg`）:
```bat
python scripts\run_firered_cli_demo.py --wav data\audio\demo.wav --use llm --minutes 0.5 --workers 0 --no_ffmpeg
```
或测试 AED-L：
```bat
python scripts\run_firered_cli_demo.py --wav data\audio\demo.wav --use aed --minutes 0.5 --workers 0 --no_ffmpeg
```
成功后把 `run_firered_stub_demo.py` 的 infer_fn 换成上述 `infer_llm_cli`/`infer_aed_cli` 即可跑整小时节目。
