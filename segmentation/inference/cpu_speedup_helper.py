# -*- coding: utf-8 -*-
"""
segmentation/inference/cpu_speedup_helper.py
- CPU 线程与亲和力调优
- 推理前的 eval() 冻结
- Encoder 线性层动态量化（qint8），对准确率影响极小、CPU 常见 1.2~1.6x
"""
import os
import logging
import torch
import torch.nn as nn

def tune_cpu(threads: int | None = None, interop: int | None = None):
    try:
        if threads:
            threads = int(threads)
            torch.set_num_threads(threads)
            os.environ["OMP_NUM_THREADS"] = str(threads)
            os.environ["MKL_NUM_THREADS"] = str(threads)
            os.environ["OPENBLAS_NUM_THREADS"] = str(threads)
            os.environ["NUMEXPR_NUM_THREADS"] = str(threads)
        if interop:
            torch.set_num_interop_threads(int(interop))
    except Exception as e:
        logging.warning("[CPU] tune threads warn: %s", e)
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    logging.info("[CPU] threads=%s interop=%s", torch.get_num_threads(), torch.get_num_interop_threads())

def apply_dynamic_quant(module: nn.Module):
    try:
        q = torch.quantization.quantize_dynamic(module, {nn.Linear}, dtype=torch.qint8)
        logging.info("[CPU] dynamic quantization applied on Linear layers")
        return q
    except Exception as e:
        logging.warning("[CPU] dynamic quantization failed: %s", e)
        return module

def optimize_for_inference(module: nn.Module):
    try:
        module.eval()
        for p in module.parameters(): 
            p.requires_grad_(False)
    except Exception:
        pass
    return module
