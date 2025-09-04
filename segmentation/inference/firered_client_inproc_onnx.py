import logging
import os
import time
from typing import Dict, List, Optional

import torch

# FireRed wrapper
from fireredasr.models.fireredasr import FireRedAsr

from .onnx_encoder_patch import patch_model_encoder_with_onnx


class FireRedInprocOnnxRunner:
    """
    In-process FireRed AED runner with ONNX Runtime encoder patch.
    One-time load, repeated transcribe(items).
    """
    def __init__(self,
                 asr_type: str,
                 model_dir: str,
                 onnx_encoder: str,
                 decode_opts: Optional[Dict] = None,
                 onnx_threads: Optional[Dict] = None,
                 providers: Optional[list] = None,
                 batch: int = 4,
                 bucket_bounds: Optional[List[int]] = None):
        self.asr_type = asr_type
        self.model_dir = model_dir
        self.onnx_encoder = onnx_encoder
        self.decode_opts = decode_opts or {}
        self.batch = int(batch)
        self.bucket_bounds = bucket_bounds or [10, 15, 20, 30, 45]
        self.onnx_threads = onnx_threads or {"intra": 8, "inter": 1}
        self.providers = providers or ["CPUExecutionProvider"]

        t0 = time.time()
        logging.info("[ONNXRunner] Loading FireRed model: type=%s dir=%s", asr_type, model_dir)
        self.model = FireRedAsr.from_pretrained(asr_type, model_dir)
        logging.info("[ONNXRunner] Model loaded in %.2fs", time.time() - t0)

        # CPU / CUDA hint
        if torch.cuda.is_available():
            logging.info("[ONNXRunner] CUDA available; decoder will use CUDA if model enabled.")
        else:
            logging.info("[ONNXRunner] CPU mode")

        # Patch encoder to ORT
        session_kwargs = {
            "intra_threads": int(self.onnx_threads.get("intra", 8)),
            "inter_threads": int(self.onnx_threads.get("inter", 1)),
            "providers": self.providers,
            "arena": "disabled",
        }
        self.model = patch_model_encoder_with_onnx(self.model, self.onnx_encoder, session_kwargs=session_kwargs)

    def _flush_batch(self, uttids: List[str], wavs: List[str]) -> List[Dict]:
        if not uttids:
            return []
        t0 = time.time()
        outs = self.model.transcribe(uttids, wavs, self.decode_opts)
        dt = time.time() - t0
        logging.info("[ONNXRunner] batch=%d took %.2fs", len(uttids), dt)
        return outs

    def infer_items(self, items: List[Dict]) -> List[Dict]:
        """
        items: list of {"uttid": str, "path": wav_path}
        """
        N = len(items)
        logging.info("[ONNXRunner] infer_items: N=%d batch=%d", N, self.batch)
        outs_all: List[Dict] = []

        # Simple bucketing by (rounded) duration to reduce padding
        def _dur_sec(p):
            try:
                bn = os.path.basename(p)
                # our cache names like *_start_end.wav; it's ok if we don't parse
                return 20.0
            except Exception:
                return 20.0

        buf_u, buf_w = [], []
        t0 = time.time()
        for it in items:
            uid = it.get("uttid") or it.get("id") or os.path.basename(it["path"]).replace(".wav", "")
            buf_u.append(uid)
            buf_w.append(it["path"])
            if len(buf_u) >= self.batch:
                outs_all.extend(self._flush_batch(buf_u, buf_w))
                buf_u, buf_w = [], []
        if buf_u:
            outs_all.extend(self._flush_batch(buf_u, buf_w))

        logging.info("[ONNXRunner] infer complete: total=%d in %.2fs", N, time.time() - t0)
        return outs_all
