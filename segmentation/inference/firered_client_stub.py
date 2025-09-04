import logging
import os
import time
from typing import Dict, List, Optional
import soundfile as sf
import numpy as np

# Defaults
DEFAULT_SEG_SEC = 30.0
DEFAULT_OVERLAP_SEC = 0.5

def _ensure_16k_mono(wav_path: str) -> np.ndarray:
    """Load arbitrary WAV and return float32 mono 16k waveform array."""
    t0 = time.time()
    data, sr = sf.read(wav_path, always_2d=False)
    if data.dtype != np.float32:
        data = data.astype(np.float32, copy=False)
    if data.ndim == 2:
        data = np.mean(data, axis=1, dtype=np.float32)
    if sr != 16000:
        # naive linear resample using numpy (good enough for stub); real impl should use torchaudio/soxr
        import math
        ratio = 16000.0 / sr
        new_len = int(math.floor(len(data) * ratio))
        x = np.linspace(0, 1, len(data), endpoint=False)
        xi = np.linspace(0, 1, new_len, endpoint=False)
        data = np.interp(xi, x, data).astype(np.float32)
    logging.debug("[Stub] _ensure_16k_mono: %s -> %d samples (%.2fs) in %.3fs", wav_path, len(data), len(data)/16000.0, time.time()-t0)
    return data

def _save_pcm16(out_wav: str, wav: np.ndarray):
    t0 = time.time()
    # scale to int16
    x = np.clip(wav, -1.0, 1.0)
    x = (x * 32767.0).astype(np.int16)
    sf.write(out_wav, x, 16000, subtype='PCM_16')
    logging.debug("[Stub] _save_pcm16: %s (%.2fs) in %.3fs", out_wav, len(x)/16000.0, time.time()-t0)

def _rms(y: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(y))) + 1e-12)

def _slice_and_cache(wav_path: str,
                     out_dir: str,
                     seg_sec: float,
                     overlap_sec: float,
                     vad_rms_thresh: float = 0.0) -> List[Dict]:
    """Slice long wav into segments, cache to PCM16 16k files, optional VAD by RMS."""
    os.makedirs(out_dir, exist_ok=True)
    wav = _ensure_16k_mono(wav_path)
    n = len(wav)
    hop = int((seg_sec - overlap_sec) * 16000)
    win = int(seg_sec * 16000)
    ovl = int(overlap_sec * 16000)
    items: List[Dict] = []
    t0 = time.time()
    idx = 0
    while idx < n:
        s = idx
        e = min(n, idx + win)
        seg = wav[s:e]
        dur = (e - s) / 16000.0
        if vad_rms_thresh > 0 and _rms(seg) < vad_rms_thresh:
            idx += hop
            continue
        start_sec = s / 16000.0
        end_sec = e / 16000.0
        bn = os.path.splitext(os.path.basename(wav_path))[0]
        out_w = os.path.join(out_dir, f"{bn}_{start_sec:.2f}_{end_sec:.2f}.wav")
        _save_pcm16(out_w, seg)
        items.append({"uttid": f"{bn}_{start_sec:.2f}_{end_sec:.2f}", "path": out_w, "start": start_sec, "end": end_sec})
        if e >= n:
            break
        idx += hop
    logging.warning("[Stub][SLOW] build_slices took %.2fs N=%d total_dur=%.1fs", time.time()-t0, len(items), n/16000.0)
    return items


class FireRedClientStub:
    """
    Thin stub for building slices and delegating inference to runner.
    """
    def __init__(self,
                 seg_sec: Optional[float] = None,
                 overlap_sec: Optional[float] = None,
                 cache_dir: str = "cache/asr_slices",
                 vad_rms_thresh: float = 0.0):
        self.seg_sec = seg_sec if seg_sec is not None else DEFAULT_SEG_SEC
        self.overlap_sec = overlap_sec if overlap_sec is not None else DEFAULT_OVERLAP_SEC
        self.cache_dir = cache_dir
        self.vad_rms_thresh = vad_rms_thresh

    def build_batches(self, wav_path: str) -> List[List[Dict]]:
        os.makedirs(self.cache_dir, exist_ok=True)
        t0 = time.time()
        items = _slice_and_cache(wav_path, self.cache_dir, self.seg_sec, self.overlap_sec, self.vad_rms_thresh)
        logging.warning("[Stub][SLOW] build_batches total took %.2fs slices=%d", time.time()-t0, len(items))
        # single batch, runner will re-batch
        return [items]

    def infer_batches(self, batches, runner):
        results = []
        for i, batch in enumerate(batches):
            logging.info("[Stub] infer batch #%d size=%d", i, len(batch))
            res = runner.infer_items(batch)
            results.extend(res)
        return results
