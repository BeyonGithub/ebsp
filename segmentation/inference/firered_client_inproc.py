# -*- coding: utf-8 -*-
"""
segmentation/inference/firered_client_inproc.py — 进程内推理（含耗时日志）
新增：
- 统一慢步骤门限 SLOW_SEC（env: ASR_LOG_SLOW_SEC，默认 3s）
- 关键环节耗时：info/重采样/保存、微分片推理每段、整批推理总时长、RTF
"""
from __future__ import annotations
import os, io, logging, tempfile, time
from typing import List, Dict, Any, Optional, Tuple

SLOW_SEC = float(os.getenv("ASR_LOG_SLOW_SEC", "3.0"))

def _log_dur(t0: float, label: str, extra: str = ""):
    dt = time.perf_counter() - t0
    msg = f"{label} took {dt:.2f}s"
    if extra:
        msg += f" {extra}"
    if dt >= SLOW_SEC:
        logging.warning("[Inproc][SLOW] %s", msg)
    else:
        logging.info("[Inproc] %s", msg)
    return dt

try:
    import torchaudio
except Exception as e:
    torchaudio = None
    logging.warning("[Inproc] torchaudio not available: %s", e)

def _save_pcm16(wav_tensor, sr, path):
    t0 = time.perf_counter()
    if torchaudio is None:
        raise RuntimeError("torchaudio not available for saving wav")
    try:
        torchaudio.save(path, wav_tensor.to('cpu'), sr, encoding='PCM_S', bits_per_sample=16)
    except TypeError:
        torchaudio.save(path, wav_tensor.to('cpu'), sr)
    _log_dur(t0, "save PCM16", f"-> {os.path.basename(path)}")

def _ensure_16k_mono(wav_path: str, cache_dir: str = None) -> str:
    """确保输入是 16k/mono/PCM16；若不满足则重采样并另存临时文件"""
    if torchaudio is None:
        return wav_path
    t0 = time.perf_counter()
    try:
        info = torchaudio.info(wav_path)
        sr = info.sample_rate
        ch = info.num_channels
        bps = getattr(info, "bits_per_sample", None)
    except Exception:
        sr = None; ch = None; bps = None
    _log_dur(t0, "torchaudio.info", f"sr={sr} ch={ch} bps={bps} file={os.path.basename(wav_path)}")

    need = (sr != 16000 or ch != 1 or bps == 24)
    if not need:
        return wav_path

    try:
        t1 = time.perf_counter()
        wav, sr0 = torchaudio.load(wav_path)
        _log_dur(t1, "load wav")
        if sr0 != 16000:
            t2 = time.perf_counter()
            wav = torchaudio.functional.resample(wav, sr0, 16000)
            _log_dur(t2, "resample", f"{sr0} -> 16000")
        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)
        base = cache_dir or os.getenv("ASR_TMP_DIR") or tempfile.gettempdir()
        os.makedirs(base, exist_ok=True)
        tmp_out = tempfile.NamedTemporaryFile(prefix="resamp16k_", suffix=".wav", delete=False, dir=base)
        tmp_out.close()
        _save_pcm16(wav, 16000, tmp_out.name)
        logging.info("[Inproc][Guard] Saved PCM16 16k mono: %s -> %s", wav_path, tmp_out.name)
        return tmp_out.name
    except Exception as e:
        logging.warning("[Inproc][Guard] resample/save failed (%s), fallback original: %s", e, wav_path)
        return wav_path

def _estimate_frames_10ms(wav_path: str) -> int:
    if torchaudio is None:
        return 0
    try:
        info = torchaudio.info(wav_path)
        sr = info.sample_rate
        nframes = info.num_frames
        if sr <= 0: return 0
        dur = nframes / float(sr)
    except Exception:
        return 0
    return int(dur * 100)

def _micro_chunk_and_decode(model, uttid: str, wav_path: str, opts: dict,
                            micro_sec: float = 15.0, micro_overlap: float = 0.5) -> Dict[str, Any]:
    """对长切片做 15s 微分片并串行解码，最后拼成一段文本（含每段耗时&RTF）"""
    if torchaudio is None:
        hyps = model.transcribe([uttid], [wav_path], opts)
        return hyps[0] if isinstance(hyps, list) else hyps

    info = torchaudio.info(wav_path)
    sr = info.sample_rate
    total = info.num_frames / float(sr)
    if total <= micro_sec + 1e-6:
        hyps = model.transcribe([uttid], [wav_path], opts)
        return hyps[0] if isinstance(hyps, list) else hyps

    step = micro_sec - micro_overlap
    times = []
    s = 0.0
    while s < total:
        e = min(total, s + micro_sec)
        times.append((s, e))
        if e >= total: break
        s = s + step
    out_text = []
    wav, _ = torchaudio.load(wav_path)
    for i, (ss, ee) in enumerate(times):
        s_idx = int(ss * sr); e_idx = int(ee * sr)
        cut = wav[:, s_idx:e_idx]
        tmp = tempfile.NamedTemporaryFile(prefix=f"micro_{i:03d}_", suffix=".wav", delete=False)
        tmp.close()
        torchaudio.save(tmp.name, cut, sr)
        t0 = time.perf_counter()
        hyp = model.transcribe([f"{uttid}_{i:03d}"], [tmp.name], opts)
        dt = _log_dur(t0, "micro-decode", f"seg={i}/{len(times)} dur={ee-ss:.2f}s RTF={dt/(ee-ss):.2f}" if (ee-ss)>0 else "")
        seg_text = ""
        if isinstance(hyp, list) and hyp:
            if isinstance(hyp[0], dict) and "text" in hyp[0]:
                seg_text = hyp[0]["text"]
            else:
                seg_text = str(hyp[0])
        elif isinstance(hyp, dict) and "text" in hyp:
            seg_text = hyp["text"]
        out_text.append(seg_text)
    return {"uttid": uttid, "text": "".join(out_text)}

def _coalesce_text_from_hyp(h) -> str:
    if h is None:
        return ""
    if isinstance(h, dict):
        if "text" in h and isinstance(h["text"], str):
            return h["text"]
        if "nbest" in h and isinstance(h["nbest"], list) and h["nbest"]:
            nb = h["nbest"][0]
            if isinstance(nb, dict) and "text" in nb:
                return nb["text"]
    if isinstance(h, list) and h and isinstance(h[0], dict) and "text" in h[0]:
        return h[0]["text"]
    if isinstance(h, str):
        return h
    return str(h)

class FireRedInprocRunner:
    def __init__(self, asr_type: str, model_dir: str, device: str = "cpu",
                 batch: int = 6, beam_size: int = 1, decode_max_len: int = 256):
        from fireredasr.models.fireredasr import FireRedAsr
        self.decode_opts = {
            "use_gpu": 0 if (device == "cpu" or str(device).startswith("cpu")) else 1,
            "beam_size": beam_size,
            "nbest": 1,
            "decode_max_len": decode_max_len,
        }
        t0 = time.perf_counter()
        logging.info("[Inproc] Loading FireRed model: type=%s dir=%s", asr_type, model_dir)
        self.model = FireRedAsr.from_pretrained(asr_type, model_dir)
        _log_dur(t0, "FireRedAsr.from_pretrained")
        logging.info("[Inproc] %s mode", "CPU" if self.decode_opts["use_gpu"] == 0 else "GPU")
        th = int(os.getenv("OMP_NUM_THREADS", "8"))
        it = int(os.getenv("MKL_NUM_THREADS", "8"))
        logging.info("[CPU] threads=%d interop=%d", th, it)
        self.batch = max(1, int(batch))

    def _run_batch(self, uttids: List[str], wavs: List[str]) -> List[Dict[str, Any]]:
        t0 = time.perf_counter()
        safe_paths = []
        for p in wavs:
            s0 = time.perf_counter()
            safe = _ensure_16k_mono(p)
            _log_dur(s0, "guard 16k/mono", os.path.basename(p))
            safe_paths.append(safe)

        outs = []
        for u, p in zip(uttids, safe_paths):
            frames = _estimate_frames_10ms(p)
            dur = frames / 100.0
            t1 = time.perf_counter()
            if frames >= 3600:  # >= 360s
                logging.warning("[Inproc][Guard] long slice (~%.1fs), micro-chunk decode.", dur)
                hyp = _micro_chunk_and_decode(self.model, u, p, self.decode_opts, micro_sec=15.0, micro_overlap=0.5)
                txt = _coalesce_text_from_hyp(hyp)
            else:
                hyp = self.model.transcribe([u], [p], self.decode_opts)
                txt = _coalesce_text_from_hyp(hyp[0] if isinstance(hyp, list) and hyp else hyp)
            dt = time.perf_counter() - t1
            rtf = dt / max(0.01, dur)
            if dt >= SLOW_SEC:
                logging.warning("[Inproc][SLOW] decode %s took %.2fs (dur=%.2fs RTF=%.2f)", u, dt, dur, rtf)
            else:
                logging.info("[Inproc] decode %s took %.2fs (dur=%.2fs RTF=%.2f)", u, dt, dur, rtf)
            outs.append({"uttid": u, "text": txt})
        _log_dur(t0, "batch total", f"batch={len(uttids)}")
        return outs

    @staticmethod
    def _pick(it: Dict[str, Any], *keys, default=None):
        for k in keys:
            if k in it and it[k] is not None:
                return it[k]
        return default

    def infer_items(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """items: [{'uttid', 'cache_wav', 'start','end',...}, ...]"""
        logging.info("[Inproc] infer_items: N=%d", len(items))
        buf_u, buf_w = [], []
        outs_all = []
        for it in items:
            uid = self._pick(it, "uttid", "uid", "id", "segment_id", "seg_id")
            wavp = self._pick(it, "cache_wav", "wav", "path", "audio_path")
            if not uid:
                base = os.path.splitext(os.path.basename(wavp or "chunk.wav"))[0]
                st = self._pick(it, "start", "st", default=0.0)
                ed = self._pick(it, "end", "et", default=0.0)
                uid = f"{base}_{float(st):.2f}-{float(ed):.2f}s"
                logging.debug("[Inproc] composed uttid=%s", uid)
            if not wavp or not os.path.exists(wavp):
                raise FileNotFoundError(f"[Inproc] invalid wav path: {wavp} (uid={uid})")
            buf_u.append(uid)
            buf_w.append(wavp)
            if len(buf_u) >= self.batch:
                outs_all.extend(self._run_batch(buf_u, buf_w))
                buf_u, buf_w = [], []
        if buf_u:
            outs_all.extend(self._run_batch(buf_u, buf_w))
        logging.info("[Inproc] infer_items done: N_out=%d", len(outs_all))
        return outs_all
