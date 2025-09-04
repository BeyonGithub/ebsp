import logging
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np

try:
    import onnxruntime as ort
except Exception as e:
    ort = None
    logging.warning("[ONNXPatch] onnxruntime not available: %s", e)

import torch


def _build_session(onnx_path: str, providers=None, intra_threads: int = 8, inter_threads: int = 1, arena_strategy: str = "disabled"):
    """
    Build an ONNX Runtime session with reasonable CPU defaults.
    """
    if ort is None:
        raise RuntimeError("onnxruntime is not installed")
    so = ort.SessionOptions()
    so.intra_op_num_threads = intra_threads
    so.inter_op_num_threads = inter_threads
    # graph opt level
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    # memory arena
    if hasattr(ort, "DisableCPUAllocatorArena") and arena_strategy == "disabled":
        try:
            so.add_session_config_entry("session.enable_cpu_mem_arena", "0")
        except Exception:
            pass

    providers = providers or ["CPUExecutionProvider"]
    t0 = time.time()
    sess = ort.InferenceSession(onnx_path, sess_options=so, providers=providers)
    logging.info("[ONNXPatch] Created ORT session in %.2fs providers=%s", time.time() - t0, providers)
    return sess


def _to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x


def _to_torch(x: np.ndarray, like: Optional[torch.Tensor] = None) -> torch.Tensor:
    t = torch.from_numpy(x)
    if like is not None:
        t = t.to(dtype=like.dtype, device=like.device)
    return t


def _make_src_mask_from_lengths(lengths: torch.Tensor, T: int) -> torch.Tensor:
    """
    Make square attention mask (N, T, T) where valid positions are 1, padded 0.
    This matches decoder usage: src_masks.unsqueeze(1).repeat(...)
    """
    N = lengths.shape[0]
    mask = torch.zeros((N, T, T), dtype=torch.bool)  # bool saves memory
    for i, L in enumerate(lengths.tolist()):
        L = int(L)
        L = max(1, min(L, T))
        mask[i, :L, :L] = True
    return mask


def patch_model_encoder_with_onnx(firered_asr_model: Any,
                                  onnx_path: str,
                                  providers=None,
                                  session_kwargs: Optional[Dict[str, Any]] = None) -> Any:
    """
    Monkey-patch FireRed AED model's encoder.forward() with ONNXRuntime session.
    Supports model object at either model.model.encoder or model.encoder.
    Returns the same model instance (patched).
    """
    logging.info("[ONNXPatch] Patching AED encoder with ONNX: %s", onnx_path)
    # Locate encoder module holder. FireRedAsr (wrapper) -> .model -> .encoder
    target_holder = None
    if hasattr(firered_asr_model, "model") and hasattr(firered_asr_model.model, "encoder"):
        target_holder = firered_asr_model.model
    elif hasattr(firered_asr_model, "encoder"):
        target_holder = firered_asr_model
    else:
        raise RuntimeError("Model has no attribute 'encoder' to patch.")

    encoder_mod = target_holder.encoder
    # Keep a reference to original forward just in case
    orig_forward = getattr(encoder_mod, "forward", None)

    # Build session
    session_kwargs = session_kwargs or {}
    intra = int(session_kwargs.get("intra_threads", 8))
    inter = int(session_kwargs.get("inter_threads", 1))
    providers = providers or session_kwargs.get("providers") or ["CPUExecutionProvider"]
    arena = session_kwargs.get("arena", "disabled")
    sess = _build_session(onnx_path, providers=providers, intra_threads=intra, inter_threads=inter, arena_strategy=arena)

    # Try to infer input and output names
    onames = [o.name for o in sess.get_outputs()]
    inames = [i.name for i in sess.get_inputs()]
    logging.info("[ONNXPatch] Inputs=%s Outputs=%s", inames, onames)

    # Heuristics
    # Assume encoder ONNX inputs: "feats" (N,T,F) float32, "lengths" (N,) int64
    # outputs: "enc_out" (N,T',D), optionally "enc_len"
    # Some exports may use different names; handle common variants.
    def _name_or_fallback(candidates, fallback_list):
        for n in candidates:
            if n in fallback_list:
                return n
        # fallback to first
        return fallback_list[0] if fallback_list else None

    in_feats = _name_or_fallback(["feats", "inputs", "x", "speech"], inames)
    in_lens  = _name_or_fallback(["lengths", "feats_lengths", "input_lengths"], inames)

    out_enc  = _name_or_fallback(["enc_out", "encoder_out", "outputs"], onames)
    out_lens = None
    for cand in ["enc_lengths", "lengths_out", "output_lengths"]:
        if cand in onames:
            out_lens = cand
            break

    if in_feats is None:
        raise RuntimeError(f"[ONNXPatch] Cannot find feats input among {inames}")
    if out_enc is None:
        raise RuntimeError(f"[ONNXPatch] Cannot find encoder output among {onames}")

    logging.info("[ONNXPatch] Bind names: feats=%s lens=%s enc_out=%s enc_len=%s",
                 in_feats, in_lens, out_enc, out_lens)

    def ort_forward(feats: torch.Tensor,
                    lengths: torch.Tensor,
                    *args, **kwargs) -> Tuple[torch.Tensor, None, torch.Tensor]:
        """
        Mimic original encoder forward: returns (enc_outputs, None, src_masks)
        - feats: (N,T,F) float
        - lengths: (N,) int
        """
        t0 = time.time()
        feed = {in_feats: _to_numpy(feats)}
        if in_lens is not None and lengths is not None:
            feed[in_lens] = _to_numpy(lengths.to(dtype=torch.int64))
        ort_outs = sess.run(None, feed)
        # Map outputs by name
        name_to_val = {name: val for name, val in zip(onames, ort_outs)}
        enc_np = name_to_val.get(out_enc, ort_outs[0])
        enc = _to_torch(enc_np, like=feats)

        # Lengths
        if out_lens is not None and out_lens in name_to_val:
            L = _to_torch(name_to_val[out_lens], like=lengths).to(dtype=torch.int64, device=lengths.device)
        else:
            # Infer: assume time axis = 1
            Tprime = enc.shape[1]
            L = torch.full((enc.shape[0],), Tprime, dtype=torch.int64, device=enc.device)

        # Build square mask (N, T', T')
        src_mask = _make_src_mask_from_lengths(L.to('cpu'), enc.shape[1]).to(device=enc.device)

        dt = time.time() - t0
        logging.debug("[ONNXPatch] encoder ORT forward took %.3fs  out_shape=%s", dt, tuple(enc.shape))
        return enc, None, src_mask

    # Monkey-patch
    setattr(encoder_mod, "forward", ort_forward)
    logging.info("[ONNXPatch] Patched encoder.forward with ORT session.")
    # stash references to allow unpatch or debug
    setattr(encoder_mod, "_ort_session", sess)
    setattr(encoder_mod, "_orig_forward", orig_forward)
    return firered_asr_model
