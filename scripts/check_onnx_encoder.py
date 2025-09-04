import argparse
import time
import logging

import numpy as np
import onnxruntime as ort


def run_once(sess, B, T, F, feat_scale=1.0):
    x = np.random.randn(B, T, F).astype("float32") * feat_scale
    ilen = np.full((B,), T, dtype="int64")
    t0 = time.time()
    enc_out, enc_len = sess.run([sess.get_outputs()[0].name, sess.get_outputs()[1].name],
                                {sess.get_inputs()[0].name: x, sess.get_inputs()[1].name: ilen})
    dt = time.time() - t0
    return enc_out, enc_len, dt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--feat_dim", type=int, default=80)
    ap.add_argument("--feat_hz", type=int, default=100)
    ap.add_argument("--seconds", type=str, default="10,20,30")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--provider", default="CPUExecutionProvider")
    ap.add_argument("--log", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(level=getattr(logging, args.log.upper(), logging.INFO),
                        format="%(asctime)s %(levelname)s %(message)s")

    providers = [args.provider]
    sess = ort.InferenceSession(args.onnx, providers=providers)
    logging.info("[Check] Providers: %s", sess.get_providers())
    logging.info("[Check] I/O: in=%s out=%s", [i.name for i in sess.get_inputs()],
                 [o.name for o in sess.get_outputs()])

    secs = [float(s) for s in args.seconds.split(",")]
    for s in secs:
        T = int(s * args.feat_hz)
        logging.info("[Check] ---- seconds=%.1f (T=%d frames) ----", s, T)
        # warmup
        for _ in range(args.warmup):
            run_once(sess, args.batch, T, args.feat_dim)
        # runs
        times = []
        total_frames = 0
        for _ in range(args.runs):
            enc_out, enc_len, dt = run_once(sess, args.batch, T, args.feat_dim)
            times.append(dt)
            total_frames += int(enc_len.sum())
        lat = np.mean(times)
        p95 = float(np.percentile(times, 95))
        fps = total_frames / sum(times)
        logging.info("[Check] enc_out shape=%s enc_len=%s", enc_out.shape, enc_len)
        logging.info("[Check] avg=%.4fs p95=%.4fs throughput=%.1f enc_frames/s", lat, p95, fps)

if __name__ == "__main__":
    main()
