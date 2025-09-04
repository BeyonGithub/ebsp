# Optional helper: you can import this at the top of any script if you
# prefer explicit control instead of sitecustomize.
import os, logging
os.environ.setdefault("TORCHAUDIO_USE_SOUNDFILE", "1")
logging.getLogger("torio").setLevel(logging.ERROR)
try:
    import torchaudio
    try:
        torchaudio.set_audio_backend("soundfile")
    except Exception:
        pass
except Exception:
    pass