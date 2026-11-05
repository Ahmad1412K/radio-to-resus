"""Transcribe the audio samples in examples/ with faster-whisper.

Writes, per file, examples/transcripts/<name>.txt (plain transcript) and
<name>.json (segments + word timestamps + word confidences, for later
error attribution and self-repair analysis).
"""
import json
import pathlib
import sys
import time

import av
import numpy as np
from faster_whisper import WhisperModel


def load_audio(path, sr=16000):
    """Decode any container to 16 kHz mono float32.

    Done here rather than by faster-whisper: its decoder calls av.open() with an
    argument PyAV 15+ no longer accepts, and older PyAV has no Python 3.12 wheel.
    """
    resampler = av.AudioResampler(format="s16", layout="mono", rate=sr)
    chunks = []
    with av.open(str(path)) as container:
        for frame in container.decode(audio=0):
            for out in resampler.resample(frame):
                chunks.append(out.to_ndarray().reshape(-1))
        for out in resampler.resample(None):
            chunks.append(out.to_ndarray().reshape(-1))
    return np.concatenate(chunks).astype(np.float32) / 32768.0


ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "examples"
OUT = SRC / "transcripts"
MODEL = sys.argv[1] if len(sys.argv) > 1 else "large-v3-turbo"

OUT.mkdir(exist_ok=True)
model = WhisperModel(MODEL, device="cpu", compute_type="int8")

for path in sorted(p for p in SRC.iterdir() if p.suffix.lower() in {".flac", ".m4a", ".mp4", ".wav", ".mp3"}):
    t0 = time.time()
    segments, info = model.transcribe(load_audio(path), language="en", beam_size=5,
                                      word_timestamps=True, vad_filter=False)
    segs = []
    for s in segments:
        segs.append({
            "start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip(),
            "avg_logprob": round(s.avg_logprob, 3), "no_speech_prob": round(s.no_speech_prob, 3),
            "words": [{"w": w.word.strip(), "start": round(w.start, 2), "end": round(w.end, 2),
                       "p": round(w.probability, 3)} for w in (s.words or [])],
        })
    text = "\n".join(s["text"] for s in segs)
    (OUT / f"{path.stem}{path.suffix.replace('.', '_')}.txt").write_text(text + "\n", encoding="utf-8")
    (OUT / f"{path.stem}{path.suffix.replace('.', '_')}.json").write_text(json.dumps({
        "file": path.name, "model": MODEL, "duration_s": round(info.duration, 1),
        "language": info.language, "language_probability": round(info.language_probability, 3),
        "segments": segs}, indent=2), encoding="utf-8")
    words = [w for s in segs for w in s["words"]]
    low = sum(1 for w in words if w["p"] < 0.5)
    print(f"{path.name:22s} {info.duration:6.1f}s audio  {time.time() - t0:6.1f}s compute  "
          f"{len(words)} words, {low} below 0.5 confidence")
