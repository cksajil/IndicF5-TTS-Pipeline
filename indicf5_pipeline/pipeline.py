from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
from transformers import AutoModel

MODEL_REPO = "ai4bharat/IndicF5"
SAMPLE_RATE = 24000

_model = None


def _load_model():
    global _model
    if _model is None:
        _model = AutoModel.from_pretrained(MODEL_REPO, trust_remote_code=True)
    return _model


def synthesize(
    texts: list[str],
    output_dir: str | Path,
    ref_audio_path: str | Path,
    ref_text: str,
) -> list[Path]:
    """Generate one WAV file per input text, cloning the voice from ref_audio_path.

    Returns the list of output paths in the same order as `texts`.
    """
    if not texts:
        raise ValueError("texts must contain at least one string")

    ref_audio_path = Path(ref_audio_path)
    if not ref_audio_path.exists():
        raise FileNotFoundError(f"Reference audio not found: {ref_audio_path}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = _load_model()

    width = max(3, len(str(len(texts))))
    out_paths = []
    for i, text in enumerate(texts, start=1):
        audio = np.asarray(model(text, ref_audio_path=str(ref_audio_path), ref_text=ref_text))
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        out_path = output_dir / f"{i:0{width}d}.wav"
        sf.write(out_path, audio.astype(np.float32), samplerate=SAMPLE_RATE)
        out_paths.append(out_path)

    return out_paths
