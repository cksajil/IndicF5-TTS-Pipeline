from __future__ import annotations

import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from transformers import AutoModel

MODEL_REPO = "ai4bharat/IndicF5"
SAMPLE_RATE = 24000
# F5-TTS's own default is 32 ODE steps; 16 is the documented "fast" preset and
# roughly halves per-call compute for a small, usually inaudible quality cost.
DEFAULT_NFE_STEP = 16

_model = None
_patched = False


def _patch_for_cpu_speed(nfe_step: int) -> None:
    """Speed up the vendored f5_tts inference path for CPU-only runs.

    Must run before AutoModel.from_pretrained() triggers the dynamic import of
    IndicF5's remote model.py, since that's the moment it does
    `from f5_tts.infer.utils_infer import infer_process, preprocess_ref_audio_text`
    and binds whatever those names currently point to.

    Note: the vendored model.py wraps the DiT model and vocoder in
    torch.compile(), but infer_batch_process() actually drives them through
    model_obj.sample(...) / vocoder.decode(...) - plain method calls that
    bypass forward()/__call__ entirely, so torch.compile never intercepts
    them and buys nothing either way. It's load-bearing only for the
    ai4bharat/IndicF5 checkpoint's state_dict, whose keys were saved with the
    "_orig_mod." prefix torch.compile adds - so we leave it untouched here
    rather than risk loading the model with random weights.
    """
    global _patched
    if _patched:
        return
    _patched = True

    if torch.cuda.is_available():
        return  # the full step count is worth it on a real GPU

    # We only ever run one synthesis at a time, so there are no independent
    # ops that benefit from an inter-op thread pool; keep all threads
    # available for the intra-op (matmul) work instead.
    torch.set_num_interop_threads(1)

    import f5_tts.infer.utils_infer as utils_infer

    original_infer_process = utils_infer.infer_process

    def fast_infer_process(*args, **kwargs):
        kwargs.setdefault("nfe_step", nfe_step)
        return original_infer_process(*args, **kwargs)

    utils_infer.infer_process = fast_infer_process

    # preprocess_ref_audio_text() re-decodes the reference clip and re-runs
    # pydub silence trimming on every call even though our ref audio/text is
    # identical across an entire batch of texts. Cache it by (path, text).
    utils_infer.preprocess_ref_audio_text = lru_cache(maxsize=8)(
        utils_infer.preprocess_ref_audio_text
    )


def _load_model(nfe_step: int):
    global _model
    if _model is None:
        _patch_for_cpu_speed(nfe_step)
        t0 = time.perf_counter()
        _model = AutoModel.from_pretrained(MODEL_REPO, trust_remote_code=True)
        print(f"Model loaded in {time.perf_counter() - t0:.2f}s")
    return _model


def synthesize(
    texts: list[str],
    output_dir: str | Path,
    ref_audio_path: str | Path,
    ref_text: str,
    nfe_step: int = DEFAULT_NFE_STEP,
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

    model = _load_model(nfe_step)

    width = max(3, len(str(len(texts))))
    out_paths = []
    total_infer_s = 0.0
    total_audio_s = 0.0
    for i, text in enumerate(texts, start=1):
        t0 = time.perf_counter()
        audio = np.asarray(model(text, ref_audio_path=str(ref_audio_path), ref_text=ref_text))
        infer_s = time.perf_counter() - t0

        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        audio = audio.astype(np.float32)

        audio_s = len(audio) / SAMPLE_RATE
        rtf = infer_s / audio_s if audio_s > 0 else float("inf")
        total_infer_s += infer_s
        total_audio_s += audio_s
        print(
            f"[{i}/{len(texts)}] inference {infer_s:.2f}s -> "
            f"{audio_s:.2f}s audio (RTF {rtf:.2f}x)"
        )

        out_path = output_dir / f"{i:0{width}d}.wav"
        sf.write(out_path, audio, samplerate=SAMPLE_RATE)
        out_paths.append(out_path)

    if len(texts) > 1:
        avg_rtf = total_infer_s / total_audio_s if total_audio_s else float("inf")
        print(
            f"Total: {total_infer_s:.2f}s inference -> {total_audio_s:.2f}s audio "
            f"(avg RTF {avg_rtf:.2f}x)"
        )

    return out_paths
