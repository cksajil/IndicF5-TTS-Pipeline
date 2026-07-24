from __future__ import annotations

import os
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from transformers import AutoModel

MODEL_REPO = "ai4bharat/IndicF5"
SAMPLE_RATE = 24000

# F5-TTS ships an "Empirically Pruned Step Sampling" (EPSS) table with
# non-uniform, quality-tuned schedules at 5/6/7/10/12/16 steps (see
# f5_tts.model.utils.get_epss_timesteps). It's active by default whenever
# nfe_step matches one of those keys, so these aren't naive step cuts -
# they're the schedules upstream specifically tuned to hold up at low NFE.
# 16 is the previous "fast" preset; 7 is a further ~2x on top of that with
# a noticeably larger (but often still acceptable) quality trade-off.
# Benchmark both on your own voice/text before committing to 7.
DEFAULT_NFE_STEP = 16
FAST_NFE_STEP = 7

_model = None
_patched = False


def _configure_cpu_threads() -> None:
    """Pin torch's intra-op thread pool to physical cores.

    The previous version set interop_threads=1 but left intraop threads at
    PyTorch's default, which is read from the OS at import time and is
    frequently wrong: containers/VMs often over-report logical (hyperthread)
    core counts, and PyTorch does not distinguish. Hyperthreads share a
    physical core's execution units, so pointing matmul threads at them
    causes contention rather than speedup - explicitly capping to physical
    core count is a common, safe win. We only do this if the user hasn't
    already set the env var / a prior torch call hasn't already configured
    it, since torch.set_num_threads() raises if intra-op work has started.
    """
    if os.environ.get("OMP_NUM_THREADS") or os.environ.get("INDICF5_NUM_THREADS"):
        try:
            n = int(os.environ.get("INDICF5_NUM_THREADS", os.environ.get("OMP_NUM_THREADS")))
            torch.set_num_threads(n)
        except (ValueError, RuntimeError):
            pass
        return

    try:
        import psutil

        physical = psutil.cpu_count(logical=False) or os.cpu_count() or 1
    except ImportError:
        # Fall back to os.cpu_count() (logical count) halved as a rough
        # physical-core estimate on typical SMT/hyperthreaded CPUs; this is
        # imprecise but still much better than leaving it unset.
        logical = os.cpu_count() or 1
        physical = max(1, logical // 2) if logical > 1 else 1

    try:
        torch.set_num_threads(max(1, physical))
    except RuntimeError:
        pass  # threads already in use elsewhere; leave torch's default


def _patch_for_cpu_speed(nfe_step: int, use_bf16: bool) -> None:
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

    _configure_cpu_threads()

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

    if use_bf16:
        # Wrap the CFM.sample call (what model_obj.sample(...) resolves to)
        # in a CPU bf16 autocast region. Modern x86 CPUs (AVX512-BF16 /
        # AMX-capable, e.g. most Intel Ice Lake-SP+ / Sapphire Rapids, and
        # recent AMD Zen4+) get a real matmul throughput boost from this;
        # older CPUs without native bf16 support still work correctly
        # (PyTorch upcasts under the hood) but may see little or no gain -
        # hence this being an opt-in flag, not a default.
        from f5_tts.model.cfm import CFM

        original_sample = CFM.sample

        def bf16_sample(self, *args, **kwargs):
            with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                return original_sample(self, *args, **kwargs)

        CFM.sample = bf16_sample


def _load_model(nfe_step: int, use_bf16: bool):
    global _model
    if _model is None:
        _patch_for_cpu_speed(nfe_step, use_bf16)
        t0 = time.perf_counter()
        _model = AutoModel.from_pretrained(MODEL_REPO, trust_remote_code=True)
        print(f"Model loaded in {time.perf_counter() - t0:.2f}s")
        if not torch.cuda.is_available():
            print(f"CPU threads: {torch.get_num_threads()}")
    return _model


def synthesize(
    texts: list[str],
    output_dir: str | Path,
    ref_audio_path: str | Path,
    ref_text: str,
    nfe_step: int = DEFAULT_NFE_STEP,
    use_bf16: bool = False,
) -> list[Path]:
    """Generate one WAV file per input text, cloning the voice from ref_audio_path.

    Args:
        nfe_step: diffusion steps per chunk. F5-TTS's EPSS schedules give
            tuned, non-uniform timesteps at 5/6/7/10/12/16/32 - prefer one
            of those over an arbitrary value. Lower = faster, more
            quality risk. Try FAST_NFE_STEP (7) for a further ~2x over the
            16-step default.
        use_bf16: run the diffusion transformer under CPU bf16 autocast.
            Meaningful speedup (roughly 1.3-1.8x on top of everything else)
            on CPUs with native bf16 support (recent Intel/AMD server and
            high-end desktop chips); negligible-to-none on older/mobile
            CPUs. Off by default because the benefit is hardware-dependent
            and it hasn't been broadly quality-validated for this model -
            benchmark on your machine before relying on it.

    Returns the list of output paths in the same order as `texts`.
    """
    if not texts:
        raise ValueError("texts must contain at least one string")

    ref_audio_path = Path(ref_audio_path)
    if not ref_audio_path.exists():
        raise FileNotFoundError(f"Reference audio not found: {ref_audio_path}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = _load_model(nfe_step, use_bf16)

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
