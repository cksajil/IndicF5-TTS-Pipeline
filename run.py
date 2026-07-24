#!/usr/bin/env python
"""CLI for the IndicF5 voice-cloning pipeline.

Examples:
    python run.py --text "Hello there" --text "Second sentence"
    python run.py --input texts.txt --outdir output/
    python run.py --text "Hello there" --fast          # ~2x faster, lower quality
    python run.py --text "Hello there" --bf16           # try CPU bf16 autocast
"""
import argparse
from pathlib import Path

from indicf5_pipeline import DEFAULT_NFE_STEP, FAST_NFE_STEP, synthesize


def read_texts(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip() and not line.strip().startswith("#")]


def main():
    parser = argparse.ArgumentParser(description="Clone your voice with IndicF5 for one or more input texts.")
    parser.add_argument("--input", type=Path, help="Path to a text file, one text to synthesize per line")
    parser.add_argument("--text", action="append", default=[], help="A single text to synthesize (repeatable)")
    parser.add_argument("--outdir", type=Path, default=Path("output"), help="Directory to write generated WAV files")
    parser.add_argument("--ref-audio", type=Path, default=Path("reference/sajil_ref_audio_1.wav"), help="Path to your reference voice recording")
    parser.add_argument("--ref-text", type=Path, default=Path("reference/transcript.txt"), help="Path to a file containing the exact transcript of the reference audio")
    parser.add_argument("--nfe-steps", type=int, default=DEFAULT_NFE_STEP, help=f"Diffusion steps per chunk on CPU; lower is faster, higher is higher quality (default {DEFAULT_NFE_STEP}, upstream default 32). Prefer one of F5-TTS's tuned EPSS step counts: 5, 6, 7, 10, 12, 16, 32.")
    parser.add_argument("--fast", action="store_true", help=f"Shortcut for --nfe-steps {FAST_NFE_STEP}: a further ~2x speedup over the default 16 steps, with a larger quality trade-off. Benchmark on your own text/voice first.")
    parser.add_argument("--bf16", action="store_true", help="Run the diffusion transformer under CPU bf16 autocast. Can give a real extra speedup on CPUs with native bf16 support (recent Intel/AMD chips); little to no effect on older/mobile CPUs. Benchmark before relying on it.")
    parser.add_argument("--threads", type=int, default=None, help="Override the number of CPU threads used for inference (default: auto-detected physical core count).")
    args = parser.parse_args()

    if args.threads is not None:
        import os

        os.environ["INDICF5_NUM_THREADS"] = str(args.threads)

    texts = list(args.text)
    if args.input:
        texts.extend(read_texts(args.input))

    if not texts:
        parser.error("Provide at least one text via --text or --input")
    if not args.ref_audio.exists():
        parser.error(f"Reference audio not found: {args.ref_audio}")
    if not args.ref_text.exists():
        parser.error(f"Reference transcript not found: {args.ref_text}")

    ref_text = args.ref_text.read_text(encoding="utf-8").strip()

    nfe_step = FAST_NFE_STEP if args.fast else args.nfe_steps

    paths = synthesize(
        texts,
        args.outdir,
        args.ref_audio,
        ref_text,
        nfe_step=nfe_step,
        use_bf16=args.bf16,
    )
    for p in paths:
        print(p)


if __name__ == "__main__":
    main()
