#!/usr/bin/env python
"""CLI for the IndicF5 voice-cloning pipeline.

Examples:
    python run.py --text "Hello there" --text "Second sentence"
    python run.py --input texts.txt --outdir output/
"""
import argparse
from pathlib import Path

from indicf5_pipeline import synthesize


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
    args = parser.parse_args()

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

    paths = synthesize(texts, args.outdir, args.ref_audio, ref_text)
    for p in paths:
        print(p)


if __name__ == "__main__":
    main()
