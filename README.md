# IndicF5 Voice Cloning Pipeline

Generate speech in your own cloned voice for one or more input texts using
[ai4bharat/IndicF5](https://huggingface.co/ai4bharat/IndicF5). Supports Assamese,
Bengali, Gujarati, Hindi, Kannada, Malayalam, Marathi, Odia, Punjabi, Tamil, and
Telugu.

Voice cloning here is zero-shot: every run needs a short reference audio clip of
the target voice plus its exact transcript. There's no separate training step.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The model repo is gated on Hugging Face:

1. Log in to huggingface.co and accept the license at
   https://huggingface.co/ai4bharat/IndicF5.
2. Authenticate locally: `huggingface-cli login` (paste an access token when
   prompted), or export `HF_TOKEN` in your shell before running anything.

## Reference voice

Place your reference clip and its transcript under `reference/`:

- `reference/sajil_ref_audio_1.wav` — a few seconds of clean speech
- `reference/transcript.txt` — the exact text spoken in that clip

## Usage

### CLI

```bash
# One or more texts directly
python run.py --text "First sentence" --text "Second sentence"

# Or a text file, one text per line (blank lines and lines starting with # are skipped)
python run.py --input texts.txt --outdir output/
```

Each input text produces one WAV file in `--outdir` (default `output/`), named
`001.wav`, `002.wav`, ... in input order.

Flags:
- `--text` — a single text to synthesize (repeatable)
- `--input` — path to a text file with one text per line
- `--outdir` — output directory (default `output/`)
- `--ref-audio` — path to the reference voice recording (default
  `reference/sajil_ref_audio_1.wav`)
- `--ref-text` — path to a file with the reference clip's transcript (default
  `reference/transcript.txt`)
- `--nfe-steps` — diffusion steps per chunk on CPU (default `16`). Lower is
  faster with a small quality cost; raise to `32` (the upstream default) for
  higher quality at roughly double the time.

Each text prints its inference time, generated audio duration, and real-time
factor (RTF = inference time / audio duration) as it's synthesized, plus a
totals line at the end of a batch.

### Python

```python
from indicf5_pipeline import synthesize

paths = synthesize(
    texts=["First sentence", "Second sentence"],
    output_dir="output",
    ref_audio_path="reference/sajil_ref_audio_1.wav",
    ref_text="<exact transcript of the reference clip>",
)
```

## Notes

- First run downloads the IndicF5 model weights from Hugging Face.
- Output audio is written at 24kHz mono WAV.
- Runs on CPU; expect noticeably slower generation than GPU. IndicF5's DiT
  model is a real ~22-layer transformer, so on a weak CPU (e.g. a dual-core
  ultra-low-power laptop chip) a real-time factor over 100x (minutes of
  compute per second of audio) is expected even after the optimizations
  below - there's a hard floor set by the hardware, not just software
  overhead.
- On CPU this pipeline: runs the diffusion sampler at 16 steps instead of
  IndicF5's default 32 (see `--nfe-steps`), caches the reference-audio
  preprocessing across all texts in a batch instead of redoing it per text,
  and keeps all CPU threads on matmul work instead of the unused inter-op
  thread pool.
