# Zero-shot industrial anomaly/PPE detection with a reasoning layer

Step 1 (zero-shot detection) + Step 2 (segmentation).  Detects things like
"person without hard hat" using text prompts alone — no training, no labeled
dataset — then segments them with SAM. Runs comfortably on an RTX 3080 10GB.

## Models

Both are pulled automatically from the Hugging Face Hub the first time you
run either script (cached under `~/.cache/huggingface`, no manual download
needed). Direct model card URLs for reference:

| Purpose | Model | URL | Approx. size |
|---|---|---|---|
| Zero-shot detection | `IDEA-Research/grounding-dino-tiny` | https://huggingface.co/IDEA-Research/grounding-dino-tiny | ~170M params, ~700MB |
| Zero-shot detection (more accurate, slower) | `IDEA-Research/grounding-dino-base` | https://huggingface.co/IDEA-Research/grounding-dino-base | ~230M params, ~900MB |
| Segmentation | `facebook/sam-vit-base` | https://huggingface.co/facebook/sam-vit-base | ~90M params, ~375MB |
| Segmentation (higher quality) | `facebook/sam-vit-large` / `sam-vit-huge` | https://huggingface.co/facebook/sam-vit-large | ~1.2GB / ~2.4GB |

The tiny + base combo (default in both scripts) comfortably fits alongside
each other on 10GB VRAM with room to spare.  The larger variants could be
used with `--dino-model` / `--sam-model` once the baseline works.

## Setup

```bash
# 1. Check your CUDA version
nvidia-smi

# 2. Create a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate

# 3. Install torch matching your CUDA version, e.g. for CUDA 13.0:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130

# 4. Install the rest
pip install -r requirements.txt
```

## Step 1: sanity-check on a single image

Grab any photo with people in it (a stock photo of a construction site works
fine for the first test) and run:

```bash
python quickstart_image_test.py --image test.jpg \
    --prompt "person. hard hat."
```

First run downloads the models (~1GB total) — after that, inference on a
single image takes a few seconds on the 3080. Check `annotated_test.png` for
boxes and labels. If nothing is detected, lower `--box-threshold` (default
0.35) or simplify the prompt.

**Prompt format matters**: Grounding DINO wants lowercase phrases separated
by periods, each phrase being a separate thing to look for. `"person."` and
`"person without hard hat."` are two different classes it'll look for
independently — it doesn't reason about the relationship between them, so
phrase it as the literal visual thing you want it to find.

## Step 2: full pipeline on your test video

Once the image test looks right, film your with/without-helmet clips (a
phone video transferred to the desktop works fine — doesn't need to be from
a connected webcam) and run:

```bash
python ppe_zero_shot_pipeline.py \
    --video worker_test.mp4 \
    --output annotated.mp4 \
    --prompt "person. hard hat. person without hard hat." \
    --every-n-frames 3
```

`--every-n-frames 3` runs detection on every 3rd frame and reuses the last
result in between — much faster for a first pass, and fine for a demo video.
Drop to `1` for a fully per-frame pass once you're tuning results, or if the
video is short.

Green boxes = compliant, red boxes = "without hard hat" (matched by the word
"without" in the label — adjust the color logic in `draw_overlay()` if you
change the prompt wording).
