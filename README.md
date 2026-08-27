# Zero-Shot PPE Compliance Detection

A small computer-vision proof of concept: detect PPE compliance ("person
wearing a hard hat" vs. "person without one") in images and video using
**zero-shot, open-vocabulary object detection** (Grounding DINO) plus
**prompt-free segmentation** (SAM) — no training run, no labeled dataset,
just text prompts against off-the-shelf foundation models. Runs comfortably
on a single consumer GPU (developed and benchmarked on an RTX 3080 10GB —
see `hardware.md`).

Included:
- `quickstart_image_test.py` — single-image sanity check.
- `ppe_zero_shot_pipeline.py` — full video pipeline with box + mask overlay.
- `test_ppe_models.py` — a benchmark harness comparing all 4 model
  combinations below on accuracy, confidence, VRAM, and throughput (see
  `RESULTS.md`).

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

# 3. Install torch matching your CUDA version, e.g. for CUDA 12.1:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 4. Install the rest
pip install -r requirements.txt
```

## Step 1: sanity-check on a single image

Two self-filmed sample stills — `test_image.jpg` (hard hat on) and
`test_image2.jpg` (no hard hat) — are fetched automatically the first time
you run either script (via `fetch_test_media.sh`), so you can run this
immediately without sourcing your own photo. There's no third-party
image-licensing question since both are frames from this repo's own test
videos. Run:

```bash
python quickstart_image_test.py --image test_image.jpg \
    --prompt "person. hard hat."
```

Or point `--image` at any photo of your own with people in it.

First run downloads the models (~1GB total) — after that, inference on a
single image takes a few seconds on the 3080. Check `annotated_test.png` for
boxes and labels. If nothing is detected, lower `--box-threshold` (default
0.35) or simplify the prompt.

**Prompt format matters**: Grounding DINO wants lowercase phrases separated
by periods, each phrase being a separate thing to look for — keep phrases
to simple, literal visual objects (see "Why the prompt is..." below for why
compound/negated phrases don't work well here).

## Step 2: full pipeline on a video

Once the image test looks right, try it on the bundled sample clips (also
fetched automatically via `fetch_test_media.sh`):

```bash
python ppe_zero_shot_pipeline.py \
    --video 20260824_174640.mp4 \
    --output annotated.mp4 \
    --prompt "person. hard hat." \
    --every-n-frames 3
```

Or film your own with/without-hard-hat clips (a phone video transferred to
the desktop works fine — doesn't need to be from a connected webcam) and
point `--video` at those instead.

`--every-n-frames 3` runs detection on every 3rd frame and reuses the last
result in between — much faster for a first pass, and fine for a demo video.
Drop to `1` for a fully per-frame pass once you're tuning results, or if the
video is short.

Add `--verbose` to print each frame's raw detections (label, score, box) to
the console as it processes — useful for tuning the prompt or threshold
without waiting to review the output video.

Green boxes = compliant ("person OK"), red boxes = "person NO HARD HAT",
orange boxes = the raw hard-hat detections themselves. Compliance is
computed geometrically, not from the label text — see below.

## Model comparison / benchmark suite

```bash
python test_ppe_models.py
```

Runs all 4 model combinations from the table above (2 Grounding DINO
variants × 2 SAM variants) against both bundled sample videos, and reports:
per-video compliance verdict + confidence, plus peak VRAM and throughput
(fps) for each combo. Full write-up and results table in `RESULTS.md`;
target-hardware specs it's checked against are in `hardware.md`.

## Why the prompt is "person. hard hat." not "person without hard hat."

Grounding DINO grounds phrases to image regions via token-level correlation,
not logical reasoning — it's weak at negation. Prompting for a compound
concept like "person without hard hat" tends to produce garbled or
duplicated labels (e.g. `"person person"`) rather than a clean detection.

Both scripts instead detect "person" and "hard hat" as independent classes,
then determine compliance themselves with a small geometric rule
(`head_has_hat()`): is a detected hard hat positioned over a detected
person's head region? Green box + "OK" if yes, red box + "NO HARD HAT" if
no. This is a cleaner and more reliable pattern generally for compliance-
style detection tasks — negation is a real, well-known limitation of
open-vocabulary grounding models, so it's more robust to work around it with
a small geometric rule than to keep fighting the prompt.

## Troubleshooting

- **`TypeError: ...got an unexpected keyword argument 'box_threshold'`**: the
  `transformers` library renamed this parameter to `threshold` in newer
  releases, and the result dict's label field moved from `"labels"` to
  `"text_labels"`. Both scripts here are updated for current `transformers`
  (checked against 5.15.1). If you hit a similar signature mismatch on a
  different call, run `python -c "import inspect, transformers; from
  transformers import GroundingDinoProcessor;
  print(inspect.signature(GroundingDinoProcessor.post_process_grounded_object_detection))"`
  to see the exact signature your installed version expects — the
  HF model card examples can lag behind released versions.

- **"unauthenticated requests to the HF Hub" warning**: harmless — models
  are already cached locally after the first run, but `from_pretrained()`
  still checks the Hub for updates by default. To force a fully offline run
  (no network calls, fails loudly if something isn't cached) once you've
  downloaded everything once:
  ```bash
  export HF_HUB_OFFLINE=1
  ```

## License

MIT — see `LICENSE`.
