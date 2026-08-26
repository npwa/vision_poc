"""
Test suite: run the PPE zero-shot pipeline's 4 README-listed models
(2 Grounding DINO detector variants x 2 SAM segmenter variants) against
both test videos, and report:

  1. A table of per-video, per-model-combo answers to "was there a person
     wearing a hard hat, or a person without one?" plus the detector's
     confidence score for that verdict.
  2. A hardware-adequacy table: peak GPU VRAM used and throughput (fps)
     for each combo, checked against the RTX 3080 10GB described in
     hardware.md.

Design note: the compliance ANSWER and its confidence score come entirely
from the Grounding DINO detection stage (person/hard-hat boxes + the
geometric head_has_hat() overlap check in ppe_zero_shot_pipeline.py). SAM
only produces segmentation masks for the overlay video -- it has no
independent way to answer a yes/no compliance question, so it does not
change the answer or confidence for a given detector. It IS still loaded
and run on every frame here (mirroring real pipeline usage) so its VRAM
and speed cost can be measured for the hardware-adequacy table.

Usage:
    python test_ppe_models.py
"""

import argparse
import json
import os
import subprocess
import time

import cv2
import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    AutoModelForZeroShotObjectDetection,
    SamModel,
    SamProcessor,
)

# Falls back to CPU so the script still runs (slowly) on a machine with no
# GPU, but every timing/VRAM number in the output is only meaningful on CUDA.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Kept identical to ppe_zero_shot_pipeline.py's defaults so this test suite
# measures the same behavior the real pipeline would produce. "person." and
# "hard hat." are prompted as two separate, independent phrases rather than
# a single negated phrase like "person without hard hat." -- see the
# head_has_hat() docstring below for why Grounding DINO can't be trusted
# with negation directly.
PROMPT = "person. hard hat."
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25

# The two PoC clips under test: one person wearing a hard hat, one without.
VIDEOS = ["20260824_174509.mp4", "20260824_174640.mp4"]

# All 4 models named in the README's model table, taken 2x2: the 2
# Grounding DINO detector variants (tiny = faster/smaller, base = more
# accurate/slower) crossed with the 2 SAM segmenter variants (base vs.
# large). "sam-vit-large" stands in for the README's combined
# "sam-vit-large / sam-vit-huge" row -- huge wasn't run, only estimated in
# RESULTS.md, since it's strictly heavier than large along both axes we
# measure (VRAM, speed).
CONFIGS = [
    {"name": "grounding-dino-tiny + sam-vit-base",  "dino": "IDEA-Research/grounding-dino-tiny", "sam": "facebook/sam-vit-base"},
    {"name": "grounding-dino-base + sam-vit-base",  "dino": "IDEA-Research/grounding-dino-base", "sam": "facebook/sam-vit-base"},
    {"name": "grounding-dino-tiny + sam-vit-large", "dino": "IDEA-Research/grounding-dino-tiny", "sam": "facebook/sam-vit-large"},
    {"name": "grounding-dino-base + sam-vit-large", "dino": "IDEA-Research/grounding-dino-base", "sam": "facebook/sam-vit-large"},
]


def ensure_test_media():
    """
    Fetch VIDEOS from almassy.com via fetch_test_media.sh if they aren't
    already present in the current directory. Kept as a shell script rather
    than inline Python so the download logic (curl retry, per-file skip) is
    reusable outside this script and stays simple to inspect/audit -- it's
    fetching test media over the network before anything else runs.
    """
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fetch_test_media.sh")
    subprocess.run(["bash", script], check=True)


def head_has_hat(person_box, hat_boxes, head_fraction=0.35, x_margin=0.15):
    """
    Geometric compliance check, copied from ppe_zero_shot_pipeline.py: is
    a detected "hard hat" box centered over this person's head region?

    This exists because Grounding DINO grounds phrases via token-level
    correlation, not logical reasoning, so it can't reliably be prompted
    with negation (e.g. "person without hard hat" tends to produce garbled
    output like "person person"). Instead we detect "person" and "hard
    hat" as independent classes and determine compliance ourselves: take
    the top head_fraction of the person's box height (plus a small x/y
    margin to tolerate a hat sitting slightly outside the person box) and
    check whether any hat box's center falls inside it.
    """
    px0, py0, px1, py1 = person_box
    head_y1 = py0 + head_fraction * (py1 - py0)
    margin = x_margin * (px1 - px0)
    head_x0, head_x1 = px0 - margin, px1 + margin
    head_y0 = py0 - margin
    for hx0, hy0, hx1, hy1 in hat_boxes:
        hcx, hcy = (hx0 + hx1) / 2, (hy0 + hy1) / 2
        if head_x0 <= hcx <= head_x1 and head_y0 <= hcy <= head_y1:
            return True
    return False


def load_models(dino_id, sam_id):
    """Download (if not cached) and load one Grounding DINO + SAM pair onto DEVICE."""
    dino_processor = AutoProcessor.from_pretrained(dino_id)
    dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(dino_id).to(DEVICE)
    sam_processor = SamProcessor.from_pretrained(sam_id)
    sam_model = SamModel.from_pretrained(sam_id).to(DEVICE)
    return dino_processor, dino_model, sam_processor, sam_model


def detect(frame_pil, dino_processor, dino_model):
    """
    Run one Grounding DINO forward pass on a single frame for PROMPT and
    return (boxes, text_labels, scores) as numpy/list, one entry per
    detected box above BOX_THRESHOLD/TEXT_THRESHOLD. This is the only
    function that determines the PPE compliance answer -- SAM (segment(),
    below) never sees or influences these results.
    """
    inputs = dino_processor(images=frame_pil, text=PROMPT, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = dino_model(**inputs)
    results = dino_processor.post_process_grounded_object_detection(
        outputs, inputs.input_ids,
        threshold=BOX_THRESHOLD, text_threshold=TEXT_THRESHOLD,
        target_sizes=[frame_pil.size[::-1]],
    )[0]
    return results["boxes"].cpu().numpy(), list(results["text_labels"]), results["scores"].cpu().numpy()


def segment(frame_pil, boxes, sam_processor, sam_model):
    """
    Run SAM on the boxes Grounding DINO found, box-prompted the same way
    ppe_zero_shot_pipeline.py does for its mask overlay. The output is
    intentionally discarded (not returned) -- this test suite only cares
    about SAM's resource cost (VRAM/time), not its masks, since it has no
    bearing on the compliance answer. Skipped when there are no boxes
    because SamProcessor errors on an empty input_boxes list.
    """
    if len(boxes) == 0:
        return
    inputs = sam_processor(frame_pil, input_boxes=[boxes.tolist()], return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        sam_model(**inputs)


def run_video(video_path, dino_processor, dino_model, sam_processor, sam_model):
    """
    Process every frame of one video through detect() + segment() and
    collect a (person_detection_score, is_compliant) pair for every
    detected person in every frame. Unlike ppe_zero_shot_pipeline.py's
    --every-n-frames option, every frame is processed here (no skipping)
    since this is a benchmark run, not a demo video render -- speed/VRAM
    numbers should reflect real per-frame cost.

    Returns (frame_results, n_frames, elapsed_seconds, fps) so the caller
    can both compute the compliance verdict (summarize()) and report
    throughput for the hardware-adequacy table.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {video_path}")

    frame_results = []  # list of (person_score, compliant) tuples, one per detected person per frame
    n_frames = 0
    t0 = time.perf_counter()

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        n_frames += 1
        # cv2 reads BGR; every downstream model (DINO, SAM, PIL) expects RGB.
        frame_pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        boxes, labels, scores = detect(frame_pil, dino_processor, dino_model)
        segment(frame_pil, boxes, sam_processor, sam_model)  # run for VRAM/timing cost only; result unused

        boxes_list = boxes.tolist()
        hat_boxes = [b for b, l in zip(boxes_list, labels) if "hat" in l.lower()]
        for box, label, score in zip(boxes_list, labels, scores):
            if "person" in label.lower():
                compliant = head_has_hat(box, hat_boxes)
                frame_results.append((float(score), compliant))

    cap.release()
    elapsed = time.perf_counter() - t0
    fps = n_frames / elapsed if elapsed > 0 else 0.0
    return frame_results, n_frames, elapsed, fps


def summarize(frame_results):
    """
    Collapse one video's per-frame (score, compliant) pairs into a single
    plain-text verdict + confidence, since the pipeline re-runs detection
    independently on every frame and can flip compliant/non-compliant
    frame-to-frame (occlusion, motion blur, a missed hat detection, etc).

    Verdict is a majority vote across all frames that had a detected
    person: whichever side (compliant vs. non-compliant) has more
    agreeing frames wins the video's verdict, and the reported confidence
    is the mean DINO detection score across just those agreeing frames
    (not all frames) -- so it reflects how confident the detector was in
    the frames that support the winning verdict, not diluted by outliers
    on the losing side. Ties favor "WITH hard hat" (>=), a deliberate
    fail-safe bias for a safety-compliance check: an ambiguous read
    should not be silently reported as a violation.

    Returns ("No person detected", None) if no person was ever detected
    in the video.
    """
    if not frame_results:
        return "No person detected", None
    compliant_scores = [s for s, c in frame_results if c]
    noncompliant_scores = [s for s, c in frame_results if not c]
    if len(compliant_scores) >= len(noncompliant_scores):
        verdict = "Person WITH hard hat"
        conf = sum(compliant_scores) / len(compliant_scores) if compliant_scores else 0.0
        agree, total = len(compliant_scores), len(frame_results)
    else:
        verdict = "Person WITHOUT hard hat"
        conf = sum(noncompliant_scores) / len(noncompliant_scores) if noncompliant_scores else 0.0
        agree, total = len(noncompliant_scores), len(frame_results)
    return verdict, {"confidence": conf, "agree_frames": agree, "total_person_frames": total}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="test_results.json")
    args = parser.parse_args()

    ensure_test_media()

    print(f"Device: {DEVICE}")
    if DEVICE == "cpu":
        print("WARNING: no CUDA GPU detected -- this will be slow.")

    all_results = []

    # Each of the 4 CONFIGS is run to completion (both videos) before moving
    # to the next, with its own model load + explicit teardown, so that
    # peak-VRAM measurement and timing for one combo can never be polluted
    # by a previous combo's models still sitting on the GPU.
    for cfg in CONFIGS:
        print(f"\n=== {cfg['name']} ===")
        # empty_cache() releases any memory PyTorch is still holding from the
        # previous combo's teardown; reset_peak_memory_stats() zeroes the
        # high-water mark so max_memory_allocated() below reflects only this
        # combo's usage, not a peak carried over from an earlier one.
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        t_load0 = time.perf_counter()
        dino_processor, dino_model, sam_processor, sam_model = load_models(cfg["dino"], cfg["sam"])
        load_time = time.perf_counter() - t_load0
        print(f"  loaded in {load_time:.1f}s")

        video_results = {}
        total_fps = []
        for video in VIDEOS:
            frame_results, n_frames, elapsed, fps = run_video(
                video, dino_processor, dino_model, sam_processor, sam_model
            )
            verdict, detail = summarize(frame_results)
            video_results[video] = {
                "verdict": verdict,
                "detail": detail,
                "n_frames": n_frames,
                "elapsed_s": elapsed,
                "fps": fps,
            }
            total_fps.append(fps)
            conf_str = f"{detail['confidence']:.2f}" if detail else "n/a"
            print(f"  {video}: {verdict} (conf={conf_str}, {fps:.1f} fps)")

        # max_memory_allocated() reports the high-water mark since the last
        # reset above, i.e. the peak VRAM this specific dino+sam combo used
        # across both videos -- this is the number the hardware-adequacy
        # table checks against the RTX 3080's 10GB budget.
        peak_vram_bytes = torch.cuda.max_memory_allocated()
        peak_vram_gb = peak_vram_bytes / (1024 ** 3)
        print(f"  peak VRAM: {peak_vram_gb:.2f} GB")

        all_results.append({
            "config": cfg["name"],
            "dino_model": cfg["dino"],
            "sam_model": cfg["sam"],
            "load_time_s": load_time,
            "peak_vram_gb": peak_vram_gb,
            "avg_fps": sum(total_fps) / len(total_fps),
            "videos": video_results,
        })

        # Drop all Python references to this combo's models so CUDA's
        # allocator can actually free their VRAM before the next combo loads
        # -- without the del, the tensors stay referenced by the local
        # variables until they're reassigned next iteration, and
        # empty_cache() can't reclaim memory that's still referenced.
        del dino_processor, dino_model, sam_processor, sam_model
        torch.cuda.empty_cache()

    with open(args.out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
