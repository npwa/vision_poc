"""
Test suite: run the PPE zero-shot pipeline's 4 README-listed models
(2 Grounding DINO detector variants x 2 SAM segmenter variants) against
every bundled test image/video, and report:

  1. Per-clip, per-model-combo compliance accuracy against hand-labeled
     ground truth (test_labels.json) -- how many people in each clip were
     correctly classified as wearing/not wearing a hard hat.
  2. A hardware-adequacy table: peak GPU VRAM used and throughput (fps)
     for each combo, checked against the RTX 3080 10GB described in
     hardware.md.

Design note: the compliance answer and its confidence score come entirely
from the Grounding DINO detector. SAM only produces segmentation masks for
the overlay video -- it has no independent way to answer a yes/no
compliance question, so swapping SAM variants never changes the answer or
confidence for a given detector. It IS still loaded and run on every frame
here (mirroring real pipeline usage) so its VRAM and speed cost can be
measured for the hardware-adequacy table.

Multi-person clips need each person tracked across frames, not pooled into
one video-level verdict -- otherwise "2 people, 1 compliant, 1 not" would
collapse into a single meaningless majority vote. GreedyIOUTracker (see
tracker.py) assigns a persistent ID to each detected person across a
clip's frames; each track gets its own majority-vote verdict, and a clip's
prediction is scored as its (compliant, non_compliant) track counts against
the (compliant, non_compliant) counts in test_labels.json.

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

from tracker import GreedyIOUTracker

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

LABELS_FILE = "test_labels.json"

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
    Fetch the media listed in test_labels.json from almassy.com via
    fetch_test_media.sh if they aren't already present in the current
    directory. Kept as a shell script rather than inline Python so the
    download logic (curl retry, per-file skip) is reusable outside this
    script and stays simple to inspect/audit.
    """
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fetch_test_media.sh")
    subprocess.run(["bash", script], check=True)


def load_ground_truth():
    with open(LABELS_FILE) as f:
        return json.load(f)


def head_has_hat(person_box, hat_boxes, head_fraction=0.35, x_margin=0.05):
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

    x_margin is deliberately tight (was 0.15): the baseline eval found a
    held-not-worn hat clip (20260826_221517.mp4) scoring compliant on 68%
    of frames, because a hat held out to the side at head height fell
    within the wider margin meant to tolerate a slightly off-center worn
    hat. 0.05 still tolerates normal detection jitter but no longer
    reaches an arm's-length hold.
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


def run_media(media_path, dino_processor, dino_model, sam_processor, sam_model):
    """
    Process every frame of one image/video through detect() + segment(),
    tracking each detected person across frames with a fresh
    GreedyIOUTracker so a multi-person clip keeps each person's
    observations separate. cv2.VideoCapture reads a still image as a
    single-frame "video," so images and videos share this same loop.

    Unlike ppe_zero_shot_pipeline.py's --every-n-frames option, every frame
    is processed here (no skipping) since this is a benchmark run, not a
    demo video render -- speed/VRAM numbers should reflect real per-frame
    cost.

    Returns (track_frame_results, n_frames, elapsed_seconds, fps).
    track_frame_results maps track_id -> list of (person_score, compliant)
    tuples, one per frame that track was seen with a hat-compliance read.
    """
    cap = cv2.VideoCapture(media_path)
    if not cap.isOpened():
        raise SystemExit(f"Could not open media: {media_path}")

    tracker = GreedyIOUTracker()
    track_frame_results = {}  # track_id -> [(score, compliant), ...]
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

        person_boxes, person_scores = [], []
        for box, label, score in zip(boxes_list, labels, scores):
            if "person" in label.lower():
                person_boxes.append(box)
                person_scores.append(float(score))

        track_ids = tracker.update(person_boxes)
        for tid, box, score in zip(track_ids, person_boxes, person_scores):
            compliant = head_has_hat(box, hat_boxes)
            track_frame_results.setdefault(tid, []).append((score, compliant))

    cap.release()
    elapsed = time.perf_counter() - t0
    fps = n_frames / elapsed if elapsed > 0 else 0.0
    return track_frame_results, n_frames, elapsed, fps


def summarize_track(frame_results):
    """
    Collapse one track's per-frame (score, compliant) pairs into a single
    verdict + confidence, since detection re-runs independently each frame
    and can flip compliant/non-compliant frame-to-frame (occlusion, motion
    blur, a missed hat detection, etc).

    Verdict is a majority vote across the track's frames: whichever side
    (compliant vs. non-compliant) has more agreeing frames wins, and
    confidence is the mean detection score across just those agreeing
    frames. Ties favor "compliant" (>=), a deliberate fail-safe bias for a
    safety check: an ambiguous read should not be silently reported as a
    violation.
    """
    compliant_scores = [s for s, c in frame_results if c]
    noncompliant_scores = [s for s, c in frame_results if not c]
    if len(compliant_scores) >= len(noncompliant_scores):
        compliant = True
        conf = sum(compliant_scores) / len(compliant_scores) if compliant_scores else 0.0
    else:
        compliant = False
        conf = sum(noncompliant_scores) / len(noncompliant_scores) if noncompliant_scores else 0.0
    return compliant, conf


MIN_TRACK_FRACTION = 0.05  # drop tracks seen in less than this fraction of a clip's frames


def score_media(track_frame_results, expected, n_frames):
    """
    Turn per-track verdicts into (compliant, non_compliant) counts and
    compare against test_labels.json's expected counts for this clip.
    Matching is by count, not by track identity -- the tracker has no way
    to know which physical person is "the compliant one," so a clip is
    scored correct if it found the right number of compliant and
    non-compliant people, regardless of which track ID got which verdict.

    Tracks seen in fewer than MIN_TRACK_FRACTION of the clip's frames are
    dropped before scoring: the baseline eval found a 4-of-148-frame
    "phantom person" (a one-off spurious detection, not a real second
    subject) inflating the predicted person count. The threshold is a
    fraction rather than a fixed frame count, floored at 1, so a
    single-frame image's only track is never filtered out.
    """
    min_track_frames = max(1, round(MIN_TRACK_FRACTION * n_frames))
    tracks = [
        summarize_track(fr) for fr in track_frame_results.values()
        if len(fr) >= min_track_frames
    ]
    predicted_compliant = sum(1 for compliant, _ in tracks if compliant)
    predicted_non_compliant = sum(1 for compliant, _ in tracks if not compliant)
    match = (
        predicted_compliant == expected["compliant"]
        and predicted_non_compliant == expected["non_compliant"]
    )
    return {
        "predicted_persons": len(tracks),
        "predicted_compliant": predicted_compliant,
        "predicted_non_compliant": predicted_non_compliant,
        "expected_persons": expected["persons"],
        "expected_compliant": expected["compliant"],
        "expected_non_compliant": expected["non_compliant"],
        "match": match,
        "tracks": [{"compliant": c, "confidence": conf} for c, conf in tracks],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="test_results.json")
    parser.add_argument(
        "--only", default=None,
        help="Run only the one CONFIGS entry whose name contains this substring "
             "(e.g. 'tiny + sam-vit-base'). Useful for running each of the 4 "
             "combos as its own short-lived process instead of one long ~15min "
             "run -- merge_results.py stitches the resulting --out files back "
             "into one test_results.json.",
    )
    args = parser.parse_args()

    ensure_test_media()
    ground_truth = load_ground_truth()
    media_files = sorted(ground_truth.keys())

    configs_to_run = CONFIGS
    if args.only:
        configs_to_run = [c for c in CONFIGS if args.only in c["name"]]
        if not configs_to_run:
            raise SystemExit(f"No config name contains {args.only!r}. Options: {[c['name'] for c in CONFIGS]}")

    print(f"Device: {DEVICE}")
    if DEVICE == "cpu":
        print("WARNING: no CUDA GPU detected -- this will be slow.")

    all_results = []

    # Each of the 4 CONFIGS is run to completion (all media) before moving
    # to the next, with its own model load + explicit teardown, so that
    # peak-VRAM measurement and timing for one combo can never be polluted
    # by a previous combo's models still sitting on the GPU.
    for cfg in configs_to_run:
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

        media_results = {}
        total_fps = []
        correct_clips = 0
        correct_persons, total_persons = 0, 0
        for media_file in media_files:
            track_frame_results, n_frames, elapsed, fps = run_media(
                media_file, dino_processor, dino_model, sam_processor, sam_model
            )
            score = score_media(track_frame_results, ground_truth[media_file], n_frames)
            media_results[media_file] = {
                **score,
                "n_frames": n_frames,
                "elapsed_s": elapsed,
                "fps": fps,
            }
            total_fps.append(fps)
            correct_clips += 1 if score["match"] else 0
            total_persons += score["expected_persons"]
            correct_persons += min(score["predicted_compliant"], score["expected_compliant"]) + \
                min(score["predicted_non_compliant"], score["expected_non_compliant"])

            status = "OK" if score["match"] else "MISS"
            print(
                f"  {media_file}: {status} predicted="
                f"{score['predicted_compliant']}c/{score['predicted_non_compliant']}nc "
                f"expected={score['expected_compliant']}c/{score['expected_non_compliant']}nc "
                f"({fps:.1f} fps)"
            )

        peak_vram_bytes = torch.cuda.max_memory_allocated()
        peak_vram_gb = peak_vram_bytes / (1024 ** 3)
        clip_accuracy = correct_clips / len(media_files)
        person_accuracy = correct_persons / total_persons if total_persons else 0.0
        print(
            f"  peak VRAM: {peak_vram_gb:.2f} GB | "
            f"clip accuracy: {correct_clips}/{len(media_files)} ({clip_accuracy:.0%}) | "
            f"person-count accuracy: {person_accuracy:.0%}"
        )

        all_results.append({
            "config": cfg["name"],
            "dino_model": cfg["dino"],
            "sam_model": cfg["sam"],
            "load_time_s": load_time,
            "peak_vram_gb": peak_vram_gb,
            "avg_fps": sum(total_fps) / len(total_fps),
            "clip_accuracy": clip_accuracy,
            "person_accuracy": person_accuracy,
            "media": media_results,
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
