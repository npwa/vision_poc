"""
Zero-shot PPE (hard hat) detection + segmentation on a VIDEO file.

Processes a video frame-by-frame with Grounding DINO (zero-shot, open-
vocabulary object detection) and SAM (box-prompted segmentation), then
writes an annotated output video with boxes, labels, and mask overlays.

Models used (auto-downloaded from Hugging Face Hub on first run, cached
under ~/.cache/huggingface):
  Detection    : IDEA-Research/grounding-dino-tiny
                 https://huggingface.co/IDEA-Research/grounding-dino-tiny
                 (swap to IDEA-Research/grounding-dino-base for higher
                 accuracy at the cost of speed/VRAM)
  Segmentation : facebook/sam-vit-base
                 https://huggingface.co/facebook/sam-vit-base
                 (swap to facebook/sam-vit-large or sam-vit-huge for
                 higher-quality masks, more VRAM/time)

Usage:
    python ppe_zero_shot_pipeline.py \
        --video worker_test_clip.mp4 \
        --output annotated.mp4 \
        --prompt "person. hard hat. person without hard hat." \
        --every-n-frames 3

Run quickstart_image_test.py first on a single frame to confirm your
setup and tune --prompt / --box-threshold before processing full video.
"""

import argparse
import cv2
import torch
import numpy as np
from PIL import Image
from transformers import (
    AutoProcessor,
    AutoModelForZeroShotObjectDetection,
    SamModel,
    SamProcessor,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_models(dino_id="IDEA-Research/grounding-dino-tiny", sam_id="facebook/sam-vit-base"):
    print(f"Device: {DEVICE}")
    print(f"Loading detector: {dino_id}")
    dino_processor = AutoProcessor.from_pretrained(dino_id)
    dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(dino_id).to(DEVICE)

    print(f"Loading segmenter: {sam_id}")
    sam_processor = SamProcessor.from_pretrained(sam_id)
    sam_model = SamModel.from_pretrained(sam_id).to(DEVICE)

    return dino_processor, dino_model, sam_processor, sam_model


def detect(frame_pil, prompt, dino_processor, dino_model, box_threshold, text_threshold):
    inputs = dino_processor(images=frame_pil, text=prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = dino_model(**inputs)

    results = dino_processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        target_sizes=[frame_pil.size[::-1]],
    )[0]

    return results["boxes"].cpu().numpy(), list(results["labels"]), results["scores"].cpu().numpy()


def segment(frame_pil, boxes, sam_processor, sam_model):
    if len(boxes) == 0:
        return []
    input_boxes = [boxes.tolist()]
    inputs = sam_processor(frame_pil, input_boxes=input_boxes, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = sam_model(**inputs)

    masks = sam_processor.image_processor.post_process_masks(
        outputs.pred_masks.cpu(),
        inputs["original_sizes"].cpu(),
        inputs["reshaped_input_sizes"].cpu(),
    )[0]
    scores = outputs.iou_scores.cpu()

    best_masks = []
    for i in range(masks.shape[0]):
        best_idx = scores[0, i].argmax().item()
        best_masks.append(masks[i, best_idx].numpy())
    return best_masks


def draw_overlay(frame_bgr, boxes, labels, scores, masks):
    overlay = frame_bgr.copy()
    mask_iter = masks if masks else [None] * len(boxes)
    for box, label, score, mask in zip(boxes, labels, scores, mask_iter):
        x0, y0, x1, y1 = box.astype(int)
        is_violation = "without" in label.lower() or "no " in label.lower()
        color = (0, 0, 255) if is_violation else (0, 255, 0)  # BGR: red vs green

        cv2.rectangle(overlay, (x0, y0), (x1, y1), color, 2)
        cv2.putText(
            overlay, f"{label} {score:.2f}", (x0, max(y0 - 8, 0)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
        )
        if mask is not None:
            m = mask.squeeze() > 0.5
            colored = np.zeros_like(frame_bgr)
            colored[m] = color
            overlay = cv2.addWeighted(overlay, 1.0, colored, 0.35, 0)
    return overlay


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--output", default="annotated.mp4")
    parser.add_argument("--prompt", default="person. hard hat. person without hard hat.")
    parser.add_argument("--box-threshold", type=float, default=0.35)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    parser.add_argument(
        "--every-n-frames", type=int, default=1,
        help="Run detection/segmentation every N frames; reuse the last "
             "result for skipped frames. Use 3-5 to speed up long clips.",
    )
    parser.add_argument("--no-sam", action="store_true", help="Boxes only, skip segmentation masks")
    parser.add_argument("--dino-model", default="IDEA-Research/grounding-dino-tiny")
    parser.add_argument("--sam-model", default="facebook/sam-vit-base")
    args = parser.parse_args()

    dino_processor, dino_model, sam_processor, sam_model = load_models(
        args.dino_model, args.sam_model
    )

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    print(f"Video: {w}x{h} @ {fps:.1f}fps, ~{total_frames} frames")

    frame_idx = 0
    last_boxes = np.empty((0, 4))
    last_labels, last_scores, last_masks = [], np.empty((0,)), []

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        if frame_idx % args.every_n_frames == 0:
            frame_pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
            boxes, labels, scores = detect(
                frame_pil, args.prompt, dino_processor, dino_model,
                args.box_threshold, args.text_threshold,
            )
            masks = [] if args.no_sam else segment(frame_pil, boxes, sam_processor, sam_model)
            last_boxes, last_labels, last_scores, last_masks = boxes, labels, scores, masks

        annotated = draw_overlay(frame_bgr, last_boxes, last_labels, last_scores, last_masks)
        writer.write(annotated)

        frame_idx += 1
        if frame_idx % 30 == 0:
            print(f"  processed {frame_idx}/{total_frames} frames...")

    cap.release()
    writer.release()
    print(f"\nDone. Wrote {args.output}")


if __name__ == "__main__":
    main()
