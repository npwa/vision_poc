"""
Quickstart: zero-shot detection + segmentation on a SINGLE IMAGE.

Run this first, before the video pipeline, to confirm your environment,
GPU, and model downloads all work. Takes ~1-2 min the first time
(model weights download and cache under ~/.cache/huggingface).

Models used (auto-downloaded from Hugging Face Hub on first run):
  Detection    : IDEA-Research/grounding-dino-tiny
                 https://huggingface.co/IDEA-Research/grounding-dino-tiny
  Segmentation : facebook/sam-vit-base
                 https://huggingface.co/facebook/sam-vit-base

Usage:
    python quickstart_image_test.py --image path/to/photo.jpg \
        --prompt "person. hard hat. person without hard hat."

The prompt format for Grounding DINO: lowercase phrases separated by
periods. Each phrase is a separate thing to detect.
"""

import argparse
import os
import subprocess

import torch
from PIL import Image, ImageDraw
from transformers import (
    AutoProcessor,
    AutoModelForZeroShotObjectDetection,
    SamModel,
    SamProcessor,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def ensure_test_media():
    """Fetch the bundled sample images/videos via fetch_test_media.sh if missing."""
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fetch_test_media.sh")
    subprocess.run(["bash", script], check=True)


def head_has_hat(person_box, hat_boxes, head_fraction=0.35, x_margin=0.15):
    """
    Geometric compliance check: is a 'hard hat' box positioned over this
    person's head region? Used instead of asking the detector to understand
    negated/compound prompts like "person without hard hat" directly, which
    Grounding DINO handles unreliably (it grounds on token correlation, not
    logical negation).
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument(
        "--prompt",
        default="person. hard hat.",
    )
    parser.add_argument("--box-threshold", type=float, default=0.35)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    parser.add_argument("--output", default="annotated_test.png")
    args = parser.parse_args()

    ensure_test_media()

    print(f"Device: {DEVICE}")
    if DEVICE == "cpu":
        print("WARNING: no CUDA GPU detected by torch — this will be slow.")

    print("Loading Grounding DINO (zero-shot detector)...")
    dino_id = "IDEA-Research/grounding-dino-tiny"
    dino_processor = AutoProcessor.from_pretrained(dino_id)
    dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(dino_id).to(DEVICE)

    print("Loading SAM (segmentation)...")
    sam_id = "facebook/sam-vit-base"
    sam_processor = SamProcessor.from_pretrained(sam_id)
    sam_model = SamModel.from_pretrained(sam_id).to(DEVICE)

    image = Image.open(args.image).convert("RGB")

    # --- Detection ---
    inputs = dino_processor(images=image, text=args.prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = dino_model(**inputs)

    results = dino_processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        target_sizes=[image.size[::-1]],
    )[0]

    boxes = results["boxes"].cpu()
    labels = results["text_labels"]
    scores = results["scores"].cpu()

    print(f"\nFound {len(boxes)} detections:")
    for box, label, score in zip(boxes, labels, scores):
        print(f"  {label!r}  score={score:.2f}  box={box.tolist()}")

    if len(boxes) == 0:
        print("\nNo detections above threshold. Try lowering --box-threshold, "
              "or check your prompt wording.")
        image.save(args.output)
        return

    # --- Segmentation (box-prompted) ---
    input_boxes = [boxes.tolist()]
    sam_inputs = sam_processor(image, input_boxes=input_boxes, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        sam_outputs = sam_model(**sam_inputs)

    masks = sam_processor.image_processor.post_process_masks(
        sam_outputs.pred_masks.cpu(),
        sam_inputs["original_sizes"].cpu(),
        sam_inputs["reshaped_input_sizes"].cpu(),
    )[0]
    iou_scores = sam_outputs.iou_scores.cpu()

    # --- Draw and save ---
    boxes_list = boxes.tolist()
    hat_boxes = [b for b, l in zip(boxes_list, labels) if "hat" in l.lower()]

    draw_img = image.copy()
    draw = ImageDraw.Draw(draw_img)
    for box, label, score in zip(boxes_list, labels, scores):
        x0, y0, x1, y1 = box
        l = label.lower()

        if "person" in l:
            compliant = head_has_hat(box, hat_boxes)
            color = "lime" if compliant else "red"
            text = f"person {'OK' if compliant else 'NO HARD HAT'} {score:.2f}"
        elif "hat" in l:
            color = "orange"
            text = f"{label} {score:.2f}"
        else:
            color = "yellow"
            text = f"{label} {score:.2f}"

        draw.rectangle([x0, y0, x1, y1], outline=color, width=3)
        draw.text((x0, max(y0 - 12, 0)), text, fill=color)

    draw_img.save(args.output)
    print(f"\nSaved annotated image to {args.output}")


if __name__ == "__main__":
    main()
