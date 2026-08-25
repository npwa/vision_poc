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
import torch
from PIL import Image, ImageDraw
from transformers import (
    AutoProcessor,
    AutoModelForZeroShotObjectDetection,
    SamModel,
    SamProcessor,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument(
        "--prompt",
        default="person. hard hat. person without hard hat.",
    )
    parser.add_argument("--box-threshold", type=float, default=0.35)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    parser.add_argument("--output", default="annotated_test.png")
    args = parser.parse_args()

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
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        target_sizes=[image.size[::-1]],
    )[0]

    boxes = results["boxes"].cpu()
    labels = results["labels"]
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
    draw_img = image.copy()
    draw = ImageDraw.Draw(draw_img)
    for i, (box, label, score) in enumerate(zip(boxes, labels, scores)):
        x0, y0, x1, y1 = box.tolist()
        color = "red" if "without" in label.lower() or "no " in label.lower() else "lime"
        draw.rectangle([x0, y0, x1, y1], outline=color, width=3)
        draw.text((x0, max(y0 - 12, 0)), f"{label} {score:.2f}", fill=color)

    draw_img.save(args.output)
    print(f"\nSaved annotated image to {args.output}")


if __name__ == "__main__":
    main()
