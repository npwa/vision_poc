#!/usr/bin/env bash
# Downloads the PPE test media (videos + sample images) from almassy.com
# into the current directory if they aren't already present. Called
# automatically by test_ppe_models.py before it runs, so a fresh checkout
# doesn't need this (large, and in the images' case, third-party-photo-
# licensing-sensitive) test media committed to git.
set -euo pipefail

BASE_URL="https://almassy.com/media"
MEDIA_FILES=(
    "20260824_174509.mp4"
    "20260824_174640.mp4"
    "test_image.jpg"
    "test_image2.jpg"
    "20260824_190302.mp4"
    "20260824_190237.mp4"
    "20260826_221411.mp4"
    "20260826_221430.mp4"
    "20260826_221451.mp4"
    "20260826_221511.mp4"
    "20260826_221517.mp4"
    "IMG-20260119-WA0022.jpg"
    "20260724_054702.jpg"
    "20260723_054550.jpg"
)

for file in "${MEDIA_FILES[@]}"; do
    if [[ -f "$file" ]]; then
        echo "Found $file, skipping download."
        continue
    fi
    echo "Downloading $file from $BASE_URL ..."
    curl -fSL --retry 3 -o "$file" "$BASE_URL/$file"
done
