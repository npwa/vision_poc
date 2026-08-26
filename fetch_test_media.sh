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
)

for file in "${MEDIA_FILES[@]}"; do
    if [[ -f "$file" ]]; then
        echo "Found $file, skipping download."
        continue
    fi
    echo "Downloading $file from $BASE_URL ..."
    curl -fSL --retry 3 -o "$file" "$BASE_URL/$file"
done
