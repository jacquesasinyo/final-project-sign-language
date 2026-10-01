#!/usr/bin/env bash
# Download the AUTSL colour videos + labels from the Hugging Face mirror
# (aipieces/AUTSL, AUTSL.zip: 36,302 *_color.mp4, no depth), verify, extract.
# The official ChaLearn server is down (2026-10); this mirror was checked
# to contain every sample listed in the official train/val/test label files.
#
# Usage: bash scripts/download_autsl.sh [DATA_DIR]   (default: data)
# Result: DATA_DIR/raw/AUTSL/{train,val,test}/*_color.mp4 + label CSVs.
# Safe to re-run: the download resumes, and a verified archive is not re-fetched.
set -euo pipefail

DATA_DIR="${1:-data}"
URL="https://huggingface.co/datasets/aipieces/AUTSL/resolve/main/AUTSL.zip"
# Git-LFS oid of AUTSL.zip = SHA-256 of the file (from the HF API, revision b2a1d74).
SHA256="9c0ae4a2c8985cca04b816d424ced071d6d73dc24e6cc85311dc8a1be731d452"
ZIP="$DATA_DIR/downloads/AUTSL.zip"
OUT="$DATA_DIR/raw"

mkdir -p "$DATA_DIR/downloads" "$OUT"

verify() { [ "$(shasum -a 256 "$ZIP" | cut -d' ' -f1)" = "$SHA256" ]; }

if [ -f "$ZIP" ] && verify; then
    echo "Archive already present and verified."
else
    # A single HF connection is slow (~1 MB/s), so fetch PARTS byte ranges in
    # parallel. Each part resumes from its current size; the SHA-256 check
    # below catches any assembly error.
    rm -f "$ZIP"
    PARTS=8
    SIZE=$(curl -sIL "$URL" | awk 'tolower($1)=="content-length:"{v=$2} END{print v+0}' | tr -d '\r')
    STEP=$(( (SIZE + PARTS - 1) / PARTS ))
    fetch_part() {  # $1 = part index
        local p="$ZIP.part$1" start=$(( $1 * STEP )) end=$(( ($1 + 1) * STEP - 1 ))
        [ "$end" -ge "$SIZE" ] && end=$(( SIZE - 1 ))
        local want=$(( end - start + 1 ))
        touch "$p"
        while [ "$(stat -f%z "$p")" -lt "$want" ]; do
            # No curl --retry: it would restart the range and append duplicates.
            curl -sL --fail -r "$(( start + $(stat -f%z "$p") ))-$end" "$URL" >> "$p" || sleep 10
        done
        [ "$(stat -f%z "$p")" -eq "$want" ] || { echo "part $1 has wrong size" >&2; return 1; }
    }
    echo "Downloading $SIZE bytes in $PARTS parallel parts..."
    pids=()
    for i in $(seq 0 $((PARTS - 1))); do fetch_part "$i" & pids+=($!); done
    for pid in "${pids[@]}"; do wait "$pid"; done
    for i in $(seq 0 $((PARTS - 1))); do cat "$ZIP.part$i"; done > "$ZIP"
    rm -f "$ZIP".part*
    echo "Verifying SHA-256..."
    verify || { echo "SHA-256 mismatch: delete $ZIP and re-run." >&2; exit 1; }
fi

echo "Testing archive integrity (CRC of every member)..."
unzip -tq "$ZIP"

echo "Extracting to $OUT ..."
unzip -oq "$ZIP" -d "$OUT"

for s in train val test; do
    echo "$s: $(find "$OUT/AUTSL/$s" -name '*_color.mp4' | wc -l | tr -d ' ') colour videos"
done
echo "Expected: train 28142, val 4418, test 3742"
