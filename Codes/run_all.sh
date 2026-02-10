#!/bin/bash
# CineMatch — Run all recommendation pipelines
# Usage: bash Codes/run_all.sh
set -e

echo "=== CineMatch Recommendation Pipeline ==="
echo ""

# Create output directories
mkdir -p output/content output/cooccur output/collab output/hybrid

# ── 1. Content-based filtering ──
echo "[1/4] Running content-based filtering..."
python Codes/content_based.py \
    -r hadoop \
    --movies dataset/movies.csv \
    dataset/ratings.csv \
    -o output/content/
echo "  Done."

# ── 2. Collaborative filtering — Phase 1: Co-occurrence matrix ──
echo "[2/4] Building co-occurrence matrix..."
python Codes/cooccurrence.py \
    -r hadoop \
    dataset/ratings.csv \
    -o output/cooccur/
echo "  Done."

# ── 3. Collaborative filtering — Phase 2: Score movies ──
echo "[3/4] Scoring collaborative recommendations..."
python Codes/collaborative.py \
    -r hadoop \
    --movies dataset/movies.csv \
    --cooccurrence output/cooccur/part-00000 \
    dataset/ratings.csv \
    -o output/collab/
echo "  Done."

# ── 4. Hybrid ──
echo "[4/4] Computing hybrid recommendations..."
python Codes/hybrid.py \
    -r hadoop \
    --content output/content/part-00000 \
    --collab output/collab/part-00000 \
    dataset/ratings.csv \
    -o output/hybrid/
echo "  Done."

# ── Consolidate outputs ──
echo ""
echo "Consolidating output files..."
cat output/content/part-* > output/contentout.txt 2>/dev/null || true
cat output/collab/part-* > output/collaborativeout.txt 2>/dev/null || true
cat output/hybrid/part-* > output/hybridout.txt 2>/dev/null || true

# ── Generate web data ──
echo "Generating website data..."
python Codes/generate_web_data.py

echo ""
echo "=== Pipeline complete ==="
echo "Output files:"
echo "  output/contentout.txt"
echo "  output/collaborativeout.txt"
echo "  output/hybridout.txt"
echo "  website/data.json"
