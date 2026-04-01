#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Setup: venv for ONNX export, inference, and benchmarking.
#
# Run once from T5-converter/:
#   bash onnx/setup.sh
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Creating onnx/.venv ==="
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip --quiet

echo "=== Installing optimum + onnxruntime (export + inference) ==="
pip install "optimum[onnxruntime]" --quiet

echo "=== Installing eval + tokeniser deps ==="
pip install "transformers" "sentencepiece" "sacrebleu" --quiet

echo ""
echo "Setup complete. Activate with:"
echo "  source onnx/.venv/bin/activate"
echo ""
echo "Then export:"
echo "  python onnx/export.py"
echo ""
echo "Then verify / benchmark:"
echo "  python onnx/verify.py"
echo "  python onnx/benchmark.py"
