#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Setup: single venv for BOTH litert/explicit and litert/stateful.
#
# litert-torch==0.8.0 requires torch==2.9.x (conflicts with executorch 2.10+).
# tensorflow is needed only for litert/stateful/convert.py.
# ai-edge-litert is the runtime used by both verify scripts.
#
# Run once from T5-converter/:
#   bash litert/setup.sh
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Creating litert/.venv ==="
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip --quiet

echo "=== Installing litert-torch (explicit KV conversion) ==="
# Pins torch==2.9.x
pip install "litert-torch==0.8.0" --quiet

echo "=== Installing TensorFlow (stateful KV conversion) ==="
pip install "tensorflow" "safetensors" --quiet

echo "=== Installing runtime + shared deps ==="
pip install "ai-edge-litert" "transformers" "sentencepiece" "sacrebleu" --quiet

echo ""
echo "Setup complete. Activate with:"
echo "  source litert/.venv/bin/activate"
echo ""
echo "Explicit KV:  python litert/explicit/convert.py"
echo "              python litert/explicit/verify.py"
echo "Stateful:     python litert/stateful/convert.py"
echo "              python litert/stateful/verify.py"
echo "Benchmark:    python litert/benchmark.py"
