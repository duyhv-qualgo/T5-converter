#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Setup: venv for ExecuTorch export + inference using optimum-executorch.
#
# optimum-executorch brings in executorch as a dependency, so both the export
# tools and the Python pybindings runtime are installed together.
#
# Run once from T5-converter/:
#   bash executorch/setup.sh
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Creating executorch/.venv ==="
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip --quiet

echo "=== Installing optimum-executorch (pulls in executorch runtime) ==="
pip install "optimum-executorch" --quiet

echo "=== Installing runtime + shared deps ==="
pip install "transformers" "sentencepiece" "safetensors" "sacrebleu" --quiet

echo ""
echo "Setup complete.  Activate with:"
echo "  source executorch/.venv/bin/activate"
echo ""
echo "Convert: python executorch/convert.py"
echo "Verify:  python executorch/verify.py"
