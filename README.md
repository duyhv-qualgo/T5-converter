# T5-converter

Reusable pipeline for exporting a T5-family checkpoint to on-device formats.

---

## Structure

```
T5-converter/
├── shared/
│   ├── testset.py          # 60-pair EN↔VI test set (shared by all benchmarks)
│   └── metrics.py          # BLEU-4 / chrF++ / exact-match helpers
│
├── litert/                 # Google LiteRT (.tflite) — two KV-cache strategies
│   ├── setup.sh            # venv: litert-torch + tensorflow + ai-edge-litert
│   ├── explicit/
│   │   ├── patch_t5_attention.py   # litert-torch v0.8.0 compatibility fixes
│   │   ├── convert.py              # PyTorch → .tflite  (KV cache as explicit tensors)
│   │   └── verify.py               # correctness + latency
│   ├── stateful/
│   │   ├── convert.py              # TF SavedModel → .tflite  (KV cache as tf.Variable)
│   │   └── verify.py               # correctness + latency
│   └── benchmark.py        # accuracy table across all litert variants
│
├── onnx/                   # ONNX (CPU via onnxruntime)
│   ├── setup.sh            # venv: optimum[onnxruntime] + sacrebleu
│   ├── export.py           # HF checkpoint → ONNX (FP32 + optional INT8)
│   ├── verify.py           # correctness + latency
│   └── benchmark.py        # accuracy table (FP32 vs INT8)
│
├── compare.py              # cross-format: ONNX vs LiteRT Explicit vs LiteRT Stateful
└── README.md
```

Output files land in:
- `litert/output/`  — `*.tflite` files
- `onnx/output/fp32/` and `onnx/output/int8/`  — `*.onnx` files

---

## Quick start

### LiteRT

```bash
# 1. Setup (once)
bash litert/setup.sh
source litert/.venv/bin/activate

# 2. Explicit KV cache (litert-torch path)
python litert/explicit/convert.py   # → litert/output/t5_mini_explicit_{fp32,int8}.tflite
python litert/explicit/verify.py

# 3. Stateful KV cache (TF SavedModel path)
python litert/stateful/convert.py   # → litert/output/t5_mini_stateful_{enc,dec}_{fp32,int8}.tflite
python litert/stateful/verify.py

# 4. Accuracy benchmark across all litert variants
python litert/benchmark.py
```

### ONNX

```bash
bash onnx/setup.sh
source onnx/.venv/bin/activate

python onnx/export.py           # → onnx/output/fp32/*.onnx
python onnx/export.py --int8    # → onnx/output/int8/*.onnx  (also runs FP32 first)
python onnx/verify.py
python onnx/benchmark.py
```

### Cross-format comparison

```bash
# Activate litert venv and add onnxruntime to it
source litert/.venv/bin/activate
pip install onnxruntime

python compare.py
```

---

## KV-cache strategies compared

| Strategy | Format | How cache works | Decode signature |
|---|---|---|---|
| **Explicit** | `.tflite` (litert-torch) | Caller owns `k_i`/`v_i` arrays, threads them as tensor args each step | `(hidden, token, pos, mask, k_0, v_0, …)` → `(logits, k_0_upd, v_0_upd, …)` |
| **Stateful** | `.tflite` (TF SavedModel) | `tf.Variable` inside interpreter; caller sees nothing | `(hidden, token, step, mask)` → `(logits,)` |
| **ONNX** | `.onnx` (optimum) | `present.*` tensors passed back as `past_key_values.*`; split encoder/decoder | Step-0 decoder + `decoder_with_past` |

---

## Adapting for a new T5 variant

### LiteRT explicit / stateful

Edit the **MODEL CONFIG block** at the top of each `convert.py`:

```python
MODEL_DIR  = Path(...)    # HuggingFace checkpoint directory
MODEL_NAME = "my_t5"      # output filename stem
SEQ_LEN    = 128

# Architecture — match your checkpoint's config.json
NUM_LAYERS = 6
NUM_HEADS  = 12
HEAD_DIM   = 64
D_MODEL    = 768
D_FF       = 3072
VOCAB_SIZE = 32128
```

The `litert/stateful/convert.py` reads `N_LAYERS`, `N_HEADS`, etc. directly
from `config.json` in `MODEL_DIR`, so only `MODEL_DIR` and `SEQ_LEN` need
changing for standard HuggingFace T5 checkpoints.

Also update `NUM_HEADS` / `HEAD_DIM` in the corresponding `verify.py` (used
to allocate the explicit KV-cache buffers).

### ONNX

Only `MODEL_DIR` needs changing in `onnx/export.py`. Optimum handles the rest.

---

## Dependencies

### litert/.venv

| Package | Version | Purpose |
|---|---|---|
| `litert-torch` | 0.8.0 | PyTorch → TFLite (explicit KV path) |
| `torch` | 2.9.x | Pinned by litert-torch |
| `tensorflow` | latest | TF SavedModel → TFLite (stateful path) |
| `ai-edge-litert` | latest | Run `.tflite` files |
| `transformers` | latest | Load HF checkpoint + tokeniser |
| `sentencepiece` | latest | T5 tokeniser backend |
| `sacrebleu` | latest | BLEU-4 / chrF++ evaluation |

> `litert-torch 0.8.0` pins `torch 2.9.x` — keep this venv separate from
> executorch (which requires torch 2.10+).

### onnx/.venv

| Package | Purpose |
|---|---|
| `optimum[onnxruntime]` | Export HF model to ONNX + INT8 quantisation |
| `onnxruntime` | CPU inference |
| `transformers`, `sentencepiece` | Tokeniser |
| `sacrebleu` | Evaluation |
