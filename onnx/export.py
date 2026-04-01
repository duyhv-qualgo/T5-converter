"""
Export a HuggingFace T5 checkpoint to ONNX via Optimum.

Produces three files in OUT_DIR:
  encoder_model.onnx            — encoder
  decoder_model.onnx            — decoder (step 0, no past KV)
  decoder_with_past_model.onnx  — decoder (step 1+, with past KV)

The decoder_with_past pattern is the standard optimum KV-cache export for
text2text-generation. At inference:
  step 0 : encoder_model + decoder_model → logits + present.*.{key,value}
  step k>0: decoder_with_past_model (past = present from previous step) → logits + updated present

Optionally also exports INT8 dynamic-quantised versions.

Usage:
  source onnx/.venv/bin/activate
  python onnx/export.py
  python onnx/export.py --int8        # also quantise to INT8
  python onnx/export.py --opset 17    # override ONNX opset
"""

import argparse
from pathlib import Path


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG  ← edit for a different checkpoint
# ══════════════════════════════════════════════════════════════════════════════

MODEL_DIR = Path(__file__).parents[2] / "translation" / "model_pt"
OUT_FP32  = Path(__file__).parent / "output" / "fp32"
OUT_INT8  = Path(__file__).parent / "output" / "int8"

# ══════════════════════════════════════════════════════════════════════════════


def export_fp32(opset: int):
    from optimum.exporters.onnx import main_export

    print(f"Exporting FP32 ONNX to {OUT_FP32} …")
    OUT_FP32.mkdir(parents=True, exist_ok=True)
    main_export(
        model_name_or_path=str(MODEL_DIR),
        output=str(OUT_FP32),
        task="text2text-generation-with-past",
        opset=opset,
    )
    files = sorted(OUT_FP32.glob("*.onnx"))
    print("  Exported:")
    for f in files:
        print(f"    {f.name:50s}  {f.stat().st_size / 1024 / 1024:.1f} MB")


def quantise_int8():
    from optimum.onnxruntime import ORTQuantizer
    from optimum.onnxruntime.configuration import AutoQuantizationConfig

    print(f"\nQuantising to INT8 in {OUT_INT8} …")
    OUT_INT8.mkdir(parents=True, exist_ok=True)

    qconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False)

    for model_file in ["encoder_model.onnx", "decoder_model.onnx",
                       "decoder_with_past_model.onnx"]:
        src = OUT_FP32 / model_file
        if not src.exists():
            print(f"  [skip] {model_file} not found in fp32/")
            continue
        quantizer = ORTQuantizer.from_pretrained(str(OUT_FP32), file_name=model_file)
        quantizer.quantize(save_dir=str(OUT_INT8), quantization_config=qconfig)
        out = OUT_INT8 / model_file
        if out.exists():
            print(f"  {model_file:50s}  {out.stat().st_size / 1024 / 1024:.1f} MB")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--int8",  action="store_true", help="Also export INT8")
    parser.add_argument("--opset", type=int, default=14, help="ONNX opset (default: 14)")
    args = parser.parse_args()

    print(f"Checkpoint : {MODEL_DIR}")
    print()

    export_fp32(args.opset)

    if args.int8:
        quantise_int8()

    print("\nDone.")


if __name__ == "__main__":
    main()
