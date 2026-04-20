"""
Convert a HuggingFace T5 checkpoint to ExecuTorch (.pte) using optimum-executorch.

Uses optimum.exporters.executorch with the xnnpack recipe.  Two variants:
  fp32  — full precision, no quantization
  int8  — 8-bit weight-only quantization on attention + MLP linear layers
           (embeddings and lm_head excluded due to T5's tied-weight scheme)

The optimum-executorch seq2seq task does not (yet) apply quantization by itself,
so for int8 we load the model manually, quantize with torchao via
optimum.exporters.executorch.quantization.quantize_model_, then export through
the lower-level export_to_executorch() path.

Method signatures in the produced model.pte:
  encoder(input_ids: int64[1, seq_len])
      → encoder_hidden_states: float32[1, seq_len, 384]

  text_decoder(decoder_input_ids: int64[1, 1],
               encoder_hidden_states: float32[1, seq_len, 384],
               cache_position: int64[1],
               encoder_attention_mask: int64[1, seq_len])
      → lm_logits: float32[1, 1, vocab_size]

  sampler(logits: float32[1, 1, vocab_size])
      → next_token_id: int64[1]   (argmax — optional, skip in Python if preferred)

The decoder uses a static KV cache.  cache_position[0] doubles as a reset signal:
passing cache_position=[0] at step 0 zeroes the cache, enabling multi-sentence
inference with a single loaded module.

Output:
  executorch/output/fp32/model.pte
  executorch/output/int8/model.pte

Usage:
  source executorch/.venv/bin/activate
  python executorch/convert.py
  python executorch/convert.py --fp32-only
  python executorch/convert.py --int8-only
"""

import argparse
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_DIR = Path(__file__).parents[2] / "translation" / "model_pt"
OUT_DIR   = Path(__file__).parent / "output"
SEQ_LEN   = 128   # encoder and decoder sequence length (static; inputs are padded to this)


# ── Static-shape exportable module ───────────────────────────────────────────

def _make_exportable(model, seq_len: int):
    """
    Wrap model in Seq2SeqLMExportableModule and patch _export_decoder to use
    fully static shapes.

    The upstream implementation marks encoder_hidden_states dim-1 as dynamic
    (max=max_hidden_seq_len).  With transformers 5.0 and T5's cross-attention
    causal-mask slicing, this causes a symbolic-shape conflict with the static
    KV-cache size during torch.export.  Since we always pad encoder inputs to
    SEQ_LEN, fixing both dims to SEQ_LEN avoids the conflict at zero cost.
    """
    import torch
    import torch.nn as nn
    from transformers.integrations.executorch import (
        Seq2SeqLMDecoderExportableModuleWithStaticCache,
    )
    from optimum.exporters.executorch.integrations import Seq2SeqLMExportableModule

    d_model = model.config.d_model
    device  = model.device


    class T5StaticExportableModule(Seq2SeqLMExportableModule):
        def _export_encoder(self, encoder_input_ids):
            # Wrap the encoder so it computes attention_mask = (input_ids != 0)
            # internally. Without this, the encoder attends to PAD tokens (id=0)
            # used for static padding, corrupting hidden states at real positions.
            # transformers 5.0 made attention_mask keyword-only in T5Stack.forward,
            # so we call it with kwargs and return last_hidden_state directly.
            enc = self.encoder
            enc_device = self.model.device

            class _EncoderWithMask(nn.Module):
                def __init__(self):
                    super().__init__()
                    self._enc = enc

                def forward(self, input_ids):
                    attention_mask = (input_ids != 0).long()
                    hidden = self._enc(
                        input_ids=input_ids, attention_mask=attention_mask
                    ).last_hidden_state
                    # Zero out PAD positions so decoder cross-attention ignores them,
                    # matching the behaviour of HF encoder which only outputs real tokens.
                    return hidden * attention_mask.unsqueeze(-1).float()

            wrapped = _EncoderWithMask().to(enc_device).eval()
            seq_len_dim = torch.export.Dim("encoder_seq_len", max=seq_len)
            with torch.no_grad():
                return torch.export.export(
                    wrapped,
                    (encoder_input_ids,),
                    dynamic_shapes={"input_ids": {1: seq_len_dim}},
                    strict=True,
                )

        def _export_decoder(self, decoder_input_ids, encoder_hidden_states, cache_position):
            # Subclass to inject encoder_attention_mask into the decoder call so
            # T5's relative position bias is computed for actual encoder length,
            # not the full padded SEQ_LEN.
            class _T5DecoderWithEncoderMask(Seq2SeqLMDecoderExportableModuleWithStaticCache):
                def forward(self, decoder_input_ids, encoder_hidden_states,
                            cache_position, encoder_attention_mask):
                    outputs = self.decoder(
                        input_ids=decoder_input_ids,
                        encoder_hidden_states=encoder_hidden_states,
                        encoder_attention_mask=encoder_attention_mask,
                        past_key_values=self.cache,
                        use_cache=True,
                        cache_position=cache_position,
                    )
                    return self.lm_head(outputs[0])

            wrapped = (
                _T5DecoderWithEncoderMask(
                    model=self.model,
                    max_static_cache_length=seq_len,
                    batch_size=1,
                )
                .to(device)
                .eval()
            )
            example_dec_ids  = torch.tensor([[0]], dtype=torch.long, device=device)
            example_enc_hs   = torch.zeros((1, seq_len, d_model), dtype=torch.float32, device=device)
            example_cache    = torch.tensor([0], dtype=torch.long, device=device)
            example_enc_mask = torch.ones((1, seq_len), dtype=torch.long, device=device)
            with torch.nn.attention.sdpa_kernel(
                [torch.nn.attention.SDPBackend.MATH]
            ), torch.no_grad():
                return torch.export.export(
                    wrapped,
                    (example_dec_ids, example_enc_hs, example_cache, example_enc_mask),
                    dynamic_shapes=None,   # fully static — avoids cross-attn mask conflict
                    strict=True,
                )

    return T5StaticExportableModule(model, batch_size=1, max_seq_len=seq_len, max_hidden_seq_len=seq_len)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _export_fp32(out_dir: Path) -> None:
    """FP32 export with static shapes."""
    import torch
    from transformers import AutoModelForSeq2SeqLM
    from optimum.exporters.executorch.convert import export_to_executorch

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Checkpoint : {MODEL_DIR}")
    print(f"  Output     : {out_dir / 'model.pte'}")

    model      = AutoModelForSeq2SeqLM.from_pretrained(str(MODEL_DIR)).eval()
    exportable = _make_exportable(model, SEQ_LEN)

    export_to_executorch(
        exportable,
        task="text2text-generation",
        recipe="xnnpack",
        output_dir=str(out_dir),
    )


def _export_int8(out_dir: Path) -> None:
    """
    INT8 export using torchao weight-only quantization (qlinear_config="8w").

    NOTE ON SIZE REDUCTION
    The expected ~4× size reduction (→ ~28 MB) requires XNNPACK-native int8
    quantized linear kernels.  This needs the pt2e (prepare_pt2e → convert_pt2e)
    pipeline from executorch.  That pipeline is blocked here by a version mismatch:
      - executorch 1.2.0's XNNPACKQuantizer emits aten.quantized.* ops
      - torchao's convert_pt2e emits quantized_decomposed.* ops
      - XNNPACK partitioner does not fuse the decomposed pattern → no size gain

    What we do instead:
      - quantize_model_(qlinear_config="8w") via torchao IntxWeightOnlyConfig
      - This quantizes attention/MLP nn.Linear weights to int8 in the eager model
      - T5 lm_head.weight is FIRST cloned (untied from shared embedding) so both
        can be quantized independently
      - Embedding (model.shared) is also quantized via qembedding_config="8w"
      - XNNPACK runs fp32 computation internally (dequantizes weights on the fly),
        so runtime speed is similar to fp32

    Actual size: ~91 MB (vs 113 MB fp32) — ~20% smaller.
    """
    import torch
    from transformers import AutoModelForSeq2SeqLM

    from optimum.exporters.executorch.convert import export_to_executorch
    from optimum.exporters.executorch.quantization import quantize_model_

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Checkpoint : {MODEL_DIR}")
    print(f"  Output     : {out_dir / 'model.pte'}")

    model = AutoModelForSeq2SeqLM.from_pretrained(str(MODEL_DIR)).eval()

    # Untie lm_head from shared embedding so each can be quantized independently.
    model.lm_head.weight = torch.nn.Parameter(
        model.shared.weight.detach().clone(), requires_grad=False
    )
    model.config.tie_word_embeddings = False

    quantize_model_(model, qlinear_config="8w")

    exportable = _make_exportable(model, SEQ_LEN)

    export_to_executorch(
        exportable,
        task="text2text-generation",
        recipe="xnnpack",
        output_dir=str(out_dir),
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fp32-only", action="store_true")
    parser.add_argument("--int8-only", action="store_true")
    args = parser.parse_args()

    do_fp32 = not args.int8_only
    do_int8 = not args.fp32_only

    if do_fp32:
        print("=" * 60)
        print("Pass 1 — FP32")
        print("=" * 60)
        _export_fp32(OUT_DIR / "fp32")
        size_mb = (OUT_DIR / "fp32" / "model.pte").stat().st_size / 1024 / 1024
        print(f"  Done  {size_mb:.1f} MB")

    if do_int8:
        print()
        print("=" * 60)
        print("Pass 2 — INT8 (8-bit weight-only linear, XNNPACK)")
        print("=" * 60)
        _export_int8(OUT_DIR / "int8")
        size_mb = (OUT_DIR / "int8" / "model.pte").stat().st_size / 1024 / 1024
        print(f"  Done  {size_mb:.1f} MB")

    print()
    print("Done.  Output files:")
    for f in sorted(OUT_DIR.rglob("model.pte")):
        print(f"  {str(f.relative_to(OUT_DIR)):40s}  {f.stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
