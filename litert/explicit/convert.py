"""
Convert a HuggingFace T5 checkpoint to .tflite with EXPLICIT KV cache.

Approach: litert-torch (PyTorch path)
  - KV cache is explicit: k_i / v_i tensors are inputs AND outputs of decode()
  - TFLite creates proper VAR_HANDLE / ASSIGN ops from the explicit pytree
  - Caller manages and threads cache tensors between decode steps

Decode signature:
  inputs:  (encoder_hidden_states, decoder_input_ids, decoder_input_pos,
             pad_mask, k_0, v_0, k_1, v_1, …, k_{N-1}, v_{N-1})
  outputs: (logits, k_0_updated, v_0_updated, …, k_{N-1}_updated, v_{N-1}_updated)

Produces:
  output/<MODEL_NAME>_explicit_int8.tflite
  output/<MODEL_NAME>_explicit_fp32.tflite

Usage:
  source litert/.venv/bin/activate
  python litert/explicit/convert.py
"""

import sys
from pathlib import Path

# Apply patch BEFORE any litert T5 import
import litert_torch.generative.examples.t5.t5_attention as _t5_attn_mod
import litert_torch.generative.examples.t5.t5 as _t5_mod
sys.path.insert(0, str(Path(__file__).parent))
from patch_t5_attention import patch as _patch, T5EncoderFixed, T5DecoderFixed
_patch(_t5_attn_mod, _t5_mod)

import re
import torch
import torch.nn as nn
import litert_torch
from litert_torch.generative.quantize import quant_recipes
import litert_torch.generative.layers.model_config as cfg
from litert_torch.generative.layers import kv_cache as kv_utils
from transformers import T5ForConditionalGeneration


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL CONFIG  ← edit this section for a different T5 variant
# ══════════════════════════════════════════════════════════════════════════════

# HuggingFace checkpoint directory (must contain config.json + weights)
MODEL_DIR = Path(__file__).parents[3] / "translation" / "model_pt"

# Output directory for .tflite files
OUT_DIR = Path(__file__).parents[1] / "output"

# Stem used in output filenames
MODEL_NAME = "t5_mini"

# Max encoder / decoder sequence length
SEQ_LEN = 128

# Architecture — match your checkpoint's config.json
NUM_LAYERS  = 4
NUM_HEADS   = 8
HEAD_DIM    = 32      # d_kv
D_MODEL     = 384     # d_model
D_FF        = 1536    # d_ff
VOCAB_SIZE  = 20008

REL_ATTN_BUCKETS  = 32
REL_ATTN_MAX_DIST = 128

# ══════════════════════════════════════════════════════════════════════════════


def get_model_config() -> cfg.ModelConfig:
    attn_config = cfg.AttentionConfig(
        num_heads=NUM_HEADS, head_dim=HEAD_DIM, num_query_groups=NUM_HEADS,
        qkv_use_bias=False,
        relative_attention_num_buckets=REL_ATTN_BUCKETS,
        relative_attention_max_distance=REL_ATTN_MAX_DIST,
        enable_kv_cache=True,
    )
    ff_config = cfg.FeedForwardConfig(
        type=cfg.FeedForwardType.SEQUENTIAL,
        activation=cfg.ActivationConfig(cfg.ActivationType.RELU),
        intermediate_size=D_FF,
    )
    norm_config = cfg.NormalizationConfig(
        type=cfg.NormalizationType.RMS_NORM, epsilon=1e-6, enable_hlfb=False,
    )
    block_config = cfg.TransformerBlockConfig(
        attn_config=attn_config, relative_attention=True, ff_config=ff_config,
        pre_attention_norm_config=norm_config, post_attention_norm_config=norm_config,
    )
    return cfg.ModelConfig(
        vocab_size=VOCAB_SIZE, num_layers=NUM_LAYERS, max_seq_len=SEQ_LEN,
        embedding_dim=D_MODEL, block_configs=block_config,
        final_norm_config=norm_config, lm_head_use_bias=False,
    )


# ── Weight loading ────────────────────────────────────────────────────────────

PROJ_MAP = {"q_projection": "q", "k_projection": "k",
            "v_projection": "v", "output_projection": "o"}


def _remap_key(key: str) -> str:
    """Map our module parameter name → corresponding HuggingFace state-dict key."""
    if key in ("encoder.embed_tokens.weight", "decoder.embed_tokens.weight"):
        return "shared.weight"
    if key == "lm_head.weight":
        return "lm_head.weight"

    m = re.match(r"(encoder|decoder)\.final_norm\.(\w+)", key)
    if m:
        return f"{m.group(1)}.final_layer_norm.{m.group(2)}"

    m = re.match(r"encoder\.transformer_blocks\.(\d+)\.(.+)", key)
    if m:
        i, rest = m.group(1), m.group(2)
        pfx = f"encoder.block.{i}"
        if rest.startswith("atten_func."):
            inner = rest[len("atten_func."):]
            if "relative_attention_bias" in inner:
                return f"{pfx}.layer.0.SelfAttention.relative_attention_bias.weight"
            if "pre_atten_norm" in inner:
                return f"{pfx}.layer.0.layer_norm.weight"
            for proj, hf_p in PROJ_MAP.items():
                if inner.startswith(proj):
                    return f"{pfx}.layer.0.SelfAttention.{hf_p}.weight"
        if rest.startswith("post_atten_norm."):
            return f"{pfx}.layer.1.layer_norm.weight"
        if rest.startswith("ff."):
            inner = rest[3:]
            if inner.startswith("w1"): return f"{pfx}.layer.1.DenseReluDense.wi.weight"
            if inner.startswith("w2"): return f"{pfx}.layer.1.DenseReluDense.wo.weight"

    m = re.match(r"decoder\.transformer_blocks\.(\d+)\.(.+)", key)
    if m:
        i, rest = m.group(1), m.group(2)
        pfx = f"decoder.block.{i}"
        if rest.startswith("atten_func."):
            inner = rest[len("atten_func."):]
            if "relative_attention_bias" in inner:
                return f"{pfx}.layer.0.SelfAttention.relative_attention_bias.weight"
            if "pre_atten_norm" in inner:
                return f"{pfx}.layer.0.layer_norm.weight"
            for proj, hf_p in PROJ_MAP.items():
                if inner.startswith(proj):
                    return f"{pfx}.layer.0.SelfAttention.{hf_p}.weight"
        if rest.startswith("cross_atten_func."):
            inner = rest[len("cross_atten_func."):]
            if "pre_atten_norm" in inner:
                return f"{pfx}.layer.1.layer_norm.weight"
            for proj, hf_p in PROJ_MAP.items():
                if inner.startswith(proj):
                    return f"{pfx}.layer.1.EncDecAttention.{hf_p}.weight"
        if rest.startswith("post_atten_norm."):
            return f"{pfx}.layer.2.layer_norm.weight"
        if rest.startswith("ff."):
            inner = rest[3:]
            if inner.startswith("w1"): return f"{pfx}.layer.2.DenseReluDense.wi.weight"
            if inner.startswith("w2"): return f"{pfx}.layer.2.DenseReluDense.wo.weight"

    return key


def load_weights(model: nn.Module, model_dir: Path) -> nn.Module:
    hf     = T5ForConditionalGeneration.from_pretrained(str(model_dir))
    hf_sd  = hf.state_dict()
    our_sd = model.state_dict()

    new_sd  = {}
    missing = []
    for our_key in our_sd:
        hf_key = _remap_key(our_key)
        if hf_key in hf_sd:
            new_sd[our_key] = hf_sd[hf_key]
        else:
            missing.append((our_key, hf_key))

    if missing:
        print(f"  [warn] {len(missing)} unmapped param(s):")
        for ok, hk in missing[:10]:
            print(f"    {ok!r:60s} → {hk!r}")

    result = model.load_state_dict(new_sd, strict=False)
    print(f"  Loaded {len(new_sd)} tensors. "
          f"Missing={len(result.missing_keys)}, Unexpected={len(result.unexpected_keys)}")
    return model


# ── Sample inputs for tracing ─────────────────────────────────────────────────

def _enc_sample(config: cfg.ModelConfig):
    tokens    = torch.full((1, SEQ_LEN), 0, dtype=torch.int)
    tokens[0, :6] = torch.tensor([20000, 3, 9, 100, 50, 1], dtype=torch.int)
    input_pos = torch.arange(SEQ_LEN, dtype=torch.int)
    pad_mask  = torch.zeros(SEQ_LEN, dtype=torch.float32)
    pad_mask[6:] = float("-inf")
    return tokens, input_pos, pad_mask


def _dec_sample(config: cfg.ModelConfig):
    hidden_states = torch.zeros(1, SEQ_LEN, config.embedding_dim, dtype=torch.float32)
    decoder_token = torch.tensor([[0]], dtype=torch.int)
    decoder_pos   = torch.tensor([0],   dtype=torch.int)
    pad_mask      = torch.zeros(SEQ_LEN, dtype=torch.float32)
    pad_mask[6:]  = float("-inf")
    kv_cache      = kv_utils.KVCache.from_model_config(
        kv_cache_max=SEQ_LEN, config=config, batch_size=1,
    )
    return hidden_states, decoder_token, decoder_pos, pad_mask, kv_cache


# ── Conversion ────────────────────────────────────────────────────────────────

def convert(config: cfg.ModelConfig, quant_config, out_path: Path) -> None:
    embedding = nn.Embedding(config.vocab_size, config.embedding_dim, padding_idx=0)

    print("  Building encoder …")
    enc = T5EncoderFixed(config, embedding)
    load_weights(enc, MODEL_DIR)
    enc.eval()

    print("  Building decoder …")
    dec = T5DecoderFixed(config, embedding)
    load_weights(dec, MODEL_DIR)
    dec.eval()

    enc_in = _enc_sample(config)
    dec_in = _dec_sample(config)
    with torch.no_grad():
        h = enc(*enc_in)
        print(f"    encoder → {tuple(h.shape)}")
        l, _ = dec(h, *dec_in[1:])
        print(f"    decoder → {tuple(l.shape)}")

    print("  Converting via litert_torch.signature …")
    edge_model = (
        litert_torch.signature("encode", enc, enc_in)
        .signature("decode", dec, dec_in)
        .convert(quant_config=quant_config)
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    edge_model.export(str(out_path))
    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"  Saved → {out_path}  ({size_mb:.1f} MB)")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"Checkpoint : {MODEL_DIR}")
    print(f"Output dir : {OUT_DIR}")
    print()

    config = get_model_config()

    print("=" * 60)
    print("Pass 1 — INT8 dynamic quantisation")
    print("=" * 60)
    convert(config, quant_recipes.full_dynamic_recipe(),
            OUT_DIR / f"{MODEL_NAME}_explicit_int8.tflite")

    print()
    print("=" * 60)
    print("Pass 2 — FP32 baseline")
    print("=" * 60)
    convert(config, None,
            OUT_DIR / f"{MODEL_NAME}_explicit_fp32.tflite")

    print()
    print("Done.  Output files:")
    for f in sorted(OUT_DIR.glob(f"{MODEL_NAME}_explicit*.tflite")):
        print(f"  {f.name:50s}  {f.stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
