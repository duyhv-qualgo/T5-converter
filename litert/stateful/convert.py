"""
Convert a HuggingFace T5 checkpoint to .tflite with STATEFUL KV cache.

Approach: TF SavedModel → TFLite (via tf.lite.TFLiteConverter)
  - T5 encoder/decoder built as tf.Module in pure TensorFlow
  - KV cache stored as tf.Variable (one per layer × k/v)
  - TFLite lowers tf.Variable → VAR_HANDLE + READ_VARIABLE + ASSIGN_VARIABLE ops
  - Decode signature has NO k/v tensors — state lives inside the interpreter

Decode signature:
  inputs:  (encoder_hidden_states, decoder_input_ids, step, pad_mask)
  outputs: (logits,)
  (KV cache updated in-place — caller sees nothing)

Cache reset between sentences: re-allocate the TFLite interpreter
  (new interpreter = fresh zero-initialised resource variables)

Produces:
  output/<MODEL_NAME>_stateful_enc_fp32.tflite
  output/<MODEL_NAME>_stateful_dec_fp32.tflite
  output/<MODEL_NAME>_stateful_dec_int8.tflite

Usage:
  source litert/.venv/bin/activate
  python litert/stateful/convert.py
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import math
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from safetensors.torch import load_file as load_safetensors


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL CONFIG  ← edit this section for a different T5 variant
# ══════════════════════════════════════════════════════════════════════════════

MODEL_DIR  = Path(__file__).parents[3] / "translation" / "model_pt"
OUT_DIR    = Path(__file__).parents[1] / "output"
MODEL_NAME = "t5_mini"

SEQ_LEN = 128

# Read architecture from checkpoint config.json (falls back to defaults)
_cfg = json.loads((MODEL_DIR / "config.json").read_text())
N_LAYERS  = _cfg.get("num_layers", 4)
N_HEADS   = _cfg.get("num_heads",  8)
D_KV      = _cfg.get("d_kv",      32)
D_MODEL   = _cfg.get("d_model",   384)
D_FF      = _cfg.get("d_ff",      1536)
VOCAB_SIZE= _cfg.get("vocab_size", 20008)
REL_BUCKETS  = _cfg.get("relative_attention_num_buckets", 32)
REL_MAX_DIST = _cfg.get("relative_attention_max_distance", 128)

# ══════════════════════════════════════════════════════════════════════════════


# ── Weight loading ────────────────────────────────────────────────────────────

def _load_hf_weights() -> dict:
    """Return {hf_key: numpy_array} from the checkpoint."""
    sf = MODEL_DIR / "model.safetensors"
    if sf.exists():
        sd = load_safetensors(str(sf))
        return {k: v.numpy() for k, v in sd.items()}
    import torch
    sd = torch.load(MODEL_DIR / "pytorch_model.bin", map_location="cpu")
    return {k: v.numpy() for k, v in sd.items()}


# ── Relative position bucket table ───────────────────────────────────────────

def _build_rel_pos_buckets(bidirectional: bool) -> np.ndarray:
    """Return (SEQ_LEN, SEQ_LEN) int32 bucket indices."""
    q_pos = np.arange(SEQ_LEN)[:, None]
    k_pos = np.arange(SEQ_LEN)[None, :]
    rel   = k_pos - q_pos

    num_buckets  = REL_BUCKETS
    max_distance = REL_MAX_DIST

    if bidirectional:
        num_buckets //= 2
        is_fwd        = (rel > 0).astype(np.int32)
        abs_rel       = np.abs(rel)
        bucket_offset = num_buckets * is_fwd
    else:
        abs_rel       = np.clip(-rel, 0, None)
        bucket_offset = np.zeros_like(abs_rel, dtype=np.int32)

    max_exact_b = num_buckets // 2
    is_small    = (abs_rel < max_exact_b)
    log_bucket  = max_exact_b + (
        np.log(np.maximum(abs_rel, 1) / max_exact_b) /
        math.log(max_distance / max_exact_b) *
        (num_buckets - max_exact_b)
    ).astype(np.int32)
    log_bucket = np.clip(log_bucket, 0, num_buckets - 1)
    buckets    = np.where(is_small, abs_rel, log_bucket)
    return (buckets + bucket_offset).astype(np.int32)


# ── TF primitives ─────────────────────────────────────────────────────────────

def _rms_norm(x, weight, eps=1e-6):
    ms = tf.reduce_mean(tf.square(x), axis=-1, keepdims=True)
    return x * tf.math.rsqrt(ms + eps) * tf.cast(weight, x.dtype)


def _t5_attention(q_w, k_w, v_w, o_w, x_q, x_kv, mask, pos_bias):
    """Pure-TF T5 attention without KV cache (used for cross-attn + encoder)."""
    q = tf.matmul(x_q,  q_w, transpose_b=True)
    k = tf.matmul(x_kv, k_w, transpose_b=True)
    v = tf.matmul(x_kv, v_w, transpose_b=True)

    def _split_heads(t, T):
        t = tf.reshape(t, [1, T, N_HEADS, D_KV])
        return tf.transpose(t, [0, 2, 1, 3])

    q = _split_heads(q, tf.shape(x_q)[1])
    k = _split_heads(k, tf.shape(x_kv)[1])
    v = _split_heads(v, tf.shape(x_kv)[1])

    scores = tf.matmul(q, k, transpose_b=True)
    if pos_bias is not None:
        scores = scores + tf.cast(pos_bias, scores.dtype)
    if mask is not None:
        scores = scores + tf.cast(mask, scores.dtype)
    attn = tf.nn.softmax(scores, axis=-1)

    out = tf.matmul(attn, v)
    out = tf.transpose(out, [0, 2, 1, 3])
    out = tf.reshape(out, [1, tf.shape(x_q)[1], N_HEADS * D_KV])
    return tf.matmul(out, o_w, transpose_b=True)


# ── T5 Encoder ────────────────────────────────────────────────────────────────

class T5EncoderTF(tf.Module):
    def __init__(self, sd: dict):
        super().__init__(name="T5EncoderTF")
        def w(k): return tf.constant(sd[k], dtype=tf.float32)

        self.embed    = tf.Variable(w("shared.weight"), trainable=False, name="embed")
        self.layers_q = [tf.Variable(w(f"encoder.block.{i}.layer.0.SelfAttention.q.weight"), trainable=False) for i in range(N_LAYERS)]
        self.layers_k = [tf.Variable(w(f"encoder.block.{i}.layer.0.SelfAttention.k.weight"), trainable=False) for i in range(N_LAYERS)]
        self.layers_v = [tf.Variable(w(f"encoder.block.{i}.layer.0.SelfAttention.v.weight"), trainable=False) for i in range(N_LAYERS)]
        self.layers_o = [tf.Variable(w(f"encoder.block.{i}.layer.0.SelfAttention.o.weight"), trainable=False) for i in range(N_LAYERS)]
        self.layers_ln0 = [tf.Variable(w(f"encoder.block.{i}.layer.0.layer_norm.weight"), trainable=False) for i in range(N_LAYERS)]
        self.layers_wi  = [tf.Variable(w(f"encoder.block.{i}.layer.1.DenseReluDense.wi.weight"), trainable=False) for i in range(N_LAYERS)]
        self.layers_wo  = [tf.Variable(w(f"encoder.block.{i}.layer.1.DenseReluDense.wo.weight"), trainable=False) for i in range(N_LAYERS)]
        self.layers_ln1 = [tf.Variable(w(f"encoder.block.{i}.layer.1.layer_norm.weight"), trainable=False) for i in range(N_LAYERS)]
        self.final_ln   = tf.Variable(w("encoder.final_layer_norm.weight"), trainable=False)

        buckets   = _build_rel_pos_buckets(bidirectional=True)
        rel_w     = sd["encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"]
        pos_bias  = rel_w[buckets].transpose(2, 0, 1)[None]   # (1, n_heads, SEQ_LEN, SEQ_LEN)
        self.enc_pos_bias  = tf.constant(pos_bias, dtype=tf.float32)
        self.enc_mask      = tf.constant(np.zeros((1, 1, SEQ_LEN, SEQ_LEN), dtype=np.float32))

    @tf.function(input_signature=[
        tf.TensorSpec([1, SEQ_LEN], tf.int32,  name="input_ids"),
        tf.TensorSpec([SEQ_LEN],    tf.int32,   name="input_pos"),
        tf.TensorSpec([SEQ_LEN],    tf.float32, name="pad_mask"),
    ])
    def encode(self, input_ids, input_pos, pad_mask):
        pad_mask_2d = tf.reshape(pad_mask, [1, 1, 1, SEQ_LEN])
        enc_mask    = self.enc_mask + pad_mask_2d
        x = tf.expand_dims(tf.gather(self.embed, input_ids[0]), 0)

        for i in range(N_LAYERS):
            xn   = _rms_norm(x, self.layers_ln0[i])
            attn = _t5_attention(self.layers_q[i], self.layers_k[i],
                                  self.layers_v[i], self.layers_o[i],
                                  xn, xn, enc_mask, self.enc_pos_bias)
            x = x + attn
            xn = _rms_norm(x, self.layers_ln1[i])
            ff = tf.nn.relu(tf.matmul(xn, self.layers_wi[i], transpose_b=True))
            ff = tf.matmul(ff, self.layers_wo[i], transpose_b=True)
            x  = x + ff

        return _rms_norm(x, self.final_ln)


# ── T5 Decoder (stateful tf.Variable KV cache) ───────────────────────────────

class T5DecoderStatefulTF(tf.Module):
    """
    T5 decoder with KV cache stored in tf.Variable.

    Each decode call:
      1. Embeds the single token
      2. For each layer: self-attn (updates k_self/v_self at `step`) → cross-attn → FF
      3. LM head → logits

    TFLite lowering: tf.Variable → VAR_HANDLE + READ_VARIABLE + ASSIGN_VARIABLE
    """

    def __init__(self, sd: dict):
        super().__init__(name="T5DecoderStatefulTF")
        def w(k): return tf.constant(sd[k], dtype=tf.float32)

        self.embed   = tf.Variable(w("shared.weight"),  trainable=False, name="embed")
        self.lm_head = tf.Variable(w("lm_head.weight"), trainable=False, name="lm_head")

        self.sq    = [tf.Variable(w(f"decoder.block.{i}.layer.0.SelfAttention.q.weight"),    trainable=False) for i in range(N_LAYERS)]
        self.sk    = [tf.Variable(w(f"decoder.block.{i}.layer.0.SelfAttention.k.weight"),    trainable=False) for i in range(N_LAYERS)]
        self.sv    = [tf.Variable(w(f"decoder.block.{i}.layer.0.SelfAttention.v.weight"),    trainable=False) for i in range(N_LAYERS)]
        self.so    = [tf.Variable(w(f"decoder.block.{i}.layer.0.SelfAttention.o.weight"),    trainable=False) for i in range(N_LAYERS)]
        self.s_ln0 = [tf.Variable(w(f"decoder.block.{i}.layer.0.layer_norm.weight"),         trainable=False) for i in range(N_LAYERS)]
        self.cq    = [tf.Variable(w(f"decoder.block.{i}.layer.1.EncDecAttention.q.weight"),  trainable=False) for i in range(N_LAYERS)]
        self.ck    = [tf.Variable(w(f"decoder.block.{i}.layer.1.EncDecAttention.k.weight"),  trainable=False) for i in range(N_LAYERS)]
        self.cv    = [tf.Variable(w(f"decoder.block.{i}.layer.1.EncDecAttention.v.weight"),  trainable=False) for i in range(N_LAYERS)]
        self.co    = [tf.Variable(w(f"decoder.block.{i}.layer.1.EncDecAttention.o.weight"),  trainable=False) for i in range(N_LAYERS)]
        self.c_ln1 = [tf.Variable(w(f"decoder.block.{i}.layer.1.layer_norm.weight"),         trainable=False) for i in range(N_LAYERS)]
        self.wi    = [tf.Variable(w(f"decoder.block.{i}.layer.2.DenseReluDense.wi.weight"),  trainable=False) for i in range(N_LAYERS)]
        self.wo    = [tf.Variable(w(f"decoder.block.{i}.layer.2.DenseReluDense.wo.weight"),  trainable=False) for i in range(N_LAYERS)]
        self.ff_ln = [tf.Variable(w(f"decoder.block.{i}.layer.2.layer_norm.weight"),         trainable=False) for i in range(N_LAYERS)]
        self.final_ln = tf.Variable(w("decoder.final_layer_norm.weight"), trainable=False)

        # Stateful KV cache — shape: (1, SEQ_LEN, N_HEADS, D_KV)
        zeros = np.zeros((1, SEQ_LEN, N_HEADS, D_KV), dtype=np.float32)
        self.k_self = [tf.Variable(zeros.copy(), trainable=False, name=f"k_self_{i}") for i in range(N_LAYERS)]
        self.v_self = [tf.Variable(zeros.copy(), trainable=False, name=f"v_self_{i}") for i in range(N_LAYERS)]

        # Precomputed tables
        dec_buckets = _build_rel_pos_buckets(bidirectional=False)
        rel_w = sd["decoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"]
        dec_pb = rel_w[dec_buckets].transpose(2, 0, 1)[None]  # (1, n_heads, SEQ_LEN, SEQ_LEN)
        self.dec_pos_bias_table = tf.constant(dec_pb, dtype=tf.float32)

        causal = np.full((SEQ_LEN, SEQ_LEN), float("-inf"), dtype=np.float32)
        for i in range(SEQ_LEN):
            causal[i, :i + 1] = 0.0
        self.causal_table = tf.constant(causal[None, None, :, :])
        self.cross_mask   = tf.constant(np.zeros((1, 1, 1, SEQ_LEN), dtype=np.float32))

    def _self_attn_with_kv_cache(self, x_norm, step, layer_idx, pos_bias, causal_mask):
        q     = tf.matmul(x_norm, self.sq[layer_idx], transpose_b=True)
        k_new = tf.matmul(x_norm, self.sk[layer_idx], transpose_b=True)
        v_new = tf.matmul(x_norm, self.sv[layer_idx], transpose_b=True)

        q     = tf.reshape(q,     [1, 1, N_HEADS, D_KV])
        k_new = tf.reshape(k_new, [1, 1, N_HEADS, D_KV])
        v_new = tf.reshape(v_new, [1, 1, N_HEADS, D_KV])

        # One-hot blend → read-modify-write (becomes ASSIGN_VARIABLE in TFLite)
        step_oh = tf.reshape(tf.one_hot(step, SEQ_LEN, dtype=tf.float32), [1, SEQ_LEN, 1, 1])
        k_cache = self.k_self[layer_idx] * (1.0 - step_oh) + k_new * step_oh
        v_cache = self.v_self[layer_idx] * (1.0 - step_oh) + v_new * step_oh
        self.k_self[layer_idx].assign(k_cache)
        self.v_self[layer_idx].assign(v_cache)

        q = tf.transpose(q,       [0, 2, 1, 3])
        k = tf.transpose(k_cache, [0, 2, 1, 3])
        v = tf.transpose(v_cache, [0, 2, 1, 3])

        scores = tf.matmul(q, k, transpose_b=True) + pos_bias + causal_mask
        attn   = tf.nn.softmax(scores, axis=-1)
        out    = tf.transpose(tf.matmul(attn, v), [0, 2, 1, 3])
        out    = tf.reshape(out, [1, 1, N_HEADS * D_KV])
        return tf.matmul(out, self.so[layer_idx], transpose_b=True)

    @tf.function(input_signature=[
        tf.TensorSpec([1, SEQ_LEN, D_MODEL], tf.float32, name="encoder_hidden_states"),
        tf.TensorSpec([1, 1],                tf.int32,   name="decoder_input_ids"),
        tf.TensorSpec([],                    tf.int32,   name="step"),
        tf.TensorSpec([SEQ_LEN],             tf.float32, name="pad_mask"),
    ])
    def decode(self, encoder_hidden_states, decoder_input_ids, step, pad_mask):
        causal_row = tf.reshape(tf.gather(self.causal_table[0, 0], step), [1, 1, 1, SEQ_LEN])
        pb_row     = tf.expand_dims(
            tf.expand_dims(tf.gather(self.dec_pos_bias_table[0], step, axis=1), 1), 0
        )  # (1, n_heads, 1, SEQ_LEN)
        cross_mask = self.cross_mask + tf.reshape(pad_mask, [1, 1, 1, SEQ_LEN])

        x = tf.expand_dims(tf.gather(self.embed, decoder_input_ids[0]), 0)

        for i in range(N_LAYERS):
            xn   = _rms_norm(x, self.s_ln0[i])
            attn = self._self_attn_with_kv_cache(xn, step, i, pb_row, causal_row)
            x    = x + attn

            xn    = _rms_norm(x, self.c_ln1[i])
            cattn = _t5_attention(self.cq[i], self.ck[i], self.cv[i], self.co[i],
                                   xn, encoder_hidden_states, cross_mask, None)
            x = x + cattn

            xn = _rms_norm(x, self.ff_ln[i])
            ff = tf.nn.relu(tf.matmul(xn, self.wi[i], transpose_b=True))
            ff = tf.matmul(ff, self.wo[i], transpose_b=True)
            x  = x + ff

        x = _rms_norm(x, self.final_ln) * tf.cast(D_MODEL ** -0.5, x.dtype)
        return tf.matmul(x, self.lm_head, transpose_b=True)


# ── SavedModel export ─────────────────────────────────────────────────────────

def build_and_save(sd: dict, saved_dir: Path):
    print("  Building TF encoder …")
    enc = T5EncoderTF(sd)
    enc.encode(tf.zeros([1, SEQ_LEN], tf.int32),
               tf.range(SEQ_LEN, tf.int32),
               tf.zeros([SEQ_LEN], tf.float32))
    print("    OK")

    print("  Building TF decoder (stateful) …")
    dec = T5DecoderStatefulTF(sd)
    dec.decode(tf.zeros([1, SEQ_LEN, D_MODEL], tf.float32),
               tf.zeros([1, 1], tf.int32),
               tf.constant(0, tf.int32),
               tf.zeros([SEQ_LEN], tf.float32))
    print("    OK")

    enc_dir = saved_dir / "encoder"
    dec_dir = saved_dir / "decoder"
    enc_dir.mkdir(parents=True, exist_ok=True)
    dec_dir.mkdir(parents=True, exist_ok=True)
    tf.saved_model.save(enc, str(enc_dir), signatures={"encode": enc.encode})
    tf.saved_model.save(dec, str(dec_dir), signatures={"decode": dec.decode})
    print(f"  SavedModels → {saved_dir}")
    return saved_dir / "encoder", saved_dir / "decoder"


# ── TFLite conversion (subprocess to avoid TF internal state issues) ──────────

def _to_tflite_subprocess(saved_subdir: Path, out_path: Path,
                           stateful: bool, quantize: bool):
    import subprocess, sys, textwrap
    script = textwrap.dedent(f"""
        import os; os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        import tensorflow as tf
        from pathlib import Path
        converter = tf.lite.TFLiteConverter.from_saved_model({str(saved_subdir)!r})
        converter.experimental_enable_resource_variables = {stateful}
        ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
        if {stateful}:
            ops.append(tf.lite.OpsSet.SELECT_TF_OPS)
        converter.target_spec.supported_ops = ops
        if {quantize}:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite = converter.convert()
        Path({str(out_path)!r}).write_bytes(tflite)
        has_var = b"VAR_HANDLE" in tflite or b"ReadVariable" in tflite
        print(f"  {{len(tflite)/1024/1024:.1f}} MB  stateful=" + str(has_var))
    """)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    if r.stdout.strip():
        print(r.stdout.strip())
    noise_keywords = ["I0000", "W0000", "gpu_device", "tf_tfl", "UserWarning", "FutureWarning"]
    err_lines = [l for l in r.stderr.splitlines()
                 if not any(x in l for x in noise_keywords)]
    if err_lines:
        print("\n".join(err_lines[-5:]))
    if r.returncode != 0:
        raise RuntimeError(f"TFLite conversion failed (exit {r.returncode})")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"Checkpoint : {MODEL_DIR}")
    print(f"Output dir : {OUT_DIR}")
    print()

    print("Loading HF weights …")
    sd = _load_hf_weights()
    print(f"  {len(sd)} tensors loaded\n")

    saved_dir = OUT_DIR / "_saved_model_stateful"
    enc_dir, dec_dir = build_and_save(sd, saved_dir)
    print()

    print("=" * 60)
    print("Pass 1 — Encoder FP32")
    print("=" * 60)
    enc_out = OUT_DIR / f"{MODEL_NAME}_stateful_enc_fp32.tflite"
    _to_tflite_subprocess(enc_dir, enc_out, stateful=False, quantize=False)
    print(f"  → {enc_out}")

    print()
    print("=" * 60)
    print("Pass 2 — Decoder FP32 (stateful VAR_HANDLE KV cache)")
    print("=" * 60)
    dec_fp32 = OUT_DIR / f"{MODEL_NAME}_stateful_dec_fp32.tflite"
    _to_tflite_subprocess(dec_dir, dec_fp32, stateful=True, quantize=False)
    print(f"  → {dec_fp32}")

    print()
    print("=" * 60)
    print("Pass 3 — Decoder INT8 (stateful)")
    print("=" * 60)
    dec_int8 = OUT_DIR / f"{MODEL_NAME}_stateful_dec_int8.tflite"
    _to_tflite_subprocess(dec_dir, dec_int8, stateful=True, quantize=True)
    print(f"  → {dec_int8}")

    print()
    print("Done.  Output files:")
    for f in sorted(OUT_DIR.glob(f"{MODEL_NAME}_stateful*.tflite")):
        print(f"  {f.name:55s}  {f.stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
