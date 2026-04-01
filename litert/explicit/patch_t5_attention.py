"""
Monkey-patch for litert_torch T5 classes.

Fixes two bugs in litert-torch v0.8.0:

1. CrossAttention constructor mismatch
   T5Attention calls: super().__init__(batch, dim, dim, config, kv_cache_max, enable_hlfb)
   CrossAttention v0.8.0 expects: (query_dim, cross_dim, hidden_dim, output_dim, config, enable_hlfb)
   → `config` (AttentionConfig) lands in the int `output_dim` slot and crashes.

2. KV cache stored as module state
   Storing kv_cache as self.kv_cache_entry makes litert treat those tensors as
   constants — they reset to zeros on every call. v0.8.0 requires KV cache to be
   an explicit function argument so the exporter creates proper TFLite
   VAR_HANDLE / ASSIGN ops.

Apply before building any T5 model:
    import litert_torch.generative.examples.t5.t5_attention as t5_attn_mod
    import litert_torch.generative.examples.t5.t5 as t5_mod
    from litert.explicit.patch_t5_attention import patch
    patch(t5_attn_mod, t5_mod)

After patching, build models with T5EncoderFixed / T5DecoderFixed.
"""

import copy
from typing import Optional, Tuple

import torch
import torch.nn as nn

from litert_torch.generative.layers.attention import CrossAttention
import litert_torch.generative.layers.builder as builder
import litert_torch.generative.layers.model_config as cfg
from litert_torch.generative.layers import kv_cache as kv_utils
import litert_torch.generative.layers.attention_utils as attn_utils


# ─────────────────────────────────────────────────────────────────────────────
# T5AttentionFixed
# ─────────────────────────────────────────────────────────────────────────────

class T5AttentionFixed(CrossAttention):
    """
    Drop-in replacement for T5Attention compatible with CrossAttention v0.8.0.
    KV cache is an explicit argument — NOT stored as module state.
    """

    def __init__(
        self,
        batch: int,
        dim: int,
        config: cfg.AttentionConfig,
        norm_config: cfg.NormalizationConfig,
        kv_cache_max: int,
        enable_hlfb: bool,
        has_relative_attention_bias: bool = False,
    ) -> None:
        hidden_dim = config.num_heads * config.head_dim
        # v0.8.0 CrossAttention: (query_dim, cross_dim, hidden_dim, output_dim, config, enable_hlfb)
        super().__init__(dim, dim, hidden_dim, dim, config, enable_hlfb)

        self._hidden_dim = hidden_dim
        self.pre_atten_norm = builder.build_norm(dim, norm_config)

        self.has_relative_attention_bias = has_relative_attention_bias
        self.relative_attention_num_buckets = config.relative_attention_num_buckets
        if self.has_relative_attention_bias:
            self.relative_attention_bias = nn.Embedding(
                self.relative_attention_num_buckets, self.n_heads
            )

    def forward(
        self,
        x: torch.Tensor,
        input_pos: Optional[torch.Tensor] = None,
        key_value_states: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        relative_position: Optional[torch.Tensor] = None,
        position_bias=None,
        kv_cache_entry: Optional[kv_utils.KVCacheEntry] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[kv_utils.KVCacheEntry]]:
        x = self.pre_atten_norm(x)
        B, T, _ = x.size()

        query_states = self.q_projection(x).reshape(B, T, -1, self.config.head_dim)

        if key_value_states is not None:
            kvB, kvT, _ = key_value_states.size()
            key_states   = self.k_projection(key_value_states).reshape(kvB, kvT, -1, self.config.head_dim)
            value_states = self.v_projection(key_value_states).reshape(kvB, kvT, -1, self.config.head_dim)
        else:
            key_states   = self.k_projection(x).reshape(B, T, -1, self.config.head_dim)
            value_states = self.v_projection(x).reshape(B, T, -1, self.config.head_dim)

        updated_kv_cache_entry = kv_cache_entry
        if key_value_states is None and kv_cache_entry is not None:
            updated_kv_cache_entry = kv_utils.update(
                kv_cache_entry, input_pos, key_states, value_states
            )
            key_states   = updated_kv_cache_entry.k_cache
            value_states = updated_kv_cache_entry.v_cache

        if position_bias is None:
            if self.has_relative_attention_bias:
                position_bias = self.relative_attention_bias(relative_position)
                position_bias = position_bias.permute([0, 1, 4, 2, 3]).squeeze(0)
            else:
                position_bias = torch.zeros_like(mask, dtype=torch.float32)

        mask = mask + position_bias
        y = self.sdpa_func(
            query_states, key_states, value_states,
            self.config.head_dim, mask=mask, scale=1.0,
        )
        y = y.reshape(B, T, self._hidden_dim)
        y = self.output_projection(y)
        return y, position_bias, updated_kv_cache_entry


# ─────────────────────────────────────────────────────────────────────────────
# EncoderDecoderBlockFixed
# ─────────────────────────────────────────────────────────────────────────────

class EncoderDecoderBlockFixed(nn.Module):
    def __init__(self, config: cfg.ModelConfig,
                 has_relative_attention_bias: bool = False,
                 is_decoder: bool = False):
        super().__init__()
        attn_cfg      = config.block_config(0).attn_config
        ff_cfg        = config.block_config(0).ff_config
        norm_cfg      = config.block_config(0).pre_attention_norm_config
        post_norm_cfg = config.block_config(0).post_attention_norm_config
        kv_max        = config.max_seq_len

        self.atten_func = T5AttentionFixed(
            batch=1, dim=config.embedding_dim, config=attn_cfg,
            norm_config=norm_cfg, kv_cache_max=kv_max, enable_hlfb=False,
            has_relative_attention_bias=has_relative_attention_bias,
        )

        self.cross_atten_func = None
        if is_decoder:
            cross_cfg = copy.deepcopy(attn_cfg)
            cross_cfg.enable_kv_cache = False
            self.cross_atten_func = T5AttentionFixed(
                batch=1, dim=config.embedding_dim, config=cross_cfg,
                norm_config=norm_cfg, kv_cache_max=kv_max, enable_hlfb=False,
                has_relative_attention_bias=False,
            )

        self.post_atten_norm = builder.build_norm(config.embedding_dim, post_norm_cfg)
        self.ff = builder.build_ff(config.embedding_dim, ff_cfg)

    def forward(self, x, input_pos=None, mask=None, relative_position=None,
                position_bias=None, encoder_hidden_states=None,
                encoder_attention_mask=None, encoder_decoder_position_bias=None,
                kv_cache_entry: Optional[kv_utils.KVCacheEntry] = None):
        hidden, position_bias, updated_kvc = self.atten_func(
            x, input_pos=input_pos, mask=mask, relative_position=relative_position,
            position_bias=position_bias, kv_cache_entry=kv_cache_entry,
        )
        attn_out = hidden + x

        if self.cross_atten_func is not None:
            hidden, encoder_decoder_position_bias, _ = self.cross_atten_func(
                attn_out, input_pos=input_pos,
                key_value_states=encoder_hidden_states,
                mask=encoder_attention_mask, relative_position=relative_position,
                position_bias=encoder_decoder_position_bias, kv_cache_entry=None,
            )
            attn_out = hidden + attn_out

        ff_out = self.post_atten_norm(attn_out)
        ff_out = self.ff(ff_out)
        return attn_out + ff_out, position_bias, encoder_decoder_position_bias, updated_kvc


# ─────────────────────────────────────────────────────────────────────────────
# T5StackFixed
# ─────────────────────────────────────────────────────────────────────────────

class T5StackFixed(nn.Module):
    def __init__(self, config: cfg.ModelConfig, embedding_layer: nn.Embedding,
                 is_decoder: bool = False):
        super().__init__()
        self.config       = config
        self.is_decoder   = is_decoder
        self.embed_tokens = embedding_layer
        self.transformer_blocks = nn.ModuleList([
            EncoderDecoderBlockFixed(config, has_relative_attention_bias=(i == 0),
                                     is_decoder=is_decoder)
            for i in range(config.num_layers)
        ])
        self.final_norm = builder.build_norm(config.embedding_dim, config.final_norm_config)

    def forward(self, input_ids, input_pos, attention_mask, relative_position,
                encoder_hidden_states=None, encoder_attention_mask=None,
                kv_cache: Optional[kv_utils.KVCache] = None):
        hidden           = self.embed_tokens(input_ids)
        position_bias    = None
        enc_dec_pos_bias = None

        for i, layer in enumerate(self.transformer_blocks):
            kvc_entry = kv_cache.caches[i] if kv_cache is not None else None
            hidden, position_bias, enc_dec_pos_bias, updated_kvc = layer(
                hidden, input_pos=input_pos, mask=attention_mask,
                relative_position=relative_position, position_bias=position_bias,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                encoder_decoder_position_bias=enc_dec_pos_bias,
                kv_cache_entry=kvc_entry,
            )
            if kv_cache is not None:
                caches = list(kv_cache.caches)
                caches[i] = updated_kvc
                kv_cache = kv_utils.KVCache(tuple(caches))

        return self.final_norm(hidden), kv_cache


# ─────────────────────────────────────────────────────────────────────────────
# T5EncoderFixed / T5DecoderFixed
# ─────────────────────────────────────────────────────────────────────────────

class T5EncoderFixed(nn.Module):
    def __init__(self, config: cfg.ModelConfig, embedding_layer: nn.Embedding):
        super().__init__()
        self.config  = config
        self.encoder = T5StackFixed(config, embedding_layer, is_decoder=False)

        attn_cfg = config.block_config(0).attn_config
        self.enc_rel_pos_mask = attn_utils.build_relative_position_buckets(
            bidirectional=True,
            query_length=config.max_seq_len, key_length=config.max_seq_len,
            num_buckets=attn_cfg.relative_attention_num_buckets,
            max_distance=attn_cfg.relative_attention_max_distance,
        )
        self.enc_attn_mask_cache = torch.zeros(
            (1, 1, config.max_seq_len, config.max_seq_len), dtype=torch.float32
        )

    @torch.inference_mode()
    def forward(self, input_ids: torch.Tensor, input_pos: torch.Tensor,
                pad_mask: torch.Tensor) -> torch.Tensor:
        enc_mask    = self.enc_attn_mask_cache.index_select(2, input_pos)
        enc_mask    = enc_mask[:, :, :, : self.config.max_seq_len]
        enc_mask[:, :, :, :] += pad_mask
        enc_rel_pos = self.enc_rel_pos_mask.index_select(2, input_pos)
        enc_rel_pos = enc_rel_pos[:, :, :, : self.config.max_seq_len]
        hidden, _   = self.encoder(input_ids=input_ids, input_pos=input_pos,
                                    attention_mask=enc_mask, relative_position=enc_rel_pos)
        return hidden


class T5DecoderFixed(nn.Module):
    def __init__(self, config: cfg.ModelConfig, embedding_layer: nn.Embedding):
        super().__init__()
        self.config  = config
        self.decoder = T5StackFixed(config, embedding_layer, is_decoder=True)
        self.lm_head = nn.Linear(config.embedding_dim, config.vocab_size,
                                  bias=config.lm_head_use_bias)

        attn_cfg = config.block_config(0).attn_config
        self.dec_rel_pos_mask = attn_utils.build_relative_position_buckets(
            bidirectional=False,
            query_length=config.max_seq_len, key_length=config.max_seq_len,
            num_buckets=attn_cfg.relative_attention_num_buckets,
            max_distance=attn_cfg.relative_attention_max_distance,
        )
        self.dec_attn_mask_cache = attn_utils.build_causal_mask_cache(size=config.max_seq_len)
        self.enc_attn_mask_cache = torch.zeros(
            (1, 1, config.max_seq_len, config.max_seq_len), dtype=torch.float32
        )

    @torch.inference_mode()
    def forward(self, encoder_hidden_states: torch.Tensor,
                decoder_input_ids: torch.Tensor, decoder_input_pos: torch.Tensor,
                pad_mask: torch.Tensor,
                kv_cache: kv_utils.KVCache) -> Tuple[torch.Tensor, kv_utils.KVCache]:
        dec_mask    = self.dec_attn_mask_cache.index_select(2, decoder_input_pos)
        dec_mask    = dec_mask[:, :, :, : self.config.max_seq_len]
        dec_rel_pos = self.dec_rel_pos_mask.index_select(2, decoder_input_pos)
        dec_rel_pos = dec_rel_pos[:, :, :, : self.config.max_seq_len]
        enc_attn_mask = self.enc_attn_mask_cache.index_select(2, decoder_input_pos) + pad_mask

        decoder_out, updated_kv_cache = self.decoder(
            input_ids=decoder_input_ids, input_pos=decoder_input_pos,
            attention_mask=dec_mask, relative_position=dec_rel_pos,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=enc_attn_mask, kv_cache=kv_cache,
        )
        sequence_output = decoder_out * (self.config.embedding_dim ** -0.5)
        return self.lm_head(sequence_output), updated_kv_cache


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def patch(t5_attn_mod, t5_mod=None) -> None:  # noqa: ARG001
    """Replace upstream T5Attention in the given litert module with T5AttentionFixed."""
    t5_attn_mod.T5Attention = T5AttentionFixed
    print("[patch] T5Attention → T5AttentionFixed  (CrossAttention v0.8.0 + external KV cache)")
    print("[patch] T5EncoderFixed / T5DecoderFixed available.")
