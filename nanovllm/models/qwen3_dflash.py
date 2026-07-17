from __future__ import annotations

import os
from dataclasses import dataclass
from glob import glob

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
from safetensors import safe_open

from nanovllm.layers.attention import Attention
from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import QKVParallelLinear, ReplicatedLinear, RowParallelLinear
from nanovllm.layers.rotary_embedding import get_rope
from nanovllm.models.qwen3 import Qwen3MLP
from nanovllm.utils.loader import default_weight_loader


def _get_rope_kwargs(config) -> tuple[float, dict | None]:
    rope_parameters = getattr(config, "rope_parameters", None) or {}
    rope_theta = rope_parameters.get("rope_theta", getattr(config, "rope_theta", 1000000))
    rope_scaling = getattr(config, "rope_scaling", None)
    if rope_scaling is None:
        rope_scaling = rope_parameters.get("rope_scaling")
    return rope_theta, rope_scaling


class DFlashQwen3Attention(nn.Module):

    def __init__(self, config) -> None:
        super().__init__()
        tp_size = dist.get_world_size()
        self.hidden_size = config.hidden_size
        self.total_num_heads = config.num_attention_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = config.num_key_value_heads
        assert self.total_num_kv_heads % tp_size == 0
        self.num_kv_heads = self.total_num_kv_heads // tp_size
        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.total_num_heads)
        rope_theta, rope_scaling = _get_rope_kwargs(config)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5
        attention_bias = getattr(config, "attention_bias", False)

        self.qkv_proj = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=attention_bias,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            bias=attention_bias,
        )
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=config.max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
        )
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        q, k, v = self.project_qkv(hidden_states)
        q = self.q_norm(q)
        k = self.k_norm(k)
        q, k = self.rotary_emb(positions, q, k)
        o = self.attn(q, k, v)
        return self.o_proj(o.flatten(1, -1))

    def project_qkv(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        return q, k, v

    def project_context_kv(self, hidden_states: torch.Tensor, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        kv_weight = self.qkv_proj.weight[self.q_size:]
        kv_bias = self.qkv_proj.bias[self.q_size:] if self.qkv_proj.bias is not None else None
        kv = F.linear(hidden_states, kv_weight, kv_bias)
        k, v = kv.split([self.kv_size, self.kv_size], dim=-1)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        k = self.k_norm(k)
        _, k = self.rotary_emb(positions, k, k)
        return k.contiguous(), v.contiguous()


class DFlashQwen3DecoderLayer(nn.Module):

    def __init__(self, config) -> None:
        super().__init__()
        self.self_attn = DFlashQwen3Attention(config)
        self.mlp = Qwen3MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class DFlashQwen3Model(nn.Module):

    def __init__(self, config, target_hidden_size: int):
        super().__init__()
        self.config = config
        dflash_config = getattr(config, "dflash_config", None) or {}
        self.use_aux_hidden_state = dflash_config.get("use_aux_hidden_state", True)
        self.mask_token_id = dflash_config.get("mask_token_id")
        self.target_layer_ids = dflash_config.get("target_layer_ids") or dflash_config.get("layer_ids") or []

        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.mask_embedding = nn.Parameter(torch.zeros(config.hidden_size), requires_grad=False)
        self.has_separate_mask_embedding = False
        self.layers = nn.ModuleList([DFlashQwen3DecoderLayer(config) for _ in range(config.num_hidden_layers)])
        if self.use_aux_hidden_state:
            num_features = len(self.target_layer_ids) if self.target_layer_ids else config.num_hidden_layers
            fc_input_size = getattr(config, "target_hidden_size", target_hidden_size) * num_features
            self.fc = ReplicatedLinear(fc_input_size, config.hidden_size, bias=False)
        self.hidden_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        if self.has_separate_mask_embedding and self.mask_token_id is not None:
            is_mask = (input_ids == self.mask_token_id).unsqueeze(-1)
            hidden_states = torch.where(is_mask, self.mask_embedding.to(hidden_states.dtype), hidden_states)
        return hidden_states

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        hidden_states = self.embed_input_ids(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

    def precompute_and_store_context_kv(
        self,
        context_states: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mapping: torch.Tensor | None,
    ) -> None:
        if context_states.numel() == 0:
            return
        normed_context = self.hidden_norm(context_states)
        for layer in self.layers:
            k, v = layer.self_attn.project_context_kv(normed_context, context_positions)
            layer.self_attn.attn.update_kv_cache(k, v, context_slot_mapping)


class DFlashQwen3ForCausalLM(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(self, config, target_hidden_size: int):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.draft_vocab_size = getattr(config, "draft_vocab_size", None) or config.vocab_size
        self.target_vocab_size = config.vocab_size
        self.model = DFlashQwen3Model(config, target_hidden_size=target_hidden_size)
        self.lm_head = ParallelLMHead(self.draft_vocab_size, self.hidden_size)
        self.register_buffer("d2t", torch.zeros(self.draft_vocab_size, dtype=torch.long))
        self.register_buffer("t2d", torch.zeros(self.target_vocab_size, dtype=torch.bool))
        self.has_d2t = False

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids, positions)

    def map_draft_logits_to_target(self, logits: torch.Tensor) -> torch.Tensor:
        if not self.has_d2t and self.draft_vocab_size == self.target_vocab_size:
            return logits
        mapped_shape = (*logits.shape[:-1], self.target_vocab_size)
        mapped_logits = logits.new_full(mapped_shape, float("-inf"))
        mapped_logits[..., self.d2t] = logits
        return mapped_logits

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.map_draft_logits_to_target(self.lm_head(hidden_states))

    def combine_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if not self.model.use_aux_hidden_state:
            return hidden_states
        return self.model.fc(hidden_states)

    def precompute_and_store_context_kv(
        self,
        context_states: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mapping: torch.Tensor | None,
    ) -> None:
        self.model.precompute_and_store_context_kv(context_states, context_positions, context_slot_mapping)

    def finalize_d2t(self, includes_d2t: bool) -> None:
        if includes_d2t:
            self.d2t.add_(torch.arange(self.draft_vocab_size, device=self.d2t.device))
            self.has_d2t = True
        elif self.draft_vocab_size != self.target_vocab_size:
            raise RuntimeError("DFlash draft_vocab_size differs from target vocab size but no d2t mapping was loaded")


@dataclass
class DFlashLoadInfo:
    includes_embed_tokens: bool = False
    includes_lm_head: bool = False
    includes_d2t: bool = False
    includes_mask_embedding: bool = False


def _load_named_tensor(model: nn.Module, name: str, loaded_weight: torch.Tensor) -> None:
    named_buffers = dict(model.named_buffers())
    if name in named_buffers:
        named_buffers[name].copy_(loaded_weight)
        return

    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    for key, (packed_name, shard_id) in packed_modules_mapping.items():
        if key not in name:
            continue
        param_name = name.replace(key, packed_name)
        param = model.get_parameter(param_name)
        weight_loader = getattr(param, "weight_loader")
        weight_loader(param, loaded_weight, shard_id)
        return

    param = model.get_parameter(name)
    weight_loader = getattr(param, "weight_loader", default_weight_loader)
    weight_loader(param, loaded_weight)


def _normalize_dflash_weight_name(name: str, info: DFlashLoadInfo) -> str | None:
    if "mask_hidden" in name:
        raise RuntimeError("DFlash checkpoints should use mask_token_id, not mask_hidden")
    if name.endswith("t2d") or ".t2d" in name:
        return None
    if name.endswith("draft_id_to_target_id"):
        info.includes_d2t = True
        return "d2t"
    if name.endswith("d2t"):
        info.includes_d2t = True
        return "d2t"
    if "midlayer." in name:
        name = name.replace("midlayer.", "layers.0.")
    if "embed_tokens" in name:
        info.includes_embed_tokens = True
    if "lm_head" in name:
        info.includes_lm_head = True
    if name.startswith("model.lm_head"):
        name = name[len("model."):]
    if name.endswith("mask_embedding") or name.endswith("mask_embedding.weight"):
        info.includes_mask_embedding = True
        return "model.mask_embedding"
    if "lm_head" not in name and not name.startswith("model."):
        name = "model." + name
    return name


def _load_mask_embedding(model: DFlashQwen3ForCausalLM, path: str, info: DFlashLoadInfo) -> None:
    mask_path = os.path.join(path, "mask_embedding.pt")
    if not os.path.exists(mask_path):
        return
    state = torch.load(mask_path, map_location="cpu")
    if isinstance(state, dict):
        mask_token_id = state.get("mask_token_id", model.model.mask_token_id)
        if mask_token_id != model.model.mask_token_id:
            raise ValueError(
                f"mask_embedding.pt mask_token_id {mask_token_id} does not match "
                f"dflash_config.mask_token_id {model.model.mask_token_id}"
            )
        state = state["embedding"]
    model.model.mask_embedding.data.copy_(state.reshape(-1))
    model.model.has_separate_mask_embedding = True
    info.includes_mask_embedding = True


def load_dflash_model(model: DFlashQwen3ForCausalLM, path: str) -> DFlashLoadInfo:
    info = DFlashLoadInfo()
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                normalized_name = _normalize_dflash_weight_name(weight_name, info)
                if normalized_name is None:
                    continue
                _load_named_tensor(model, normalized_name, f.get_tensor(weight_name))
    _load_mask_embedding(model, path, info)
    if info.includes_mask_embedding:
        model.model.has_separate_mask_embedding = True
    model.finalize_d2t(info.includes_d2t)
    return info
