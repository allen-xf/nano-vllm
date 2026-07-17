import os
from glob import glob

import torch
from torch import nn
from safetensors import safe_open

from nanovllm.layers.embed_head import ParallelLMHead, VocabParallelEmbedding
from nanovllm.models.qwen3_dflash import (
    DFlashLoadInfo,
    DFlashQwen3ForCausalLM,
    DFlashQwen3Model,
    _load_mask_embedding,
    _load_named_tensor,
)


def ensure_dspark_dflash_config(config) -> dict:
    dflash_config = dict(getattr(config, "dflash_config", None) or {})
    for key in ("mask_token_id", "target_layer_ids", "layer_ids", "use_aux_hidden_state"):
        value = getattr(config, key, None)
        if value is not None:
            dflash_config[key] = value
    config.dflash_config = dflash_config
    return dflash_config


class DSparkMarkovHead(nn.Module):

    def __init__(self, vocab_size: int, draft_vocab_size: int, markov_rank: int) -> None:
        super().__init__()
        self.markov_w1 = VocabParallelEmbedding(vocab_size, markov_rank)
        self.markov_w2 = ParallelLMHead(draft_vocab_size, markov_rank)

    def embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.markov_w1(token_ids)

    def bias(self, markov_embed: torch.Tensor) -> torch.Tensor:
        return self.markov_w2.forward_all(markov_embed)


class Qwen3DSparkModel(DFlashQwen3Model):

    def __init__(self, config, target_hidden_size: int):
        ensure_dspark_dflash_config(config)
        super().__init__(config, target_hidden_size)
        draft_vocab_size = getattr(config, "draft_vocab_size", None) or config.vocab_size
        markov_rank = getattr(config, "markov_rank", None)
        if markov_rank is None:
            raise ValueError("DSpark draft model requires markov_rank")
        self.markov_head = DSparkMarkovHead(config.vocab_size, draft_vocab_size, markov_rank)


class Qwen3DSparkForCausalLM(DFlashQwen3ForCausalLM):

    def __init__(self, config, target_hidden_size: int):
        nn.Module.__init__(self)
        ensure_dspark_dflash_config(config)
        self.config = config
        self.hidden_size = config.hidden_size
        self.draft_vocab_size = getattr(config, "draft_vocab_size", None) or config.vocab_size
        self.target_vocab_size = config.vocab_size
        self.model = Qwen3DSparkModel(config, target_hidden_size=target_hidden_size)
        self.lm_head = ParallelLMHead(self.draft_vocab_size, self.hidden_size)
        self.register_buffer("d2t", torch.zeros(self.draft_vocab_size, dtype=torch.long))
        self.register_buffer("t2d", torch.zeros(self.target_vocab_size, dtype=torch.bool))
        self.has_d2t = False

    def compute_draft_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head.forward_all(hidden_states)

    def map_draft_to_target(self, draft_ids: torch.Tensor) -> torch.Tensor:
        if not self.has_d2t and self.draft_vocab_size == self.target_vocab_size:
            return draft_ids
        return self.d2t[draft_ids]

    def markov_embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.model.markov_head.embed(token_ids)

    def markov_bias(self, markov_embed: torch.Tensor) -> torch.Tensor:
        return self.model.markov_head.bias(markov_embed)


def _normalize_dspark_weight_name(name: str, info: DFlashLoadInfo) -> str | None:
    if "mask_hidden" in name:
        raise RuntimeError("DSpark checkpoints should use mask_token_id, not mask_hidden")
    if "confidence_head" in name:
        return None
    if name.endswith("t2d") or ".t2d" in name:
        return None
    if name.endswith("draft_id_to_target_id"):
        info.includes_d2t = True
        return "d2t"
    if name.endswith("d2t") or ".d2t" in name:
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


def load_dspark_model(model: Qwen3DSparkForCausalLM, path: str) -> DFlashLoadInfo:
    info = DFlashLoadInfo()
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                normalized_name = _normalize_dspark_weight_name(weight_name, info)
                if normalized_name is None:
                    continue
                _load_named_tensor(model, normalized_name, f.get_tensor(weight_name))
    _load_mask_embedding(model, path, info)
    if info.includes_mask_embedding:
        model.model.has_separate_mask_embedding = True
    model.finalize_d2t(info.includes_d2t)
    return info
