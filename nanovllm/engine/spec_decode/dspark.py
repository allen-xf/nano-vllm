from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from nanovllm.engine.spec_decode.dflash import DFlashProposalInfo, DFlashSpecBackend
from nanovllm.models.qwen3_dspark import (
    Qwen3DSparkForCausalLM,
    ensure_dspark_dflash_config,
    load_dspark_model,
)
from nanovllm.utils.context import reset_context, set_context

if TYPE_CHECKING:
    from nanovllm.engine.model_runner import ModelRunner


class DSparkSpecBackend(DFlashSpecBackend):
    """Qwen3 DSpark speculative decoding backend.

    Phase 1 keeps DFlash's target-side contract (target hidden capture, context-KV
    precompute, and target verify), and only swaps the proposal path to native
    DSpark query layout plus sequential Markov sampling.
    """

    def __init__(self, runner: "ModelRunner") -> None:
        self.runner = runner
        self.config = runner.config
        self.block_size = runner.block_size
        self.enforce_eager = runner.enforce_eager
        self.spec_profile = runner.spec_profile
        self.spec_debug = runner.spec_debug
        self.rank = runner.rank
        self.model = runner.model
        self.sampler = runner.sampler

        config = runner.config
        hf_config = config.hf_config
        self.dflash_config = ensure_dspark_dflash_config(config.draft_hf_config)
        raw_target_layer_ids = self.dflash_config.get("target_layer_ids") or self.dflash_config.get("layer_ids")
        self.target_layer_ids = [layer_id + 1 for layer_id in raw_target_layer_ids]
        self.mask_token_id = self.dflash_config["mask_token_id"]
        self.num_spec_tokens = config.num_spec_tokens
        self.sample_from_anchor = not getattr(config.draft_hf_config, "dspark_bonus_anchor", False)
        self.num_query_per_req = self.num_spec_tokens if self.sample_from_anchor else self.num_spec_tokens + 1
        self.eagle3_fuse_layers = self.target_layer_ids
        self.dflash_causal = False

        self.draft_model = Qwen3DSparkForCausalLM(
            config.draft_hf_config,
            target_hidden_size=hf_config.hidden_size,
        )
        load_info = load_dspark_model(self.draft_model, config.draft_model)
        if not load_info.includes_embed_tokens:
            target_embed = self.model.model.embed_tokens
            draft_embed = self.draft_model.model.embed_tokens
            if target_embed.weight.shape != draft_embed.weight.shape:
                raise RuntimeError("DSpark checkpoint omitted embed_tokens but target/draft embeddings are incompatible")
            self.draft_model.model.embed_tokens = target_embed

        if not load_info.includes_lm_head:
            target_lm_head = self.model.lm_head
            draft_lm_head = self.draft_model.lm_head
            if target_lm_head.weight.shape != draft_lm_head.weight.shape:
                raise RuntimeError("DSpark checkpoint omitted lm_head but target/draft lm_head are incompatible")
            self.draft_model.lm_head = target_lm_head

        runner.draft_model = self.draft_model
        runner.num_spec_tokens = self.num_spec_tokens
        runner.dflash_target_layer_ids = self.target_layer_ids
        self.reset_acceptance_metrics()

    def _verify_spec_decode_rows(self, decode_seqs, decode_rows, logits_selected):
        # temp=1 直接复用 propose 阶段缓存的 draft logits(q)，与采样严格同源，
        # 无需二次 draft forward，既消除上下文不一致又更快。
        need_draft_logits = any(row["seq"].temperature == 1 for row in decode_rows)
        draft_target_logits = None
        if need_draft_logits:
            draft_target_logits = torch.stack(
                [row["seq"].prev_draft_logits for row in decode_rows], dim=0
            )
        return super()._verify_spec_decode_rows(
            decode_seqs,
            decode_rows,
            logits_selected,
            draft_target_logits=draft_target_logits,
        )

    def _run_spec_draft_sync_and_propose(self, proposal_infos: list[DFlashProposalInfo]) -> None:
        K = self.num_spec_tokens
        if K <= 0 or not proposal_infos:
            return

        sample_from_anchor = getattr(self, "sample_from_anchor", True)
        num_query_per_req = getattr(
            self,
            "num_query_per_req",
            K if sample_from_anchor else K + 1,
        )

        query_input_ids = []
        query_positions = []
        query_slot_mapping = []
        query_context_lens = []
        sample_indices = []

        for info in proposal_infos:
            row_start = len(query_input_ids)
            slots = self.block_manager.append_n_slots(info.seq, num_query_per_req, start_pos=info.query_start_position)
            query_input_ids.append(info.bonus_token)
            if sample_from_anchor:
                query_input_ids.extend([self.mask_token_id] * (K - 1))
                sample_indices.extend(range(row_start, row_start + K))
            else:
                query_input_ids.extend([self.mask_token_id] * K)
                sample_indices.extend(range(row_start + 1, row_start + K + 1))
            query_positions.extend(range(info.query_start_position, info.query_start_position + num_query_per_req))
            query_slot_mapping.extend(slots)
            query_context_lens.extend([info.query_start_position + num_query_per_req] * num_query_per_req)

        query_input_ids_t = torch.tensor(query_input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        query_positions_t = torch.tensor(query_positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        query_slot_mapping_t = torch.tensor(query_slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        query_context_lens_t = torch.tensor(query_context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        query_block_tables = self.prepare_block_tables([info.seq for info in proposal_infos]).repeat_interleave(num_query_per_req, dim=0)

        set_context(False,
                    decode_slot_mapping=query_slot_mapping_t,
                    decode_context_lens=query_context_lens_t,
                    block_tables=query_block_tables,
                    num_decode_seqs=len(query_input_ids),
                    causal=False)
        hidden_states = self.draft_model(query_input_ids_t, query_positions_t)
        reset_context()

        sample_indices_t = torch.tensor(sample_indices, dtype=torch.int64, device=hidden_states.device)
        base_logits = self.draft_model.compute_draft_logits(hidden_states.index_select(0, sample_indices_t)).view(len(proposal_infos), K, -1)
        temperatures = self.prepare_sample([info.seq for info in proposal_infos])
        prev = torch.tensor([info.bonus_token for info in proposal_infos], dtype=torch.int64, device=hidden_states.device)
        draft_tokens = []
        draft_step_logits = []  # 缓存每步采样所用的 target 词表 logits(q)，供 temp=1 verify 复用
        for step in range(K):
            markov_embed = self.draft_model.markov_embed(prev)
            bias = self.draft_model.markov_bias(markov_embed)
            logits = base_logits[:, step, :] + bias
            draft_step_logits.append(self.draft_model.map_draft_logits_to_target(logits))
            draft_ids = self.sampler(logits, temperatures)
            target_ids = self.draft_model.map_draft_to_target(draft_ids)
            draft_tokens.append(target_ids)
            prev = target_ids

        draft_tokens = torch.stack(draft_tokens, dim=1).cpu().tolist()
        # [B, K, target_vocab]：与 draft_tokens 严格同源，消除 verify 二次 forward 的上下文不一致
        draft_step_logits = torch.stack(draft_step_logits, dim=1)
        for i, (info, tokens) in enumerate(zip(proposal_infos, draft_tokens)):
            info.seq.prev_draft_tokens = tokens
            info.seq.prev_draft_logits = draft_step_logits[i]

        for info in proposal_infos:
            self.block_manager.rollback_blocks(info.seq, info.rollback_num_blocks)
