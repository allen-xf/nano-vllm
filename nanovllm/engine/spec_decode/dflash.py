from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from nanovllm.engine.sequence import Sequence
from nanovllm.engine.spec_decode.eagle3 import Eagle3SpecBackend
from nanovllm.models.qwen3_dflash import DFlashQwen3ForCausalLM, load_dflash_model
from nanovllm.utils.context import set_context, reset_context

if TYPE_CHECKING:
    from nanovllm.engine.model_runner import ModelRunner


@dataclass
class DFlashProposalInfo:
    seq: Sequence
    bonus_token: int
    query_start_position: int
    rollback_num_blocks: int


class DFlashSpecBackend(Eagle3SpecBackend):
    """Qwen3 DFlash speculative decoding backend.

    This first implementation keeps the existing target verify / accept-reject
    contract from EAGLE3 and only changes the proposal path: target hidden states
    are precomputed into the draft KV cache, then `[bonus] + mask * K` is run as
    a query-only draft forward.
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
        self.dflash_config = getattr(config.draft_hf_config, "dflash_config", None) or {}
        self.target_layer_ids = self.dflash_config.get("target_layer_ids") or self.dflash_config.get("layer_ids")
        self.mask_token_id = self.dflash_config["mask_token_id"]
        self.dflash_causal = self.dflash_config.get("causal", False)
        self.num_spec_tokens = config.num_spec_tokens
        # Reuse EAGLE3 target-verify CUDA graph helpers; DFlash captures these
        # target hidden layers for context KV precompute instead of EAGLE fusion.
        self.eagle3_fuse_layers = self.target_layer_ids

        self.draft_model = DFlashQwen3ForCausalLM(
            config.draft_hf_config,
            target_hidden_size=hf_config.hidden_size,
        )
        load_info = load_dflash_model(self.draft_model, config.draft_model)
        if not load_info.includes_embed_tokens:
            target_embed = self.model.model.embed_tokens
            draft_embed = self.draft_model.model.embed_tokens
            if target_embed.weight.shape != draft_embed.weight.shape:
                raise RuntimeError("DFlash checkpoint omitted embed_tokens but target/draft embeddings are incompatible")
            self.draft_model.model.embed_tokens = target_embed

        # DFlash checkpoint 未提供 lm_head 且 draft config tie_word_embeddings=True，
        # 需共享 target model 的 lm_head，否则 draft 使用随机权重导致提议 token 全部被拒。
        if not load_info.includes_lm_head:
            target_lm_head = self.model.lm_head
            draft_lm_head = self.draft_model.lm_head
            if target_lm_head.weight.shape != draft_lm_head.weight.shape:
                raise RuntimeError("DFlash checkpoint omitted lm_head but target/draft lm_head are incompatible")
            self.draft_model.lm_head = target_lm_head

        runner.draft_model = self.draft_model
        runner.num_spec_tokens = self.num_spec_tokens
        runner.dflash_target_layer_ids = self.target_layer_ids
        self.reset_acceptance_metrics()

    def _combine_captured_hidden(self, captured: dict[int, torch.Tensor], indices: list[int] | None = None):
        if not self.draft_model.model.use_aux_hidden_state:
            layer_id = self.target_layer_ids[-1]
            if indices is None:
                hidden_states = captured[layer_id]
            else:
                indices_t = torch.tensor(indices, dtype=torch.int64, device=captured[layer_id].device)
                hidden_states = captured[layer_id].index_select(0, indices_t)
            return self.draft_model.combine_hidden_states(hidden_states)
        if indices is None:
            hidden = [captured[layer_id] for layer_id in self.target_layer_ids]
        else:
            indices_t = torch.tensor(
                indices,
                dtype=torch.int64,
                device=captured[self.target_layer_ids[0]].device,
            )
            hidden = [captured[layer_id].index_select(0, indices_t) for layer_id in self.target_layer_ids]
        return self.draft_model.combine_hidden_states(torch.cat(hidden, dim=-1))

    def _run_spec_target_forward(self, use_spec_graph: bool, rows: list[dict],
                                 input_ids: torch.Tensor, positions: torch.Tensor,
                                 cu_q: torch.Tensor, cu_k: torch.Tensor, max_q: int, max_k: int,
                                 slot_mapping: torch.Tensor, block_tables: torch.Tensor | None):
        if use_spec_graph:
            return self._run_spec_decode_graph(
                rows, input_ids, positions, cu_k, max_k, slot_mapping, block_tables
            )
        set_context(True, cu_q, cu_k, max_q, max_k, slot_mapping,
                    block_tables=block_tables,
                    num_prefill_tokens=input_ids.size(0),
                    num_decode_seqs=0,
                    finishing_prefill_indices=[])
        return self.model(input_ids, positions, capture_layers=self.target_layer_ids)

    def _append_context_range(self, row: dict, start_pos: int, count: int,
                              context_indices: list[int], context_positions: list[int],
                              context_slots: list[int]) -> None:
        if count <= 0:
            return
        row_start = row["row_start"]
        seq = row["seq"]
        context_indices.extend(range(row_start, row_start + count))
        context_positions.extend(range(start_pos, start_pos + count))
        for pos in range(start_pos, start_pos + count):
            context_slots.append(self._slot_for_position(seq, pos))

    def _precompute_context_kv(self, captured: dict[int, torch.Tensor], context_indices: list[int],
                               context_positions: list[int], context_slots: list[int]) -> None:
        if not context_indices:
            return
        context_states = self._combine_captured_hidden(captured, context_indices)
        context_positions_t = torch.tensor(
            context_positions,
            dtype=torch.int64,
            pin_memory=True,
        ).cuda(non_blocking=True)
        context_slots_t = torch.tensor(
            context_slots,
            dtype=torch.int32,
            pin_memory=True,
        ).cuda(non_blocking=True)
        self.draft_model.precompute_and_store_context_kv(context_states, context_positions_t, context_slots_t)

    def _build_spec_draft_sync_batch(self, rows: list[dict], input_ids: torch.Tensor, captured: dict[int, torch.Tensor]):
        context_indices = []
        context_positions = []
        context_slots = []
        proposal_infos: list[DFlashProposalInfo] = []

        for row in rows:
            seq = row["seq"]
            if row["kind"] == "prefill":
                q_len = row["end"] - row["start"]
                self._append_context_range(row, row["start"], q_len, context_indices, context_positions, context_slots)
                if not row["is_finishing_prefill"]:
                    continue
                sampled_token = row["sampled_token"]
                will_finish = (
                    (not seq.ignore_eos and sampled_token == self.config.eos)
                    or seq.num_completion_tokens + 1 >= seq.max_tokens
                )
                if will_finish:
                    seq.prev_draft_tokens = []
                    continue
                proposal_infos.append(DFlashProposalInfo(
                    seq=seq,
                    bonus_token=sampled_token,
                    query_start_position=row["end"],
                    rollback_num_blocks=seq.num_blocks,
                ))
            else:
                accepted = row["accepted_tokens"]
                M = row["num_accepted"]
                original_len = row["original_len"]
                self._append_context_range(
                    row,
                    original_len - 1,
                    M,
                    context_indices,
                    context_positions,
                    context_slots,
                )

                final_len = original_len + M
                final_num_blocks = (final_len + self.block_size - 1) // self.block_size
                row["final_num_blocks"] = final_num_blocks
                will_finish = (
                    (not seq.ignore_eos and accepted[-1] == self.config.eos)
                    or seq.num_completion_tokens + M >= seq.max_tokens
                )
                if will_finish:
                    self.block_manager.rollback_blocks(seq, final_num_blocks)
                    seq.prev_draft_tokens = []
                    continue

                proposal_infos.append(DFlashProposalInfo(
                    seq=seq,
                    bonus_token=accepted[-1],
                    query_start_position=original_len - 1 + M,
                    rollback_num_blocks=final_num_blocks,
                ))

        self._precompute_context_kv(captured, context_indices, context_positions, context_slots)
        return (proposal_infos,)

    def _run_spec_draft_sync_and_propose(self, proposal_infos: list[DFlashProposalInfo]) -> None:
        K = self.num_spec_tokens
        if K <= 0 or not proposal_infos:
            return

        query_input_ids = []
        query_positions = []
        query_slot_mapping = []
        query_context_lens = []
        sample_indices = []

        for info in proposal_infos:
            row_start = len(query_input_ids)
            slots = self.block_manager.append_n_slots(info.seq, K + 1, start_pos=info.query_start_position)
            query_input_ids.append(info.bonus_token)
            query_input_ids.extend([self.mask_token_id] * K)
            query_positions.extend(range(info.query_start_position, info.query_start_position + K + 1))
            query_slot_mapping.extend(slots)
            sample_indices.extend(range(row_start + 1, row_start + K + 1))

            if self.dflash_causal:
                query_context_lens.extend(range(info.query_start_position + 1, info.query_start_position + K + 2))
            else:
                query_context_lens.extend([info.query_start_position + K + 1] * (K + 1))

        query_input_ids_t = torch.tensor(query_input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        query_positions_t = torch.tensor(query_positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        query_slot_mapping_t = torch.tensor(query_slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        query_context_lens_t = torch.tensor(query_context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        base_block_tables = self.prepare_block_tables([info.seq for info in proposal_infos])
        query_block_tables = base_block_tables.repeat_interleave(K + 1, dim=0)

        set_context(False,
                    decode_slot_mapping=query_slot_mapping_t,
                    decode_context_lens=query_context_lens_t,
                    block_tables=query_block_tables,
                    num_decode_seqs=len(query_input_ids),
                    causal=self.dflash_causal)
        hidden_states = self.draft_model(query_input_ids_t, query_positions_t)
        reset_context()

        sample_indices_t = torch.tensor(sample_indices, dtype=torch.int64, device=hidden_states.device)
        logits = self.draft_model.compute_logits(hidden_states.index_select(0, sample_indices_t))
        draft_tokens = logits.argmax(dim=-1).view(len(proposal_infos), K).cpu().tolist()
        for info, tokens in zip(proposal_infos, draft_tokens):
            info.seq.prev_draft_tokens = tokens

        for info in proposal_infos:
            self.block_manager.rollback_blocks(info.seq, info.rollback_num_blocks)

    @torch.inference_mode()
    def capture_decode_cudagraph(self):
        super().capture_decode_cudagraph()
