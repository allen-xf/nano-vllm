from __future__ import annotations

import time as _time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from nanovllm.engine.sequence import Sequence
from nanovllm.models.eagle3 import Eagle3DraftModel
from nanovllm.utils.context import set_context, reset_context
from nanovllm.utils.loader import load_model

if TYPE_CHECKING:
    from nanovllm.engine.model_runner import ModelRunner


@dataclass
class SpecProposalInfo:
    seq: Sequence
    last_sync_index: int
    start_position: int
    rollback_num_blocks: int


class Eagle3SpecBackend:
    """EAGLE3 speculative decoding backend.

    This class is a mechanical extraction of the previous EAGLE3-specific
    orchestration from ModelRunner. It intentionally keeps the existing row
    layout, draft-token state, slot reservation, and CUDA graph behavior.
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
        self.draft_model = Eagle3DraftModel(
            config.draft_hf_config,
            target_hidden_size=hf_config.hidden_size,
            num_fuse_layers=len(config.eagle3_fuse_layers),
        )
        self.draft_model.midlayer.spec_debug = self.spec_debug
        load_model(self.draft_model, config.draft_model)
        # d2t 在 checkpoint 中存储的是 offset（vLLM 约定: target_id = draft_id + d2t[draft_id]）
        # 转换为绝对 target ID，简化后续使用。
        self.draft_model.d2t.add_(torch.arange(self.draft_model.draft_vocab_size, device="cuda"))
        # 共享 target model 的 embedding。
        self.draft_model.embed_tokens = self.model.model.embed_tokens
        self.eagle3_fuse_layers = config.eagle3_fuse_layers
        self.num_spec_tokens = config.num_spec_tokens

        # Compatibility attributes for ModelRunner allocation code during this refactor.
        runner.draft_model = self.draft_model
        runner.eagle3_fuse_layers = self.eagle3_fuse_layers
        runner.num_spec_tokens = self.num_spec_tokens
        self.reset_acceptance_metrics()

    def __getattr__(self, name):
        return getattr(self.runner, name)

    @property
    def block_manager(self):
        return self.runner.block_manager

    def reset_acceptance_metrics(self):
        self._acceptance_metrics = {
            "num_rows": 0,
            "accepted_prefix_hist": [0] * (self.num_spec_tokens + 1),
            "emitted_token_hist": [0] * (self.num_spec_tokens + 2),
            "position_accepts": [0] * self.num_spec_tokens,
            "bonus_accepts": 0,
        }

    def _record_acceptance_metrics(self, prefix_len: int, emitted_len: int):
        metrics = self._acceptance_metrics
        metrics["num_rows"] += 1
        metrics["accepted_prefix_hist"][prefix_len] += 1
        metrics["emitted_token_hist"][min(emitted_len, self.num_spec_tokens + 1)] += 1
        for pos in range(prefix_len):
            metrics["position_accepts"][pos] += 1
        if prefix_len == self.num_spec_tokens and emitted_len > self.num_spec_tokens:
            metrics["bonus_accepts"] += 1

    def get_acceptance_metrics(self) -> dict:
        if self.rank != 0:
            return {}
        metrics = self._acceptance_metrics
        num_rows = metrics["num_rows"]
        if num_rows == 0:
            return {
                "num_rows": 0,
                "avg_accepted_prefix_len": 0.0,
                "accepted_prefix_hist": metrics["accepted_prefix_hist"],
                "emitted_token_hist": metrics["emitted_token_hist"],
                "position_accept_rates": [0.0] * self.num_spec_tokens,
                "bonus_rate": 0.0,
            }
        avg_prefix = sum(
            prefix_len * count
            for prefix_len, count in enumerate(metrics["accepted_prefix_hist"])
        ) / num_rows
        position_accept_rates = [
            count / num_rows for count in metrics["position_accepts"]
        ]
        return {
            "num_rows": num_rows,
            "avg_accepted_prefix_len": avg_prefix,
            "accepted_prefix_hist": metrics["accepted_prefix_hist"],
            "emitted_token_hist": metrics["emitted_token_hist"],
            "position_accept_rates": position_accept_rates,
            "bonus_rate": metrics["bonus_accepts"] / num_rows,
        }

    def _fuse_captured_hidden(self, captured, indices=None):
        if indices is None:
            hidden = [captured[l] for l in self.eagle3_fuse_layers]
        else:
            if not isinstance(indices, torch.Tensor):
                indices = torch.tensor(
                    indices,
                    dtype=torch.int64,
                    device=captured[self.eagle3_fuse_layers[0]].device,
                )
            hidden = [captured[l].index_select(0, indices) for l in self.eagle3_fuse_layers]
        return self.draft_model.fc(torch.cat(hidden, dim=-1))

    def _assign_prev_draft_tokens(self, seqs: list[Sequence], draft_tokens_all: list[torch.Tensor]):
        if not seqs:
            return
        draft_tokens = torch.stack(draft_tokens_all, dim=1).cpu().tolist()
        for seq, tokens in zip(seqs, draft_tokens):
            seq.prev_draft_tokens = tokens

    @torch.inference_mode()
    def _generate_draft_tokens_from_state(self, seqs, current_input, current_hidden,
                                          start_positions, num_steps,
                                          initial_tokens=None, draft_slots=None,
                                          rollback_num_blocks=None):
        """从给定 EAGLE 状态继续 serial 生成 draft tokens。"""
        draft_tokens_all = list(initial_tokens) if initial_tokens is not None else []
        if num_steps <= 0:
            if rollback_num_blocks is not None:
                for seq, num_blocks in zip(seqs, rollback_num_blocks):
                    self.block_manager.rollback_blocks(seq, num_blocks)
            return draft_tokens_all

        N = len(seqs)
        if isinstance(start_positions, torch.Tensor):
            start_positions_t = start_positions.to(dtype=torch.int64, device="cuda")
        else:
            start_positions_t = torch.tensor(start_positions, dtype=torch.int64, device="cuda")

        if draft_slots is None:
            raise ValueError("draft_slots is required when num_steps > 0")
        draft_slots_t = torch.tensor(draft_slots, dtype=torch.int32, device="cuda")
        draft_block_tables = self.prepare_block_tables(seqs)

        for k in range(num_steps):
            draft_positions = start_positions_t + k
            draft_slot_mapping = draft_slots_t[:, k].contiguous()
            draft_context_lens = (draft_positions + 1).to(torch.int32)

            set_context(False, decode_slot_mapping=draft_slot_mapping,
                        decode_context_lens=draft_context_lens,
                        block_tables=draft_block_tables, num_decode_seqs=N)
            draft_logits, current_hidden = self.draft_model(current_input, draft_positions, current_hidden)
            reset_context()

            draft_token = draft_logits.argmax(dim=-1)
            target_token = self.draft_model.d2t[draft_token]

            if self.spec_debug and not hasattr(self, "_debug_logits_printed"):
                if not hasattr(self, "_debug_logits_count"):
                    self._debug_logits_count = 0
                self._debug_logits_count += 1
                if self._debug_logits_count <= 3:
                    logits_0 = draft_logits[0]
                    topk_vals, topk_ids = logits_0.topk(5)
                    topk_target = self.draft_model.d2t[topk_ids]
                    print(f"    [DEBUG] draft k={k}: raw_argmax={draft_token[0].item()}, "
                          f"top5_draft={topk_ids.tolist()}, top5_target={topk_target.tolist()}, "
                          f"top5_vals={[f'{v:.2f}' for v in topk_vals.tolist()]}, "
                          f"logits_std={logits_0.std().item():.4f}, hidden_norm={current_hidden[0].norm().item():.2f}")
                if self._debug_logits_count >= 3:
                    self._debug_logits_printed = True

            current_input = target_token
            draft_tokens_all.append(target_token)

        if rollback_num_blocks is not None:
            for seq, num_blocks in zip(seqs, rollback_num_blocks):
                self.block_manager.rollback_blocks(seq, num_blocks)

        return draft_tokens_all

    @torch.inference_mode()
    def _run_spec_proposal_batch(self, proposal_infos: list[SpecProposalInfo], sync_logits: torch.Tensor,
                                 sync_hidden_out: torch.Tensor) -> None:
        """从 draft sync 的输出统一生成下一轮 prev_draft_tokens。"""
        K = self.num_spec_tokens
        if K <= 0 or not proposal_infos:
            return

        seqs = [info.seq for info in proposal_infos]
        last_indices = torch.tensor(
            [info.last_sync_index for info in proposal_infos],
            dtype=torch.int64,
            device="cuda",
        )
        start_positions = [info.start_position for info in proposal_infos]
        rollback_num_blocks = [info.rollback_num_blocks for info in proposal_infos]

        current_hidden = sync_hidden_out[last_indices]
        draft_token = sync_logits[last_indices].argmax(dim=-1)
        first_draft_token = self.draft_model.d2t[draft_token]

        draft_slots = None
        if K > 1:
            draft_slots = [
                self.block_manager.append_n_slots(info.seq, K - 1, start_pos=info.start_position)
                for info in proposal_infos
            ]

        draft_tokens_all = self._generate_draft_tokens_from_state(
            seqs=seqs,
            current_input=first_draft_token,
            current_hidden=current_hidden,
            start_positions=start_positions,
            num_steps=K - 1,
            initial_tokens=[first_draft_token],
            draft_slots=draft_slots,
            rollback_num_blocks=rollback_num_blocks,
        )
        self._assign_prev_draft_tokens(seqs, draft_tokens_all)

    def _slot_for_position(self, seq: Sequence, pos: int) -> int:
        block_idx = pos // self.block_size
        block_offset = pos % self.block_size
        return seq.block_table[block_idx] * self.block_size + block_offset

    def _build_spec_target_batch(self, prefill_seqs: list[Sequence], decode_seqs: list[Sequence]):
        K = self.num_spec_tokens
        rows = []
        seqs = []
        input_ids = []
        positions = []
        slot_mapping = []
        cu_q = [0]
        cu_k = [0]
        max_q = 0
        max_k = 0

        for prefill_index, seq in enumerate(prefill_seqs):
            start = seq.num_computed_tokens
            end = start + seq.scheduled_chunk_size
            q_len = end - start
            k_len = end
            row_start = len(input_ids)
            input_ids.extend(seq[start:end])
            positions.extend(range(start, end))
            for pos in range(start, end):
                slot_mapping.append(self._slot_for_position(seq, pos))
            row_end = len(input_ids)
            rows.append({
                "kind": "prefill",
                "seq": seq,
                "prefill_index": prefill_index,
                "row_start": row_start,
                "row_end": row_end,
                "start": start,
                "end": end,
                "original_len": len(seq),
                "is_finishing_prefill": end >= seq.num_tokens,
            })
            seqs.append(seq)
            cu_q.append(cu_q[-1] + q_len)
            cu_k.append(cu_k[-1] + k_len)
            max_q = max(max_q, q_len)
            max_k = max(max_k, k_len)

        for decode_index, seq in enumerate(decode_seqs):
            if len(seq.prev_draft_tokens) != K:
                raise RuntimeError(
                    f"spec decode seq {seq.seq_id} missing prev_draft_tokens: "
                    f"got {len(seq.prev_draft_tokens)}, expected {K}"
                )
            original_len = len(seq)
            self.block_manager.append_n_slots(seq, K)
            row_start = len(input_ids)
            input_ids.append(seq.last_token)
            input_ids.extend(seq.prev_draft_tokens)
            positions.extend(range(original_len - 1, original_len + K))
            for pos in range(original_len - 1, original_len + K):
                slot_mapping.append(self._slot_for_position(seq, pos))
            row_end = len(input_ids)
            q_len = K + 1
            k_len = original_len + K
            rows.append({
                "kind": "decode_verify",
                "seq": seq,
                "decode_index": decode_index,
                "row_start": row_start,
                "row_end": row_end,
                "original_len": original_len,
                "num_prev_drafts": K,
            })
            seqs.append(seq)
            cu_q.append(cu_q[-1] + q_len)
            cu_k.append(cu_k[-1] + k_len)
            max_q = max(max_q, q_len)
            max_k = max(max_k, k_len)

        input_ids_t = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions_t = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping_t = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_q_t = torch.tensor(cu_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_k_t = torch.tensor(cu_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs) if cu_k[-1] > cu_q[-1] else None
        return rows, input_ids_t, positions_t, cu_q_t, cu_k_t, max_q, max_k, slot_mapping_t, block_tables

    def _can_run_spec_decode_graph(self, prefill_seqs: list[Sequence], decode_seqs: list[Sequence],
                                   block_tables: torch.Tensor | None, max_k: int) -> bool:
        if self.enforce_eager or prefill_seqs or not decode_seqs:
            return False
        if not hasattr(self, "spec_graphs") or not self.spec_graphs:
            return False
        if block_tables is None:
            return False
        if len(decode_seqs) > self.spec_graph_bs[-1]:
            return False
        if max_k > self.spec_graph_max_seqlen_k:
            return False
        return True

    def _run_spec_decode_graph(self, rows: list[dict], input_ids: torch.Tensor, positions: torch.Tensor,
                               cu_k: torch.Tensor, max_k: int, slot_mapping: torch.Tensor,
                               block_tables: torch.Tensor) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
        bs = len(rows)
        graph_bs = next(x for x in self.spec_graph_bs if x >= bs)
        q_len = self.spec_graph_query_len
        num_tokens = bs * q_len
        graph_tokens = graph_bs * q_len
        graph = self.spec_graphs[graph_bs]
        graph_vars = self.spec_graph_vars

        graph_vars["input_ids"][:num_tokens] = input_ids
        graph_vars["positions"][:num_tokens] = positions
        graph_vars["slot_mapping"][:graph_tokens].fill_(-1)
        graph_vars["slot_mapping"][:num_tokens] = slot_mapping
        graph_vars["cu_k"][:bs + 1] = cu_k
        if graph_bs > bs:
            cu_k_pad = [0]
            for row in rows:
                cu_k_pad.append(cu_k_pad[-1] + row["original_len"] + self.num_spec_tokens)
            while len(cu_k_pad) <= graph_bs:
                cu_k_pad.append(cu_k_pad[-1] + q_len)
            cu_k_pad_t = torch.tensor(cu_k_pad, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
            graph_vars["cu_k"][:graph_bs + 1] = cu_k_pad_t
        graph_vars["block_tables"][:graph_bs].zero_()
        graph_vars["block_tables"][:bs, :block_tables.size(1)] = block_tables

        graph.replay()
        hidden_out = graph_vars["outputs"][:num_tokens]
        captured = {
            layer: graph_vars["captured"][layer][:num_tokens]
            for layer in self.eagle3_fuse_layers
        }
        return hidden_out, captured

    def _run_spec_target_forward(self, use_spec_graph: bool, rows: list[dict],
                                 input_ids: torch.Tensor, positions: torch.Tensor,
                                 cu_q: torch.Tensor, cu_k: torch.Tensor, max_q: int, max_k: int,
                                 slot_mapping: torch.Tensor, block_tables: torch.Tensor | None):
        if use_spec_graph:
            hidden_out, captured = self._run_spec_decode_graph(
                rows, input_ids, positions, cu_k, max_k, slot_mapping, block_tables
            )
        else:
            set_context(True, cu_q, cu_k, max_q, max_k, slot_mapping,
                        block_tables=block_tables,
                        num_prefill_tokens=input_ids.size(0),
                        num_decode_seqs=0,
                        finishing_prefill_indices=[])
            hidden_out, captured = self.model(input_ids, positions, capture_layers=self.eagle3_fuse_layers)
        return hidden_out, captured

    def _select_spec_logits(self, rows: list[dict], hidden_out: torch.Tensor):
        finishing_rows = [row for row in rows if row["kind"] == "prefill" and row["is_finishing_prefill"]]
        decode_rows = [row for row in rows if row["kind"] == "decode_verify"]

        logit_indices = []
        for row in finishing_rows:
            row["sample_logit_index"] = len(logit_indices)
            logit_indices.append(row["row_end"] - 1)
        for row in decode_rows:
            row["logit_start"] = len(logit_indices)
            logit_indices.extend(range(row["row_start"], row["row_end"]))
            row["logit_end"] = len(logit_indices)

        logits_selected = None
        if logit_indices:
            logit_indices_t = torch.tensor(logit_indices, dtype=torch.int64, device=hidden_out.device)
            logits_selected = self.model.lm_head.forward_all(hidden_out.index_select(0, logit_indices_t))
        return finishing_rows, decode_rows, logits_selected

    def _sample_spec_prefill_rows(self, prefill_seqs: list[Sequence], finishing_rows: list[dict],
                                  logits_selected: torch.Tensor | None) -> list[int | None]:
        prefill_token_ids = [None] * len(prefill_seqs)
        if finishing_rows:
            if self.rank == 0:
                sample_logits = torch.stack([logits_selected[row["sample_logit_index"]] for row in finishing_rows])
                temperatures = self.prepare_sample([row["seq"] for row in finishing_rows])
                sampled_tokens = self.sampler(sample_logits, temperatures).tolist()
            else:
                sampled_tokens = [0] * len(finishing_rows)
            for row, token_id in zip(finishing_rows, sampled_tokens):
                row["sampled_token"] = token_id
                if self.rank == 0:
                    prefill_token_ids[row["prefill_index"]] = token_id
        return prefill_token_ids

    def _verify_spec_decode_rows_greedy(self, decode_seqs: list[Sequence], decode_rows: list[dict],
                                        logits_selected: torch.Tensor | None) -> list[list[int]]:
        K = self.num_spec_tokens
        decode_target_token_ids = []
        if self.rank == 0 and decode_rows:
            decode_logits_start = decode_rows[0]["logit_start"]
            decode_target_token_ids = (
                logits_selected[decode_logits_start:]
                .argmax(dim=-1)
                .view(len(decode_rows), K + 1)
                .cpu()
                .tolist()
            )

        decode_accepted_tokens = [[] for _ in decode_seqs]
        for row_idx, row in enumerate(decode_rows):
            seq = row["seq"]
            prefix_len = 0
            if self.rank == 0:
                target_token_ids = decode_target_token_ids[row_idx]
                accepted = []
                for j in range(K):
                    target_pred = target_token_ids[j]
                    draft_tok = seq.prev_draft_tokens[j]
                    if target_pred == draft_tok:
                        accepted.append(draft_tok)
                        prefix_len += 1
                    else:
                        accepted.append(target_pred)
                        break
                else:
                    accepted.append(target_token_ids[K])
                self._record_acceptance_metrics(prefix_len, len(accepted))
            else:
                accepted = [0]
            if self.rank == 0:
                if not seq.ignore_eos and self.config.eos in accepted:
                    accepted = accepted[:accepted.index(self.config.eos) + 1]
                remaining_tokens = seq.max_tokens - seq.num_completion_tokens
                accepted = accepted[:remaining_tokens]
            row["accepted_tokens"] = accepted
            row["num_accepted"] = len(accepted)
            if self.rank == 0:
                decode_accepted_tokens[row["decode_index"]] = accepted
        return decode_accepted_tokens

    def _sample_temp1_token(self, logits: torch.Tensor) -> int:
        # verify 逐行采样走 eager multinomial，避免调用 @torch.compile 的 sampler
        # 触发新形状重编译（BackendCompilerFailed）。
        probs = torch.softmax(logits.float(), dim=-1)
        return torch.multinomial(probs, num_samples=1).item()

    def _sample_probs_token(self, probs: torch.Tensor) -> int:
        # probs 已是归一化分布（residual/recovered），直接采样。
        return torch.multinomial(probs, num_samples=1).item()

    def _verify_temp1_row_delta(self, row_logits: torch.Tensor, draft_tokens: list[int]) -> tuple[list[int], int]:
        accepted = []
        prefix_len = 0
        for j, draft_tok in enumerate(draft_tokens):
            target_probs = torch.softmax(row_logits[j].float(), dim=-1)
            if torch.rand((), device=target_probs.device).item() < target_probs[draft_tok].item():
                accepted.append(draft_tok)
                prefix_len += 1
                continue

            recovered_probs = target_probs.clone()
            recovered_probs[draft_tok] = 0
            recovered_probs = recovered_probs / recovered_probs.sum()
            accepted.append(self._sample_probs_token(recovered_probs))
            break
        else:
            accepted.append(self._sample_temp1_token(row_logits[len(draft_tokens)].float()))
        return accepted, prefix_len

    def _verify_temp1_row_standard(self, row_logits: torch.Tensor, draft_logits: torch.Tensor,
                                   draft_tokens: list[int]) -> tuple[list[int], int]:
        accepted = []
        prefix_len = 0
        for j, draft_tok in enumerate(draft_tokens):
            target_probs = torch.softmax(row_logits[j].float(), dim=-1)
            draft_probs = torch.softmax(draft_logits[j].float(), dim=-1)
            draft_prob = draft_probs[draft_tok].item()
            target_prob = target_probs[draft_tok].item()
            accept_prob = min(1.0, target_prob / max(draft_prob, 1e-12))
            if torch.rand((), device=target_probs.device).item() < accept_prob:
                accepted.append(draft_tok)
                prefix_len += 1
                continue

            residual_probs = (target_probs - draft_probs).clamp_min_(0)
            residual_mass = residual_probs.sum()
            if residual_mass.item() <= 0:
                accepted.append(self._sample_temp1_token(row_logits[j].float()))
            else:
                accepted.append(self._sample_probs_token(residual_probs / residual_mass))
            break
        else:
            accepted.append(self._sample_temp1_token(row_logits[len(draft_tokens)].float()))
        return accepted, prefix_len

    def _verify_spec_decode_rows(self, decode_seqs: list[Sequence], decode_rows: list[dict],
                                 logits_selected: torch.Tensor | None,
                                 draft_target_logits: torch.Tensor | None = None) -> list[list[int]]:
        if not decode_rows:
            return [[] for _ in decode_seqs]

        if self.rank == 0:
            temperatures = self.prepare_sample([row["seq"] for row in decode_rows])
            if not torch.all((temperatures == 0) | (temperatures == 1)):
                unsupported = temperatures[(temperatures != 0) & (temperatures != 1)].cpu().tolist()
                raise NotImplementedError(
                    f"spec decoding currently supports temperature 0 or 1 only, got {unsupported}"
                )
            if (temperatures == 0).all():
                return self._verify_spec_decode_rows_greedy(decode_seqs, decode_rows, logits_selected)

            decode_logits_start = decode_rows[0]["logit_start"]
            decode_logits = logits_selected[decode_logits_start:].view(
                len(decode_rows), self.num_spec_tokens + 1, -1
            )
            temperature_values = temperatures.cpu().tolist()
        else:
            decode_logits = None
            temperature_values = None

        K = self.num_spec_tokens
        decode_accepted_tokens = [[] for _ in decode_seqs]
        for row_idx, row in enumerate(decode_rows):
            seq = row["seq"]
            if self.rank == 0:
                row_logits = decode_logits[row_idx]
                temperature = temperature_values[row_idx]
                if temperature == 0:
                    target_token_ids = row_logits.argmax(dim=-1).tolist()
                    accepted = []
                    prefix_len = 0
                    for j in range(K):
                        target_pred = target_token_ids[j]
                        draft_tok = seq.prev_draft_tokens[j]
                        if target_pred == draft_tok:
                            accepted.append(draft_tok)
                            prefix_len += 1
                        else:
                            accepted.append(target_pred)
                            break
                    else:
                        accepted.append(target_token_ids[K])
                elif draft_target_logits is not None:
                    accepted, prefix_len = self._verify_temp1_row_standard(
                        row_logits,
                        draft_target_logits[row_idx],
                        seq.prev_draft_tokens,
                    )
                else:
                    accepted, prefix_len = self._verify_temp1_row_delta(row_logits, seq.prev_draft_tokens)
                self._record_acceptance_metrics(prefix_len, len(accepted))
            else:
                accepted = [0]
            if self.rank == 0:
                if not seq.ignore_eos and self.config.eos in accepted:
                    accepted = accepted[:accepted.index(self.config.eos) + 1]
                remaining_tokens = seq.max_tokens - seq.num_completion_tokens
                accepted = accepted[:remaining_tokens]
            row["accepted_tokens"] = accepted
            row["num_accepted"] = len(accepted)
            if self.rank == 0:
                decode_accepted_tokens[row["decode_index"]] = accepted
        return decode_accepted_tokens

    def _build_spec_draft_sync_batch(self, rows: list[dict], input_ids: torch.Tensor, captured: dict[int, torch.Tensor]):
        sync_input_ids = []
        sync_positions = []
        sync_slot_mapping = []
        sync_fused_indices = []
        sync_cu_q = [0]
        sync_cu_k = [0]
        sync_seqs = []
        sync_rows = []
        max_sync_q = 0
        max_sync_k = 0

        for row in rows:
            seq = row["seq"]
            if row["kind"] == "prefill":
                start = row["start"]
                end = row["end"]
                q_len = end - start
                shifted = list(seq.token_ids[start + 1:end])
                if row["is_finishing_prefill"]:
                    shifted.append(row["sampled_token"])
                else:
                    shifted.append(seq.token_ids[end])
                row_sync_start = len(sync_input_ids)
                sync_input_ids.extend(shifted)
                sync_positions.extend(range(start, end))
                for pos in range(start, end):
                    sync_slot_mapping.append(self._slot_for_position(seq, pos))
                sync_fused_indices.extend(range(row["row_start"], row["row_end"]))
                row_sync_end = len(sync_input_ids)
                sync_rows.append({
                    "kind": "prefill",
                    "seq": seq,
                    "source_row": row,
                    "row_start": row_sync_start,
                    "row_end": row_sync_end,
                })
                sync_seqs.append(seq)
                sync_cu_q.append(sync_cu_q[-1] + q_len)
                sync_cu_k.append(sync_cu_k[-1] + end)
                max_sync_q = max(max_sync_q, q_len)
                max_sync_k = max(max_sync_k, end)
            else:
                accepted = row["accepted_tokens"]
                M = row["num_accepted"]
                original_len = row["original_len"]
                final_len = original_len + M
                final_num_blocks = (final_len + self.block_size - 1) // self.block_size
                row["final_num_blocks"] = final_num_blocks
                self.block_manager.rollback_blocks(seq, final_num_blocks)

                will_finish = (
                    (not seq.ignore_eos and accepted[-1] == self.config.eos)
                    or seq.num_completion_tokens + M >= seq.max_tokens
                )
                if will_finish:
                    seq.prev_draft_tokens = []
                    continue

                row_sync_start = len(sync_input_ids)
                sync_input_ids.extend(accepted)
                sync_positions.extend(range(original_len - 1, original_len - 1 + M))
                for pos in range(original_len - 1, original_len - 1 + M):
                    sync_slot_mapping.append(self._slot_for_position(seq, pos))
                sync_fused_indices.extend(range(row["row_start"], row["row_start"] + M))
                row_sync_end = len(sync_input_ids)
                sync_rows.append({
                    "kind": "decode_verify",
                    "seq": seq,
                    "source_row": row,
                    "row_start": row_sync_start,
                    "row_end": row_sync_end,
                })
                sync_seqs.append(seq)
                sync_cu_q.append(sync_cu_q[-1] + M)
                sync_cu_k.append(sync_cu_k[-1] + original_len - 1 + M)
                max_sync_q = max(max_sync_q, M)
                max_sync_k = max(max_sync_k, original_len - 1 + M)

        sync_fused_hidden = None
        if sync_input_ids:
            use_full_fuse = (
                len(sync_fused_indices) == input_ids.size(0)
                and all(i == idx for idx, i in enumerate(sync_fused_indices))
            )
            sync_fused_hidden = self._fuse_captured_hidden(
                captured,
                None if use_full_fuse else sync_fused_indices,
            )

        return (
            sync_input_ids,
            sync_positions,
            sync_slot_mapping,
            sync_fused_hidden,
            sync_cu_q,
            sync_cu_k,
            sync_seqs,
            sync_rows,
            max_sync_q,
            max_sync_k,
        )

    def _run_spec_draft_sync_and_propose(self, sync_input_ids: list[int], sync_positions: list[int],
                                         sync_slot_mapping: list[int], sync_fused_hidden: torch.Tensor | None,
                                         sync_cu_q: list[int], sync_cu_k: list[int], sync_seqs: list[Sequence],
                                         sync_rows: list[dict], max_sync_q: int, max_sync_k: int) -> None:
        if sync_input_ids:
            sync_input_ids_t = torch.tensor(sync_input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
            sync_positions_t = torch.tensor(sync_positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
            sync_slot_mapping_t = torch.tensor(sync_slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
            sync_cu_q_t = torch.tensor(sync_cu_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
            sync_cu_k_t = torch.tensor(sync_cu_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
            sync_block_tables = self.prepare_block_tables(sync_seqs) if sync_cu_k[-1] > sync_cu_q[-1] else None

            set_context(True, sync_cu_q_t, sync_cu_k_t, max_sync_q, max_sync_k,
                        sync_slot_mapping_t, block_tables=sync_block_tables,
                        num_prefill_tokens=len(sync_input_ids), num_decode_seqs=0,
                        finishing_prefill_indices=[])
            sync_logits, sync_hidden_out = self.draft_model(sync_input_ids_t, sync_positions_t, sync_fused_hidden)
            reset_context()

            proposal_infos = []
            for sync_row in sync_rows:
                source = sync_row["source_row"]
                if sync_row["kind"] == "prefill" and source["is_finishing_prefill"]:
                    seq = sync_row["seq"]
                    sampled_token = source["sampled_token"]
                    if (not seq.ignore_eos and sampled_token == self.config.eos) or seq.num_completion_tokens + 1 >= seq.max_tokens:
                        seq.prev_draft_tokens = []
                        continue
                    proposal_infos.append(SpecProposalInfo(
                        seq=seq,
                        last_sync_index=sync_row["row_end"] - 1,
                        start_position=len(seq),
                        rollback_num_blocks=seq.num_blocks,
                    ))
                elif sync_row["kind"] == "decode_verify":
                    seq = sync_row["seq"]
                    start_position = source["original_len"] + source["num_accepted"] - 1
                    proposal_infos.append(SpecProposalInfo(
                        seq=seq,
                        last_sync_index=sync_row["row_end"] - 1,
                        start_position=start_position,
                        rollback_num_blocks=source["final_num_blocks"],
                    ))

            self._run_spec_proposal_batch(proposal_infos, sync_logits, sync_hidden_out)
        else:
            reset_context()

    def _record_spec_profile(self, profile_start_time: float, after_build_batch_time: float,
                             after_target_forward_time: float, after_verify_time: float,
                             after_fuse_build_sync_time: float, profile_end_time: float,
                             use_spec_graph: bool) -> None:
        if not hasattr(self, "_spec_timings"):
            self._spec_timings = {
                "build_batch": 0,
                "target_fwd": 0,
                "verify": 0,
                "fuse_build_sync": 0,
                "draft_all": 0,
                "count": 0,
            }
        t = self._spec_timings
        step_build_batch = after_build_batch_time - profile_start_time
        step_target_fwd = after_target_forward_time - after_build_batch_time
        step_verify = after_verify_time - after_target_forward_time
        step_fuse_build_sync = after_fuse_build_sync_time - after_verify_time
        step_draft_all = profile_end_time - after_fuse_build_sync_time
        t["build_batch"] += step_build_batch
        t["target_fwd"] += step_target_fwd
        t["verify"] += step_verify
        t["fuse_build_sync"] += step_fuse_build_sync
        t["draft_all"] += step_draft_all
        t["count"] += 1
        if t["count"] % 10 == 0:
            n = t["count"]
            print(f"  [SPEC TIMING] steps={n} mode={'graph' if use_spec_graph else 'eager'} | "
                  f"last build={step_build_batch*1000:.1f}ms target={step_target_fwd*1000:.1f}ms "
                  f"verify={step_verify*1000:.1f}ms fuse+sync={step_fuse_build_sync*1000:.1f}ms "
                  f"draft={step_draft_all*1000:.1f}ms total={(profile_end_time-profile_start_time)*1000:.1f}ms | "
                  f"avg build={t['build_batch']/n*1000:.1f}ms target={t['target_fwd']/n*1000:.1f}ms "
                  f"verify={t['verify']/n*1000:.1f}ms fuse+sync={t['fuse_build_sync']/n*1000:.1f}ms "
                  f"draft={t['draft_all']/n*1000:.1f}ms")

    @torch.inference_mode()
    def run_step(self, prefill_seqs: list[Sequence], decode_seqs: list[Sequence]) -> dict | None:
        profile = self.spec_profile

        if profile:
            torch.cuda.synchronize()
            profile_start_time = _time.perf_counter()

        rows, input_ids, positions, cu_q, cu_k, max_q, max_k, slot_mapping, block_tables = \
            self._build_spec_target_batch(prefill_seqs, decode_seqs)
        use_spec_graph = self._can_run_spec_decode_graph(prefill_seqs, decode_seqs, block_tables, max_k)
        if profile:
            torch.cuda.synchronize()
            after_build_batch_time = _time.perf_counter()

        hidden_out, captured = self._run_spec_target_forward(
            use_spec_graph, rows, input_ids, positions,
            cu_q, cu_k, max_q, max_k, slot_mapping, block_tables,
        )

        if profile:
            torch.cuda.synchronize()
            after_target_forward_time = _time.perf_counter()

        finishing_rows, decode_rows, logits_selected = self._select_spec_logits(rows, hidden_out)
        prefill_token_ids = self._sample_spec_prefill_rows(prefill_seqs, finishing_rows, logits_selected)
        decode_accepted_tokens = self._verify_spec_decode_rows(decode_seqs, decode_rows, logits_selected)

        if profile:
            torch.cuda.synchronize()
            after_verify_time = _time.perf_counter()

        reset_context()

        sync_batch = self._build_spec_draft_sync_batch(rows, input_ids, captured)

        if profile:
            torch.cuda.synchronize()
            after_fuse_build_sync_time = _time.perf_counter()

        self._run_spec_draft_sync_and_propose(*sync_batch)

        if profile:
            torch.cuda.synchronize()
            profile_end_time = _time.perf_counter()
            self._record_spec_profile(
                profile_start_time,
                after_build_batch_time,
                after_target_forward_time,
                after_verify_time,
                after_fuse_build_sync_time,
                profile_end_time,
                use_spec_graph,
            )

        if self.rank != 0:
            return None
        return {
            "prefill_seqs": prefill_seqs,
            "prefill_token_ids": prefill_token_ids,
            "decode_seqs": decode_seqs,
            "decode_accepted_tokens": decode_accepted_tokens,
        }

    def reset_profile_metrics(self):
        if hasattr(self, "_spec_timings"):
            del self._spec_timings

    @torch.inference_mode()
    def capture_decode_cudagraph(self):
        config = self.config
        hf_config = config.hf_config
        q_len = self.num_spec_tokens + 1
        max_spec_bs = min(
            config.max_num_seqs,
            config.max_num_batched_tokens // q_len,
            512 // q_len,
        )
        if max_spec_bs < 1:
            self.spec_graphs = {}
            self.spec_graph_bs = []
            return

        graph_bs = [1, 2, 4, 8] + list(range(16, max_spec_bs + 1, 16))
        graph_bs.append(max_spec_bs)
        self.spec_graph_bs = sorted({bs for bs in graph_bs if 1 <= bs <= max_spec_bs})
        self.spec_graph_query_len = q_len
        self.spec_graph_max_seqlen_k = config.max_model_len + self.num_spec_tokens
        max_spec_tokens = self.spec_graph_bs[-1] * q_len
        max_num_blocks = (self.spec_graph_max_seqlen_k + self.block_size - 1) // self.block_size

        input_ids = torch.zeros(max_spec_tokens, dtype=torch.int64, device="cuda")
        positions = torch.arange(max_spec_tokens, dtype=torch.int64, device="cuda")
        slot_mapping = torch.full((max_spec_tokens,), -1, dtype=torch.int32, device="cuda")
        cu_q = torch.arange(self.spec_graph_bs[-1] + 1, dtype=torch.int32, device="cuda") * q_len
        cu_k = torch.arange(self.spec_graph_bs[-1] + 1, dtype=torch.int32, device="cuda") * q_len
        block_tables = torch.zeros(self.spec_graph_bs[-1], max_num_blocks, dtype=torch.int32, device="cuda")
        outputs = torch.zeros(max_spec_tokens, hf_config.hidden_size, dtype=hf_config.torch_dtype, device="cuda")
        captured_outputs = {
            layer: torch.zeros(max_spec_tokens, hf_config.hidden_size, dtype=hf_config.torch_dtype, device="cuda")
            for layer in self.eagle3_fuse_layers
        }

        self.spec_graphs = {}
        self.spec_graph_pool = None
        for bs in reversed(self.spec_graph_bs):
            num_tokens = bs * q_len

            def forward_graph():
                hidden, captured = self.model(
                    input_ids[:num_tokens],
                    positions[:num_tokens],
                    capture_layers=self.eagle3_fuse_layers,
                )
                outputs[:num_tokens] = hidden
                for layer in self.eagle3_fuse_layers:
                    captured_outputs[layer][:num_tokens] = captured[layer]

            set_context(True, cu_q[:bs + 1], cu_k[:bs + 1], q_len, self.spec_graph_max_seqlen_k,
                        slot_mapping[:num_tokens], block_tables=block_tables[:bs],
                        num_prefill_tokens=num_tokens, num_decode_seqs=0,
                        finishing_prefill_indices=[])
            forward_graph()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, self.spec_graph_pool):
                forward_graph()
            if self.spec_graph_pool is None:
                self.spec_graph_pool = graph.pool()
            self.spec_graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.spec_graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            cu_q=cu_q,
            cu_k=cu_k,
            block_tables=block_tables,
            outputs=outputs,
            captured=captured_outputs,
        )
