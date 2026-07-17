from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:

    def __init__(self, config: Config, collect_metrics: bool = False):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.enable_chunked_prefill = config.enable_chunked_prefill
        self.eos = config.eos
        self.has_spec = config.draft_model is not None
        self.num_spec_tokens = config.num_spec_tokens
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        # 可选指标收集
        self.collect_metrics = collect_metrics
        self._metrics = None
        if collect_metrics:
            self.reset_metrics()

    def reset_metrics(self):
        self._metrics = {
            "step_count": 0,
            "pure_prefill_steps": 0,
            "pure_decode_steps": 0,
            "mixed_steps": 0,
            "total_utilization": 0.0,
        }

    def record_step(self, prefill_seqs: list[Sequence], decode_seqs: list[Sequence]):
        if not self.collect_metrics:
            return
        prefill_tokens = sum(s.scheduled_chunk_size for s in prefill_seqs) if prefill_seqs else 0
        decode_tokens = sum(self.decode_target_cost(s) for s in decode_seqs) if decode_seqs else 0
        utilization = (prefill_tokens + decode_tokens) / self.max_num_batched_tokens
        m = self._metrics
        m["step_count"] += 1
        m["total_utilization"] += utilization
        if prefill_seqs and decode_seqs:
            m["mixed_steps"] += 1
        elif prefill_seqs:
            m["pure_prefill_steps"] += 1
        else:
            m["pure_decode_steps"] += 1

    def get_metrics(self) -> dict:
        if not self._metrics:
            return {}
        m = self._metrics
        avg_util = m["total_utilization"] / m["step_count"] * 100 if m["step_count"] else 0
        return {
            "step_count": m["step_count"],
            "pure_prefill_steps": m["pure_prefill_steps"],
            "pure_decode_steps": m["pure_decode_steps"],
            "mixed_steps": m["mixed_steps"],
            "avg_utilization": f"{avg_util:.1f}%",
        }

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], list[Sequence]]:
        if self.enable_chunked_prefill:
            return self._schedule_chunked()
        else:
            return self._schedule_non_chunked()

    def decode_target_cost(self, seq: Sequence) -> int:
        if not self.has_spec:
            return 1
        if len(seq.prev_draft_tokens) != self.num_spec_tokens:
            raise RuntimeError(
                f"spec decode seq {seq.seq_id} missing prev_draft_tokens: "
                f"got {len(seq.prev_draft_tokens)}, expected {self.num_spec_tokens}"
            )
        return self.num_spec_tokens + 1

    def _schedule_non_chunked(self) -> tuple[list[Sequence], list[Sequence]]:
        """非 chunked 模式：prefill 优先，prefill/decode 互斥"""
        prefill_seqs = []
        decode_seqs = []
        num_seqs = 0
        num_batched_tokens = 0

        # 1. 先从 waiting 调度 prefill（prefill 优先）
        while self.waiting and num_seqs < self.max_num_seqs:
            seq = self.waiting[0]
            if num_batched_tokens + seq.num_uncomputed_tokens > self.max_num_batched_tokens:
                break  # budget 不够，等下一步
            if not seq.block_table:
                if not self.block_manager.can_allocate(seq):
                    break
                self.block_manager.allocate(seq)
                seq.num_computed_tokens = seq.num_cached_tokens
            seq.scheduled_chunk_size = seq.num_uncomputed_tokens
            num_batched_tokens += seq.num_uncomputed_tokens
            num_seqs += 1
            seq.status = SequenceStatus.RUNNING
            self.waiting.popleft()
            prefill_seqs.append(seq)

        if prefill_seqs:
            self.running.extendleft(reversed(prefill_seqs))
            return prefill_seqs, decode_seqs

        # 2. waiting 为空或 budget 不够，调度 decode
        scheduled_running = []
        while self.running and num_seqs < self.max_num_seqs:
            seq = self.running.popleft()
            decode_cost = self.decode_target_cost(seq)
            if num_batched_tokens + decode_cost > self.max_num_batched_tokens:
                scheduled_running.append(seq)  # budget 不够
                continue
            if self.has_spec:
                while not self.block_manager.can_append_n(seq, decode_cost):
                    if self.running:
                        self.preempt(self.running.pop())
                    else:
                        self.preempt(seq)
                        break
                else:
                    decode_seqs.append(seq)
                    num_batched_tokens += decode_cost
                    num_seqs += 1
            else:
                while not self.block_manager.can_append(seq):
                    if self.running:
                        self.preempt(self.running.pop())
                    else:
                        self.preempt(seq)
                        break
                else:
                    self.block_manager.may_append(seq)
                    decode_seqs.append(seq)
                    num_batched_tokens += decode_cost
                    num_seqs += 1
        for seq in scheduled_running:
            self.running.append(seq)

        self.running.extendleft(reversed(decode_seqs))
        assert prefill_seqs or decode_seqs
        return prefill_seqs, decode_seqs

    def _schedule_chunked(self) -> tuple[list[Sequence], list[Sequence]]:
        """Chunked prefill 模式：decode 优先，prefill/decode 可混合"""
        prefill_seqs = []
        decode_seqs = []
        num_seqs = 0
        num_batched_tokens = 0

        # 1. 调度 running 队列（decode 优先）
        scheduled_running = []
        while self.running:
            if num_seqs >= self.max_num_seqs:
                break
            seq = self.running.popleft()
            if seq.is_prefill:
                # chunked prefill 没做完，继续
                chunk_size = min(seq.num_uncomputed_tokens, self.max_num_batched_tokens - num_batched_tokens)
                if chunk_size <= 0:
                    scheduled_running.append(seq)
                    continue
                seq.scheduled_chunk_size = chunk_size
                num_batched_tokens += chunk_size
                num_seqs += 1
                prefill_seqs.append(seq)
            else:
                # decode
                decode_cost = self.decode_target_cost(seq)
                if num_batched_tokens + decode_cost > self.max_num_batched_tokens:
                    scheduled_running.append(seq)
                    continue
                if self.has_spec:
                    while not self.block_manager.can_append_n(seq, decode_cost):
                        if self.running:
                            self.preempt(self.running.pop())
                        else:
                            self.preempt(seq)
                            break
                    else:
                        decode_seqs.append(seq)
                        num_batched_tokens += decode_cost
                        num_seqs += 1
                else:
                    while not self.block_manager.can_append(seq):
                        if self.running:
                            self.preempt(self.running.pop())
                        else:
                            self.preempt(seq)
                            break
                    else:
                        self.block_manager.may_append(seq)
                        decode_seqs.append(seq)
                        num_batched_tokens += decode_cost
                        num_seqs += 1
        # 因为 budget / block 不够 / max_num_seqs 限制，回 self.running
        for seq in scheduled_running:
            self.running.append(seq)

        # 2. 从 waiting 调度新 prefill（用剩余 budget）
        while self.waiting and num_seqs < self.max_num_seqs:
            seq = self.waiting[0]
            remaining_budget = self.max_num_batched_tokens - num_batched_tokens
            if remaining_budget <= 0:
                break
            if not seq.block_table:
                if not self.block_manager.can_allocate(seq):
                    break
                self.block_manager.allocate(seq)
                seq.num_computed_tokens = seq.num_cached_tokens
            chunk_size = min(seq.num_uncomputed_tokens, remaining_budget)
            if chunk_size <= 0:
                break
            seq.scheduled_chunk_size = chunk_size
            num_batched_tokens += chunk_size
            num_seqs += 1
            seq.status = SequenceStatus.RUNNING
            self.waiting.popleft()
            prefill_seqs.append(seq)

        all_scheduled = prefill_seqs + decode_seqs
        self.running.extendleft(reversed(all_scheduled))
        assert prefill_seqs or decode_seqs
        return prefill_seqs, decode_seqs

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        seq.num_computed_tokens = 0
        seq.prev_draft_tokens = []
        seq.prev_draft_logits = None
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess_speculative_step(self, result: dict):
        """统一 spec step 后处理：prefill/decode 都由 model runner 维护 draft 状态。"""
        prefill_seqs = result["prefill_seqs"]
        prefill_token_ids = result["prefill_token_ids"]
        decode_seqs = result["decode_seqs"]
        accepted_tokens_per_seq = result["decode_accepted_tokens"]

        for seq, token_id in zip(prefill_seqs, prefill_token_ids):
            seq.num_computed_tokens += seq.scheduled_chunk_size
            if not seq.is_prefill:
                assert token_id is not None
                seq.append_token(token_id)
                if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens >= seq.max_tokens:
                    seq.status = SequenceStatus.FINISHED
                    self.block_manager.deallocate(seq)
                    self.running.remove(seq)

        for seq, accepted_tokens in zip(decode_seqs, accepted_tokens_per_seq):
            seq.append_tokens(accepted_tokens)
            last_token = accepted_tokens[-1]
            if (not seq.ignore_eos and last_token == self.eos) or seq.num_completion_tokens >= seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)

    def postprocess(self, prefill_seqs: list[Sequence], decode_seqs: list[Sequence], token_ids: list[int]):
        token_idx = 0
        for seq in prefill_seqs:
            seq.num_computed_tokens += seq.scheduled_chunk_size
            if not seq.is_prefill:
                # prefill 完成，追加生成的 token
                seq.append_token(token_ids[token_idx])
                token_idx += 1
                if (not seq.ignore_eos and token_ids[token_idx - 1] == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                    seq.status = SequenceStatus.FINISHED
                    self.block_manager.deallocate(seq)
                    self.running.remove(seq)
        for seq in decode_seqs:
            seq.prev_draft_tokens = []  # 走了 run() 路径，draft tokens 过时，清空
            seq.prev_draft_logits = None
            seq.append_token(token_ids[token_idx])
            token_idx += 1
            if (not seq.ignore_eos and token_ids[token_idx - 1] == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
