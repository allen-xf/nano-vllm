import ast
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]


def load_module_from_path(relative_path: str, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_row_parallel_weight_loader_accepts_unsharded_bias():
    linear = load_module_from_path("nanovllm/layers/linear.py", "linear_for_dflash_test")
    param = nn.Parameter(torch.zeros(4))
    loaded_weight = torch.arange(4, dtype=param.dtype)

    linear.RowParallelLinear.weight_loader(object(), param, loaded_weight)

    assert torch.equal(param.data, loaded_weight)


def test_context_defaults_to_causal_and_allows_override():
    context = load_module_from_path("nanovllm/utils/context.py", "context_for_dflash_test")

    assert context.get_context().causal is True

    context.set_context(False, causal=False)
    assert context.get_context().causal is False

    context.reset_context()
    assert context.get_context().causal is True


def test_dflash_context_kv_projection_does_not_compute_query_projection():
    source = (ROOT / "nanovllm/models/qwen3_dflash.py").read_text()
    tree = ast.parse(source)
    project_context_kv = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "project_context_kv"
    )

    calls_full_qkv_projection = any(
        isinstance(node, ast.Attribute) and node.attr == "project_qkv"
        for node in ast.walk(project_context_kv)
    )
    uses_direct_linear = any(
        isinstance(node, ast.Attribute) and node.attr == "linear"
        for node in ast.walk(project_context_kv)
    )

    assert not calls_full_qkv_projection
    assert uses_direct_linear


def test_dspark_config_normalizer_promotes_top_level_dflash_fields():
    dspark = load_module_from_path("nanovllm/models/qwen3_dspark.py", "qwen3_dspark_for_test")

    class DummyConfig:
        pass

    config = DummyConfig()
    config.dflash_config = {"mask_token_id": 7}
    config.target_layer_ids = [2, 8]
    config.layer_ids = None
    config.use_aux_hidden_state = False

    merged = dspark.ensure_dspark_dflash_config(config)

    assert merged["mask_token_id"] == 7
    assert merged["target_layer_ids"] == [2, 8]
    assert merged["use_aux_hidden_state"] is False


def test_dspark_weight_normalizer_skips_confidence_head():
    dspark = load_module_from_path("nanovllm/models/qwen3_dspark.py", "qwen3_dspark_weights_for_test")
    info = dspark.DFlashLoadInfo()

    assert dspark._normalize_dspark_weight_name("confidence_head.weight", info) is None
    assert dspark._normalize_dspark_weight_name("draft_id_to_target_id", info) == "d2t"
    assert info.includes_d2t is True


def test_dspark_backend_uses_non_causal_markov_sampling():
    source = (ROOT / "nanovllm/engine/spec_decode/dspark.py").read_text()
    tree = ast.parse(source)
    propose = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_run_spec_draft_sync_and_propose"
    )

    called_attrs = {
        node.attr
        for node in ast.walk(propose)
        if isinstance(node, ast.Attribute)
    }
    sets_non_causal_context = any(
        isinstance(node, ast.keyword)
        and node.arg == "causal"
        and isinstance(node.value, ast.Constant)
        and node.value.value is False
        for node in ast.walk(propose)
    )

    assert "markov_embed" in called_attrs
    assert "markov_bias" in called_attrs
    assert sets_non_causal_context


def test_dspark_backend_runtime_writes_prev_draft_tokens(monkeypatch):
    dspark = load_module_from_path("nanovllm/engine/spec_decode/dspark.py", "dspark_backend_runtime_for_test")
    real_tensor = torch.tensor

    def cpu_tensor(data, *args, **kwargs):
        kwargs.pop("pin_memory", None)
        return real_tensor(data, *args, **kwargs)

    monkeypatch.setattr(dspark.torch, "tensor", cpu_tensor)
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, non_blocking=True: self, raising=False)
    monkeypatch.setattr(dspark, "set_context", lambda *args, **kwargs: None)
    monkeypatch.setattr(dspark, "reset_context", lambda: None)

    class DummyBlockManager:

        def __init__(self):
            self.append_calls = []
            self.rollback_calls = []

        def append_n_slots(self, seq, n, start_pos):
            self.append_calls.append((seq, n, start_pos))
            return list(range(start_pos, start_pos + n))

        def rollback_blocks(self, seq, num_blocks):
            self.rollback_calls.append((seq, num_blocks))

    class DummyDraftModel:

        def __call__(self, input_ids, positions):
            return torch.zeros(input_ids.numel(), 1)

        def compute_draft_logits(self, hidden_states):
            return torch.zeros(hidden_states.size(0), 5)

        def markov_embed(self, prev):
            return prev

        def markov_bias(self, markov_embed):
            vocab_size = 5
            bias = torch.zeros(markov_embed.size(0), vocab_size)
            next_ids = ((markov_embed + 1) % vocab_size).unsqueeze(1)
            bias.scatter_(1, next_ids, 1.0)
            return bias

        def map_draft_to_target(self, draft_ids):
            return draft_ids

    backend = dspark.DSparkSpecBackend.__new__(dspark.DSparkSpecBackend)
    backend.num_spec_tokens = 3
    backend.mask_token_id = 99
    backend.runner = SimpleNamespace(block_manager=DummyBlockManager())
    backend.draft_model = DummyDraftModel()
    backend.prepare_block_tables = lambda seqs: torch.zeros(len(seqs), 1, dtype=torch.int32)

    seq1 = SimpleNamespace(prev_draft_tokens=None)
    seq2 = SimpleNamespace(prev_draft_tokens=None)
    proposal_infos = [
        dspark.DFlashProposalInfo(seq=seq1, bonus_token=1, query_start_position=10, rollback_num_blocks=3),
        dspark.DFlashProposalInfo(seq=seq2, bonus_token=3, query_start_position=20, rollback_num_blocks=5),
    ]

    backend._run_spec_draft_sync_and_propose(proposal_infos)

    assert seq1.prev_draft_tokens == [2, 3, 4]
    assert seq2.prev_draft_tokens == [4, 0, 1]
    assert backend.runner.block_manager.append_calls == [(seq1, 3, 10), (seq2, 3, 20)]
    assert backend.runner.block_manager.rollback_calls == [(seq1, 3), (seq2, 5)]
