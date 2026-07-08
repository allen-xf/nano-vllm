import ast
import importlib.util
import sys
from pathlib import Path

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
