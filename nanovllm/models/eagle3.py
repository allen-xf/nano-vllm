import torch
from torch import nn
import torch.distributed as dist

from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.attention import Attention
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import QKVParallelLinear, MergedColumnParallelLinear, RowParallelLinear, ReplicatedLinear
from nanovllm.layers.rotary_embedding import get_rope


class Eagle3Attention(nn.Module):
    """EAGLE3 Attention: QKV 输入维度为 2 * hidden_size（embedding + hidden 拼接后）"""

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
        self.head_dim = config.head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5

        # QKV 输入维度为 2 * hidden_size（dual-norm 拼接后的结果）
        self.qkv_proj = QKVParallelLinear(
            config.hidden_size * 2,  # 关键区别：输入维度翻倍
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=getattr(config, 'attention_bias', False),
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            config.hidden_size,
            bias=False,
        )
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=config.max_position_embeddings,
            base=getattr(config, "rope_theta", 1000000),
            rope_scaling=getattr(config, "rope_scaling", None),
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
        )

    def forward(self, positions: torch.Tensor, concat_input: torch.Tensor) -> torch.Tensor:
        """concat_input: [N, 2 * hidden_size] = cat(normed_embed, normed_hidden)"""
        qkv = self.qkv_proj(concat_input)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        q, k = self.rotary_emb(positions, q, k)
        o = self.attn(q, k, v)
        output = self.o_proj(o.flatten(1, -1))
        return output


class Eagle3DecoderLayer(nn.Module):
    """EAGLE3 的修改版 decoder layer：双输入（embedding + hidden），分别 norm 后拼接"""

    def __init__(self, config) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hidden_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = Eagle3Attention(config)
        self.mlp = Eagle3MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.spec_debug = False

    def forward(self, positions: torch.Tensor, token_embeds: torch.Tensor, fused_hidden: torch.Tensor) -> torch.Tensor:
        normed_embeds = self.input_layernorm(token_embeds)
        normed_hidden = self.hidden_norm(fused_hidden)
        concat = torch.cat([normed_embeds, normed_hidden], dim=-1)  # [N, 2 * hidden_size]
        attn_out = self.self_attn(positions, concat)

        # DEBUG: 首次调用时检查 attention 贡献度
        if self.spec_debug:
            if not hasattr(self, '_debug_attn_count'):
                self._debug_attn_count = 0
            if self._debug_attn_count < 3:
                self._debug_attn_count += 1
                ratio = attn_out.norm() / (fused_hidden.norm() + 1e-8)
                print(f"    [DEBUG-ATTN] attn_out_norm={attn_out.norm().item():.2f}, "
                      f"fused_hidden_norm={fused_hidden.norm().item():.2f}, "
                      f"ratio={ratio.item():.4f}, "
                      f"embed_norm={token_embeds.norm().item():.2f}")

        hidden = attn_out + fused_hidden  # residual on hidden
        normed = self.post_attention_layernorm(hidden)
        mlp_out = self.mlp(normed)
        output = mlp_out + hidden  # residual
        return output


class Eagle3MLP(nn.Module):

    def __init__(self, hidden_size: int, intermediate_size: int, hidden_act: str) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
        )
        assert hidden_act == "silu"
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x = self.down_proj(x)
        return x


class Eagle3DraftModel(nn.Module):
    """
    EAGLE3 Draft Model:
    - fc: 融合 target model 多层 hidden states
    - midlayer: 单层修改版 decoder layer（双输入）
    - lm_head: 投影到 draft_vocab_size，通过 d2t 映射回 target vocab
    - embed_tokens: 从 target model 共享（外部赋值）
    """
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(self, config, target_hidden_size: int, num_fuse_layers: int = 3):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.draft_vocab_size = config.draft_vocab_size

        # FC fusion: target_hidden_size * num_fuse_layers -> hidden_size
        self.fc = ReplicatedLinear(target_hidden_size * num_fuse_layers, self.hidden_size, bias=False)

        # 单层 decoder layer
        self.midlayer = Eagle3DecoderLayer(config)

        # 输出
        self.norm = RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = ReplicatedLinear(self.hidden_size, self.draft_vocab_size, bias=False)

        # draft vocab <-> target vocab 映射
        self.register_buffer("d2t", torch.zeros(self.draft_vocab_size, dtype=torch.long))
        self.register_buffer("t2d", torch.zeros(config.vocab_size, dtype=torch.bool))

        # embed_tokens 从 target model 共享，外部赋值
        self.embed_tokens = None

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor,
                fused_hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            input_ids: [N] token ids
            positions: [N] position indices
            fused_hidden: [N, hidden_size] 融合后的 hidden states（来自 fc 或上一步 draft 输出）
        Returns:
            logits: [N, draft_vocab_size]
            hidden_states: [N, hidden_size] 用于下一步 draft
        """
        token_embeds = self.embed_tokens(input_ids)  # [N, target_hidden_size]
        hidden_states = self.midlayer(positions, token_embeds, fused_hidden)
        normed = self.norm(hidden_states)
        logits = self.lm_head(normed)
        return logits, hidden_states
