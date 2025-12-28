import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from nanovllm.utils.context import get_context


class VocabParallelEmbedding(nn.Module):

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
    ):
        super().__init__()
        self.tp_rank = dist.get_rank() #当前进程在哪个GPU上
        self.tp_size = dist.get_world_size() #总共有多少GPU Tensor Parallelism, TP
        assert num_embeddings % self.tp_size == 0
        self.num_embeddings = num_embeddings
        self.num_embeddings_per_partition = self.num_embeddings // self.tp_size
        self.vocab_start_idx = self.num_embeddings_per_partition * self.tp_rank
        self.vocab_end_idx = self.vocab_start_idx + self.num_embeddings_per_partition
        self.weight = nn.Parameter(torch.empty(self.num_embeddings_per_partition, embedding_dim))
        self.weight.weight_loader = self.weight_loader

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(0)
        start_idx = self.tp_rank * shard_size
        # {tensor.narrow(dimension, start, length)}
        loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor):
        if self.tp_size > 1:
            mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx)
            x = mask * (x - self.vocab_start_idx)
        # 它的工作原理就像查字典：你给它一个单词的“页码”（索引），它还给你那一页上的“具体内容”（嵌入向量）
        y = F.embedding(x, self.weight)
        if self.tp_size > 1:
            y = mask.unsqueeze(1) * y #是不是一个bug unsqueeze是为了维度对齐， 之前mask的维度比y少了一个维度
            # all_reduce 操作会将所有参与并行计算的 GPU 上的 y 张量按对应位置相加（默认是 SUM
            dist.all_reduce(y)
        return y


class ParallelLMHead(VocabParallelEmbedding):

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        bias: bool = False,
    ):
        assert not bias
        super().__init__(num_embeddings, embedding_dim)

    def forward(self, x: torch.Tensor):
        context = get_context()
        if context.is_prefill:
            # 在“预填充（Prefill）”阶段，丢弃没用的中间预测，只保留每个序列最后一个 Token 的特征
            last_indices = context.cu_seqlens_q[1:] - 1
            # 在内存中把这些挑出来的向量重新排整齐
            x = x[last_indices].contiguous()
        logits = F.linear(x, self.weight)
        if self.tp_size > 1:
            all_logits = [torch.empty_like(logits) for _ in range(self.tp_size)] if self.tp_rank == 0 else None
            '''
            为什么这里不用 all_reduce？
            你之前在 Embedding 层看到了 all_reduce，而这里用 gather，这是因为它们的并行数学逻辑不同：

            Embedding (Row Parallel)：每个 GPU 算出的结果是“不完整的特征”，需要把大家的值加起来（Sum）才是正确的。所以用 all_reduce。

            LM Head (Column Parallel)：每个 GPU 算出的结果是“部分词的分数”，它们不需要相加，而是需要拼接（Concatenate）在一起。所以用 gather。
            '''
            dist.gather(logits, all_logits, 0)
            logits = torch.cat(all_logits, -1) if self.tp_rank == 0 else None
        return logits
