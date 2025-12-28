import torch
from torch import nn
import torch.nn.functional as F


class SiluAndMul(nn.Module):

    def __init__(self):
        super().__init__()

    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 2: 表示将张量平均分成 2 份。
        # -1: 表示在 最后一个维度（即特征维度）进行切分 (batch, output)
        x, y = x.chunk(2, -1)
        return F.silu(x) * y
