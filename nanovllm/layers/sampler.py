import torch
from torch import nn


class Sampler(nn.Module):

    def __init__(self):
        super().__init__()

    @torch.compile
    # logits:batch_size * 词库  temperatures：batch
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        '''
        先转成概率（Softmax）。
        用这些概率去加噪（或者像你代码里那样被指数分布除）。
        这时再 argmax，原本概率大的选项胜出的机会大，但概率小的也有机会“逆袭”
        '''
        logits = logits.float().div_(temperatures.unsqueeze(dim=1))
        probs = torch.softmax(logits, dim=-1)
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
        return sample_tokens
