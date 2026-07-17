import torch
from torch import nn


class Sampler(nn.Module):

    def __init__(self):
        super().__init__()

    # logits:batch_size * 词库  temperatures：batch
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        greedy_mask = temperatures == 0
        if greedy_mask.all():
            return logits.argmax(dim=-1)
        if not greedy_mask.any():
            return self._sample(logits, temperatures)

        sample_tokens = torch.empty(logits.size(0), dtype=torch.int64, device=logits.device)
        sample_tokens[greedy_mask] = logits[greedy_mask].argmax(dim=-1)
        sample_mask = ~greedy_mask
        sample_tokens[sample_mask] = self._sample(logits[sample_mask], temperatures[sample_mask])
        return sample_tokens

    @torch.compile
    def _sample(self, logits: torch.Tensor, temperatures: torch.Tensor):
        '''
        先转成概率（Softmax）。
        用这些概率去加噪（或者像你代码里那样被指数分布除）。
        这时再 argmax，原本概率大的选项胜出的机会大，但概率小的也有机会”逆袭”
        '''
        logits = logits.float().div_(temperatures.unsqueeze(dim=1))
        probs = torch.softmax(logits, dim=-1)
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
        return sample_tokens
