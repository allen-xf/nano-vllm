"""
compute_rewards 逻辑解析

核心公式：
    reward_t = -β * (log_π(a_t|s_t) - log_π_ref(a_t|s_t))
             = β * (log_π_ref(a_t|s_t) - log_π(a_t|s_t))

含义：ref_model 对 actor 生成的 token 认可度越高（ref 给的概率 >= actor 给的概率），
      奖励越大。这就是 KL 惩罚的实现方式。

最终 reward：
    - 中间 token: β * (ref_log_prob - actor_log_prob)
    - 最后 token: β * (ref_log_prob - actor_log_prob) + RM_score

来源：https://zhuanlan.zhihu.com/p/677607581 (作者：猛猿)
"""

import torch


def compute_rewards(prompts, log_probs, ref_log_probs, reward_score, action_mask,
                    kl_ctl=0.1, clip_reward_value=5.0):
    """
    Args:
        prompts:        (batch_size, prompt_len) - prompt token ids，用于确定 response 起始位置
        log_probs:      (batch_size, seq_len) - actor 模型对每个 token 的 log prob（只有 response 部分有意义，prompt 部分为 0 或被 mask 掉）
        ref_log_probs:  (batch_size, seq_len) - ref 模型对每个 token 的 log prob（同上）
        reward_score:   (batch_size,) - RM 给整条 response 的打分（标量）
        action_mask:    (batch_size, seq_len) - 哪些位置是 response token（1=response, 0=padding/prompt）
        kl_ctl:         KL 惩罚系数 β
        clip_reward_value: reward score 的截断范围

    Returns:
        rewards: (batch_size, seq_len) - 每个 token 位置的最终 reward
    """

    # Step 1: 计算每个 token 的 KL 惩罚/奖励
    # = β * (ref_log_prob - actor_log_prob)
    # ref 认可度高 → 正奖励；actor 偏离 ref 太多 → 负奖励（惩罚）
    #  - 正：ref_log_prob > actor_log_prob → ref 觉得这个 token 概率更高，说明 actor 没跑偏，给正奖励
    #  - 负：ref_log_prob < actor_log_prob → actor 给这个 token 的概率比 ref 高，说明 actor 开始偏离 ref 了，给负惩罚
    kl_divergence_estimate = -kl_ctl * (log_probs - ref_log_probs)
    rewards = kl_divergence_estimate  # (batch_size, seq_len)

    # Step 2: 确定 response 的起始和结束位置
    # prompt 已经 padding 到同一长度，所以 response 起始位置一致
    start = prompts.shape[1] - 1  # int，所有样本共享

    # 每条数据的 response 长度不同，结束位置不同
    ends = start + action_mask[:, start:].sum(1) + 1  # (batch_size,)

    # Step 3: 截断 RM reward score
    reward_clip = torch.clamp(reward_score, -clip_reward_value, clip_reward_value)

    # Step 4: 把 RM score 加到每条数据 response 的最后一个有效 token 上
    # 注意：不能用 rewards[j][-1]，因为 seq_len 包含 padding，[-1] 可能指向 padding 位置
    # 必须用 rewards[j, start:ends[j]][-1] 定位到 response 的最后一个有效 token
    # RM 对整条 response 打一个分，放最后一个 token 是约定，GAE 会把奖励向前传播
    batch_size = log_probs.shape[0]
    for j in range(batch_size):
        rewards[j, start:ends[j]][-1] += reward_clip[j]

    return rewards


# ============================================================
# Step 2: GAE (Generalized Advantage Estimation)
# ============================================================
# 拿到 per-token rewards 后，下一步是算每个 token 的 Advantage
#
# 不引入 GAE 时的优势值：
#   delta_t = r_t + γ * V(s_{t+1}) - V(s_t)
#   其中 V(s_t) 是 Critic 网络对"从 t 开始能拿多少总收益"的预估
#   delta_t 就是"实际比预估好多少"
#
# 引入 GAE 后：
#   A_t = delta_t + γλ * A_{t+1}
#   不仅看当前时刻的优势，还考虑未来的优势（用 λ 衰减）
#   从后往前倒推：A_T = delta_T（最后一个时刻没有未来）
#
# returns（实际收益）= A_t + V(s_t)，用于更新 Critic
#
# 来源：https://zhuanlan.zhihu.com/p/677607581 (作者：猛猿)


def get_advantages_and_returns(values, rewards, start, gamma=1.0, lam=0.95):
    """
    Args:
        values:   (batch_size, seq_len) - Critic 网络对每个位置的价值估计 V(s_t)
        rewards:  (batch_size, seq_len) - compute_rewards 算出的 per-token reward
        start:    int - response 开始的位置（prompt 的最后一个 token 位置）
        gamma:    折扣因子（通常为 1.0，因为 response 不长，不需要折扣）
        lam:      GAE 的 λ 参数，控制 bias-variance trade-off（λ=1 纯 MC，λ=0 纯 TD）

    Returns:
        advantages: (batch_size, response_len) - 每个 response token 的优势值
        returns:    (batch_size, response_len) - 每个 response token 的实际收益（用于训练 Critic）
    """
    lastgaelam = 0
    advantages_reversed = []
    length = rewards.size()[-1]

    # 从后往前倒推（动态规划）
    for t in reversed(range(start, length)):
        # 最后一个时刻没有 V(s_{t+1})，设为 0
        nextvalues = values[:, t + 1] if t < length - 1 else 0.0

        # delta_t = r_t + γ * V(s_{t+1}) - V(s_t)
        # "实际拿到的 + 未来预估" vs "当前预估"
        delta = rewards[:, t] + gamma * nextvalues - values[:, t]

        # A_t = delta_t + γλ * A_{t+1}
        lastgaelam = delta + gamma * lam * lastgaelam
        advantages_reversed.append(lastgaelam)

    # 翻转回正序
    advantages = torch.stack(advantages_reversed[::-1], dim=1)

    # returns = A_t + V(s_t)，即实际收益，用于 Critic 的 loss: (returns - V(s_t))^2
    returns = advantages + values[:, start:]

    return advantages.detach(), returns


# ============================================================
# 示例：直观理解
# ============================================================
if __name__ == "__main__":
    batch_size = 2
    prompt_len = 4
    response_len = 5
    seq_len = prompt_len + response_len

    # 模拟数据
    prompts = torch.zeros(batch_size, prompt_len, dtype=torch.long)

    # actor 和 ref 的 log probs（只有 response 部分有意义）
    log_probs = torch.randn(batch_size, seq_len) * 0.1
    ref_log_probs = torch.randn(batch_size, seq_len) * 0.1

    # RM 给的分数
    reward_score = torch.tensor([0.8, -0.3])

    # action mask: prompt 部分为 0，response 部分为 1
    # 第一条 response 长度 5，第二条长度 3（后面 padding）
    action_mask = torch.zeros(batch_size, seq_len)
    action_mask[0, prompt_len:prompt_len + 5] = 1  # 第一条：5 个 response token
    action_mask[1, prompt_len:prompt_len + 3] = 1  # 第二条：3 个 response token

    rewards = compute_rewards(prompts, log_probs, ref_log_probs, reward_score, action_mask)

    print("=== Rewards ===")
    print(f"Shape: {rewards.shape}")
    print()
    for j in range(batch_size):
        print(f"样本 {j}:")
        print(f"  KL 部分 (所有位置): β * (ref - actor) = {(-0.1 * (log_probs[j] - ref_log_probs[j])).tolist()[:seq_len]}")
        print(f"  最终 rewards:        {rewards[j].tolist()}")
        print(f"  RM score (clipped):  {torch.clamp(reward_score[j], -5, 5).item():.2f} (加在 response 最后一个 token)")
        print()
