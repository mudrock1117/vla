from __future__ import annotations
import logging
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

class SemanticExtractor(nn.Module):
    """从 LLM text_model 某层 hidden states 提取 wording-invariant 语义表示。
    
    Old path: instruction → regex parse → abstract_ids → AbstractTaskEncoder → sem_pooled
    New path: instruction → input_ids → 冻结text_model中间层 → SemanticExtractor → sem_pooled
    
    原理: LLM 在海量指令上预训练, "pick"/"grab"/"lift" 的 hidden states 天然相似。
    冻结 text_model, 只训这个小 MLP → 训练快, 不破坏 LLM 能力。
    """

    def __init__(self, hidden_size: int = 960, bottleneck: int = 256):
        super().__init__()
        self.hidden_size = hidden_size
        self.bottleneck = bottleneck

        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, bottleneck),
            nn.LayerNorm(bottleneck),
            nn.ReLU(),
            nn.Linear(bottleneck, hidden_size),
        )

        total = sum(p.numel() for p in self.parameters())
        logger.info(f"SemanticExtractor created: hidden={hidden_size}, bottleneck={bottleneck}, params={total/1e6:.2f}M")

    def forward(self, text_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (text_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        pooled = self.mlp(pooled)
        pooled = pooled / (pooled.norm(dim=-1, keepdim=True) + 1e-6)
        return pooled.unsqueeze(1)