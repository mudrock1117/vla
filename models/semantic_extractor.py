from __future__ import annotations
import logging
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

class SemanticExtractor(nn.Module):
    def __init__(self, hidden_size: int = 960, bottleneck: int = 1024):
        super().__init__()
        self.hidden_size = hidden_size
        self.bottleneck = bottleneck
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, bottleneck),
            nn.ReLU(),
            nn.Linear(bottleneck, hidden_size),
        )

        total = sum(p.numel() for p in self.parameters())
        logger.info(f"SemanticExtractor created: hidden={hidden_size}, bottleneck={bottleneck}, params={total/1e6:.2f}M")

    def forward(self, text_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        last_idx = attention_mask.sum(dim=1) - 1
        B = text_hidden.size(0)
        pooled = text_hidden[torch.arange(B, device=text_hidden.device), last_idx]
        pooled = pooled / (pooled.norm(dim=-1, keepdim=True) + 1e-6)
        return pooled.unsqueeze(1)