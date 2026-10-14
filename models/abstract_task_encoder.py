"""
MVP Abstract Task Encoder.

Converts de-instantiated AbstractTask → continuous embedding [B, D].
Replaces raw language instruction in the VLM backbone.

Architecture:
  AbstractTask → [CLS, ACT:xx, OBJ:xx, TGT:xx, REL:xx...]
    → Embedding + PosEmbedding → small Transformer → CLS token = task_embedding

参数量 ~0.5–1.0M，冻结 SmolVLM 后可快速训练。
"""

from __future__ import annotations
import logging
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from .semantic_schema import (
    AbstractTask, Action, ObjectCat, TargetCat, Relation,
    ACTION_LIST, OBJECT_CAT_LIST, TARGET_CAT_LIST, RELATION_LIST,
)

logger = logging.getLogger(__name__)


class AbstractTaskEncoder(nn.Module):
    """
    Tiny Transformer encoder for MVP abstract task tokens.

    Input:  LongTensor [B, L]   (from AbstractTaskTokenizer.tokenize_batch)
    Output:
      task_embedding:  [B, hidden_size]   ← CLS token, 全局任务表示
      task_features:   [B, L, hidden_size] ← 每个 token 的上下文化表示
      attention_mask:  [B, L]             ← 0=pad, 1=real
    """

    def __init__(
        self,
        hidden_size: int = 256,
        max_len: int = 16,
        num_heads: int = 4,
        num_layers: int = 2,
        vocab_size: int = 20,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_len = max_len

        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.pos_embedding = nn.Embedding(max_len, hidden_size)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 2,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        logger.info(
            f"AbstractTaskEncoder (MVP) created: "
            f"hidden={hidden_size}, heads={num_heads}, layers={num_layers}, "
            f"vocab={vocab_size}, max_len={max_len}"
        )

    def forward(self, task_ids: torch.LongTensor) -> Dict[str, torch.Tensor]:
        B, L = task_ids.shape
        device = task_ids.device

        pad_id = 0
        attention_mask = (task_ids != pad_id).bool()  # [B, L]

        positions = torch.arange(L, device=device).unsqueeze(0).expand(B, L)
        x = self.token_embedding(task_ids) + self.pos_embedding(positions)

        # PyTorch Transformer: src_key_padding_mask = True means "ignore" (pad)
        key_pad_mask = ~attention_mask  # [B, L], True = pad position
        encoded = self.encoder(x, src_key_padding_mask=key_pad_mask)

        cls_embeds = encoded[:, 0, :]  # [B, hidden_size]

        return {
            "task_embedding": cls_embeds,         # [B, hidden_size]
            "task_features": encoded,              # [B, L, hidden_size]
            "attention_mask": attention_mask,      # [B, L]
        }


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from models.semantic_schema import AbstractTaskTokenizer, parse_instruction

    tokenizer = AbstractTaskTokenizer()
    print(f"Vocab size: {tokenizer.vocab_size}")
    tokenizer.print_vocab()

    encoder = AbstractTaskEncoder(
        hidden_size=64, max_len=16, num_heads=4, num_layers=1,
        vocab_size=tokenizer.vocab_size,
    )
    total = sum(p.numel() for p in encoder.parameters())
    print(f"\nEncoder params: {total/1e6:.2f}M")

    tests = [
        "Pick up the red mug and place it on the tray",
        "Pick up the blue mug and place it on the tray",
        "Pick up the black bowl and place it on the plate",
        "Put the cream cheese in the bowl",
        "Open the middle drawer of the cabinet",
    ]

    tasks = [parse_instruction(t) for t in tests]
    batch = tokenizer.tokenize_batch(tasks, 16)
    out = encoder(batch)

    print(f"\ntask_embedding: {out['task_embedding'].shape}")
    print(f"task_features:  {out['task_features'].shape}")
    print(f"attention_mask: {out['attention_mask'].shape}")

    emb = out["task_embedding"]
    cos = torch.nn.functional.cosine_similarity(emb.unsqueeze(1), emb.unsqueeze(0), dim=-1)
    print(f"\nCosine similarity matrix:")
    for i, t in enumerate(tests):
        print(f"  [{i}] {t[:45]:<45s}: {cos[i].tolist()}")

    print("\n✅ MVP encoder works!")