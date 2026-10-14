"""
MVP Semantic Abstractor: 极简 parser + 可选 GPT-6 Astra API。

用 semantic_schema.parse_instruction() 做本地 fallback，
或者用 GPT-6 Astra 做高精度语义理解。
"""

from __future__ import annotations
import json
import re
import os
import logging
import urllib.request
import urllib.error
from typing import Optional, List, Dict, Any

from .semantic_schema import (
    AbstractTask,
    parse_instruction as _local_parse,
)

logger = logging.getLogger(__name__)


GPT_ASTRA_SYSTEM_PROMPT = """\
Convert robot manipulation instructions into de-instantiated category-level task JSON.

Categories (use EXACTLY these strings):
  operations:       pick_and_place, open
  object_category:  graspable_container, bowl_like, box_container, food_item, articulated
  target_category:  flat_surface, box_container, articulated
  relations:        on, in, from_drawer

Rules:
  - Remove all specific instance info (colors, sizes, brand names).
  - mug/cup/bottle/casserole → graspable_container
  - bowl/ramekin → bowl_like
  - basket/box → box_container
  - alphabet_soup/cream_cheese/butter/ketchup/... → food_item
  - tray/plate/rack/stove/table → flat_surface
  - drawer/cabinet/microwave → articulated (as object or target)
  - Output ONLY valid JSON, no other text.

Output format:
{"operation": "...", "object_category": "...", "target_category": "...", "relations": ["..."]}
"""


def _call_vlm_api(
    instruction: str,
    api_key: str = "",
    model: str = "gpt-6-astra",
    api_base: str = "https://api.openai.com/v1",
    timeout: int = 15,
) -> Optional[Dict[str, Any]]:
    if not api_key:
        return None

    messages = [
        {"role": "system", "content": GPT_ASTRA_SYSTEM_PROMPT},
        {"role": "user", "content": instruction},
    ]
    payload = json.dumps({
        "model": model,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": 128,
    }).encode()

    req = urllib.request.Request(
        f"{api_base}/chat/completions", data=payload,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            text = json.loads(resp.read())["choices"][0]["message"]["content"]
            text = re.sub(r"^```json\n?|```$", "", text.strip(), flags=re.MULTILINE)
            return json.loads(text)
    except Exception as e:
        logger.warning(f"VLM API call failed: {e}")
        return None


class SemanticAbstractor:
    """
    instruction → AbstractTask

    mode:
      "local"   → 极简关键词 parser，无需外部依赖
      "gpt_astra" → GPT-6 Astra API，精度更高（需要 API key）
    """

    def __init__(
        self,
        mode: str = "local",
        api_key: str = "",
        model: str = "gpt-6-astra",
        api_base: str = "https://api.openai.com/v1",
        cache_dir: Optional[str] = None,
    ):
        self.mode = mode.lower()
        self.api_key = api_key or os.environ.get("LLM_API_KEY", "")
        self.model = model
        self.api_base = api_base
        self.cache_dir = cache_dir

        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

        if self.mode == "gpt_astra" and not self.api_key:
            logger.warning("No API key for GPT-6 Astra, falling back to local")
            self.mode = "local"

        logger.info(f"SemanticAbstractor (MVP) initialized, mode={self.mode}")

    def abstract(self, instruction: str) -> AbstractTask:
        if self.mode == "gpt_astra":
            result = _call_vlm_api(
                instruction, api_key=self.api_key,
                model=self.model, api_base=self.api_base,
            )
            if result is not None:
                return AbstractTask.from_dict(result)

        return _local_parse(instruction)

    def abstract_batch(self, instructions: List[str]) -> List[AbstractTask]:
        return [self.abstract(i) for i in instructions]