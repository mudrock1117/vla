"""
SmolVLM-VLA Model

HuggingFace-compatible Vision-Language-Action policy using SmolVLM-500M-Instruct
as the visual-language backbone.

Key differences from FlorenceVLA:
  - Uses SmolVLM-500M-Instruct (efficient 500M parameter model)
  - 512x512 image input (SmolVLM-500M uses 512x512 patches)
  - All views processed together by SmolVLM, no aux_visual_inputs
  - Unified VLM output for multi-view inputs
"""

from __future__ import annotations

import logging
import traceback
from typing import Any, Dict

import numpy as np
import torch
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from PIL import Image
import uvicorn
import json_numpy
import cv2

from transformers import PreTrainedModel, AutoProcessor, AutoModelForImageTextToText
from .transformer_smolvlm import SmolVLMActionTransformer
from .action_hub import build_action_space
from .configuration_smolvlm_vla import SmolVLMVLAConfig


class SmolVLMVLA(PreTrainedModel):
    """
    SmolVLM-VLA: HuggingFace-compatible Vision-Language-Action policy.

    Components:
      • SmolVLM-500M-Instruct backbone (vision-language)
      • SmolVLMActionTransformer (flow matching action head)
      • Action space (pre/post-processing + loss)
      
    Key differences from FlorenceVLA:
      • All camera views are input to VLM together (no aux_visual_inputs)
      • 512x512 image resolution (SmolVLM-500M uses 512x512 patches)
      • Efficient 500M parameter model
    """
    config_class = SmolVLMVLAConfig
    base_model_prefix = "smolvlm_vla"
    supports_gradient_checkpointing = True

    def __init__(self, config: SmolVLMVLAConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)

        # Core settings
        self.num_actions: int = config.num_actions
        self.use_proprio: bool = config.use_proprio
        self.action_mode: str = config.action_mode.lower()
        self.image_size: int = config.image_size
        self.num_views: int = config.num_views
        
        # Action space
        self.action_space = build_action_space(config.action_mode.lower())
        dim_action = self.action_space.dim_action
        dim_proprio = getattr(self.action_space, "dim_proprio", dim_action)

        # SmolVLM backbone (force out of meta device context for nested from_pretrained)
        logging.info(f"Loading SmolVLM from: {config.smolvlm_model_path}")
        _cpu_ctx = torch.device("cpu")
        with torch.device(_cpu_ctx):
            self.vlm = AutoModelForImageTextToText.from_pretrained(
                config.smolvlm_model_path,
                torch_dtype=torch.float32,
                trust_remote_code=True,
            )
            self.vlm_processor = AutoProcessor.from_pretrained(
                config.smolvlm_model_path,
                trust_remote_code=True,
            )
        
        # Get SmolVLM hidden size from model config
        # SmolVLM-500M has hidden_size from text_config
        vlm_hidden_size = self.vlm.config.text_config.hidden_size
        logging.info(f"SmolVLM hidden size: {vlm_hidden_size}")

        # DiT/AdaLN mode setting
        self.use_adaln = getattr(config, 'use_adaln', False)
        
        # Flow matching action head (SmolVLM version - no aux_visual)
        self.transformer = SmolVLMActionTransformer(
            hidden_size=config.hidden_size,
            vlm_hidden_size=vlm_hidden_size,
            depth=config.depth,
            num_heads=config.num_heads,
            mlp_ratio=config.mlp_ratio,
            dim_action=dim_action,
            dim_propio=dim_proprio,
            dim_time=config.dim_time,
            max_len_seq=config.max_len_seq,
            use_adaln=self.use_adaln,
        )
        
        if self.use_adaln:
            logging.info("✓ DiT/AdaLN mode enabled: conditions injected via Adaptive Layer Norm")
        else:
            logging.info("✓ Concat mode enabled: conditions concatenated to sequence")

        # Semantic abstraction encoder (de-instantiated task representation)
        self.use_semantic: bool = getattr(config, 'use_semantic', False)
        self.semantic_task_encoder = None
        self.semantic_proj = None
        if self.use_semantic:
            from .abstract_task_encoder import AbstractTaskEncoder
            self.semantic_task_encoder = AbstractTaskEncoder(
                hidden_size=config.semantic_hidden_size,
                max_len=config.semantic_max_len,
                num_layers=config.semantic_num_layers,
                num_heads=config.semantic_num_heads,
                vocab_size=config.semantic_vocab_size,
            )
            if config.semantic_hidden_size != vlm_hidden_size:
                import torch.nn as nn
                self.semantic_proj = nn.Linear(config.semantic_hidden_size, vlm_hidden_size)
            import torch.nn as nn
            self.semantic_alpha = nn.Parameter(torch.tensor(0.0))
            logging.info(f"✓ Semantic abstraction enabled: "
                         f"hidden={config.semantic_hidden_size}, "
                         f"proj={'needed' if self.semantic_proj is not None else 'direct'}, "
                         f"alpha=0 (sigmoid=0.5)")

        # Visual Grounding Semantic Encoder (detected objects + counts + positions)
        self.use_visual_semantic: bool = getattr(config, 'use_visual_semantic', False)
        self.visual_semantic_encoder = None
        self.visual_proj = None
        if self.use_visual_semantic:
            from .visual_grounding import VisualSemanticEncoder
            self.visual_semantic_encoder = VisualSemanticEncoder(
                hidden_size=getattr(config, 'visual_semantic_hidden_size', 256),
                max_objects=getattr(config, 'visual_max_objects', 8),
            )
            vs_hidden = getattr(config, 'visual_semantic_hidden_size', 256)
            if vs_hidden != vlm_hidden_size:
                import torch.nn as nn
                self.visual_proj = nn.Linear(vs_hidden, vlm_hidden_size)
            import torch.nn as nn
            self.visual_alpha = nn.Parameter(torch.tensor(0.0))
            logging.info(f"✓ Visual Semantic enabled: "
                         f"hidden={vs_hidden}, "
                         f"proj={'needed' if self.visual_proj is not None else 'direct'}, "
                         f"alpha=0 (sigmoid=0.5)")

        # Deferred FastAPI app
        self.app: FastAPI | None = None

    # ============================= SmolVLM encoder =============================
    def forward_vlm(
        self,
        pixel_values: torch.FloatTensor,    # [B, V, C, H, W] - multi-view images
        image_mask: torch.Tensor,           # [B, V] (bool or 0/1)
        language_instruction: list[str] | None = None,  # Optional text prompts
    ) -> Dict[str, torch.Tensor]:
        """
        Encode multi-view images via SmolVLM2.
        
        All views are processed together by SmolVLM, producing unified features.
        No aux_visual_inputs needed - everything goes through VLM.

        Returns:
          { "vlm_features": [B, T_enc, D] }
        """
        if pixel_values.dim() == 6:
            if pixel_values.size(2) == 1:
                pixel_values = pixel_values.squeeze(2)
            else:
                pixel_values = pixel_values[:, :, 0]
            
        B, V, C, H, W = pixel_values.shape
        device = pixel_values.device
        
        # Prepare images for SmolVLM - flatten views and filter by mask
        # SmolVLM can handle multiple images as part of multi-image inference
        batch_features = []
        
        for b in range(B):
            # Get valid images for this sample
            valid_mask = image_mask[b].bool()
            valid_images = pixel_values[b][valid_mask]  # [num_valid, C, H, W]
            
            if valid_images.shape[0] == 0:
                raise ValueError("At least one image view must be valid per batch.")
            
            # Convert to PIL images for SmolVLM processor
            pil_images = []
            for img_tensor in valid_images:
                # Denormalize and convert to PIL
                img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
                # Assuming normalized with ImageNet stats, denormalize
                img_np = img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
                pil_images.append(Image.fromarray(img_np))
            
            # Build message for SmolVLM with multiple images
            content = []
            for i, img in enumerate(pil_images):
                content.append({"type": "image", "image": img})
            
            # Add text prompt if provided
            if language_instruction is not None and b < len(language_instruction):
                content.append({"type": "text", "text": language_instruction[b]})
            else:
                content.append({"type": "text", "text": "Describe the robot's observation."})
            
            messages = [{"role": "user", "content": content}]
            
            # Process with SmolVLM
            inputs = self.vlm_processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(device)
            
            # Get encoder outputs (hidden states) instead of generating text
            with torch.no_grad():
                outputs = self.vlm(
                    **inputs,
                    output_hidden_states=True,
                    return_dict=True,
                )
            
            # Use the last hidden state as features
            # Shape: [1, seq_len, hidden_size]
            hidden_states = outputs.hidden_states[-1]
            batch_features.append(hidden_states.squeeze(0))  # [seq_len, hidden_size]
        
        # Pad to same length and stack
        max_len = max(f.shape[0] for f in batch_features)
        hidden_size = batch_features[0].shape[-1]
        
        padded_features = torch.zeros(B, max_len, hidden_size, device=device, dtype=batch_features[0].dtype)
        for b, feat in enumerate(batch_features):
            padded_features[b, :feat.shape[0]] = feat
        
        return {"vlm_features": padded_features}

    def forward_vlm_efficient(
        self,
        pixel_values: torch.FloatTensor,    # [B, V, C, H, W] - Already preprocessed
        image_mask: torch.Tensor,           # [B, V]
        input_ids: torch.LongTensor | None = None,  # [B, L] - Pre-tokenized text
    ) -> Dict[str, torch.Tensor]:
        """
        Efficient VLM forward for training - uses FULL VLM to fuse vision and language.
        
        Key improvement: Uses complete VLM forward (vision encoder + language model)
        to get features that fuse visual and linguistic information, rather than
        just using the vision encoder alone.
        
        Pipeline:
          pixel_values → vision_encoder → image_features
                                               ↓
          input_ids → text_embeddings ─────────┤
                                               ↓
                                 [image_feats, text_embeds] (concat)
                                               ↓
                                 language_model forward
                                               ↓
                                 fused VLM features → return
        
        Returns:
          { "vlm_features": [B, T_enc, D] }
        """
        if pixel_values.dim() == 6:
            if pixel_values.size(2) == 1:
                pixel_values = pixel_values.squeeze(2)
            else:
                pixel_values = pixel_values[:, :, 0]
        B, V, C, H, W = pixel_values.shape
        device = pixel_values.device
        dtype = pixel_values.dtype
        
        # ========== Step 1: Get vision features ==========
        # Flatten images: [B, V, C, H, W] -> [B*V, C, H, W]
        flat_images = pixel_values.flatten(0, 1)
        flat_mask = image_mask.view(-1).bool()
        
        # Get valid images
        valid_images = flat_images[flat_mask]  # [num_valid, C, H, W]
        
        if valid_images.shape[0] == 0:
            raise ValueError("At least one image view must be valid.")
        
        # Encode images through SmolVLM's vision encoder (SigLIP)
        vision_outputs = self.vlm.model.vision_model(
            pixel_values=valid_images,
            output_hidden_states=True,
            return_dict=True,
        )
        
        # Get image features and project to LM space
        image_features = vision_outputs.last_hidden_state  # [num_valid, num_patches, vision_hidden]
        
        # Project to language model space using the connector/projector
        if hasattr(self.vlm.model, 'connector'):
            image_features = self.vlm.model.connector(image_features)
        elif hasattr(self.vlm.model, 'multi_modal_projector'):
            image_features = self.vlm.model.multi_modal_projector(image_features)
        
        # ========== Step 2: Get text embeddings ==========
        # Idefics3 (SmolVLM) uses 'text_model' instead of 'language_model'
        text_embeds = self.vlm.model.text_model.get_input_embeddings()(input_ids)  # [B, L, D]
        
        # ========== Step 3: Build combined sequence per sample ==========
        # For each sample, concatenate: [image_features_view1, ..., image_features_viewN, text_embeds]
        hidden_size = image_features.shape[-1]
        num_patches = image_features.shape[1]
        
        # Reconstruct image features with batch structure
        full_image_features = image_features.new_zeros(B * V, num_patches, hidden_size)
        full_image_features[flat_mask] = image_features
        full_image_features = full_image_features.view(B, V, num_patches, hidden_size)
        
        # Count valid views per sample for proper concatenation
        valid_per_sample = image_mask.sum(dim=1).int()  # [B]
        
        batch_inputs_embeds = []
        max_seq_len = 0
        
        for b in range(B):
            # Get valid image features for this sample
            num_valid = valid_per_sample[b].item()
            sample_image_feats = full_image_features[b, :num_valid]  # [num_valid, num_patches, D]
            sample_image_feats = sample_image_feats.reshape(-1, hidden_size)  # [num_valid*num_patches, D]
            
            # Get text embeddings for this sample
            sample_text_embeds = text_embeds[b]  # [L, D]
            
            # Concatenate: [image_features, text_embeds]
            combined = torch.cat([sample_image_feats, sample_text_embeds], dim=0)  # [T, D]
            batch_inputs_embeds.append(combined)
            max_seq_len = max(max_seq_len, combined.shape[0])
        
        # ========== Step 4: Pad and stack ==========
        padded_inputs_embeds = torch.zeros(B, max_seq_len, hidden_size, device=device, dtype=dtype)
        attention_mask = torch.zeros(B, max_seq_len, device=device, dtype=torch.long)
        
        for b, embeds in enumerate(batch_inputs_embeds):
            seq_len = embeds.shape[0]
            padded_inputs_embeds[b, :seq_len] = embeds
            attention_mask[b, :seq_len] = 1
        
        # ========== Step 5: Forward through text model (Idefics3/SmolVLM) ==========
        # This fuses visual and linguistic information through the full transformer
        lm_outputs = self.vlm.model.text_model(
            inputs_embeds=padded_inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        
        # Use the last hidden state as VLM features
        # This now contains fused vision-language representations
        vlm_features = lm_outputs.last_hidden_state  # [B, max_seq_len, D]
        
        return {"vlm_features": vlm_features}

    # ========================== Semantic abstraction VLM encoder ==========================
    def forward_vlm_semantic(
        self,
        pixel_values: torch.FloatTensor,
        image_mask: torch.Tensor,
        input_ids: torch.LongTensor,
        abstract_task_ids: torch.LongTensor,
        visual_semantics: list | None = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Residual Semantic Adapter (Extended):
          text_embeds + α_sem * sem_pooled + α_vis * vis_pooled

        If use_visual_semantic=True and visual_semantics provided → triple residual.
        Otherwise → original dual residual (text + semantic only).
        """
        if not self.use_semantic or self.semantic_task_encoder is None:
            raise RuntimeError("Semantic mode not enabled.")

        if pixel_values.dim() == 6:
            if pixel_values.size(2) == 1:
                pixel_values = pixel_values.squeeze(2)
            else:
                pixel_values = pixel_values[:, :, 0]
        B, V, C, H, W = pixel_values.shape
        device = pixel_values.device
        dtype = pixel_values.dtype

        # ========== Vision features ==========
        flat_images = pixel_values.flatten(0, 1)
        flat_mask = image_mask.view(-1).bool()
        valid_images = flat_images[flat_mask]
        vision_outputs = self.vlm.model.vision_model(
            pixel_values=valid_images, output_hidden_states=True, return_dict=True,
        )
        image_features = vision_outputs.last_hidden_state
        if hasattr(self.vlm.model, 'connector'):
            image_features = self.vlm.model.connector(image_features)
        elif hasattr(self.vlm.model, 'multi_modal_projector'):
            image_features = self.vlm.model.multi_modal_projector(image_features)

        hidden_size = image_features.shape[-1]
        num_patches = image_features.shape[1]
        full_image_features = image_features.new_zeros(B * V, num_patches, hidden_size)
        full_image_features[flat_mask] = image_features
        full_image_features = full_image_features.view(B, V, num_patches, hidden_size)
        valid_per_sample = image_mask.sum(dim=1).int()

        # ========== Text embeddings (ORIGINAL, keep baseline capability) ==========
        text_embeds = self.vlm.model.text_model.get_input_embeddings()(input_ids)
        text_embeds = text_embeds.to(dtype=dtype)

        # ========== Text Semantic residual ==========
        sem_out = self.semantic_task_encoder(abstract_task_ids)
        sem_embeds = sem_out["task_features"]
        if self.semantic_proj is not None:
            sem_embeds = self.semantic_proj(sem_embeds)
        sem_embeds = sem_embeds.to(device=device, dtype=dtype)

        sem_pooled = sem_embeds.mean(dim=1, keepdim=True)
        alpha_sem = torch.sigmoid(self.semantic_alpha)

        combined_text = text_embeds + alpha_sem * sem_pooled

        # ========== Visual Semantic: object tokens appended to image features ==========
        alpha_vis = torch.tensor(0.0, device=device, dtype=dtype)
        extra_vis_tokens = None
        if self.use_visual_semantic and self.visual_semantic_encoder is not None and visual_semantics is not None:
            vis_tokens = self.visual_semantic_encoder(visual_semantics)
            if self.visual_proj is not None:
                vis_tokens = self.visual_proj(vis_tokens)
            vis_tokens = vis_tokens.to(device=device, dtype=dtype)
            alpha_vis = torch.sigmoid(self.visual_alpha)
            extra_vis_tokens = alpha_vis * vis_tokens

            vis_mask = getattr(self.visual_semantic_encoder, '_last_mask', None)
            if vis_mask is None:
                vis_mask = torch.ones(B, vis_tokens.size(1), device=device, dtype=torch.long)

        # ========== Build combined sequence per sample ==========
        batch_inputs_embeds = []
        batch_attention_masks = []
        max_seq_len = 0
        for b in range(B):
            num_valid = valid_per_sample[b].item()
            sample_image_feats = full_image_features[b, :num_valid].reshape(-1, hidden_size)
            sample_text_feats = combined_text[b]

            parts = [sample_image_feats]
            if extra_vis_tokens is not None:
                vm = vis_mask[b]
                valid_vis = extra_vis_tokens[b][vm.bool()]
                if valid_vis.size(0) > 0:
                    parts.append(valid_vis)

            sample_vis_masks = [torch.ones(sample_image_feats.size(0), device=device, dtype=torch.long)]
            if extra_vis_tokens is not None:
                vm = vis_mask[b]
                sample_vis_masks.append(vm[vm.bool()])

            combined = torch.cat(parts, dim=0) if len(parts) > 1 else parts[0]
            text_mask = torch.ones(sample_text_feats.size(0), device=device, dtype=torch.long)
            full_mask = torch.cat(sample_vis_masks + [text_mask], dim=0)

            batch_inputs_embeds.append(torch.cat([combined, sample_text_feats], dim=0))
            batch_attention_masks.append(full_mask)
            max_seq_len = max(max_seq_len, batch_inputs_embeds[-1].shape[0])

        padded_inputs_embeds = torch.zeros(B, max_seq_len, hidden_size, device=device, dtype=dtype)
        attention_mask = torch.zeros(B, max_seq_len, device=device, dtype=torch.long)
        for b, embeds in enumerate(batch_inputs_embeds):
            padded_inputs_embeds[b, :embeds.shape[0]] = embeds
            attention_mask[b, :embeds.shape[0]] = batch_attention_masks[b]

        lm_outputs = self.vlm.model.text_model(
            inputs_embeds=padded_inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        return {"vlm_features": lm_outputs.last_hidden_state,
                "alpha_sem": alpha_sem, "alpha_vis": alpha_vis}

    # ================================= training =================================
    def forward(
        self,
        input_ids: torch.LongTensor,        # [B, L] - tokenized language instruction
        image_input: torch.FloatTensor,     # [B, V, C, H, W]
        image_mask: torch.Tensor,           # [B, V]
        proprio: torch.Tensor,              # [B, dim_proprio]
        action: torch.Tensor,               # [B, T=num_actions, D=dim_action]
        abstract_task_ids: torch.LongTensor | None = None,  # [B, L_sem] - semantic mode
    ) -> Dict[str, torch.Tensor]:
        """
        Flow Matching training.

        Two modes:
          - Text mode (default): input_ids 用于 text embedding
          - Semantic mode: 如果提供 abstract_task_ids 且 self.use_semantic=True,
            用 de-instantiated abstract task embeddings 取代原始文本

        1) Time sampling: t ~ Beta(1.5, 1) * 0.999 + 0.001
        2) Interpolation: x_t = t * noise + (1-t) * actions
        3) Target: velocity u_t = noise - actions
        4) Model predicts v_t, compute MSE(v_t, u_t)
        """
        if abstract_task_ids is not None and self.use_semantic:
            enc = self.forward_vlm_semantic(image_input, image_mask, abstract_task_ids)
        else:
            enc = self.forward_vlm_efficient(image_input, image_mask, input_ids)

        B = image_input.shape[0]
        device = input_ids.device
        
        # Beta(1.5, 1) time sampling
        beta_dist = torch.distributions.Beta(
            torch.tensor(1.5, device=device), 
            torch.tensor(1.0, device=device)
        )
        t = beta_dist.sample((B,)) * 0.999 + 0.001

        # Normalize action and proprio
        if hasattr(self.action_space, 'normalize_action'):
            action_norm = self.action_space.normalize_action(action)
        elif hasattr(self.action_space, 'normalize'):
            action_norm = self.action_space.normalize(action)
        else:
            action_norm = action
            
        if hasattr(self.action_space, 'normalize_state'):
            proprio_norm = self.action_space.normalize_state(proprio)
        elif hasattr(self.action_space, 'normalize'):
            proprio_norm = self.action_space.normalize(proprio)
        else:
            proprio_norm = proprio
        
        # Flow Matching
        noise = torch.randn_like(action_norm)
        t_expanded = t.view(-1, 1, 1)
        x_t = t_expanded * noise + (1 - t_expanded) * action_norm
        u_t = noise - action_norm

        # Model prediction (no aux_visual_inputs for SmolVLM)
        v_t = self.transformer(
            vlm_features=enc["vlm_features"],
            action_with_noise=x_t,
            t=t,
            proprio=proprio_norm,
        )
        
        # MSE loss
        velocity_loss = torch.mean(torch.square(v_t - u_t))
        
        return {"velocity_loss": velocity_loss}

    # ================================= inference =================================
    @torch.no_grad()
    def generate_actions(
        self,
        input_ids: torch.LongTensor,
        image_input: torch.FloatTensor,
        image_mask: torch.Tensor,
        proprio: torch.Tensor,
        steps: int = 10,
        abstract_task_ids: torch.LongTensor | None = None,
        visual_semantics: list | None = None,
    ) -> torch.Tensor:
        """
        Flow Matching inference (Euler integration).

        Supports 3 modes:
          - Baseline: no abstract_task_ids
          - Semantic: abstract_task_ids provided (text semantic)
          - Visual+Semantic: both abstract_task_ids and visual_semantics (triple residual)
        """
        self.eval()
        if abstract_task_ids is not None and self.use_semantic:
            enc = self.forward_vlm_semantic(image_input, image_mask, input_ids,
                                           abstract_task_ids, visual_semantics)
        else:
            enc = self.forward_vlm_efficient(image_input, image_mask, input_ids)

        B = image_input.shape[0]
        D = self.action_space.dim_action
        device = proprio.device
        dtype = proprio.dtype

        # Normalize proprio
        if hasattr(self.action_space, 'normalize_state'):
            proprio_norm = self.action_space.normalize_state(proprio)
        elif hasattr(self.action_space, 'normalize'):
            proprio_norm = self.action_space.normalize(proprio)
        else:
            proprio_norm = proprio

        # Euler integration
        steps = max(1, int(steps))
        dt = -1.0 / steps
        
        x_t = torch.randn(B, self.num_actions, D, device=device, dtype=dtype)
        t = 1.0
        
        while t > -dt / 2:
            t_tensor = torch.full((B,), t, device=device, dtype=dtype)
            
            v_t = self.transformer(
                vlm_features=enc["vlm_features"],
                action_with_noise=x_t,
                proprio=proprio_norm,
                t=t_tensor,
            )
        
            x_t = x_t + dt * v_t
            t = t + dt
        
        return self.action_space.postprocess(x_t)

    # =============================== FastAPI service =============================
    def _build_app(self, processor):
        """Build FastAPI app for SmolVLM-VLA inference."""
        if self.app is not None:
            return

        app = FastAPI()

        @app.post("/act")
        def act(payload: Dict[str, Any]):
            try:
                self.eval()
                # Decode images
                images = []
                for key in ("image0", "image1", "image2"):
                    if key not in payload:
                        continue
                    v = json_numpy.loads(payload[key])
                    if isinstance(v, np.ndarray):
                        if v.ndim == 1:
                            v = cv2.imdecode(v, cv2.IMREAD_COLOR)
                        images.append(Image.fromarray(v))
                    elif isinstance(v, (list, tuple)):
                        images.append(Image.fromarray(np.array(v)))
                    elif isinstance(v, str):
                        images.append(Image.open(v))
                        
                if not images:
                    return JSONResponse({"error": "No valid images found."}, status_code=400)

                # Process inputs
                inputs = processor(images, payload["language_instruction"])
                if not {"input_ids", "image_input", "image_mask"}.issubset(inputs):
                    return JSONResponse({"error": "Processor returned incomplete inputs."}, status_code=400)

                # Build proprio tensor
                proprio = torch.as_tensor(np.asarray(json_numpy.loads(payload["proprio"])))

                # Align to model device/dtype
                device = next(self.parameters()).device
                dtype = next(self.parameters()).dtype

                def to_model(t: torch.Tensor) -> torch.Tensor:
                    if not isinstance(t, torch.Tensor):
                        t = torch.as_tensor(t)
                    return t.to(device=device, dtype=dtype) if t.is_floating_point() else t.to(device=device)

                inputs = {k: to_model(v) for k, v in inputs.items()}
                inputs["proprio"] = to_model(proprio.unsqueeze(0))

                # Inference
                steps = int(payload.get("steps", 10))
                action = self.generate_actions(**inputs, steps=steps).squeeze(0).float().cpu().numpy()
                return JSONResponse({"action": action.tolist()})

            except Exception:
                logging.error(traceback.format_exc())
                return JSONResponse({"error": "Request failed"}, status_code=400)

        self.app = app

    def run(self, processor, host: str = "0.0.0.0", port: int = 8000):
        """Launch the FastAPI service."""
        self._build_app(processor)
        assert self.app is not None
        uvicorn.run(self.app, host=host, port=port)