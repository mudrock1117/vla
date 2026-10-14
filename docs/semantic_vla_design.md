# Vision-Grounded Semantic VLA — 设计工作书

## 1. 问题诊断：Baseline SimVLA 的失败模式

### 1.1 失败分类树

```
                    Baseline Failure
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
        ▼                  ▼                  ▼
   Language            Object             Instance
   Understanding       Recognition        Grounding
        │                  │                  │
        ▼                  ▼                  ▼
  指令理解错误        类别识别错误       找错具体对象
                                           
        ┌──────────────────┼──────────────────┐
        │                  │                  │
        ▼                  ▼                  ▼
     Counting          Spatial           Long-horizon
     Problem           Grounding           Task
        │                  │                  │
        ▼                  ▼                  ▼
   多个相同物体         位置关系错误        多步骤执行错误
```

### 1.2 实测失败任务定位

| 任务 | 失败类型 | 具体表现 |
|---|---|---|
| KITCHEN_SCENE8 put_both_moka_pots | Counting + Instance Grounding | 两个相同物体，baseline 无法区分、无法正确放置两个目标 |
| put_mug_in_microwave | Spatial Grounding | 新 target category（微波炉内部），baseline 理解为"在上面" |
| put_both 系列 (3/4) | Counting | 双物体任务，baseline 只能放一个 |

### 1.3 根因分析

Baseline SimVLA 的文本编码器将自然语言压缩为单一 embedding，**丢失了结构化信息**：
- 物体数量（1 vs 2）被压缩
- 物体类别（food_item vs graspable_container）被模糊
- 空间关系（in vs on vs inside）被弱化
- 多步任务的结构被打散

**关键洞察**：baseline 的失败不是"不懂语言"，而是"无法从单一 embedding 中解耦多维度约束"。

---

## 2. 核心思想：Vision-Grounded Semantic Representation

### 2.1 架构总览

```
Instruction ──────┬──────────────────────────────┐
                  │                              │
                  ▼                              │
        ┌──────────────────┐                     │
        │ Semantic         │                     │
        │ Abstractor       │                     │
        └────────┬─────────┘                     │
                 │                               │
                 ▼                               │
       Task-level Semantics                       │
                 │                               │
RGB Image ───────┤                               │
                 ▼                               │
        ┌──────────────────┐                     │
        │ Visual Scene     │                     │
        │ Understanding    │                     │
        └────────┬─────────┘                     │
                 │                               │
                 ▼                               │
          Visual Semantics                       │
                 │                               │
                 └──────────┐                   │
                            ▼                   ▼
                 ┌──────────────────┐    ┌───────────────┐
                 │ Semantic Task    │    │ Baseline Text │
                 │ Representation   │    │   Embedding   │
                 └────────┬─────────┘    └───────┬───────┘
                          │                       │
                          ▼                       ▼
                    Semantic Task          Vision-Language
                      Encoder                Fusion (SimVLA)
                          │                       │
                          └───────────┬───────────┘
                                      ▼
                                 Action Output
```

### 2.2 与 Baseline 的关键区别

| 维度 | Baseline SimVLA | Semantic VLA (本方案) |
|---|---|---|
| 文本编码 | 单一 VLM text_model embedding | 保留 + **结构化语义 embedding 残差** |
| 语义来源 | 仅自然语言文本 | **文本解析 + 视觉场景理解** |
| 物体数量 | 压缩到 embedding 中 | **显式 NUM:1/NUM:2 token** |
| 空间关系 | 模糊编码 | **显式 REL:in/on/inside 解析** |
| Instance ID | 无 | **Phase 2 引入视觉 grounding** |

---

## 3. Structured Task Representation 设计

### 3.1 第一版：Task-Level Semantics（仅从文本提取）

先从文本解析结构化语义，验证有效后再叠加视觉 grounding。

```yaml
Task Representation:
    Action: put
    
    Source Object:
        category: moka_pot              # 从文本解析
        quantity: 2                      # 从 "both" 解析
    
    Target Object:
        category: flat_surface           # 从文本解析 "stove"
        spatial_relation: on             # 从介词解析
```

**Object Categories（建议 3-5 个）**：

| Category | 包含物体 | 文本关键词 |
|---|---|---|
| graspable_container | moka_pot, mug, bowl | pot, mug, bowl, bottle |
| food_item | alphabet_soup, cream_cheese | soup, cheese, butter, sauce |
| box_container | basket, tray | basket, tray, box |
| flat_surface | stove, plate, cabinet | stove, plate, cabinet, table |
| drawer/compartment | drawer, compartment | drawer, compartment |

**Target Categories（建议 2-3 个）**：

| Category | 包含物体 |
|---|---|
| flat_surface | stove, plate, cabinet, table |
| box_container | basket, tray |
| enclosure | drawer, compartment, microwave |

**Actions（1-2 个）**：

| Action | 文本关键词 |
|---|---|
| put / pick_and_place | put, place, pick up |
| toggle | turn on, turn off, press |

**Relations**：

| Relation | 文本关键词 |
|---|---|
| in | in, inside |
| on | on, on top of |
| from_drawer | from the drawer |

### 3.2 第二版：引入视觉 Grounding

**在第一版验证有效后，增加**：

```yaml
Task Representation:
    Action: put
    
    Source Object:
        category: moka_pot
        quantity: 2
        instance_id: [moka_1, moka_2]    # ← 视觉检测
        position: [bbox_1, bbox_2]       # ← 视觉定位
    
    Target Object:
        category: tray
        position: bbox_target            # ← 视觉定位
    
    Spatial Relation: on
```

### 3.3 完整层级表示

```
Task Representation
    │
    ├── Action
    │
    ├── Source Object
    │     ├── Category
    │     ├── Quantity
    │     ├── Instance ID        (Phase 2)
    │     └── Position           (Phase 2)
    │
    ├── Target Object
    │     ├── Category
    │     ├── Instance ID        (Phase 2)
    │     └── Position           (Phase 2)
    │
    └── Spatial Relation
```

---

## 4. 模型架构

### 4.1 Semantic Task Encoder

```
Structured Task (token sequence)
       ↓
AbstractTaskTokenizer
       ↓
[PAD, ACT:put, OBJ:moka_pot, TGT:flat_surface, REL:on, NUM:2, PAD]
       ↓
Token Embedding (nn.Embedding) + Position Embedding
       ↓
Transformer Encoder (2 layers, d_model=256, nhead=4)
       ↓
Pool → Semantic Task Embedding (d=256)
       ↓
Projection Layer (if needed) → match VLM hidden size (d=?)
```

### 4.2 残差融合模块

```python
# 原始文本 embedding（保留 baseline 能力）
text_embeds = vlm.text_model.embeddings(input_ids)

# 语义 task embedding
sem_embeds = semantic_task_encoder(abstract_task_ids)
if semantic_proj is not None:
    sem_embeds = semantic_proj(sem_embeds)

# 语义残差融合
# 选择：mean pooling semantic → broadcast 到每个 token
sem_pooled = sem_embeds.mean(dim=1, keepdim=True)  # [B, 1, D]

# 门控（sigmoid 保证范围 [0,1]）
alpha = torch.sigmoid(semantic_alpha)  # scalar, init=0.0 → sigmoid=0.5

combined_text = text_embeds + alpha * sem_pooled  # [B, L_text, D]

# 后续进入 VLM 融合 + Action Head
```

### 4.3 为什么用残差而不是替换

- 替换方案导致 Semantic 完全接管文本理解 → baseline 能力丢失
- 残差方案让模型 **自主学习** semantic 信号的权重
- alpha 初始 0.5（sigmoid），训练中可以学到 0（不需要 semantic）或 1（完全依赖 semantic）

---

## 5. 三阶段训练流程

### Stage 1：冻结 VLM，训练 Semantic Task Encoder

```
SimVLA Baseline (冻结所有 VLM 参数)
      ↓
冻结原始 VLA (eval 模式)
      ↓
训练 Semantic Task Encoder + alpha + semantic_proj
      ↓
收集轨迹 → 从 3 个 split 收集（ID + OOD-Composition + OOD-Both）
  - ID: 学习常规语义
  - OOD-Composition: 学习新组合的语义
  - OOD-Both: 学习 NUM:2 / MULTI:yes 等 token
```

**训练设置**：
- Optimizer: AdamW, lr=1e-4
- Batch size: 16
- Steps: 800-1500
- Loss: MSE(predicted_actions, ground_truth_actions)
- 可训练参数: ~1M（semantic encoder + proj + alpha）

**预期产出**：`checkpoints/SimVLA-LIBERO/semantic_encoder.pt`（29 keys）

### Stage 2：训练 Semantic Encoder + Fusion Module

```
Stage 1 Semantic Encoder (预训练好)
      +
Fusion Module (残差 alpha)
      ↓
解冻 VLM 的最后 N 层 (LoRA 或 Adapter)
      ↓
轻量联合微调
```

**训练设置**：
- 解冻：VLM text_model 最后 2 层 + VLA action head
- Optimizer: AdamW, lr=5e-5（更小，避免遗忘）
- LoRA rank: 8
- Steps: 500-1000

### Stage 3：完整微调（可选）

```
全模型 (Semantic + VLM + VLA Head)
      ↓
LoRA / Adapter 轻量微调
      ↓
在完整 40-90 任务上微调
```

---

## 6. 实验计划

### 6.1 当前已完成

| 项目 | 状态 |
|---|---|
| OOD 数据分割 (4 split) | ✅ |
| Baseline 10 trials 复现 | ✅ OOD-Both: 81.4% |
| Semantic v1-v3 (替换方案) | ❌ 3.3%，失败 |
| Semantic v4 残差 + alpha=-5 | ❌ alpha 训不动，≈ baseline |
| Semantic v5 残差 + alpha=0 | ✅ alpha=0.49，OOD-Both: 82.9% |
| **Semantic vs Baseline 真对比** | ✅ +1.5% |

### 6.2 当前瓶颈

- **K8 put_both_moka_pots**: Baseline 20%, Semantic 20% → 文本语义不足以解决 Counting + Instance Grounding
- **文本 semantic 本质冗余**：从同一条 instruction 提取 → baseline text_model 已包含相同信息
- **缺少视觉 grounding**：文本无法告诉模型"哪个 moka pot 在哪个位置"

### 6.3 下一阶段

| 优先级 | 项目 | 描述 |
|---|---|---|
| P0 | **视觉 grounding 引入** | 检测图像中物体类别/数量/位置，补充到 semantic 表示 |
| P1 | **Stage 2 联合微调** | 解冻 VLM 部分层 + semantic encoder 联合训练 |
| P2 | **Instance ID 跟踪** | 多相同物体的区分（Phase 2） |
| P3 | **Position encoding** | 物体位置 bbox → embedding，解决 spatial grounding |

---

## 7. 文件索引

```
models/
├── semantic_schema.py          # AbstractTask 数据类 + tokenizer
├── abstract_task_encoder.py    # Transformer encoder
├── modeling_smolvlm_vla.py     # 修改版：semantic encoder + residual fusion
└── processing_smolvlm_vla.py   # 数据处理

eval/
├── eval_ood_splits.py          # 多 split 评估脚本
└── splits/                     # ID / OOD-Instance / OOD-Composition / OOD-Both json

train/
└── train_semantic_ablation.py  # Stage 1 训练脚本

checkpoints/
└── SimVLA-LIBERO/
    ├── model.safetensors       # Baseline VLA 参数
    ├── config.json
    └── semantic_encoder.pt     # Semantic 参数（29 keys）

docs/
└── semantic_vla_design.md      # 本文件
```

---

## 8. 关键技术决策记录

| 决策 | 选择 | 理由 |
|---|---|---|
| 融合方式 | 残差加法 `text + α * sem` | 替换方案丢失 baseline 能力，残差让模型自主学习权重 |
| alpha 初始化 | sigmoid(0) = 0.5 | sigmoid(-5) ≈ 0.007 导致梯度消失，alpha 训不动 |
| Semantic pool | mean over sequence | 简单有效，避免额外注意力参数 |
| 训练数据 | ID + OOD-Composition + OOD-Both | 确保 NUM:2, MULTI:yes 等 token 有足够训练样本 |
| 分阶段训练 | Stage 1 冻结 VLM | 避免小数据量下全量微调遗忘 baseline 能力 |