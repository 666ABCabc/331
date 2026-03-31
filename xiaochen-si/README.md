# 儿童语音识别挑战赛 - Word Track

DrivenData "On Top of Pasketti" 儿童语音识别挑战赛 Word 赛道方案。

- 比赛页面: https://www.drivendata.org/competitions/308/childrens-word-asr/
- 评估指标: WER (Word Error Rate)
- 截止日期: 2026年4月6日

## 成绩

| 方案 | WER | 说明 |
|------|-----|------|
| 官方 Baseline (Parakeet-TDT-0.6B + adapter dim=32) | 0.237 | 仅 65K 可训练参数 |
| **本方案 (Whisper-large-v3-turbo + LoRA)** | **0.3994** | Smoke test 分数，排名 81 |

## 方案概述

基于 `openai/whisper-large-v3-turbo` (809M 参数) 的 LoRA 微调方案：

- **微调方式**: LoRA (rank=32, alpha=64)，应用于 encoder+decoder 所有 attention 层的 q/k/v/out_proj + FFN
- **可训练参数**: 27.8M (3.3%)，比 baseline 的 65K 增加了 400 倍
- **训练数据**: 248K 条儿童语音 (DrivenData 95K + TalkBank 153K，过滤损坏/缺失后)
- **数据工程**: 按 child_id 分组验证集、过滤损坏/缺失音频、过滤异常样本
- **训练配置**: 1 epoch, batch_size=64, lr=1e-4, cosine scheduler, bf16, gradient checkpointing

## 项目结构

```
xiaochen-si/
│
├── README.md                          # 本文件
│
│  ==================== 训练代码 ====================
│
├── train_whisper.py                   # ★ [核心] Whisper LoRA 微调训练脚本（最终使用的方案）
├── train_lora.py                      #   [备选] Parakeet-TDT-1.1B 大 Adapter + 部分 encoder 解冻训练脚本
├── train_orthographic.py              #   [baseline] 官方 baseline 训练脚本 (Parakeet-TDT-0.6B + adapter)
│
│  ==================== 推理 & 提交 ====================
│
├── orthographic_submission/
│   └── main.py                        # ★ [核心] 提交用的推理入口脚本（Whisper 单模型推理）
├── pack_submission.py                 #   打包 submission.zip（main.py + 模型文件）
├── submission.zip                     #   最终提交文件 (1.2 GB)（服务器上生成）
│
│  ==================== 工具库 ====================
│
├── asr_benchmark/                     #   共享工具库
│   ├── __init__.py
│   ├── config.py                      #   路径常量 (PROJECT_ROOT, DATA_ROOT)
│   ├── data_utils.py                  #   ★ 数据加载/过滤/分割/NeMo manifest/增强配置
│   ├── score.py                       #   WER 评分 + Whisper 英文文本归一化
│   ├── nemo_adapter.py                #   NeMo adapter 工具函数（Parakeet 方案用）
│   └── assets/
│       └── asr_adaptation.yaml        #   NeMo adapter 训练配置（Parakeet 方案用）
│
│  ==================== 其他工具 ====================
│
├── rover_ensemble.py                  #   纯 Python ROVER 多模型融合（备用，未在最终提交中使用）
├── test_local.py                      #   本地推理测试脚本
├── setup_env.sh                       #   环境配置脚本
├── justfile                           #   任务自动化（来自 benchmark repo）
├── pyproject.toml                     #   项目依赖定义
│
│  ==================== 数据 ====================
│
├── data-demo/                         #   3 条 demo 数据，用于本地测试推理格式
│   └── word/
│       ├── audio/                     #     3 个 .flac 文件
│       ├── utterance_metadata.jsonl
│       └── submission_format.jsonl
│
├── data/                              #   训练数据（不提交，需自行下载）
│   ├── raw/
│   │   ├── drivendata/                #     DrivenData 语料 (95K 条)
│   │   │   ├── audio/                 #       .flac 音频文件
│   │   │   └── train_word_transcripts.jsonl
│   │   ├── talkbank/                  #     TalkBank 语料 (255K 条)
│   │   │   ├── audio/
│   │   │   └── train_word_transcripts.jsonl
│   │   └── noise/                     #     RealClass 噪声数据 (可选)
│   │       └── audio/
│   └── processed/                     #     训练时自动生成的 manifest
│
└── models/                            #   训练产出（不提交）
    └── whisper_lora/
        ├── merged_model/              #     ★ LoRA 合并后的完整模型（推理用）
        ├── lora_adapter/              #     LoRA adapter 权重
        └── checkpoint-*/              #     训练 checkpoint
```

## 文件详细说明

### 核心文件（最终提交方案）

| 文件 | 行数 | 说明 |
|------|------|------|
| `train_whisper.py` | 474 | **Whisper LoRA 训练脚本**。包含：ChildSpeechDataset（支持损坏音频跳过、速度扰动、噪声增强）、DataCollatorSpeechSeq2Seq、WhisperForConditionalGeneration.forward 补丁（兼容 HF Trainer 的 input_ids 问题）、LoRA 配置、训练参数、自动保存 LoRA adapter 和合并模型。 |
| `orthographic_submission/main.py` | 120 | **提交用推理脚本**。加载合并后的 Whisper 模型 (bf16)，批量推理 (batch=32)，soundfile 优先加载音频 + librosa 回退 + 静音兜底。输出 JSONL 格式。在 DrivenData 的 A100 80GB 上运行。 |
| `asr_benchmark/data_utils.py` | 259 | **数据工程**。合并 DrivenData + TalkBank 数据、时长/空文本/缺失音频/异常样本过滤、按 child_id 分组的 GroupShuffleSplit 验证集划分、NeMo manifest 生成、速度扰动/噪声增强配置。 |

### 备选方案文件（Parakeet，未用于最终提交）

| 文件 | 说明 |
|------|------|
| `train_lora.py` | Parakeet-TDT-1.1B/0.6B 大 Adapter + encoder 部分解冻训练。因 numba CUDA 兼容性问题未能在 3090 上完成训练。在 2080 Ti 上 val WER 达到 0.213。 |
| `train_orthographic.py` | 官方 baseline 训练脚本（Parakeet-TDT-0.6B + LinearAdapter dim=32）。 |
| `asr_benchmark/nemo_adapter.py` | NeMo adapter 设置工具（encoder target class 替换、config 合并、lhotse 补丁）。 |
| `asr_benchmark/assets/asr_adaptation.yaml` | NeMo adapter 训练完整配置（SpecAugment、optimizer、scheduler 等）。 |
| `rover_ensemble.py` | 纯 Python ROVER 实现，用于多模型词级融合。备用方案，未在最终提交中使用。 |

### 工具文件

| 文件 | 说明 |
|------|------|
| `asr_benchmark/config.py` | `PROJECT_ROOT` 和 `DATA_ROOT` 路径常量 |
| `asr_benchmark/score.py` | WER 评分，使用 Whisper EnglishTextNormalizer + 英式→美式拼写映射 |
| `pack_submission.py` | 打包 submission.zip，支持 Parakeet 单模型 / Whisper 单模型 / 双模型 |
| `test_local.py` | 在 data-demo 上跑推理验证格式 |
| `setup_env.sh` | GPU 服务器环境配置脚本 (pip install torch, nemo, transformers 等) |
| `pyproject.toml` | 项目依赖定义 |
| `justfile` | 来自 benchmark repo 的任务自动化 |

## 提交文件结构

`submission.zip` 内容（上传到 DrivenData）：

```
submission.zip (1.2 GB)
├── main.py                            # 推理入口（必须在 zip 根目录）
└── whisper_merged/                    # 微调后的 Whisper 模型
    ├── model.safetensors (1.5 GB)     #   模型权重
    ├── config.json                    #   模型配置
    ├── generation_config.json         #   生成配置
    ├── preprocessor_config.json       #   音频预处理配置
    ├── tokenizer.json                 #   分词器
    ├── tokenizer_config.json
    ├── vocab.json
    ├── merges.txt
    ├── normalizer.json
    ├── added_tokens.json
    └── special_tokens_map.json
```

## 复现步骤

### 1. 环境配置

需要 NVIDIA GPU (≥ 24GB VRAM)，Python 3.11，CUDA 12+。

```bash
pip install torch torchaudio torchvision
pip install "transformers>=4.40.0" peft librosa soundfile pandas scikit-learn loguru jiwer
```

### 2. 数据准备

从 DrivenData 和 TalkBank 下载数据：

```
data/raw/drivendata/audio/                   # 解压 audio_part_0/1/2.zip
data/raw/drivendata/train_word_transcripts.jsonl
data/raw/talkbank/audio/                     # 解压 TalkBank audio.zip
data/raw/talkbank/train_word_transcripts.jsonl
```

### 3. 训练

```bash
# Whisper LoRA 训练 (~15h on RTX 3090)
python train_whisper.py \
  --model openai/whisper-large-v3-turbo \
  --lora-rank 32 --lora-alpha 64 \
  --epochs 1 --batch-size 64 --grad-accum 1 --lr 1e-4 \
  --precision bf16 --num-workers 4 --max-duration 15.0
```

模型保存到 `models/whisper_lora/merged_model/`。

### 4. 打包提交

```bash
python pack_submission.py --whisper models/whisper_lora/merged_model
```

生成 `submission.zip`，上传到 DrivenData 提交。先 smoke test 再正式提交。

## 相对于 Baseline 的改进

| 改进项 | Baseline | 本方案 |
|--------|----------|--------|
| 模型 | Parakeet-TDT-0.6B (Transducer) | Whisper-large-v3-turbo (Encoder-Decoder) |
| 微调方式 | Linear Adapter (dim=32) | LoRA (rank=32, 6 个 target module) |
| 可训练参数 | 65K (0.01%) | 27.8M (3.3%) — 400 倍 |
| 验证集划分 | 随机 20% | 按 child_id 分组 10% |
| 数据过滤 | 仅时长 >25s | 时长 + 空文本 + 缺失音频 + 异常样本 |
| 音频加载 | pydub (需 ffprobe) | soundfile 优先 + librosa 回退 + 损坏跳过 |
| 训练精度 | bf16-mixed | bf16 + gradient checkpointing |
