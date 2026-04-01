# DrivenData Children's Speech Recognition Challenge - Code Explanation 
## Project Structure 
```
xiaochen-si/
│
├── README.md                          # Project description document
│
│  ==================== Training code ====================
│
├── train_whisper.py                   # [Core] Whisper LoRA fine-tuning training script
├── train_lora.py                      # [Alternative] Parakeet Large Adapter training script
├── train_orthographic.py              # [Reference] Official baseline training script
│
│  ==================== Inference & Submission ====================
│ ├── orthographic_submission/
│   └── main.py                        # [Core] The inference entry script for submission submission
├── pack_submission.py                 # The tool for packaging submission.zip
│
│  ==================== Library ====================
│
├── asr_benchmark/                     # Shared library
│   ├── config.py                      # Path constants configuration
│   ├── data_utils.py                  # Data loading / filtering / splitting / enhancement
│   ├── score.py                       # WER scoring + text normalization
│   ├── nemo_adapter.py                # NeMo adapter tool (for Parakeet) │   └── assets/
│       └── asr_adaptation.yaml        # NeMo training configuration
│
│  ==================== Other Tools ====================
│
├── rover_ensemble.py                  # Multi-model fusion (ROVER algorithm)
├── test_local.py                      # Local inference test script
├── setup_env.sh                       # Environment configuration script
├── justfile                           # Task automation
├── pyproject.toml                     # Project dependencies definition
│
│  ==================== Data ====================
│
├── data-demo/                         # 3 demo data sets for local testing │   └── word/
│       ├── audio/                     # 3 .flac audio files
│       ├── utterance_metadata.jsonl   # Test data metadata
│       └── submission_format.jsonl    # Submission format template
│
└── models/                            # Training outputs (not submitted) └── whisper_lora/
├── merged_model/              # The complete model after LoRA merging
├── lora_adapter/              # LoRA adapter weights
└── runs/                      # Training logs (TensorBoard) ```

## Detailed Explanation of Core Documents 
### 1. train_whisper.py (474 lines)
**Function**: Whisper LoRA fine-tuning training script 
**Main Components**:
- `ChildSpeechDataset`: Custom dataset class, supporting skipping of damaged audio, speed perturbation, and noise enhancement
- `DataCollatorSpeechSeq2Seq`: Data collator, handling batch processing of audio and text
- LoRA configuration: rank=32, alpha=64, applied to all attention layers
- Training parameters: 1 epoch, batch_size=64, lr=1e-4, cosine scheduler
- Automatic saving: Automatically saves LoRA adapter and merged model after training is completed 
**Innovative Features**:
- Patch to WhisperForConditionalGeneration.forward to address the input_ids issue in HF Trainer
- Grouping of validation sets by child_id to prevent data leakage
- Merging of multiple data sources (DrivenData + TalkBank) 
### 2. orthographic_submission/main.py (120 lines)
**Function**: The inference script submitted to DrivenData 
**Work Flow**:
1. **Model Loading**: Load the merged Whisper model (in bf16 precision)
2. **Data Loading**: Read utterance_metadata.jsonl
3. **Audio Loading**: Use soundfile first, fallback to librosa, and add 1-second silence as a backup
4. **Batch Inference**: Set batch_size to 32 and sort by audio length in descending order
5. **Result Generation**: Use model.generate() to generate the transcription text
6. **Output Format**: Write the results in the submission_format.jsonl format 
**Technical Features**:
- Robust audio loading mechanism
- Optimized batch processing for enhanced inference speed
- Memory optimization (bf16 precision) 
### 3. asr_benchmark/data_utils.py (Line 259)
**Function**: Core module of data engineering 
**Main Functions**:
- `load_all_transcripts()`: Merge DrivenData and TalkBank data
- `filter_data()`: Filter out damaged/missing audio and abnormal samples
- `split_by_child()`: Group and divide the validation set by child_id
- Generate NeMo manifest file
- Speed perturbation and noise enhancement configuration 
**Data Processing Flow**:
1. Load the original transcription data
2. Filter out abnormal samples
3. Group by child_id
4. Divide into training/verification sets
5. Generate the manifest file required for training 
### 4. pack_submission.py
**Function**: Package and submit files 
**Work Flow**:
1. Read the model file
2. Copy main.py to the packaging directory
3. Copy the model file to the packaging directory
4. Compress into submission.zip 
**Supported Models**:
- Parakeet Single Model
- Whisper Single Model
- Dual Models (for ROVER Fusion)


# DrivenData 儿童语音识别挑战赛 - 代码讲解

## 项目结构

```
xiaochen-si/
│
├── README.md                          # 项目说明文档
│
│  ==================== 训练代码 ====================
│
├── train_whisper.py                   # [核心] Whisper LoRA 微调训练脚本
├── train_lora.py                      # [备选] Parakeet 大 Adapter 训练脚本
├── train_orthographic.py              # [参考] 官方 baseline 训练脚本
│
│  ==================== 推理 & 提交 ====================
│
├── orthographic_submission/
│   └── main.py                        # [核心] 提交用的推理入口脚本
├── pack_submission.py                 # 打包 submission.zip 的工具
│
│  ==================== 工具库 ====================
│
├── asr_benchmark/                     # 共享工具库
│   ├── config.py                      # 路径常量配置
│   ├── data_utils.py                  # 数据加载/过滤/分割/增强
│   ├── score.py                       # WER 评分 + 文本归一化
│   ├── nemo_adapter.py                # NeMo adapter 工具（Parakeet用）
│   └── assets/
│       └── asr_adaptation.yaml        # NeMo 训练配置
│
│  ==================== 其他工具 ====================
│
├── rover_ensemble.py                  # 多模型融合（ROVER算法）
├── test_local.py                      # 本地推理测试脚本
├── setup_env.sh                       # 环境配置脚本
├── justfile                           # 任务自动化
├── pyproject.toml                     # 项目依赖定义
│
│  ==================== 数据 ====================
│
├── data-demo/                         # 3条 demo 数据，用于本地测试
│   └── word/
│       ├── audio/                     # 3个 .flac 音频文件
│       ├── utterance_metadata.jsonl   # 测试数据元信息
│       └── submission_format.jsonl    # 提交格式模板
│
└── models/                            # 训练产出（不提交）
    └── whisper_lora/
        ├── merged_model/              # LoRA 合并后的完整模型
        ├── lora_adapter/              # LoRA adapter 权重
        └── runs/                      # 训练日志（TensorBoard）
```

## 核心文件详解

### 1. train_whisper.py（474行）
**功能**：Whisper LoRA 微调训练脚本

**主要组件**：
- `ChildSpeechDataset`：自定义数据集类，支持损坏音频跳过、速度扰动、噪声增强
- `DataCollatorSpeechSeq2Seq`：数据整理器，处理音频和文本的批处理
- LoRA 配置：rank=32, alpha=64，应用于所有 attention 层
- 训练参数：1 epoch, batch_size=64, lr=1e-4, cosine scheduler
- 自动保存：训练完成后自动保存 LoRA adapter 和合并模型

**创新点**：
- WhisperForConditionalGeneration.forward 补丁，解决 HF Trainer 的 input_ids 问题
- 按 child_id 分组的验证集划分，避免数据泄露
- 多数据源合并（DrivenData + TalkBank）

### 2. orthographic_submission/main.py（120行）
**功能**：提交到 DrivenData 的推理脚本

**工作流程**：
1. **模型加载**：加载合并后的 Whisper 模型 (bf16 精度)
2. **数据加载**：读取 utterance_metadata.jsonl
3. **音频加载**：soundfile 优先 + librosa 回退 + 1秒静音兜底
4. **批量推理**：batch_size=32，按音频长度降序排列
5. **结果生成**：使用 model.generate() 生成转录文本
6. **输出格式**：按照 submission_format.jsonl 格式写入结果

**技术特点**：
- 鲁棒的音频加载机制
- 批量处理优化推理速度
- 内存优化（bf16 精度）

### 3. asr_benchmark/data_utils.py（259行）
**功能**：数据工程核心模块

**主要功能**：
- `load_all_transcripts()`：合并 DrivenData 和 TalkBank 数据
- `filter_data()`：过滤损坏/缺失音频、异常样本
- `split_by_child()`：按 child_id 分组划分验证集
- 生成 NeMo manifest 文件
- 速度扰动和噪声增强配置

**数据处理流程**：
1. 加载原始转录数据
2. 过滤异常样本
3. 按 child_id 分组
4. 划分训练/验证集
5. 生成训练所需的 manifest 文件

### 4. pack_submission.py
**功能**：打包提交文件

**工作流程**：
1. 读取模型文件
2. 复制 main.py 到打包目录
3. 复制模型文件到打包目录
4. 压缩为 submission.zip

**支持的模型**：
- Parakeet 单模型
- Whisper 单模型
- 双模型（用于 ROVER 融合）

## 技术栈

| 技术 | 版本/说明 | 用途 |
|------|-----------|------|
| PyTorch | 2.0+ | 深度学习框架 |
| Transformers | 4.40.0+ | HuggingFace 模型库 |
| PEFT | - | LoRA 实现 |
| Whisper | large-v3-turbo | 基础语音识别模型 |
| Librosa | - | 音频处理 |
| SoundFile | - | 音频文件读取 |
| NumPy | - | 数据处理 |
| Pandas | - | 数据管理 |
| Scikit-learn | - | 数据分割 |

## 训练流程

1. **环境配置**：安装依赖包
2. **数据准备**：下载 DrivenData 和 TalkBank 数据
3. **数据处理**：过滤、分组、分割
4. **模型训练**：Whisper + LoRA 微调
5. **模型合并**：将 LoRA 权重合并到原始模型
6. **打包提交**：生成 submission.zip
7. **线上测试**：DrivenData 平台评测

## 推理流程

1. **加载模型**：从 whisper_merged/ 加载模型
2. **读取数据**：从 utterance_metadata.jsonl 读取测试数据
3. **音频处理**：加载并预处理音频
4. **批量推理**：模型生成转录文本
5. **结果输出**：按格式写入 submission.jsonl

## 技术创新点

1. **模型选择**：使用 Whisper-large-v3-turbo 替代官方的 Parakeet
2. **微调策略**：LoRA 只训练 3.3% 的参数，高效且防过拟合
3. **数据工程**：合并多个数据源，按 child_id 分组验证
4. **鲁棒性设计**：三重音频加载保障，应对损坏文件
5. **推理优化**：批量处理 + 音频长度排序，提升速度

## 目录结构说明

| 目录/文件 | 类型 | 作用 | 是否提交 |
|-----------|------|------|----------|
| xiaochen-si/ | 目录 | 主项目目录 | - |
| __MACOSX/ | 目录 | Mac 系统生成的隐藏文件 | 不需要 |
| orthographic_submission/ | 目录 | 提交用的推理代码 | 部分（main.py） |
| asr_benchmark/ | 目录 | 工具库 | 否 |
| data-demo/ | 目录 | 测试数据 | 否 |
| models/ | 目录 | 训练产出 | 否 |
| submission.zip | 文件 | 最终提交包 | 是 |

## 关键参数

| 参数 | 值 | 说明 |
|------|-----|------|
| 模型 | Whisper-large-v3-turbo | 809M 参数 |
| LoRA rank | 32 | 低秩矩阵维度 |
| LoRA alpha | 64 | 学习率缩放因子 |
| 可训练参数 | 27.8M | 占总参数的 3.3% |
| 训练数据 | 248K | DrivenData 95K + TalkBank 153K |
| 训练 epoch | 1 | 避免过拟合 |
| 批量大小 | 64 | 高效训练 |
| 学习率 | 1e-4 | 优化器设置 |
| 推理批量 | 32 | 线上评测速度 |

## 性能指标

| 指标 | 值 | 说明 |
|------|-----|------|
| WER | 0.3994 | 词错误率（越低越好） |
| 排名 | 81 | DrivenData 平台排名 |
| 训练时间 | ~15h | RTX 3090 上的训练时间 |
| 模型大小 | 1.2GB | submission.zip 大小 |
| 推理时间 | <2小时 | DrivenData A100 上的评测时间 |
