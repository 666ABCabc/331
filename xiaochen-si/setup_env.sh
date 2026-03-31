#!/bin/bash
# 环境配置脚本 - 在云GPU服务器上运行
# 用法: bash setup_env.sh

set -euo pipefail

echo "=========================================="
echo "Children's ASR Challenge - 环境配置"
echo "=========================================="

# 检查 Python 版本
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# 安装 uv（如果没有）
if ! command -v uv &> /dev/null; then
    echo "Installing uv package manager..."
    pip install uv
fi

# 创建虚拟环境
echo "Creating virtual environment..."
uv venv --python 3.11 .venv || python3 -m venv .venv

echo "Activating environment..."
source .venv/bin/activate

# 安装 PyTorch（CUDA 12.6）
echo "Installing PyTorch with CUDA support..."
uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu126

# 安装 NeMo 和其他依赖
echo "Installing NeMo toolkit and dependencies..."
uv pip install "nemo-toolkit[asr]>=2.0.0"
uv pip install "transformers>=4.40.0"
uv pip install "lightning>=2.0.0"
uv pip install "omegaconf>=2.3.0"
uv pip install "librosa>=0.10.0"
uv pip install "pandas>=2.0.0"
uv pip install "scikit-learn>=1.3.0"
uv pip install "loguru>=0.7.0"
uv pip install "jiwer>=3.0.0"
uv pip install "matplotlib>=3.7.0"

# 验证安装
echo ""
echo "Verifying installation..."
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA device: {torch.cuda.get_device_name(0)}')
    print(f'GPU memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')

import nemo.collections.asr as nemo_asr
print(f'NeMo ASR: OK')

from transformers.models.whisper.english_normalizer import EnglishTextNormalizer
print(f'Transformers: OK')

print()
print('All dependencies installed successfully!')
"

echo ""
echo "=========================================="
echo "环境配置完成！"
echo "=========================================="
echo ""
echo "下一步:"
echo "  1. 激活环境:  source .venv/bin/activate"
echo "  2. 下载数据到 data/raw/ 目录"
echo "  3. 开始训练:  python train_orthographic.py --max-steps 5000 --batch-size 32"
