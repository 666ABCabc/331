# childrens-speech-recognition-benchmark-pub

python_version := "3.11"

# List available commands (default when you run `just`)
default:
    just --list

# Install Python dependencies
requirements:
    uv sync

# Delete all compiled Python files
clean:
    find . -type f -name "*.py[co]" -delete
    find . -type d -name "__pycache__" -delete

# Lint using ruff (use `just format` to do formatting)
lint:
    ruff format --check
    ruff check

# Format source code with ruff
format:
    ruff check --fix
    ruff format

# Run notebooks/phonetic.ipynb to train the phonetic Wav2Vec2 model
train-phonetic:
    uv run papermill notebooks/phonetic.ipynb notebooks/phonetic.ipynb --log-output

# Run notebooks/orthographic.ipynb to train the orthographic NeMo adapter model
train-orthographic:
    uv run papermill notebooks/orthographic.ipynb notebooks/orthographic.ipynb --log-output

# Run inference using data-demo/phonetic/ to test phonetic submission
test-phonetic:
    uv run phonetic_submission/main.py models/wav2vec2-phonetic-final/ data-demo/phonetic/utterance_metadata.jsonl

# Run inference using data-demo/word/ to test orthographic submission
test-orthographic:
    uv run orthographic_submission/main.py models/orthographic_benchmark_nemo/ASR-Adapter-best.nemo data-demo/word/utterance_metadata.jsonl

# Create zip for phonetic submission to challenge
pack-phonetic:
    rm -f phonetic_submission.zip && \
    (cd phonetic_submission && zip -r ../phonetic_submission.zip main.py) && \
    (cd models && zip -r ../phonetic_submission.zip wav2vec2-phonetic-final/)

# Create zip for orthographic submission to challenge
pack-orthographic:
    #!/usr/bin/env bash
    set -euo pipefail
    latest=$(ls -td models/orthographic_benchmark_nemo/ASR-Adapter/*/checkpoints/ASR-Adapter.nemo | head -1)
    ln -sf "${latest#models/orthographic_benchmark_nemo/}" models/orthographic_benchmark_nemo/ASR-Adapter-best.nemo
    echo "Updated ASR-Adapter-best.nemo -> $latest"
    rm -f orthographic_submission.zip
    (cd orthographic_submission && zip -r ../orthographic_submission.zip main.py)
    (cd models/orthographic_benchmark_nemo && zip -r ../../orthographic_submission.zip ASR-Adapter-best.nemo)

# Set up Python interpreter environment
create-environment:
    uv venv --python {{ python_version }}
    echo ">>> New uv virtual environment created. Activate with:"
    echo ">>> Windows: .\\.venv\\Scripts\\activate"
    echo ">>> Unix/macOS: source ./.venv/bin/activate"
