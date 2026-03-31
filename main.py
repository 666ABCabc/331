"""Whisper-v3-turbo inference for DrivenData Children's ASR Word Track.

Runtime: A100 80GB, 24 vCPU, 220GB RAM, 2-hour limit, no network.
"""

import json
import os
import sys
import time
from itertools import islice
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
from loguru import logger
from transformers import WhisperForConditionalGeneration, WhisperProcessor

BATCH_SIZE = 32
SCRIPT_DIR = Path(__file__).resolve().parent


def batched(iterable, n):
    iterator = iter(iterable)
    while batch := tuple(islice(iterator, n)):
        yield batch


def load_audio(path, sr=16000):
    """Load audio file robustly."""
    try:
        audio, file_sr = sf.read(path, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if file_sr != sr:
            audio = librosa.resample(audio, orig_sr=file_sr, target_sr=sr)
        return audio
    except Exception:
        try:
            audio, _ = librosa.load(path, sr=sr, mono=True)
            return audio
        except Exception:
            return np.zeros(sr, dtype=np.float32)  # 1s silence fallback


def main(data_manifest: Path):
    logger.info("Torch: {} | CUDA: {}", torch.__version__, torch.cuda.is_available())
    torch.set_float32_matmul_precision("high")

    # Load model
    model_path = SCRIPT_DIR / "whisper_merged"
    logger.info("Loading Whisper model from: {}", model_path)
    t0 = time.time()

    processor = WhisperProcessor.from_pretrained(str(model_path))
    model = WhisperForConditionalGeneration.from_pretrained(
        str(model_path), dtype=torch.bfloat16
    ).cuda().eval()
    model.generation_config.language = "en"
    model.generation_config.task = "transcribe"

    logger.info("Model loaded in {:.1f}s", time.time() - t0)

    # Load test data
    data_dir = data_manifest.parent
    with data_manifest.open("r") as f:
        items = [json.loads(line) for line in f]
    items.sort(key=lambda x: x["audio_duration_sec"], reverse=True)
    logger.info("Test set: {} utterances", len(items))

    # Run inference
    predictions = {}
    step = max(1, len(items) // 20)
    next_log = step
    processed = 0

    for batch_items in batched(items, BATCH_SIZE):
        batch_audio = [load_audio(str(data_dir / item["audio_path"])) for item in batch_items]

        inputs = processor.feature_extractor(
            batch_audio, sampling_rate=16000, return_tensors="pt", padding=True
        ).input_features.to("cuda", dtype=torch.bfloat16)

        with torch.no_grad():
            predicted_ids = model.generate(inputs, max_new_tokens=256)

        texts = processor.tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)

        for item, text in zip(batch_items, texts):
            predictions[item["utterance_id"]] = text.strip()

        processed += len(batch_items)
        if processed >= next_log:
            logger.info("Processed {}/{}", processed, len(items))
            next_log += step

    logger.info("Inference done: {} utterances", len(predictions))

    # Write submission
    submission_format = data_dir / "submission_format.jsonl"
    submission_path = Path("submission") / "submission.jsonl"
    submission_path.parent.mkdir(parents=True, exist_ok=True)

    with submission_format.open("r") as fr, submission_path.open("w") as fw:
        for line in fr:
            item = json.loads(line)
            uid = item["utterance_id"]
            item["orthographic_text"] = predictions.get(uid, "")
            fw.write(json.dumps(item) + "\n")

    logger.info("Wrote {}", submission_path)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        manifest = Path(sys.argv[1])
    else:
        manifest = Path("data/utterance_metadata.jsonl")
    main(manifest)
