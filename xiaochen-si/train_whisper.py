"""
训练脚本：Whisper large-v3-turbo LoRA 微调
用于 DrivenData 儿童语音识别挑战赛 - Word Track（第二模型，用于 ROVER 融合）

核心创新：
  1. 使用 HuggingFace transformers + peft LoRA
  2. Whisper 是 encoder-decoder attention 架构，与 Parakeet (transducer) 互补
  3. LoRA rank=32 应用于 encoder+decoder 所有 attention 层
  4. 速度扰动 + 噪声增强在 data collator 中在线执行

用法:
    # 快速测试
    python train_whisper.py --sample 200 --max-steps 50

    # 正式训练 (A100 80GB)
    python train_whisper.py \\
        --model openai/whisper-large-v3-turbo \\
        --lora-rank 32 --lora-alpha 64 \\
        --epochs 5 --batch-size 8 --lr 1e-4 \\
        --speed-perturb --noise-augment
"""

import argparse
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from loguru import logger

from asr_benchmark.config import DATA_ROOT, PROJECT_ROOT
from asr_benchmark.data_utils import (
    filter_data,
    load_all_transcripts,
    split_by_child,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Whisper for children's speech")

    # Model
    parser.add_argument("--model", type=str, default="openai/whisper-large-v3-turbo",
                        help="Whisper model name")

    # LoRA
    parser.add_argument("--lora-rank", type=int, default=32,
                        help="LoRA rank (default: 32)")
    parser.add_argument("--lora-alpha", type=int, default=64,
                        help="LoRA alpha (default: 64)")
    parser.add_argument("--lora-dropout", type=float, default=0.05,
                        help="LoRA dropout (default: 0.05)")

    # Training
    parser.add_argument("--epochs", type=int, default=5,
                        help="Training epochs (default: 5)")
    parser.add_argument("--max-steps", type=int, default=-1,
                        help="Max training steps (-1 = use epochs)")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Per-device batch size (default: 8)")
    parser.add_argument("--grad-accum", type=int, default=4,
                        help="Gradient accumulation steps (default: 4)")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate (default: 1e-4)")
    parser.add_argument("--warmup-ratio", type=float, default=0.1,
                        help="Warmup ratio (default: 0.1)")
    parser.add_argument("--precision", type=str, default="bf16",
                        choices=["bf16", "fp16", "fp32"])

    # Data
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--max-duration", type=float, default=30.0)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=4)

    # Augmentation
    parser.add_argument("--speed-perturb", action="store_true")
    parser.add_argument("--noise-augment", action="store_true")

    # Eval
    parser.add_argument("--skip-eval", action="store_true")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ChildSpeechDataset(torch.utils.data.Dataset):
    """Whisper-compatible dataset for children's speech."""

    def __init__(
        self,
        df,
        feature_extractor,
        tokenizer,
        speed_perturb: bool = False,
        noise_augment: bool = False,
        noise_dir: Optional[Path] = None,
        max_length: int = 448,
    ):
        self.entries = df.to_dict("records")
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.speed_perturb = speed_perturb
        self.noise_augment = noise_augment
        self.max_length = max_length

        # Load noise file paths
        self.noise_files = []
        if noise_augment:
            if noise_dir is None:
                noise_dir = DATA_ROOT / "raw" / "noise" / "audio"
            if noise_dir.exists():
                self.noise_files = sorted(noise_dir.glob("*.flac"))
                logger.info(f"Loaded {len(self.noise_files)} noise files for augmentation")

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        import librosa
        import soundfile as sf

        # Try loading this sample; on any error, fall back to a random other sample
        for attempt in range(5):
            try:
                return self._load_sample(idx if attempt == 0 else random.randint(0, len(self.entries) - 1))
            except Exception:
                continue
        # Last resort: return a tiny silent sample
        dummy_audio = np.zeros(16000, dtype=np.float32)
        inputs = self.feature_extractor(dummy_audio, sampling_rate=16000, return_tensors="np")
        labels = self.tokenizer("", return_tensors="np", padding=False).input_ids[0]
        return {"input_features": inputs.input_features[0], "labels": labels}

    def _load_sample(self, idx):
        import librosa
        import soundfile as sf

        entry = self.entries[idx]
        audio_path = entry["audio_path"]
        text = entry["orthographic_text"]

        # Load audio at 16kHz mono - use soundfile first (faster, handles FLAC well)
        try:
            audio, sr = sf.read(audio_path, dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if sr != 16000:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        except Exception:
            audio, sr = librosa.load(audio_path, sr=16000, mono=True)

        # Speed perturbation
        if self.speed_perturb and random.random() < 0.5:
            speed_factor = random.uniform(0.9, 1.1)
            audio = librosa.effects.time_stretch(audio, rate=speed_factor)

        # Noise augmentation
        if self.noise_augment and self.noise_files and random.random() < 0.3:
            noise_file = random.choice(self.noise_files)
            noise, _ = librosa.load(str(noise_file), sr=16000, mono=True)
            # Trim or tile noise to match audio length
            if len(noise) < len(audio):
                repeats = (len(audio) // len(noise)) + 1
                noise = np.tile(noise, repeats)
            noise = noise[:len(audio)]
            # Random SNR between 5-25 dB
            snr_db = random.uniform(5, 25)
            audio_power = np.mean(audio ** 2)
            noise_power = np.mean(noise ** 2)
            if noise_power > 0:
                scale = np.sqrt(audio_power / (noise_power * (10 ** (snr_db / 10))))
                audio = audio + scale * noise

        # Normalize
        if np.abs(audio).max() > 0:
            audio = audio / np.abs(audio).max() * 0.95

        # Extract features
        inputs = self.feature_extractor(
            audio, sampling_rate=16000, return_tensors="np"
        )
        input_features = inputs.input_features[0]

        # Tokenize labels
        labels = self.tokenizer(
            text,
            return_tensors="np",
            padding=False,
            truncation=True,
            max_length=self.max_length,
        ).input_ids[0]

        return {
            "input_features": input_features,
            "labels": labels,
        }


# ---------------------------------------------------------------------------
# Data Collator
# ---------------------------------------------------------------------------

@dataclass
class DataCollatorSpeechSeq2Seq:
    """Custom collator for Whisper seq2seq training."""
    feature_extractor: Any
    tokenizer: Any
    padding: str = "longest"

    def __call__(self, features):
        import torch

        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.feature_extractor.pad(input_features, return_tensors="pt")

        # Pad labels
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.tokenizer.pad(label_features, return_tensors="pt")

        # Replace padding token id with -100 so it's ignored by loss
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # Remove the BOS token if Whisper adds one
        if (labels[:, 0] == self.tokenizer.bos_token_id).all():
            labels = labels[:, 1:]

        # Only return input_features and labels (no input_ids)
        return {"input_features": batch["input_features"], "labels": labels}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args):
    from peft import LoraConfig, get_peft_model, TaskType
    from transformers import (
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        WhisperForConditionalGeneration,
        WhisperFeatureExtractor,
        WhisperProcessor,
        WhisperTokenizer,
    )

    # --- Load data ---
    logger.info("Loading data...")
    df = load_all_transcripts()
    df = filter_data(df, max_duration=args.max_duration)

    if args.sample:
        df = df.sample(args.sample, random_state=42)

    train_df, val_df = split_by_child(df, val_ratio=args.val_ratio)

    # --- Load model ---
    logger.info(f"Loading Whisper model: {args.model}")
    processor = WhisperProcessor.from_pretrained(args.model)
    feature_extractor = processor.feature_extractor
    tokenizer = processor.tokenizer

    model = WhisperForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if args.precision == "bf16" else torch.float32,
    )

    # Force English language and transcribe task
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.generation_config.language = "en"
    model.generation_config.task = "transcribe"

    # --- Apply LoRA ---
    logger.info(f"Applying LoRA (rank={args.lora_rank}, alpha={args.lora_alpha})")
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2"],
        task_type=TaskType.SEQ_2_SEQ_LM,
    )
    # Patch Whisper forward BEFORE wrapping with PEFT to filter unexpected kwargs
    _valid_keys = {"input_features", "labels", "decoder_input_ids", "attention_mask",
                   "decoder_attention_mask", "head_mask", "decoder_head_mask",
                   "cross_attn_head_mask", "encoder_outputs", "past_key_values",
                   "decoder_inputs_embeds", "use_cache", "output_attentions",
                   "output_hidden_states", "return_dict", "cache_position"}
    _orig_fwd = WhisperForConditionalGeneration.forward
    def _safe_forward(self, **kwargs):
        kwargs = {k: v for k, v in kwargs.items() if k in _valid_keys}
        return _orig_fwd(self, **kwargs)
    WhisperForConditionalGeneration.forward = _safe_forward

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # --- Create datasets ---
    logger.info("Creating datasets...")
    train_dataset = ChildSpeechDataset(
        train_df, feature_extractor, tokenizer,
        speed_perturb=args.speed_perturb,
        noise_augment=args.noise_augment,
    )
    val_dataset = ChildSpeechDataset(
        val_df, feature_extractor, tokenizer,
        speed_perturb=False,
        noise_augment=False,
    )

    data_collator = DataCollatorSpeechSeq2Seq(
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
    )

    # --- Training args ---
    output_dir = str(PROJECT_ROOT / "models" / "whisper_lora")

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        bf16=args.precision == "bf16",
        fp16=args.precision == "fp16",
        eval_strategy="steps",
        eval_steps=1000,
        save_strategy="steps",
        save_steps=1000,
        save_total_limit=3,
        logging_steps=50,
        predict_with_generate=True,
        generation_max_length=448,
        report_to="tensorboard",
        dataloader_num_workers=args.num_workers,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False,
        label_names=["labels"],
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )

    # --- Trainer ---
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
    )

    logger.info("Starting Whisper LoRA training...")
    trainer.train()
    logger.success("Whisper training complete!")

    # Save LoRA weights
    lora_save_path = os.path.join(output_dir, "lora_adapter")
    model.save_pretrained(lora_save_path)
    processor.save_pretrained(lora_save_path)
    logger.info(f"Saved LoRA adapter to {lora_save_path}")

    # Also save merged model for easier inference
    logger.info("Merging LoRA weights into base model...")
    merged_model = model.merge_and_unload()
    merged_save_path = os.path.join(output_dir, "merged_model")
    merged_model.save_pretrained(merged_save_path)
    processor.save_pretrained(merged_save_path)
    logger.info(f"Saved merged model to {merged_save_path}")

    return output_dir


def evaluate_whisper(model_path: str, args):
    """Evaluate the trained Whisper model."""
    from transformers import (
        WhisperForConditionalGeneration,
        WhisperProcessor,
    )
    from transformers.models.whisper.english_normalizer import EnglishTextNormalizer
    from asr_benchmark.score import english_spelling_normalizer, score_wer
    import librosa

    logger.info("Evaluating Whisper model...")

    # Load merged model
    merged_path = os.path.join(model_path, "merged_model")
    if os.path.exists(merged_path):
        model_load_path = merged_path
    else:
        model_load_path = model_path

    processor = WhisperProcessor.from_pretrained(model_load_path)
    model = WhisperForConditionalGeneration.from_pretrained(
        model_load_path, torch_dtype=torch.bfloat16
    ).cuda()

    # Load validation data
    df = load_all_transcripts()
    df = filter_data(df, max_duration=args.max_duration)
    if args.sample:
        df = df.sample(args.sample, random_state=42)
    _, val_df = split_by_child(df, val_ratio=args.val_ratio)

    # Run inference
    predictions = []
    references = []

    model.eval()
    for i, (_, row) in enumerate(val_df.iterrows()):
        audio, sr = librosa.load(row["audio_path"], sr=16000, mono=True)
        inputs = processor.feature_extractor(
            audio, sampling_rate=16000, return_tensors="pt"
        ).input_features.to("cuda", dtype=torch.bfloat16)

        with torch.no_grad():
            predicted_ids = model.generate(inputs, max_new_tokens=448)

        text = processor.tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        predictions.append(text.strip())
        references.append(row["orthographic_text"])

        if (i + 1) % 500 == 0:
            logger.info(f"Processed {i + 1}/{len(val_df)}")

    normalizer = EnglishTextNormalizer(english_spelling_normalizer)
    filtered = [(r, p) for r, p in zip(references, predictions) if normalizer(r) != ""]
    refs_f, preds_f = zip(*filtered)

    wer = score_wer(refs_f, preds_f)
    logger.success(f"Whisper Validation WER: {wer:.4f}")

    for ref, pred in zip(refs_f[:10], preds_f[:10]):
        logger.info(f"  REF:  {ref}")
        logger.info(f"  PRED: {pred}")

    return wer


def main():
    args = parse_args()

    logger.info("=" * 70)
    logger.info("Children's ASR - Whisper LoRA Fine-tuning")
    logger.info("=" * 70)
    logger.info(f"Model: {args.model}")
    logger.info(f"LoRA: rank={args.lora_rank}, alpha={args.lora_alpha}")
    logger.info(f"Augmentation: speed={args.speed_perturb}, noise={args.noise_augment}")

    output_dir = train(args)

    if not args.skip_eval:
        evaluate_whisper(output_dir, args)

    logger.success("All done!")


if __name__ == "__main__":
    main()
