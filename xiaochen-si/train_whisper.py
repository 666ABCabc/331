"""
Training script: Whisper large-v3-turbo LoRA fine-tuning
For DrivenData Children's ASR Word Track challenge (second model for ROVER ensemble)

Core innovations:
  1. Using HuggingFace transformers + peft LoRA for parameter-efficient fine-tuning
  2. Whisper's encoder-decoder attention architecture complements Parakeet (transducer)
  3. LoRA rank=32 applied to all attention layers in encoder and decoder
  4. Online speed perturbation + noise augmentation implemented in data collator

Usage:
    # Quick test
    python train_whisper.py --sample 200 --max-steps 50

    # Full training (A100 80GB)
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
    """Parse command-line arguments for training."""
    parser = argparse.ArgumentParser(description="Train Whisper for children's speech")

    # Model configuration
    parser.add_argument("--model", type=str, default="openai/whisper-large-v3-turbo",
                        help="Whisper model name")

    # LoRA configuration
    parser.add_argument("--lora-rank", type=int, default=32,
                        help="LoRA rank (default: 32)")
    parser.add_argument("--lora-alpha", type=int, default=64,
                        help="LoRA alpha (default: 64)")
    parser.add_argument("--lora-dropout", type=float, default=0.05,
                        help="LoRA dropout (default: 0.05)")

    # Training hyperparameters
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
                        choices=["bf16", "fp16", "fp32"],
                        help="Training precision")

    # Data configuration
    parser.add_argument("--sample", type=int, default=None,
                        help="Sample size for quick testing")
    parser.add_argument("--max-duration", type=float, default=30.0,
                        help="Maximum audio duration in seconds")
    parser.add_argument("--val-ratio", type=float, default=0.1,
                        help="Validation set ratio")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of data loader workers")

    # Data augmentation
    parser.add_argument("--speed-perturb", action="store_true",
                        help="Enable speed perturbation augmentation")
    parser.add_argument("--noise-augment", action="store_true",
                        help="Enable noise augmentation")

    # Evaluation
    parser.add_argument("--skip-eval", action="store_true",
                        help="Skip evaluation after training")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ChildSpeechDataset(torch.utils.data.Dataset):
    """Whisper-compatible dataset for children's speech.
    
    Handles audio loading, augmentation, and feature extraction for training.
    Implements robust error handling to ensure training continues even with problematic audio files.
    """

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
        """Initialize the dataset.
        
        Args:
            df: DataFrame containing audio paths and transcripts
            feature_extractor: Whisper feature extractor
            tokenizer: Whisper tokenizer
            speed_perturb: Whether to enable speed perturbation augmentation
            noise_augment: Whether to enable noise augmentation
            noise_dir: Directory containing noise files for augmentation
            max_length: Maximum token length for transcripts
        """
        self.entries = df.to_dict("records")
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.speed_perturb = speed_perturb
        self.noise_augment = noise_augment
        self.max_length = max_length

        # Load noise file paths for augmentation
        self.noise_files = []
        if noise_augment:
            if noise_dir is None:
                noise_dir = DATA_ROOT / "raw" / "noise" / "audio"
            if noise_dir.exists():
                self.noise_files = sorted(noise_dir.glob("*.flac"))
                logger.info(f"Loaded {len(self.noise_files)} noise files for augmentation")

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.entries)

    def __getitem__(self, idx):
        """Get a sample from the dataset with robust error handling.
        
        If loading fails, tries up to 5 times with random samples as fallback.
        If all attempts fail, returns a dummy silent sample.
        
        Args:
            idx: Index of the sample to load
            
        Returns:
            Dictionary with input features and labels
        """
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
        """Load and process a single audio sample.
        
        Args:
            idx: Index of the sample to load
            
        Returns:
            Dictionary with input features and labels
        """
        import librosa
        import soundfile as sf

        entry = self.entries[idx]
        audio_path = entry["audio_path"]
        text = entry["orthographic_text"]

        # Load audio at 16kHz mono - use soundfile first (faster, handles FLAC well)
        try:
            audio, sr = sf.read(audio_path, dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)  # Convert to mono
            if sr != 16000:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        except Exception:
            # Fallback to librosa if soundfile fails
            audio, sr = librosa.load(audio_path, sr=16000, mono=True)

        # Speed perturbation augmentation
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

        # Normalize audio to [-0.95, 0.95]
        if np.abs(audio).max() > 0:
            audio = audio / np.abs(audio).max() * 0.95

        # Extract mel features for Whisper
        inputs = self.feature_extractor(
            audio, sampling_rate=16000, return_tensors="np"
        )
        input_features = inputs.input_features[0]

        # Tokenize transcript for training
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
    """Custom collator for Whisper seq2seq training.
    
    Handles batching of audio features and labels, including padding and special token handling.
    """
    feature_extractor: Any  # Whisper feature extractor
    tokenizer: Any  # Whisper tokenizer
    padding: str = "longest"  # Padding strategy

    def __call__(self, features):
        """Collate a list of features into a batch.
        
        Args:
            features: List of dictionaries containing input features and labels
            
        Returns:
            Dictionary with batched input features and labels
        """
        import torch

        # Pad input features (mel spectrograms)
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.feature_extractor.pad(input_features, return_tensors="pt")

        # Pad labels (tokenized transcripts)
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.tokenizer.pad(label_features, return_tensors="pt")

        # Replace padding token id with -100 so it's ignored by loss function
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # Remove the BOS token if Whisper adds one (common in some Whisper versions)
        if (labels[:, 0] == self.tokenizer.bos_token_id).all():
            labels = labels[:, 1:]

        # Only return input_features and labels (no input_ids)
        return {"input_features": batch["input_features"], "labels": labels}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args):
    """Train Whisper model with LoRA for children's speech recognition.
    
    Implements parameter-efficient fine-tuning using LoRA (Low-Rank Adaptation)
    with HuggingFace's Seq2SeqTrainer for streamlined training workflow.
    
    Args:
        args: Command-line arguments containing training configuration
        
    Returns:
        str: Path to the directory where trained models are saved
    """
    from peft import LoraConfig, get_peft_model, TaskType
    from transformers import (
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        WhisperForConditionalGeneration,
        WhisperFeatureExtractor,
        WhisperProcessor,
        WhisperTokenizer,
    )

    # --- Load and prepare data ---
    logger.info("Loading data...")
    # Load all transcript data
    df = load_all_transcripts()
    # Filter out audio files exceeding maximum duration
    df = filter_data(df, max_duration=args.max_duration)

    # Sample data for quick testing if specified
    if args.sample:
        df = df.sample(args.sample, random_state=42)

    # Split data into training and validation sets by child (to avoid data leakage)
    train_df, val_df = split_by_child(df, val_ratio=args.val_ratio)

    # --- Load Whisper model and processor ---
    logger.info(f"Loading Whisper model: {args.model}")
    # Load processor (combines feature extractor and tokenizer)
    processor = WhisperProcessor.from_pretrained(args.model)
    feature_extractor = processor.feature_extractor
    tokenizer = processor.tokenizer

    # Load the Whisper model with specified precision
    model = WhisperForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if args.precision == "bf16" else torch.float32,
    )

    # Configure model for English transcription task
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.generation_config.language = "en"
    model.generation_config.task = "transcribe"

    # --- Apply LoRA for parameter-efficient fine-tuning ---
    logger.info(f"Applying LoRA (rank={args.lora_rank}, alpha={args.lora_alpha})")
    lora_config = LoraConfig(
        r=args.lora_rank,  # Rank of low-rank matrices
        lora_alpha=args.lora_alpha,  # Scaling factor
        lora_dropout=args.lora_dropout,  # Dropout for LoRA layers
        # Target all attention and feed-forward layers in encoder/decoder
        target_modules=["q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2"],
        task_type=TaskType.SEQ_2_SEQ_LM,  # Task type for sequence-to-sequence language modeling
    )
    
    # Patch Whisper's forward method to filter unexpected kwargs (prevents PEFT compatibility issues)
    _valid_keys = {"input_features", "labels", "decoder_input_ids", "attention_mask",
                   "decoder_attention_mask", "head_mask", "decoder_head_mask",
                   "cross_attn_head_mask", "encoder_outputs", "past_key_values",
                   "decoder_inputs_embeds", "use_cache", "output_attentions",
                   "output_hidden_states", "return_dict", "cache_position"}
    _orig_fwd = WhisperForConditionalGeneration.forward
    def _safe_forward(self, **kwargs):
        # Filter out unexpected keyword arguments
        kwargs = {k: v for k, v in kwargs.items() if k in _valid_keys}
        return _orig_fwd(self, **kwargs)
    WhisperForConditionalGeneration.forward = _safe_forward

    # Apply LoRA to the model
    model = get_peft_model(model, lora_config)
    # Print trainable parameter information
    model.print_trainable_parameters()

    # --- Create datasets with augmentation ---
    logger.info("Creating datasets...")
    # Training dataset with data augmentation
    train_dataset = ChildSpeechDataset(
        train_df, feature_extractor, tokenizer,
        speed_perturb=args.speed_perturb,  # Enable speed perturbation
        noise_augment=args.noise_augment,  # Enable noise augmentation
    )
    # Validation dataset without augmentation
    val_dataset = ChildSpeechDataset(
        val_df, feature_extractor, tokenizer,
        speed_perturb=False,
        noise_augment=False,
    )

    # Create data collator for batching
    data_collator = DataCollatorSpeechSeq2Seq(
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
    )

    # --- Configure training arguments ---
    output_dir = str(PROJECT_ROOT / "models" / "whisper_lora")

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,  # Directory to save model checkpoints
        num_train_epochs=args.epochs,  # Number of training epochs
        max_steps=args.max_steps,  # Maximum training steps (overrides epochs)
        per_device_train_batch_size=args.batch_size,  # Batch size per device
        per_device_eval_batch_size=args.batch_size,  # Batch size for evaluation
        gradient_accumulation_steps=args.grad_accum,  # Gradient accumulation steps
        learning_rate=args.lr,  # Learning rate
        warmup_ratio=args.warmup_ratio,  # Learning rate warmup ratio
        lr_scheduler_type="cosine",  # Cosine learning rate scheduler
        bf16=args.precision == "bf16",  # Use bfloat16 precision if specified
        fp16=args.precision == "fp16",  # Use float16 precision if specified
        eval_strategy="steps",  # Evaluate at specified steps
        eval_steps=1000,  # Evaluate every 1000 steps
        save_strategy="steps",  # Save checkpoints at specified steps
        save_steps=1000,  # Save every 1000 steps
        save_total_limit=3,  # Keep only the last 3 checkpoints
        logging_steps=50,  # Log every 50 steps
        predict_with_generate=True,  # Use generate for evaluation
        generation_max_length=448,  # Maximum generation length
        report_to="tensorboard",  # Report metrics to TensorBoard
        dataloader_num_workers=args.num_workers,  # Number of data loader workers
        gradient_checkpointing=True,  # Enable gradient checkpointing to save memory
        gradient_checkpointing_kwargs={"use_reentrant": False},  # Non-reentrant checkpointing
        remove_unused_columns=False,  # Keep all columns in the dataset
        label_names=["labels"],  # Specify label column name
        load_best_model_at_end=True,  # Load best model at the end of training
        metric_for_best_model="eval_loss",  # Metric to use for model selection
    )

    # --- Initialize and run trainer ---
    trainer = Seq2SeqTrainer(
        model=model,  # Model to train
        args=training_args,  # Training configuration
        train_dataset=train_dataset,  # Training dataset
        eval_dataset=val_dataset,  # Validation dataset
        data_collator=data_collator,  # Data collator for batching
        processing_class=tokenizer,  # Tokenizer for processing
    )

    logger.info("Starting Whisper LoRA training...")
    # Run training
    trainer.train()
    logger.success("Whisper training complete!")

    # --- Save trained models ---
    # Save LoRA adapter weights
    lora_save_path = os.path.join(output_dir, "lora_adapter")
    model.save_pretrained(lora_save_path)
    processor.save_pretrained(lora_save_path)
    logger.info(f"Saved LoRA adapter to {lora_save_path}")

    # Save merged model (LoRA weights merged into base model) for easier inference
    logger.info("Merging LoRA weights into base model...")
    merged_model = model.merge_and_unload()
    merged_save_path = os.path.join(output_dir, "merged_model")
    merged_model.save_pretrained(merged_save_path)
    processor.save_pretrained(merged_save_path)
    logger.info(f"Saved merged model to {merged_save_path}")

    return output_dir


def evaluate_whisper(model_path: str, args):
    """Evaluate the trained Whisper model on validation data.
    
    Args:
        model_path: Path to the directory containing the trained model
        args: Command-line arguments containing evaluation configuration
        
    Returns:
        float: Word Error Rate (WER) on validation data
    """
    from transformers import (
        WhisperForConditionalGeneration,
        WhisperProcessor,
    )
    from transformers.models.whisper.english_normalizer import EnglishTextNormalizer
    from asr_benchmark.score import english_spelling_normalizer, score_wer
    import librosa

    logger.info("Evaluating Whisper model...")

    # Load merged model (LoRA weights merged into base model) if available
    merged_path = os.path.join(model_path, "merged_model")
    if os.path.exists(merged_path):
        model_load_path = merged_path
    else:
        model_load_path = model_path

    # Load processor and model
    processor = WhisperProcessor.from_pretrained(model_load_path)
    model = WhisperForConditionalGeneration.from_pretrained(
        model_load_path, torch_dtype=torch.bfloat16
    ).cuda()

    # Load and prepare validation data
    df = load_all_transcripts()
    df = filter_data(df, max_duration=args.max_duration)
    if args.sample:
        df = df.sample(args.sample, random_state=42)
    _, val_df = split_by_child(df, val_ratio=args.val_ratio)

    # Run inference on validation data
    predictions = []  # List to store predicted transcripts
    references = []  # List to store reference transcripts

    model.eval()  # Set model to evaluation mode
    for i, (_, row) in enumerate(val_df.iterrows()):
        # Load and resample audio to 16kHz
        audio, sr = librosa.load(row["audio_path"], sr=16000, mono=True)
        # Extract features and move to CUDA
        inputs = processor.feature_extractor(
            audio, sampling_rate=16000, return_tensors="pt"
        ).input_features.to("cuda", dtype=torch.bfloat16)

        # Generate transcriptions without gradient calculation
        with torch.no_grad():
            predicted_ids = model.generate(inputs, max_new_tokens=448)

        # Decode predicted tokens to text
        text = processor.tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        predictions.append(text.strip())
        references.append(row["orthographic_text"])

        # Log progress every 500 samples
        if (i + 1) % 500 == 0:
            logger.info(f"Processed {i + 1}/{len(val_df)}")

    # Normalize text and filter out empty references
    normalizer = EnglishTextNormalizer(english_spelling_normalizer)
    filtered = [(r, p) for r, p in zip(references, predictions) if normalizer(r) != ""]
    refs_f, preds_f = zip(*filtered)

    # Calculate Word Error Rate
    wer = score_wer(refs_f, preds_f)
    logger.success(f"Whisper Validation WER: {wer:.4f}")

    # Log sample predictions for inspection
    for ref, pred in zip(refs_f[:10], preds_f[:10]):
        logger.info(f"  REF:  {ref}")
        logger.info(f"  PRED: {pred}")

    return wer


def main():
    """Main function to run Whisper LoRA fine-tuning and evaluation.
    
    Parses command-line arguments, runs training, and evaluates the model.
    """
    # Parse command-line arguments
    args = parse_args()

    # Log training configuration
    logger.info("=" * 70)
    logger.info("Children's ASR - Whisper LoRA Fine-tuning")
    logger.info("=" * 70)
    logger.info(f"Model: {args.model}")
    logger.info(f"LoRA: rank={args.lora_rank}, alpha={args.lora_alpha}")
    logger.info(f"Augmentation: speed={args.speed_perturb}, noise={args.noise_augment}")

    # Run training
    output_dir = train(args)

    # Evaluate model if not skipped
    if not args.skip_eval:
        evaluate_whisper(output_dir, args)

    logger.success("All done!")


if __name__ == "__main__":
    main()
