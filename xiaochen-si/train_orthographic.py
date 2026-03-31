
"""
Training script: NeMo Adapter fine-tuning based on Parakeet-TDT-0.6B
for DrivenData Children's Speech Recognition Challenge - Word Track

Usage:
    python train_orthographic.py [--sample N] [--max-steps N] [--batch-size N] [--devices N] [--num-workers N]

Run on cloud GPU server:
    python train_orthographic.py --max-steps 5000 --batch-size 32 --devices 1

Core innovations:
1. Parameter-efficient fine-tuning using NeMo Adapters
2. Adapter architecture for targeted model adaptation
3. Multi-corpus training with DrivenData and TalkBank data
4. Spec augmentation for robustness improvement
5. Efficient batch processing and mixed precision training
"""

import argparse
import json
import os
from pathlib import Path

import lightning.pytorch as pl
import pandas as pd
import torch
from loguru import logger
from nemo.collections.asr.models import ASRModel
from nemo.utils.exp_manager import exp_manager
from nemo.utils.trainer_utils import resolve_trainer_cfg
from omegaconf import OmegaConf, open_dict
from sklearn.model_selection import train_test_split
from transformers.models.whisper.english_normalizer import EnglishTextNormalizer

from asr_benchmark.config import DATA_ROOT, PROJECT_ROOT
from asr_benchmark.nemo_adapter import (
    add_global_adapter_cfg,
    patch_transcribe_lhotse,
    update_model_cfg,
    update_model_config_to_support_adapter,
)
from asr_benchmark.score import english_spelling_normalizer, score_wer


def parse_args():
    parser = argparse.ArgumentParser(description="Train ASR adapter for children's speech")
    parser.add_argument("--sample", type=int, default=None, help="Use a subset of N samples for quick testing")
    parser.add_argument("--max-steps", type=int, default=5000, help="Max training steps (default: 5000)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--devices", type=int, default=1, help="Number of GPUs (default: 1)")
    parser.add_argument("--num-workers", type=int, default=8, help="DataLoader workers (default: 8)")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate (default: 0.001)")
    parser.add_argument("--precision", type=str, default="bf16-mixed", help="Training precision (default: bf16-mixed)")
    parser.add_argument("--clip-max-duration", type=float, default=25.0, help="Max audio duration in seconds")
    parser.add_argument("--val-check-interval", type=int, default=500, help="Validation check interval in steps")
    parser.add_argument("--skip-eval", action="store_true", help="Skip evaluation after training")
    return parser.parse_args()


def read_transcripts(data_dir: Path) -> pd.DataFrame:
    """Read JSONL transcript file into a DataFrame and convert audio paths to absolute paths."""
    transcript_path = data_dir / "train_word_transcripts.jsonl"
    if not transcript_path.exists():
        logger.warning(f"Transcript file not found: {transcript_path}")
        return pd.DataFrame()
    df = pd.read_json(transcript_path, lines=True)
    logger.info(f"Loaded {len(df)} utterance transcripts from {data_dir.name}")
    df["audio_relpath"] = df["audio_path"]
    df["audio_path"] = df["audio_relpath"].map(lambda p: str(data_dir / p))
    return df


def prepare_data(args):
    """Load and prepare training data from multiple corpora.
    
    Args:
        args: Command-line arguments with data configuration
        
    Returns:
        Tuple of (train_manifest_path, val_manifest_path) for NeMo training
    """
    # Create output directory for manifests
    manifest_dir = DATA_ROOT / "processed" / "ortho_dataset"
    train_manifest_path = manifest_dir / "train_manifest.jsonl"
    val_manifest_path = manifest_dir / "val_manifest.jsonl"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    # Load both DrivenData and TalkBank corpora
    df_dd = read_transcripts(DATA_ROOT / "raw" / "drivendata")
    df_tb = read_transcripts(DATA_ROOT / "raw" / "talkbank")
    df = pd.concat([df_dd, df_tb], ignore_index=True)

    if df.empty:
        raise FileNotFoundError(
            "No training data found! Please download data from DrivenData and TalkBank "
            "and place them in data/raw/drivendata/ and data/raw/talkbank/"
        )

    # Log data statistics
    logger.info(f"Total utterances: {len(df)}")
    logger.info(f"Unique children: {df.child_id.nunique()}")
    logger.info(f"Total audio hours: {df.audio_duration_sec.sum() / 3600:.1f}")

    # Convert to NeMo manifest format (required by NeMo ASR)
    df = df[["audio_path", "audio_duration_sec", "orthographic_text"]].rename(
        columns={
            "audio_path": "audio_filepath",
            "audio_duration_sec": "duration",
            "orthographic_text": "text",
        }
    )

    # Remove long audio clips to avoid memory issues
    over_max = df["duration"] > args.clip_max_duration
    logger.info(f"Removing {over_max.sum()} samples with duration > {args.clip_max_duration}s")
    df = df[~over_max]

    # Sample data for quick testing if specified
    if args.sample:
        logger.info(f"Sampling {args.sample} utterances for quick testing")
        df = df.sample(args.sample, random_state=10)

    # Split data into training (80%) and validation (20%) sets
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=10)
    logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}")

    # Save manifests in JSONL format
    train_df.to_json(train_manifest_path, orient="records", lines=True)
    val_df.to_json(val_manifest_path, orient="records", lines=True)

    return train_manifest_path, val_manifest_path


def train(args, train_manifest_path, val_manifest_path):
    """Run NeMo Adapter training for children's speech recognition.
    
    Args:
        args: Command-line arguments with training configuration
        train_manifest_path: Path to training manifest file
        val_manifest_path: Path to validation manifest file
        
    Returns:
        Tuple of (exp_log_dir, cfg) - experiment directory and configuration
    """
    # Set high precision for matrix multiplications
    torch.set_float32_matmul_precision("high")

    # Load NeMo adapter configuration defaults
    yaml_path = PROJECT_ROOT / "asr_benchmark" / "assets" / "asr_adaptation.yaml"
    cfg = OmegaConf.load(yaml_path)

    # Override configuration with command-line arguments
    overrides = OmegaConf.create(
        {
            "model": {
                "pretrained_model": "nvidia/parakeet-tdt-0.6b-v2",  # Base model
                "adapter": {
                    "adapter_name": "asr_children_orthographic",  # Adapter name
                    "adapter_module_name": "encoder",  # Target module for adaptation
                    "linear": {"in_features": 1024},  # Adapter configuration
                },
                "train_ds": {
                    "manifest_filepath": str(train_manifest_path),
                    "batch_size": args.batch_size,
                    "num_workers": args.num_workers,
                    "use_lhotse": False,
                    "channel_selector": "average",
                },
                "validation_ds": {
                    "manifest_filepath": str(val_manifest_path),
                    "batch_size": args.batch_size,
                    "num_workers": args.num_workers,
                    "use_lhotse": False,
                    "channel_selector": "average",
                },
                "optim": {
                    "lr": args.lr,  # Learning rate
                    "weight_decay": 0.0,  # No weight decay
                },
            },
            "trainer": {
                "devices": args.devices,  # Number of GPUs
                "precision": args.precision,  # Mixed precision training
                "strategy": "auto",
                "max_epochs": 1 if args.sample else None,  # Quick testing
                "max_steps": -1 if args.sample else args.max_steps,  # Training steps
                "val_check_interval": 1.0 if args.sample else args.val_check_interval,  # Validation interval
                "enable_progress_bar": True,
            },
            "exp_manager": {
                "exp_dir": str(PROJECT_ROOT / "models" / "orthographic_benchmark_nemo"),  # Output directory
            },
        }
    )

    # Merge configurations
    cfg = OmegaConf.merge(cfg, overrides)

    # Setup PyTorch Lightning Trainer
    logger.info("Setting up Trainer...")
    trainer = pl.Trainer(**resolve_trainer_cfg(cfg.trainer))
    exp_log_dir = exp_manager(trainer, cfg.get("exp_manager", None))

    # Load pretrained Parakeet model with adapter support
    logger.info("Loading pretrained model: nvidia/parakeet-tdt-0.6b-v2")
    model_cfg = ASRModel.from_pretrained(cfg.model.pretrained_model, return_config=True)
    update_model_config_to_support_adapter(model_cfg, cfg)
    model = ASRModel.from_pretrained(
        cfg.model.pretrained_model,
        override_config_path=model_cfg,
        trainer=trainer,
    )

    # Disable CUDA graph decoder (incompatible with current PyTorch)
    with open_dict(model.cfg):
        model.cfg.decoding.greedy.use_cuda_graph_decoder = False
    model.change_decoding_strategy(model.cfg.decoding)

    # Setup training and validation data
    cfg.model.train_ds = update_model_cfg(model.cfg.train_ds, cfg.model.train_ds)
    model.setup_training_data(cfg.model.train_ds)

    cfg.model.validation_ds = update_model_cfg(model.cfg.validation_ds, cfg.model.validation_ds)
    model.setup_multiple_validation_data(cfg.model.validation_ds)

    # Setup optimizer
    model.setup_optimization(cfg.model.optim)

    # Configure spec augmentation for robustness
    if "spec_augment" in cfg.model:
        model.spec_augmentation = model.from_config_dict(cfg.model.spec_augment)
    else:
        model.spec_augmentation = None
        del model.cfg.spec_augment

    # Setup NeMo Adapter
    logger.info("Setting up adapter...")
    with open_dict(cfg.model.adapter):
        adapter_name = cfg.model.adapter.pop("adapter_name")
        adapter_type = cfg.model.adapter.pop("adapter_type")
        adapter_module_name = cfg.model.adapter.pop("adapter_module_name", None)
        adapter_state_dict_name = cfg.model.adapter.pop("adapter_state_dict_name", None)

        adapter_type_cfg = cfg.model.adapter[adapter_type]

        # Format adapter name with module
        if adapter_module_name is not None and ":" not in adapter_name:
            adapter_name = f"{adapter_module_name}:{adapter_name}"

        # Add global adapter configuration if specified
        adapter_global_cfg = cfg.model.adapter.pop(model.adapter_global_cfg_key, None)
        if adapter_global_cfg is not None:
            add_global_adapter_cfg(model, adapter_global_cfg)

    # Add adapter to model
    model.add_adapter(adapter_name, cfg=adapter_type_cfg)
    assert model.is_adapter_available()

    # Enable only the adapter (freeze base model)
    model.set_enabled_adapters(enabled=False)
    model.set_enabled_adapters(adapter_name, enabled=True)

    # Freeze base model and unfreeze only adapter layers
    model.freeze()
    model = model.train()
    model.unfreeze_enabled_adapters()

    # Log parameter statistics
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters:     {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")

    # Run training
    logger.info("Starting training...")
    trainer.fit(model)
    logger.success("Training complete!")

    # Save adapter weights
    if adapter_state_dict_name is not None:
        state_path = exp_log_dir if exp_log_dir is not None else os.getcwd()
        ckpt_path = os.path.join(state_path, "checkpoints")
        if os.path.exists(ckpt_path):
            state_path = ckpt_path
        state_path = os.path.join(state_path, adapter_state_dict_name)
        model.save_adapters(str(state_path))
        logger.info(f"Saved adapter weights to {state_path}")

    return exp_log_dir, cfg


def evaluate(exp_log_dir, cfg, batch_size):
    """Evaluate the trained model on the validation set."""
    logger.info("Starting evaluation...")

    # Find best checkpoint
    nemo_ckpts = sorted((exp_log_dir / "checkpoints").glob("*.nemo"))
    if not nemo_ckpts:
        raise FileNotFoundError(f"No .nemo checkpoints found in {exp_log_dir}/checkpoints/")

    best_ckpt = nemo_ckpts[-1]
    logger.info(f"Loading checkpoint: {best_ckpt}")
    eval_model = ASRModel.restore_from(str(best_ckpt), map_location="cuda")

    with open_dict(eval_model.cfg):
        eval_model.cfg.decoding.greedy.use_cuda_graph_decoder = False
    eval_model.change_decoding_strategy(eval_model.cfg.decoding)
    patch_transcribe_lhotse(eval_model)

    # Load validation data
    val_manifest_path = cfg.model.validation_ds.manifest_filepath
    with open(val_manifest_path) as f:
        val_entries = [json.loads(line) for line in f]

    audio_files = [e["audio_filepath"] for e in val_entries]
    references = [e["text"] for e in val_entries]

    logger.info(f"Running inference on {len(audio_files)} validation utterances...")
    raw = eval_model.transcribe(
        audio_files, batch_size=batch_size, channel_selector="average", verbose=False
    )
    if isinstance(raw, tuple):
        raw = raw[0]

    predictions = [h.text if hasattr(h, "text") else h for h in raw]

    # Score
    normalizer = EnglishTextNormalizer(english_spelling_normalizer)
    filtered = [(r, p) for r, p in zip(references, predictions) if normalizer(r) != ""]
    references_filtered, predictions_filtered = zip(*filtered)

    wer = score_wer(references_filtered, predictions_filtered)
    logger.success(f"Validation WER: {wer:.4f}")

    # Print samples
    logger.info("Sample predictions:")
    for ref, pred in zip(references_filtered[:5], predictions_filtered[:5]):
        logger.info(f"  REF:  {ref}")
        logger.info(f"  PRED: {pred}")

    return wer


def main():
    args = parse_args()

    logger.info("=" * 60)
    logger.info("Children's Speech Recognition - Word Track Training")
    logger.info("=" * 60)

    # Step 1: Prepare data
    logger.info("Step 1: Preparing data...")
    train_manifest_path, val_manifest_path = prepare_data(args)

    # Step 2: Train
    logger.info("Step 2: Training adapter...")
    exp_log_dir, cfg = train(args, train_manifest_path, val_manifest_path)

    # Step 3: Evaluate
    if not args.skip_eval:
        logger.info("Step 3: Evaluating...")
        wer = evaluate(Path(exp_log_dir), cfg, args.batch_size)
    else:
        logger.info("Step 3: Skipping evaluation (--skip-eval)")

    logger.success("All done!")
    logger.info(f"Model checkpoints saved to: {exp_log_dir}")
    logger.info("Next steps:")
    logger.info("  1. Run: python pack_submission.py")
    logger.info("  2. Submit orthographic_submission.zip to DrivenData")


if __name__ == "__main__":
    main()
