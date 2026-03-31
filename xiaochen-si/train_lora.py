"""
训练脚本：Parakeet-TDT-1.1B 深度微调（大 Adapter + 部分 Encoder 解冻）
用于 DrivenData 儿童语音识别挑战赛 - Word Track

核心创新：
  1. 使用 1.1B 模型（比 baseline 的 0.6B 更强）
  2. Adapter dim=256（baseline 仅 32）→ ~1M+ 可训练参数
  3. 可选解冻 encoder 最后 N 层 → 更深层适配
  4. SpecAugment + 速度扰动 + 噪声增强
  5. 按 child_id 分组验证集（更可靠的泛化评估）

用法:
    # 快速测试
    python train_lora.py --sample 200 --max-steps 10

    # 正式训练 (A100 80GB)
    python train_lora.py \\
        --model nvidia/parakeet-tdt-1.1b \\
        --adapter-dim 256 --max-steps 20000 --batch-size 16 --lr 3e-4 \\
        --speed-perturb --noise-augment --spec-augment \\
        --unfreeze-layers 6

    # 分阶段训练: 先 adapter-only，再部分解冻
    python train_lora.py --max-steps 15000 --adapter-dim 256 --spec-augment --speed-perturb
    python train_lora.py --resume <checkpoint> --unfreeze-layers 6 --lr 5e-5 --max-steps 5000
"""

import argparse
import json
import os
from pathlib import Path

import lightning.pytorch as pl
import torch
from loguru import logger
from nemo.collections.asr.models import ASRModel
from nemo.utils.exp_manager import exp_manager
from nemo.utils.trainer_utils import resolve_trainer_cfg
from omegaconf import OmegaConf, open_dict
from transformers.models.whisper.english_normalizer import EnglishTextNormalizer

from asr_benchmark.config import DATA_ROOT, PROJECT_ROOT
from asr_benchmark.data_utils import (
    create_nemo_manifest,
    filter_data,
    get_noise_augment_config,
    get_spec_augment_config,
    get_speed_perturb_config,
    load_all_transcripts,
    split_by_child,
)
from asr_benchmark.nemo_adapter import (
    add_global_adapter_cfg,
    patch_transcribe_lhotse,
    update_model_cfg,
    update_model_config_to_support_adapter,
)
from asr_benchmark.score import english_spelling_normalizer, score_wer


def parse_args():
    parser = argparse.ArgumentParser(description="Train Parakeet ASR for children's speech")

    # Model
    parser.add_argument("--model", type=str, default="nvidia/parakeet-tdt-1.1b",
                        help="Pretrained model name (default: nvidia/parakeet-tdt-1.1b)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from a .nemo checkpoint path")

    # Adapter config
    parser.add_argument("--adapter-dim", type=int, default=256,
                        help="Adapter hidden dimension (default: 256, baseline was 32)")
    parser.add_argument("--adapter-type", type=str, default="linear",
                        choices=["linear", "tiny_attn"],
                        help="Adapter type (default: linear)")

    # Encoder unfreezing
    parser.add_argument("--unfreeze-layers", type=int, default=0,
                        help="Number of encoder layers to unfreeze from the end (default: 0)")

    # Training
    parser.add_argument("--max-steps", type=int, default=20000,
                        help="Max training steps (default: 20000)")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size (default: 16)")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate (default: 3e-4)")
    parser.add_argument("--weight-decay", type=float, default=0.01,
                        help="Weight decay (default: 0.01)")
    parser.add_argument("--precision", type=str, default="bf16-mixed",
                        help="Training precision (default: bf16-mixed)")
    parser.add_argument("--devices", type=int, default=1, help="Number of GPUs")
    parser.add_argument("--num-workers", type=int, default=8, help="DataLoader workers")
    parser.add_argument("--val-check-interval", type=int, default=1000,
                        help="Validation check interval in steps")
    parser.add_argument("--grad-accum", type=int, default=2,
                        help="Gradient accumulation steps (default: 2, effective batch = batch-size * grad-accum)")

    # Data
    parser.add_argument("--sample", type=int, default=None,
                        help="Use N samples for quick testing")
    parser.add_argument("--max-duration", type=float, default=25.0,
                        help="Max audio duration in seconds")
    parser.add_argument("--val-ratio", type=float, default=0.1,
                        help="Validation split ratio (default: 0.1)")

    # Augmentation
    parser.add_argument("--spec-augment", action="store_true",
                        help="Enable SpecAugment")
    parser.add_argument("--speed-perturb", action="store_true",
                        help="Enable speed perturbation (0.9x-1.1x)")
    parser.add_argument("--noise-augment", action="store_true",
                        help="Enable RealClass noise augmentation")

    # Eval
    parser.add_argument("--skip-eval", action="store_true",
                        help="Skip evaluation after training")

    return parser.parse_args()


def prepare_data(args):
    """Load, filter, split data and write NeMo manifests."""
    manifest_dir = DATA_ROOT / "processed" / "parakeet_v2"
    train_manifest = manifest_dir / "train_manifest.jsonl"
    val_manifest = manifest_dir / "val_manifest.jsonl"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    df = load_all_transcripts()
    df = filter_data(df, max_duration=args.max_duration)

    if args.sample:
        logger.info(f"Sampling {args.sample} utterances for quick testing")
        df = df.sample(args.sample, random_state=42)

    train_df, val_df = split_by_child(df, val_ratio=args.val_ratio)

    create_nemo_manifest(train_df, train_manifest)
    create_nemo_manifest(val_df, val_manifest)

    return train_manifest, val_manifest


def get_encoder_dim(model_name: str) -> int:
    """Get the encoder output dimension for different Parakeet models."""
    dim_map = {
        "nvidia/parakeet-tdt-0.6b-v2": 1024,
        "nvidia/parakeet-tdt-0.6b-v3": 1024,
        "nvidia/parakeet-tdt-1.1b": 1024,
    }
    for key, dim in dim_map.items():
        if key in model_name:
            return dim
    logger.warning(f"Unknown model {model_name}, defaulting to encoder dim=1024")
    return 1024


def unfreeze_encoder_layers(model, n_layers: int):
    """Unfreeze the last N conformer layers in the encoder.

    This allows deeper adaptation beyond just the adapter weights.
    """
    if n_layers <= 0:
        return 0

    encoder = model.encoder

    # FastConformer/Conformer encoders store layers in .layers
    layers = None
    if hasattr(encoder, 'layers'):
        layers = encoder.layers
    elif hasattr(encoder, 'encoder') and hasattr(encoder.encoder, 'layers'):
        layers = encoder.encoder.layers

    if layers is None:
        logger.warning("Could not find encoder layers for unfreezing. "
                        "Trying to unfreeze by parameter name pattern.")
        # Fallback: unfreeze params matching layer indices
        total_unfrozen = 0
        for name, param in model.named_parameters():
            if "encoder" in name and not param.requires_grad:
                # Try to extract layer index from name
                parts = name.split(".")
                for i, part in enumerate(parts):
                    if part == "layers" and i + 1 < len(parts):
                        try:
                            layer_idx = int(parts[i + 1])
                            # Assume we don't know total layers, skip
                            break
                        except ValueError:
                            pass
        return total_unfrozen

    total_layers = len(layers)
    start_idx = max(0, total_layers - n_layers)
    logger.info(f"Encoder has {total_layers} layers. Unfreezing layers [{start_idx}, {total_layers})")

    unfrozen = 0
    for idx in range(start_idx, total_layers):
        for param in layers[idx].parameters():
            param.requires_grad = True
            unfrozen += param.numel()

    logger.info(f"Unfroze {unfrozen:,} additional encoder parameters from {n_layers} layers")
    return unfrozen


def train(args, train_manifest, val_manifest):
    """Run training."""
    torch.set_float32_matmul_precision("high")

    yaml_path = PROJECT_ROOT / "asr_benchmark" / "assets" / "asr_adaptation.yaml"
    cfg = OmegaConf.load(yaml_path)

    encoder_dim = get_encoder_dim(args.model)
    adapter_name = "asr_children_orthographic"

    # Build augmentor config
    augmentor_cfg = {}
    if args.speed_perturb:
        augmentor_cfg.update(get_speed_perturb_config())
        logger.info("Speed perturbation enabled (0.9x-1.1x)")
    if args.noise_augment:
        noise_cfg = get_noise_augment_config()
        if noise_cfg:
            augmentor_cfg.update(noise_cfg)
            logger.info("Noise augmentation enabled (RealClass)")

    # Training overrides
    train_ds_cfg = {
        "manifest_filepath": str(train_manifest),
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "use_lhotse": False,
        "channel_selector": "average",
    }
    if augmentor_cfg:
        train_ds_cfg["augmentor"] = augmentor_cfg

    # SpecAugment
    spec_augment_cfg = get_spec_augment_config(
        freq_masks=2, time_masks=10
    ) if args.spec_augment else get_spec_augment_config(
        freq_masks=0, time_masks=0
    )

    overrides = OmegaConf.create({
        "model": {
            "pretrained_model": args.model,
            "adapter": {
                "adapter_name": adapter_name,
                "adapter_module_name": "encoder",
                "adapter_type": args.adapter_type,
                "linear": {
                    "in_features": encoder_dim,
                    "dim": args.adapter_dim,
                },
                "tiny_attn": {
                    "n_feat": encoder_dim,
                },
            },
            "train_ds": train_ds_cfg,
            "validation_ds": {
                "manifest_filepath": str(val_manifest),
                "batch_size": args.batch_size,
                "num_workers": args.num_workers,
                "use_lhotse": False,
                "channel_selector": "average",
            },
            "spec_augment": spec_augment_cfg,
            "optim": {
                "lr": args.lr,
                "weight_decay": args.weight_decay,
            },
        },
        "trainer": {
            "devices": args.devices,
            "precision": args.precision,
            "strategy": "auto",
            "max_epochs": 1 if args.sample else None,
            "max_steps": -1 if args.sample else args.max_steps,
            "val_check_interval": 1.0 if args.sample else args.val_check_interval,
            "accumulate_grad_batches": args.grad_accum,
            "enable_progress_bar": True,
            "gradient_clip_val": 1.0,
        },
        "exp_manager": {
            "exp_dir": str(PROJECT_ROOT / "models" / "parakeet_v2"),
        },
    })

    cfg = OmegaConf.merge(cfg, overrides)

    # --- Setup Trainer ---
    logger.info("Setting up Trainer...")
    trainer = pl.Trainer(**resolve_trainer_cfg(cfg.trainer))
    exp_log_dir = exp_manager(trainer, cfg.get("exp_manager", None))

    # --- Load Model ---
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        model = ASRModel.restore_from(args.resume, trainer=trainer, map_location="cuda")
    else:
        logger.info(f"Loading pretrained model: {args.model}")
        model_cfg = ASRModel.from_pretrained(args.model, return_config=True)
        update_model_config_to_support_adapter(model_cfg, cfg)
        model = ASRModel.from_pretrained(
            args.model,
            override_config_path=model_cfg,
            trainer=trainer,
        )

    # Disable CUDA graph decoder
    with open_dict(model.cfg):
        model.cfg.decoding.greedy.use_cuda_graph_decoder = False
    model.change_decoding_strategy(model.cfg.decoding)

    # --- Setup Data ---
    cfg.model.train_ds = update_model_cfg(model.cfg.train_ds, cfg.model.train_ds)
    model.setup_training_data(cfg.model.train_ds)

    cfg.model.validation_ds = update_model_cfg(model.cfg.validation_ds, cfg.model.validation_ds)
    model.setup_multiple_validation_data(cfg.model.validation_ds)

    # --- Setup Optimizer ---
    model.setup_optimization(cfg.model.optim)

    # --- SpecAugment ---
    if args.spec_augment:
        logger.info("SpecAugment enabled (freq_masks=2, time_masks=10)")
        model.spec_augmentation = model.from_config_dict(cfg.model.spec_augment)
    else:
        model.spec_augmentation = model.from_config_dict(cfg.model.spec_augment)

    # --- Setup Adapter (only if not resuming) ---
    if not args.resume:
        logger.info(f"Setting up adapter (type={args.adapter_type}, dim={args.adapter_dim})...")
        with open_dict(cfg.model.adapter):
            a_name = cfg.model.adapter.pop("adapter_name")
            a_type = cfg.model.adapter.pop("adapter_type")
            a_module = cfg.model.adapter.pop("adapter_module_name", None)
            cfg.model.adapter.pop("adapter_state_dict_name", None)
            adapter_type_cfg = cfg.model.adapter[a_type]

            if a_module is not None and ":" not in a_name:
                a_name = f"{a_module}:{a_name}"

            adapter_global_cfg = cfg.model.adapter.pop(model.adapter_global_cfg_key, None)
            if adapter_global_cfg is not None:
                add_global_adapter_cfg(model, adapter_global_cfg)

        model.add_adapter(a_name, cfg=adapter_type_cfg)
        assert model.is_adapter_available()

        model.set_enabled_adapters(enabled=False)
        model.set_enabled_adapters(a_name, enabled=True)

    # --- Freeze / Unfreeze ---
    model.freeze()
    model = model.train()
    model.unfreeze_enabled_adapters()

    # Optionally unfreeze encoder layers
    if args.unfreeze_layers > 0:
        unfreeze_encoder_layers(model, args.unfreeze_layers)

    # --- Log trainable params ---
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters:     {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
    logger.info(f"Effective batch size: {args.batch_size * args.grad_accum}")

    # --- Train ---
    logger.info("Starting training...")
    trainer.fit(model)
    logger.success("Training complete!")

    # Save adapter state
    state_path = exp_log_dir if exp_log_dir is not None else os.getcwd()
    ckpt_path = os.path.join(state_path, "checkpoints")
    if os.path.exists(ckpt_path):
        state_path = ckpt_path
    adapter_path = os.path.join(state_path, "adapters.pt")
    model.save_adapters(adapter_path)
    logger.info(f"Saved adapter weights to {adapter_path}")

    return exp_log_dir, cfg


def evaluate(exp_log_dir, cfg, batch_size):
    """Evaluate trained model on validation set."""
    logger.info("Starting evaluation...")

    nemo_ckpts = sorted(Path(exp_log_dir / "checkpoints").glob("*.nemo"))
    if not nemo_ckpts:
        raise FileNotFoundError(f"No .nemo checkpoints in {exp_log_dir}/checkpoints/")

    best_ckpt = nemo_ckpts[-1]
    logger.info(f"Loading checkpoint: {best_ckpt}")
    eval_model = ASRModel.restore_from(str(best_ckpt), map_location="cuda")

    with open_dict(eval_model.cfg):
        eval_model.cfg.decoding.greedy.use_cuda_graph_decoder = False
    eval_model.change_decoding_strategy(eval_model.cfg.decoding)
    patch_transcribe_lhotse(eval_model)

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

    normalizer = EnglishTextNormalizer(english_spelling_normalizer)
    filtered = [(r, p) for r, p in zip(references, predictions) if normalizer(r) != ""]
    refs_f, preds_f = zip(*filtered)

    wer = score_wer(refs_f, preds_f)
    logger.success(f"Validation WER: {wer:.4f}")

    # Print samples
    logger.info("Sample predictions:")
    for ref, pred in zip(refs_f[:10], preds_f[:10]):
        logger.info(f"  REF:  {ref}")
        logger.info(f"  PRED: {pred}")
        logger.info("")

    return wer


def main():
    args = parse_args()

    logger.info("=" * 70)
    logger.info("Children's ASR - Parakeet-TDT Deep Fine-tuning")
    logger.info("=" * 70)
    logger.info(f"Model: {args.model}")
    logger.info(f"Adapter: type={args.adapter_type}, dim={args.adapter_dim}")
    logger.info(f"Unfreeze layers: {args.unfreeze_layers}")
    logger.info(f"Augmentation: spec={args.spec_augment}, speed={args.speed_perturb}, noise={args.noise_augment}")

    # Step 1: Prepare data
    logger.info("Step 1: Preparing data...")
    train_manifest, val_manifest = prepare_data(args)

    # Step 2: Train
    logger.info("Step 2: Training...")
    exp_log_dir, cfg = train(args, train_manifest, val_manifest)

    # Step 3: Evaluate
    if not args.skip_eval:
        logger.info("Step 3: Evaluating...")
        wer = evaluate(Path(exp_log_dir), cfg, args.batch_size)
    else:
        logger.info("Step 3: Skipping evaluation (--skip-eval)")

    logger.success("All done!")
    logger.info(f"Checkpoints: {exp_log_dir}")
    logger.info("Next: python pack_submission.py")


if __name__ == "__main__":
    main()
