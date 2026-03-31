"""Data loading, filtering, augmentation utilities for children's ASR training.

Shared by both Parakeet and Whisper training pipelines.
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.model_selection import GroupShuffleSplit

from asr_benchmark.config import DATA_ROOT


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def load_all_transcripts() -> pd.DataFrame:
    """Load and merge transcripts from DrivenData + TalkBank (extla_data).

    Returns a DataFrame with columns:
        utterance_id, child_id, session_id, audio_path (absolute),
        audio_duration_sec, age_bucket, orthographic_text
    """
    sources = [
        ("drivendata", DATA_ROOT / "raw" / "drivendata"),
        ("talkbank", DATA_ROOT / "raw" / "talkbank"),
    ]

    # Only use extla_data as fallback if raw/talkbank doesn't exist
    if not (DATA_ROOT / "raw" / "talkbank" / "train_word_transcripts.jsonl").exists():
        sources.append(("talkbank_ext", DATA_ROOT / "extla_data"))

    frames = []
    for name, data_dir in sources:
        transcript_path = data_dir / "train_word_transcripts.jsonl"
        if not transcript_path.exists():
            continue

        df = pd.read_json(transcript_path, lines=True)
        # Convert relative audio_path to absolute
        df["audio_path"] = df["audio_path"].map(lambda p: str(data_dir / p))
        df["source"] = name
        frames.append(df)
        logger.info(f"[{name}] Loaded {len(df)} utterances from {data_dir}")

    if not frames:
        raise FileNotFoundError(
            "No training data found! Place data in data/raw/drivendata/ "
            "and data/raw/talkbank/ (or data/extla_data/)"
        )

    df = pd.concat(frames, ignore_index=True)
    logger.info(f"Total: {len(df)} utterances, {df.child_id.nunique()} children, "
                f"{df.audio_duration_sec.sum() / 3600:.1f} hours")
    return df


# ---------------------------------------------------------------------------
# Data Filtering
# ---------------------------------------------------------------------------

def filter_data(
    df: pd.DataFrame,
    max_duration: float = 30.0,
    min_duration: float = 0.1,
    remove_empty_text: bool = True,
    remove_suspicious: bool = True,
) -> pd.DataFrame:
    """Filter out problematic training samples.

    Args:
        max_duration: Remove clips longer than this (seconds)
        min_duration: Remove clips shorter than this (seconds)
        remove_empty_text: Remove samples with empty transcripts
        remove_suspicious: Remove likely annotation errors
            (e.g., very long audio with very short transcript)
    """
    n_before = len(df)

    # Duration bounds
    mask = (df["audio_duration_sec"] >= min_duration) & (df["audio_duration_sec"] <= max_duration)
    df = df[mask]
    logger.info(f"Duration filter [{min_duration}s, {max_duration}s]: "
                f"{n_before} → {len(df)} ({n_before - len(df)} removed)")

    # Empty transcripts
    if remove_empty_text:
        n = len(df)
        df = df[df["orthographic_text"].str.strip().str.len() > 0]
        if len(df) < n:
            logger.info(f"Empty text filter: removed {n - len(df)}")

    # Suspicious: long audio (>5s) with very short text (1 word or less)
    if remove_suspicious:
        n = len(df)
        word_count = df["orthographic_text"].str.split().str.len()
        suspicious = (df["audio_duration_sec"] > 5.0) & (word_count <= 1)
        df = df[~suspicious]
        if len(df) < n:
            logger.info(f"Suspicious sample filter: removed {n - len(df)}")

    # Remove entries with missing audio files
    logger.info("Checking audio file existence (this may take a moment)...")
    exists_mask = df["audio_path"].map(lambda p: Path(p).exists())
    n_missing = (~exists_mask).sum()
    if n_missing > 0:
        logger.warning(f"Removing {n_missing} entries with missing audio files")
        df = df[exists_mask]

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Train/Val Split (grouped by child_id)
# ---------------------------------------------------------------------------

def split_by_child(
    df: pd.DataFrame,
    val_ratio: float = 0.1,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split data ensuring same child never appears in both train and val.

    Uses GroupShuffleSplit on child_id for proper generalization estimation.
    """
    gss = GroupShuffleSplit(n_splits=1, test_size=val_ratio, random_state=random_state)
    train_idx, val_idx = next(gss.split(df, groups=df["child_id"]))
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)

    logger.info(f"Split by child_id: train={len(train_df)} ({train_df.child_id.nunique()} children), "
                f"val={len(val_df)} ({val_df.child_id.nunique()} children)")

    # Log age distribution
    for name, split_df in [("train", train_df), ("val", val_df)]:
        age_dist = split_df["age_bucket"].value_counts().to_dict()
        logger.info(f"  {name} age distribution: {age_dist}")

    return train_df, val_df


# ---------------------------------------------------------------------------
# NeMo Manifest Creation
# ---------------------------------------------------------------------------

def create_nemo_manifest(
    df: pd.DataFrame,
    output_path: Path,
) -> Path:
    """Write a NeMo-compatible JSONL manifest file.

    NeMo format: {"audio_filepath": ..., "duration": ..., "text": ...}
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for _, row in df.iterrows():
            entry = {
                "audio_filepath": row["audio_path"],
                "duration": row["audio_duration_sec"],
                "text": row["orthographic_text"],
            }
            f.write(json.dumps(entry) + "\n")

    logger.info(f"Wrote NeMo manifest: {output_path} ({len(df)} entries)")
    return output_path


# ---------------------------------------------------------------------------
# NeMo Augmentation Configs
# ---------------------------------------------------------------------------

def get_speed_perturb_config() -> dict:
    """NeMo speed perturbation augmentor config."""
    return {
        "speed": {
            "prob": 0.5,
            "sr": 16000,
            "resample_type": "kaiser_fast",
            "min_speed_rate": 0.9,
            "max_speed_rate": 1.1,
        }
    }


def get_noise_augment_config(noise_dir: Optional[Path] = None) -> dict:
    """NeMo noise augmentor config using RealClass noise data."""
    if noise_dir is None:
        noise_dir = DATA_ROOT / "raw" / "noise" / "audio"

    if not noise_dir.exists():
        logger.warning(f"Noise directory not found: {noise_dir}")
        return {}

    # Create a noise manifest for NeMo
    noise_manifest = noise_dir.parent / "noise_manifest.jsonl"
    if not noise_manifest.exists():
        noise_files = sorted(noise_dir.glob("*.flac"))
        logger.info(f"Creating noise manifest with {len(noise_files)} files")
        with open(noise_manifest, "w") as f:
            for nf in noise_files:
                # We don't know exact duration, estimate from file size
                # FLAC at 48kHz stereo 16bit: ~200KB/sec roughly
                est_duration = nf.stat().st_size / 200000
                entry = {
                    "audio_filepath": str(nf),
                    "duration": est_duration,
                    "text": "",
                    "offset": 0,
                }
                f.write(json.dumps(entry) + "\n")

    return {
        "noise": {
            "prob": 0.3,
            "manifest_path": str(noise_manifest),
            "min_snr_db": 5,
            "max_snr_db": 25,
        }
    }


def get_spec_augment_config(
    freq_masks: int = 2,
    time_masks: int = 10,
    freq_width: int = 27,
    time_width: float = 0.05,
) -> dict:
    """SpecAugment config for NeMo."""
    return {
        "_target_": "nemo.collections.asr.modules.SpectrogramAugmentation",
        "freq_masks": freq_masks,
        "time_masks": time_masks,
        "freq_width": freq_width,
        "time_width": time_width,
    }


# ---------------------------------------------------------------------------
# Whisper Dataset Helpers
# ---------------------------------------------------------------------------

def prepare_whisper_dataset(df: pd.DataFrame) -> list[dict]:
    """Convert DataFrame to list of dicts for Whisper fine-tuning.

    Returns: [{"audio_path": str, "text": str, "duration": float}, ...]
    """
    records = []
    for _, row in df.iterrows():
        records.append({
            "audio_path": row["audio_path"],
            "text": row["orthographic_text"],
            "duration": row["audio_duration_sec"],
        })
    return records
