"""
打包提交文件：支持单模型或双模型（Parakeet + Whisper）打包

用法:
    # 仅 Parakeet
    python pack_submission.py --parakeet models/parakeet_v2/.../best.nemo

    # Parakeet + Whisper 融合
    python pack_submission.py \\
        --parakeet models/parakeet_v2/.../best.nemo \\
        --whisper models/whisper_lora/merged_model

    # 自动查找最新 checkpoint
    python pack_submission.py
"""

import argparse
import os
import shutil
import zipfile
from pathlib import Path

from loguru import logger


def find_latest_parakeet():
    """查找最新的 Parakeet .nemo checkpoint。"""
    for search_dir in ["models/parakeet_v2", "models/orthographic_benchmark_nemo"]:
        d = Path(search_dir)
        if d.exists():
            nemo_files = sorted(d.rglob("*.nemo"), key=os.path.getmtime)
            if nemo_files:
                logger.info(f"Found Parakeet checkpoint: {nemo_files[-1]}")
                return nemo_files[-1]
    return None


def find_latest_whisper():
    """查找最新的 Whisper merged model。"""
    whisper_dir = Path("models/whisper_lora/merged_model")
    if whisper_dir.exists() and (whisper_dir / "config.json").exists():
        logger.info(f"Found Whisper model: {whisper_dir}")
        return whisper_dir
    return None


def add_directory_to_zip(zf: zipfile.ZipFile, src_dir: Path, arc_prefix: str):
    """Recursively add a directory to a zip file."""
    for f in sorted(src_dir.rglob("*")):
        if f.is_file():
            arcname = f"{arc_prefix}/{f.relative_to(src_dir)}"
            zf.write(f, arcname)


def pack_submission(parakeet_path: Path = None, whisper_path: Path = None):
    """打包 main.py + 模型 checkpoint(s) 为 submission.zip。"""
    main_py = Path("orthographic_submission/main.py")
    if not main_py.exists():
        raise FileNotFoundError(f"main.py not found: {main_py}")

    if parakeet_path is None and whisper_path is None:
        raise FileNotFoundError("No model checkpoints specified or found!")

    output_zip = Path("orthographic_submission.zip")

    logger.info("Packing submission...")
    logger.info(f"  main.py: {main_py}")
    if parakeet_path:
        logger.info(f"  Parakeet: {parakeet_path}")
    if whisper_path:
        logger.info(f"  Whisper: {whisper_path}")

    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        # main.py must be at root
        zf.write(main_py, "main.py")

        # Parakeet checkpoint
        if parakeet_path:
            zf.write(parakeet_path, "parakeet_best.nemo")

        # Whisper merged model (directory)
        if whisper_path:
            add_directory_to_zip(zf, whisper_path, "whisper_merged")

    size_mb = output_zip.stat().st_size / (1024 * 1024)
    size_gb = size_mb / 1024

    if size_gb > 1:
        logger.success(f"Created {output_zip} ({size_gb:.2f} GB)")
    else:
        logger.success(f"Created {output_zip} ({size_mb:.1f} MB)")

    # List contents
    with zipfile.ZipFile(output_zip, "r") as zf:
        names = zf.namelist()
        logger.info(f"ZIP contains {len(names)} files:")
        for name in names[:20]:
            info = zf.getinfo(name)
            logger.info(f"  {name} ({info.file_size / 1024 / 1024:.1f} MB)")
        if len(names) > 20:
            logger.info(f"  ... and {len(names) - 20} more files")

    logger.info("")
    logger.info("Next steps:")
    logger.info("  1. Submit as smoke test first")
    logger.info("  2. Check logs on Code jobs page")
    logger.info("  3. Once passed, submit for full scoring")


def main():
    parser = argparse.ArgumentParser(description="Pack submission for DrivenData")
    parser.add_argument("--parakeet", type=str, default=None,
                        help="Path to Parakeet .nemo checkpoint")
    parser.add_argument("--whisper", type=str, default=None,
                        help="Path to Whisper merged model directory")
    args = parser.parse_args()

    parakeet_path = Path(args.parakeet) if args.parakeet else find_latest_parakeet()
    whisper_path = Path(args.whisper) if args.whisper else find_latest_whisper()

    if parakeet_path and not parakeet_path.exists():
        raise FileNotFoundError(f"Parakeet checkpoint not found: {parakeet_path}")
    if whisper_path and not whisper_path.exists():
        raise FileNotFoundError(f"Whisper model not found: {whisper_path}")

    pack_submission(parakeet_path, whisper_path)


if __name__ == "__main__":
    main()
