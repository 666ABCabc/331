"""
本地测试推理：在 data-demo/word/ 上运行推理，验证提交格式正确。

用法:
    python test_local.py [--checkpoint PATH] [--data-dir PATH]
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from loguru import logger


def main():
    parser = argparse.ArgumentParser(description="Test inference locally on demo data")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to .nemo checkpoint (auto-detect if not specified)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data-demo/word",
        help="Path to test data directory",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    manifest = data_dir / "utterance_metadata.jsonl"
    if not manifest.exists():
        logger.error(f"Manifest not found: {manifest}")
        sys.exit(1)

    # Find checkpoint
    if args.checkpoint:
        checkpoint = args.checkpoint
    else:
        from pack_submission import find_latest_checkpoint
        checkpoint = str(find_latest_checkpoint())

    logger.info(f"Running inference with checkpoint: {checkpoint}")
    logger.info(f"Data directory: {data_dir}")

    # Run main.py
    result = subprocess.run(
        [sys.executable, "orthographic_submission/main.py", checkpoint, str(manifest)],
        capture_output=False,
    )

    if result.returncode != 0:
        logger.error("Inference failed!")
        sys.exit(1)

    # Verify output
    submission_path = Path("submission/submission.jsonl")
    if not submission_path.exists():
        logger.error(f"Submission file not created: {submission_path}")
        sys.exit(1)

    with submission_path.open() as f:
        lines = f.readlines()

    logger.success(f"Submission file created with {len(lines)} predictions:")
    for line in lines:
        item = json.loads(line)
        logger.info(f"  {item['utterance_id']}: '{item['orthographic_text']}'")


if __name__ == "__main__":
    main()
