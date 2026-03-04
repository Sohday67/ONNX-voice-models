#!/usr/bin/env python3
"""
Convert NVIDIA Parakeet-TDT 110M to ONNX format.

This script downloads the Parakeet-TDT CTC 110M model from NVIDIA's
HuggingFace repository and exports it to ONNX format using NeMo's
built-in export functionality.

Source: https://huggingface.co/nvidia/parakeet-tdt_ctc-110m

Usage:
    python scripts/convert_parakeet_tdt.py [--output-dir OUTPUT_DIR]
"""

import argparse
import sys
from pathlib import Path

import nemo.collections.asr as nemo_asr


MODEL_NAME = "nvidia/parakeet-tdt_ctc-110m"
DEFAULT_OUTPUT_DIR = "models/parakeet-tdt-110m"


def convert_model(output_dir: str) -> None:
    """Download and convert Parakeet-TDT 110M to ONNX format.

    Args:
        output_dir: Directory to save the ONNX model files.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    onnx_filepath = output_path / "model.onnx"

    print(f"Downloading {MODEL_NAME} from HuggingFace...")
    model = nemo_asr.models.ASRModel.from_pretrained(model_name=MODEL_NAME)

    print(f"Exporting to ONNX: {onnx_filepath.resolve()}")
    model.export(str(onnx_filepath))

    print(f"Conversion complete. ONNX files saved to {output_path.resolve()}")
    print("\nGenerated files:")
    for f in sorted(output_path.rglob("*")):
        if f.is_file():
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"  {f.relative_to(output_path)} ({size_mb:.1f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert NVIDIA Parakeet-TDT 110M to ONNX format"
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for ONNX files (default: {DEFAULT_OUTPUT_DIR})",
    )
    args = parser.parse_args()

    try:
        convert_model(args.output_dir)
    except Exception as e:
        print(f"Error during conversion: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
