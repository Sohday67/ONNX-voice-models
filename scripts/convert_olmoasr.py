#!/usr/bin/env python3
"""
Convert OLMoASR tiny.en and base.en to ONNX format.

This script downloads the OLMoASR models from Allen AI's HuggingFace
repository and exports them to ONNX format using the Optimum library.
OLMoASR uses a Whisper-compatible encoder-decoder architecture
(WhisperForConditionalGeneration), so it can be exported with the same
Optimum pipeline used for Whisper/distil-whisper models.

Source: https://huggingface.co/allenai/OLMoASR

Usage:
    python scripts/convert_olmoasr.py [--output-dir OUTPUT_DIR] [--variant VARIANT]
"""

import argparse
import sys
from pathlib import Path

from optimum.onnxruntime import ORTModelForSpeechSeq2Seq
from transformers import AutoProcessor


MODEL_REPO = "allenai/OLMoASR"
VARIANTS = ["tiny", "base"]
DEFAULT_OUTPUT_DIR = "models/olmoasr"


def convert_variant(variant: str, output_dir: str) -> None:
    """Download and convert an OLMoASR model variant to ONNX.

    The models are stored under subfolders within the allenai/OLMoASR repo
    on HuggingFace (e.g. models/OLMoASR-tiny.en). Since Optimum expects a
    top-level model, we first download the weights with transformers and
    then export via Optimum.

    Args:
        variant: Model variant name ('tiny' or 'base').
        output_dir: Base directory for output files.
    """
    variant_dir = Path(output_dir) / f"olmoasr-{variant}-en"
    variant_dir.mkdir(parents=True, exist_ok=True)

    subfolder = f"models/OLMoASR-{variant}.en"
    model_id = MODEL_REPO

    print(f"\nConverting OLMoASR-{variant}.en...")
    print(f"  Source: {model_id} (subfolder: {subfolder})")
    print(f"  Output: {variant_dir.resolve()}")

    # Export to ONNX using Optimum (Whisper-compatible architecture)
    model = ORTModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        subfolder=subfolder,
        export=True,
    )
    model.save_pretrained(str(variant_dir))

    # Save the processor (tokenizer + feature extractor)
    processor = AutoProcessor.from_pretrained(model_id, subfolder=subfolder)
    processor.save_pretrained(str(variant_dir))

    print(f"\nGenerated files for OLMoASR-{variant}.en:")
    for f in sorted(variant_dir.rglob("*")):
        if f.is_file():
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"  {f.relative_to(variant_dir)} ({size_mb:.1f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert OLMoASR models to ONNX format"
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for ONNX files (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--variant",
        choices=VARIANTS + ["all"],
        default="all",
        help="Model variant to convert (default: all)",
    )
    args = parser.parse_args()

    variants = VARIANTS if args.variant == "all" else [args.variant]

    try:
        for variant in variants:
            convert_variant(variant, args.output_dir)
        print("\nAll conversions complete.")
    except Exception as e:
        print(f"Error during conversion: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
