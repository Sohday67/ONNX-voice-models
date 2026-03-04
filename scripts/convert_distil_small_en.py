#!/usr/bin/env python3
"""
Convert distil-whisper/distil-small.en to ONNX format.

This script downloads the distil-small.en model from Hugging Face and exports
it to ONNX format using the Optimum library. The resulting ONNX files can be
used with ONNX Runtime (including ONNX Runtime Web for browser deployment).

Source: https://huggingface.co/distil-whisper/distil-small.en

Usage:
    python scripts/convert_distil_small_en.py [--output-dir OUTPUT_DIR] [--quantize]
"""

import argparse
import sys
from pathlib import Path

from optimum.onnxruntime import ORTModelForSpeechSeq2Seq
from transformers import AutoProcessor


MODEL_ID = "distil-whisper/distil-small.en"
DEFAULT_OUTPUT_DIR = "models/distil-small-en"


def convert_model(output_dir: str, quantize: bool = False) -> None:
    """Download and convert distil-small.en to ONNX format.

    Args:
        output_dir: Directory to save the ONNX model files.
        quantize: If True, apply dynamic quantization to reduce model size.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Downloading and converting {MODEL_ID} to ONNX...")
    print(f"Output directory: {output_path.resolve()}")

    # Export the model to ONNX using Optimum
    model = ORTModelForSpeechSeq2Seq.from_pretrained(
        MODEL_ID,
        export=True,
    )

    # Save the ONNX model
    model.save_pretrained(str(output_path))

    # Save the processor (tokenizer + feature extractor)
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    processor.save_pretrained(str(output_path))

    if quantize:
        from optimum.onnxruntime import ORTQuantizer
        from optimum.onnxruntime.configuration import AutoQuantizationConfig

        print("Applying dynamic quantization...")
        qconfig = AutoQuantizationConfig.avx512_vnni(is_static=False)

        for onnx_file in output_path.glob("*.onnx"):
            quantizer = ORTQuantizer.from_pretrained(str(output_path), file_name=onnx_file.name)
            quantizer.quantize(save_dir=str(output_path / "quantized"), quantization_config=qconfig)

        print(f"Quantized model saved to {output_path / 'quantized'}")

    print(f"Conversion complete. ONNX files saved to {output_path.resolve()}")
    print("\nGenerated files:")
    for f in sorted(output_path.rglob("*")):
        if f.is_file():
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"  {f.relative_to(output_path)} ({size_mb:.1f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert distil-whisper/distil-small.en to ONNX format"
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for ONNX files (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Apply dynamic quantization to reduce model size for browser use",
    )
    args = parser.parse_args()

    try:
        convert_model(args.output_dir, args.quantize)
    except Exception as e:
        print(f"Error during conversion: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
