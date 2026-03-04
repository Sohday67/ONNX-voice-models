#!/usr/bin/env python3
"""
Convert OLMoASR tiny.en and base.en to ONNX format.

This script downloads the OLMoASR models from Allen AI's HuggingFace
repository and exports them to ONNX format using PyTorch's ONNX export.
OLMoASR uses a Whisper-compatible encoder-decoder architecture.

Source: https://huggingface.co/allenai/OLMoASR

Usage:
    python scripts/convert_olmoasr.py [--output-dir OUTPUT_DIR] [--variant VARIANT]
"""

import argparse
import sys
from pathlib import Path

import onnx
import olmoasr
import torch


MODEL_REPO = "allenai/OLMoASR"
VARIANTS = ["tiny", "base"]
DEFAULT_OUTPUT_DIR = "models/olmoasr"

# Audio parameters matching Whisper/OLMoASR configuration
N_MELS = 80
MAX_AUDIO_FRAMES = 3000  # ~30 seconds of audio


def export_encoder(model: torch.nn.Module, output_path: Path, opset_version: int = 17) -> None:
    """Export the encoder component to ONNX.

    Args:
        model: The loaded OLMoASR model.
        output_path: Path to save the encoder ONNX file.
        opset_version: ONNX opset version.
    """
    encoder = model.encoder if hasattr(model, "encoder") else model

    dummy_input = torch.randn(1, N_MELS, MAX_AUDIO_FRAMES)

    torch.onnx.export(
        encoder,
        (dummy_input,),
        str(output_path),
        input_names=["input_features"],
        output_names=["encoder_output"],
        dynamic_axes={
            "input_features": {0: "batch_size"},
            "encoder_output": {0: "batch_size"},
        },
        opset_version=opset_version,
        do_constant_folding=True,
    )
    print(f"  Encoder saved to {output_path}")


def export_decoder(model: torch.nn.Module, output_path: Path, opset_version: int = 17) -> None:
    """Export the decoder component to ONNX.

    Args:
        model: The loaded OLMoASR model.
        output_path: Path to save the decoder ONNX file.
        opset_version: ONNX opset version.
    """
    decoder = model.decoder if hasattr(model, "decoder") else model

    encoder_dim = model.dims.n_audio_state if hasattr(model, "dims") else 384
    max_encoder_len = MAX_AUDIO_FRAMES // 2  # encoder downsamples by 2

    dummy_tokens = torch.tensor([[50257]], dtype=torch.long)  # start token
    dummy_encoder_output = torch.randn(1, max_encoder_len, encoder_dim)

    torch.onnx.export(
        decoder,
        (dummy_tokens, dummy_encoder_output),
        str(output_path),
        input_names=["input_ids", "encoder_hidden_states"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "encoder_hidden_states": {0: "batch_size", 1: "encoder_length"},
            "logits": {0: "batch_size", 1: "sequence_length"},
        },
        opset_version=opset_version,
        do_constant_folding=True,
    )
    print(f"  Decoder saved to {output_path}")


def convert_variant(variant: str, output_dir: str) -> None:
    """Download and convert an OLMoASR model variant to ONNX.

    Args:
        variant: Model variant name ('tiny' or 'base').
        output_dir: Base directory for output files.
    """
    variant_dir = Path(output_dir) / f"olmoasr-{variant}-en"
    variant_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nConverting OLMoASR-{variant}.en...")
    model = olmoasr.load_model(variant, inference=True)
    model.eval()

    # Export encoder and decoder separately for browser use
    export_encoder(model, variant_dir / "encoder_model.onnx")
    export_decoder(model, variant_dir / "decoder_model.onnx")

    # Verify the exported models
    for onnx_file in variant_dir.glob("*.onnx"):
        onnx_model = onnx.load(str(onnx_file))
        onnx.checker.check_model(onnx_model)
        print(f"  Verified {onnx_file.name}: valid ONNX model")

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
