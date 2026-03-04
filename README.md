# ONNX Voice Models

Pre-converted ONNX speech recognition models for browser-based inference.

## Converted Models

All models are in the `converted/` folder, exported to ONNX format for use with ONNX Runtime Web.

| Model | Source | Architecture | ONNX Files |
|-------|--------|-------------|------------|
| **distil-small.en** | [distil-whisper/distil-small.en](https://huggingface.co/distil-whisper/distil-small.en) | Whisper (encoder-decoder) | encoder_model.onnx, decoder_model_merged.onnx |
| **distil-medium.en** | [distil-whisper/distil-medium.en](https://huggingface.co/distil-whisper/distil-medium.en) | Whisper (encoder-decoder) | encoder_model.onnx, decoder_model_merged.onnx |
| **parakeet-tdt-110m** | [nvidia/parakeet-tdt_ctc-110m](https://huggingface.co/nvidia/parakeet-tdt_ctc-110m) | FastConformer TDT (encoder + decoder-joint) | encoder-model.onnx, decoder_joint-model.onnx |
| **OLMoASR-tiny.en** | [allenai/OLMoASR](https://huggingface.co/allenai/OLMoASR) | Whisper (encoder-decoder) | encoder_model.onnx, decoder_model_merged.onnx |
| **OLMoASR-base.en** | [allenai/OLMoASR](https://huggingface.co/allenai/OLMoASR) | Whisper (encoder-decoder) | encoder_model.onnx, decoder_model_merged.onnx |

## Usage

These models can be loaded with [ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/) or [Transformers.js](https://huggingface.co/docs/transformers.js) for in-browser speech recognition.

## Conversion Tools

- **Whisper-based models**: Converted using [Hugging Face Optimum](https://huggingface.co/docs/optimum) (`optimum-cli export onnx`)
- **Parakeet TDT**: Converted using [NVIDIA NeMo](https://github.com/NVIDIA/NeMo) (`model.export()`)