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

## Quantized Models (int8)

Each model also has a `_q8` variant with dynamic int8 quantization for smaller download sizes and faster browser loading:

| Model | FP32 Size | int8 Size | Reduction |
|-------|-----------|-----------|-----------|
| **OLMoASR-tiny.en_q8** | 226 MB | 60 MB | 73% |
| **OLMoASR-base.en_q8** | 384 MB | 103 MB | 73% |
| **distil-small.en_q8** | 792 MB | 211 MB | 73% |
| **distil-medium.en_q8** | 1.7 GB | 454 MB | 74% |
| **parakeet-tdt-110m_q8** | 477 MB | 131 MB | 73% |

## Usage

These models can be loaded with [ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/) or [Transformers.js](https://huggingface.co/docs/transformers.js) for in-browser speech recognition. Use the `_q8` variants for faster loading in browser environments.

## Conversion Tools

- **Whisper-based models**: Converted using [Hugging Face Optimum](https://huggingface.co/docs/optimum) (`optimum-cli export onnx`)
- **Parakeet TDT**: Converted using [NVIDIA NeMo](https://github.com/NVIDIA/NeMo) (`model.export()`)
- **Quantization**: Dynamic int8 quantization via [ONNX Runtime](https://onnxruntime.ai/docs/performance/quantization.html) and [Optimum ORTQuantizer](https://huggingface.co/docs/optimum/onnxruntime/usage_guides/quantization)