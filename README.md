# ONNX Voice Models

Conversion scripts for exporting speech recognition models to ONNX format for use in web browsers via [ONNX Runtime Web](https://onnxruntime.ai/).

## Models

| Model | Source | Parameters | Description |
|-------|--------|------------|-------------|
| distil-small.en | [distil-whisper/distil-small.en](https://huggingface.co/distil-whisper/distil-small.en) | ~166M | Distilled Whisper small — 6× faster, English-only ASR |
| Parakeet-TDT 110M | [nvidia/parakeet-tdt_ctc-110m](https://huggingface.co/nvidia/parakeet-tdt_ctc-110m) | 110M | FastConformer + CTC decoder with punctuation & capitalization |
| OLMoASR-tiny.en | [allenai/OLMoASR](https://huggingface.co/allenai/OLMoASR) | 39M | Open, Whisper-compatible tiny model |
| OLMoASR-base.en | [allenai/OLMoASR](https://huggingface.co/allenai/OLMoASR) | 74M | Open, Whisper-compatible base model |

## Setup

```bash
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

## Conversion

Each script downloads the model from Hugging Face and exports it to ONNX.

### distil-small.en

```bash
python scripts/convert_distil_small_en.py

# With quantization for smaller browser downloads
python scripts/convert_distil_small_en.py --quantize

# Custom output directory
python scripts/convert_distil_small_en.py --output-dir ./my-models/distil-small-en
```

Output files (in `models/distil-small-en/`):
- `encoder_model.onnx` — Audio encoder
- `decoder_model_merged.onnx` — Text decoder (with KV-cache support)
- `config.json`, tokenizer files — For pre/post-processing

### NVIDIA Parakeet-TDT 110M

```bash
python scripts/convert_parakeet_tdt.py

# Custom output directory
python scripts/convert_parakeet_tdt.py --output-dir ./my-models/parakeet-tdt
```

Output files (in `models/parakeet-tdt-110m/`):
- `model.onnx` — Full ASR model

### OLMoASR (tiny.en & base.en)

```bash
# Convert both variants
python scripts/convert_olmoasr.py

# Convert only tiny.en
python scripts/convert_olmoasr.py --variant tiny

# Convert only base.en
python scripts/convert_olmoasr.py --variant base
```

Output files (in `models/olmoasr/olmoasr-{tiny,base}-en/`):
- `encoder_model.onnx` — Audio encoder
- `decoder_model.onnx` — Text decoder
- `decoder_with_past_model.onnx` — Text decoder (with KV-cache support)
- `config.json`, tokenizer files — For pre/post-processing

## Pre-Converted Models

The `converted/` folder contains pre-converted ONNX models ready for browser use. These files are tracked with [Git LFS](https://git-lfs.github.com/).

To clone with the ONNX files:
```bash
git lfs install
git clone https://github.com/Sohday67/ONNX-voice-models.git
```

## Browser Usage

The exported ONNX models can be loaded in a web browser using [ONNX Runtime Web](https://www.npmjs.com/package/onnxruntime-web):

```javascript
import * as ort from 'onnxruntime-web';

// Load the encoder model
const encoder = await ort.InferenceSession.create('./encoder_model.onnx');

// Prepare input: log-mel spectrogram with shape [1, 80, 3000]
const inputFeatures = new ort.Tensor('float32', audioFeatures, [1, 80, 3000]);

// Run encoder
const encoderOutput = await encoder.run({ input_features: inputFeatures });

// Load and run decoder in an autoregressive loop
const decoder = await ort.InferenceSession.create('./decoder_model_merged.onnx');
// ... token-by-token decoding
```

## Project Structure

```
├── README.md
├── requirements.txt
├── .gitignore
├── .gitattributes
├── scripts/
│   ├── convert_distil_small_en.py    # distil-whisper/distil-small.en
│   ├── convert_parakeet_tdt.py       # nvidia/parakeet-tdt_ctc-110m
│   └── convert_olmoasr.py            # allenai/OLMoASR tiny.en & base.en
└── converted/                        # Pre-converted ONNX models (Git LFS)
    ├── distil-small-en/
    │   ├── encoder_model.onnx
    │   ├── decoder_model.onnx
    │   ├── decoder_with_past_model.onnx
    │   └── config/tokenizer files
    ├── parakeet-tdt-110m/
    │   ├── encoder-model.onnx
    │   └── decoder_joint-model.onnx
    └── olmoasr/
        ├── olmoasr-tiny-en/
        │   ├── encoder_model.onnx
        │   ├── decoder_model.onnx
        │   ├── decoder_with_past_model.onnx
        │   └── config/tokenizer files
        └── olmoasr-base-en/
            ├── encoder_model.onnx
            ├── decoder_model.onnx
            ├── decoder_with_past_model.onnx
            └── config/tokenizer files
```

## Requirements

- Python 3.9+
- ~8 GB RAM (for the larger models during conversion)
- Internet connection (to download models from Hugging Face)
- [Git LFS](https://git-lfs.github.com/) (to clone the pre-converted ONNX files)