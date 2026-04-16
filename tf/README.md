# DeepFilterNet2 — TensorFlow / TFLite Conversion

TensorFlow/Keras reimplementation of the [DeepFilterNet2](https://github.com/Rikorose/DeepFilterNet) neural speech enhancement model, with streaming (frame-by-frame) support and TFLite conversion (float32, int16x8, int8).

## Overview

This directory contains a complete PyTorch → TensorFlow conversion pipeline:

| File | Description |
|---|---|
| `deepfilternet2_tf.py` | Batch TF/Keras model (all layers) |
| `deepfilternet2_tf_streaming.py` | Stateful streaming model (single-frame processing) |
| `weight_transfer.py` | Transfer weights from PyTorch checkpoint to TF model |
| `convert_tflite.py` | Convert streaming model to TFLite (float32 / int16x8 / int8) |
| `compare_pt_tf.py` | Validate TF model against PyTorch (mask diff ~2e-6) |
| `compare_batch_streaming.py` | Validate streaming vs batch equivalence (mask diff ~7.5e-7) |
| `test_deepfilternet2_tf.py` | Unit tests for batch model (23 tests) |
| `test_streaming.py` | Unit tests for streaming model (10 tests) |
| `test_tflite_conversion.py` | TFLite conversion tests |

## Model Architecture

DeepFilterNet2 is a real-time speech enhancement model operating in the spectral domain. It does **not** include STFT/ISTFT — only the neural network from spectral features to enhanced spectrum.

- **Encoder**: Conv2D stack on ERB features + Conv2D stack on complex spectrogram + GRU embedding
- **ERB Decoder**: GRU → ConvTranspose2D chain producing a [0, 1] gain mask over ERB bands
- **DF Decoder**: GRU → Conv2D chain producing deep filtering coefficients + blending alpha
- **GRU**: Built-in `tf.keras.layers.GRU` with `unroll=True, reset_after=True` (batch) / `tf.keras.layers.GRUCell` (streaming)
- **ConvTranspose2d**: Implemented as `tf.pad` (zero-insert upsample) + `Conv2D` for TFLite int16x8 compatibility

### Default Hyperparameters

| Parameter | Value | Description |
|---|---|---|
| `sr` | 48000 | Sample rate (Hz) |
| `fft_size` | 960 | FFT window size |
| `hop_size` | 480 | Hop size (10 ms frames) |
| `nb_erb` | 32 | Number of ERB bands |
| `nb_df` | 96 | Number of DF frequency bins |
| `df_order` | 5 | Deep filtering order (temporal taps) |
| `df_lookahead` | 2 | Lookahead frames |
| `conv_ch` | 64 | Convolution channels |
| `emb_hidden_dim` | 256 | Embedding GRU hidden size |
| `emb_num_layers` | 3 | Embedding GRU layers (encoder=1, ERB decoder=2) |
| `df_hidden_dim` | 256 | DF decoder GRU hidden size |
| `df_num_layers` | 2 | DF decoder GRU layers |
| `gru_groups` | 8 | Grouped GRU groups (encoder only) |
| `lin_groups` | 8 | Grouped linear projection groups |

## TFLite Model — Inputs & Outputs

The TFLite model is a **streaming (stateful)** model that processes **one frame at a time**. All RNN hidden states and convolution buffers are packed into a single flat state tensor that gets passed in and out each frame.

### Inputs

| Name | Shape | Type (float32) | Type (int16x8) | Description |
|---|---|---|---|---|
| `feat_erb` | `[1, 1, 1, 32]` | `float32` | `int16` | ERB-scaled log-power features for one frame. Layout: `[batch, channel, time=1, erb_bands]` |
| `feat_spec` | `[1, 1, 1, 96, 2]` | `float32` | `int16` | Complex spectrogram (DF bins only) for one frame. Layout: `[batch, channel, time=1, df_bins, real_imag]` where `[..., 0]` = real, `[..., 1]` = imaginary |
| `state` | `[1, 26304]` | `float32` | `int16` | Packed hidden state. Initialize to all zeros at start. Pass `new_state` from previous frame |

### Outputs

| Name | Shape | Type (float32) | Type (int16x8) | Description |
|---|---|---|---|---|
| `mask` | `[1, 1, 1, 32]` | `float32` | `int16` | ERB gain mask in [0, 1]. Apply to ERB-scaled magnitude to suppress noise |
| `lsnr` | `[1, 1]` | `float32` | `int16` | Local SNR estimate in dB (range: -15 to 35). Indicates speech activity |
| `df_coefs` | `[1, 5, 1, 96, 2]` | `float32` | `int16` | Deep filtering coefficients. Layout: `[batch, df_order, time=1, df_bins, real_imag]` |
| `alpha` | `[1, 1]` | `float32` | `int16` | DF blending factor in [0, 1]. Interpolates between DF-enhanced and mask-only output |
| `new_state` | `[1, 26304]` | `float32` | `int16` | Updated hidden state. Feed back as `state` input for the next frame |

### State Layout (26304 floats total)

The flat state vector is partitioned as:

| Offset | Size | Component |
|---|---|---|
| 0 | 256 | Encoder GRU hidden state (1 layer × 256) |
| 256 | 512 | ERB decoder GRU hidden state (2 layers × 256) |
| 768 | 512 | DF decoder GRU hidden state (2 layers × 256) |
| 1280 | 64 | ERB conv0 input buffer (2 frames × 32 ERB × 1 ch) |
| 1344 | 384 | DF conv0 input buffer (2 frames × 96 bins × 2 ch) |
| 1728 | 24576 | DF convp input buffer (4 frames × 96 bins × 64 ch) |

### Applying the Model Output

For each frame:
1. **ERB masking**: Multiply the noisy ERB-band magnitudes by `mask`
2. **Deep filtering**: Apply `df_coefs` as complex FIR filter over the last `df_order` frames for the first 96 bins
3. **Blending**: Use `alpha` to interpolate: `enhanced = alpha * df_output + (1 - alpha) * masked_output`
4. **SNR gating**: Use `lsnr` to gate enhancement (low SNR → more suppression)

## Quick Start

### Convert PyTorch checkpoint to TFLite

```bash
cd tf/
python convert_tflite.py --output-dir ../models/tflite/
```

This produces `dfnet2_float32.tflite`, `dfnet2_int16x8.tflite`, and `dfnet2_int8.tflite`.

### Run validation

```bash
# Unit tests
python -m pytest test_deepfilternet2_tf.py test_streaming.py -v

# PyTorch vs TensorFlow comparison
python compare_pt_tf.py

# Batch vs streaming equivalence
python compare_batch_streaming.py
```

### Use the streaming model in Python

```python
import tensorflow as tf
from deepfilternet2_tf_streaming import DfNetStreaming
from deepfilternet2_tf import DEFAULT_PARAMS

# Build model and load weights (see convert_tflite.py for full example)
model = DfNetStreaming(DEFAULT_PARAMS)
state = model.get_initial_state(batch_size=1)

# Per-frame inference loop
for frame_idx in range(num_frames):
    erb_frame = ...   # [1, 1, 1, 32]
    spec_frame = ...  # [1, 1, 1, 96, 2]
    mask, lsnr, df_coefs, alpha, state = model(erb_frame, spec_frame, state)
    # Apply mask and df_coefs to reconstruct enhanced audio
```

### Use the TFLite model

```python
import numpy as np
import tensorflow as tf

interpreter = tf.lite.Interpreter(model_path="dfnet2_float32.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Initialize state to zeros
state = np.zeros([1, 26304], dtype=np.float32)

for frame_idx in range(num_frames):
    interpreter.set_tensor(input_details[0]['index'], feat_erb_frame)   # [1,1,1,32]
    interpreter.set_tensor(input_details[1]['index'], feat_spec_frame)  # [1,1,1,96,2]
    interpreter.set_tensor(input_details[2]['index'], state)            # [1,26304]

    interpreter.invoke()

    mask = interpreter.get_tensor(output_details[0]['index'])      # [1,1,1,32]
    lsnr = interpreter.get_tensor(output_details[1]['index'])      # [1,1]
    df_coefs = interpreter.get_tensor(output_details[2]['index'])  # [1,5,1,96,2]
    alpha = interpreter.get_tensor(output_details[3]['index'])     # [1,1]
    state = interpreter.get_tensor(output_details[4]['index'])     # [1,26304]
```

## Model Sizes

| Variant | Size |
|---|---|
| Float32 | ~9.2 MB |
| Int16x8 | ~3.6 MB |
| Int8 | ~3.6 MB |

## Validation Results

| Comparison | Max Diff |
|---|---|
| PyTorch vs TensorFlow (mask) | ~2e-6 |
| Batch vs Streaming (mask) | ~7.5e-7 |
| TF vs TFLite float32 (mask) | ~1.4e-5 |

## Requirements

- Python 3.10+
- TensorFlow 2.15+ (tested with 2.20.0)
- PyTorch (for weight transfer only)
- NumPy

## Key Design Decisions

- **Built-in Keras GRU** (`unroll=True, reset_after=True`) instead of custom GRU — required for TFLite int16x8 quantization
- **`tf.pad` for ConvTranspose upsample** instead of `tf.zeros` with dynamic shapes — avoids SIGABRT in TFLite int16x8 quantizer
- **Gate reorder in weight transfer**: PyTorch GRU uses `[r, z, n]` gate order; Keras uses `[z, r, h]`. Bias is stacked as `(2, 3*H)` for `reset_after=True`
- **NHWC everywhere**: All conv layers use channels-last format for TFLite compatibility
- **Flat state vector**: All GRU hidden states and conv buffers packed into `[B, 26304]` for simple TFLite I/O
