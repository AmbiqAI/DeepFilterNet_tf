# DeepFilterNet/df TensorFlow Conversion Guide

This directory contains the **current PyTorch → TensorFlow → TFLite conversion path** for DeepFilterNet2.

The repository is now centered on the **stateful streaming** export path. Older explicit-streaming export files were removed during cleanup, and the active workflow is:

1. Re-implement PyTorch modules in TensorFlow
2. Load PyTorch weights into the TF model
3. Build a stateful streaming model
4. Export float32 / float16 / dynamic / int8 / int16x8 TFLite models
5. Run real audio through the exported stateful models

---

## What is current

The active TensorFlow/TFLite files are:

- `tf_modules.py`
- `tf_stateful_streaming.py`
- `export_stateful_tflite.py`
- `test_tf_conversion.py`
- `test_tf_stateful_streaming.py`
- `test_streaming_vs_pytorch.py`
- `test_tflite_vs_pytorch.py`

The active exported models are written to:

- `models/stateful/dfnet2_stateful_float32.tflite`
- `models/stateful/dfnet2_stateful_float16.tflite`
- `models/stateful/dfnet2_stateful_dynamic.tflite`
- `models/stateful/dfnet2_stateful_int8.tflite`
- `models/stateful/dfnet2_stateful_int16x8.tflite`

The current real-audio source files are:

- `models/streaming_audio_samples/noisy.wav`
- `models/streaming_audio_samples/clean.wav`

---

## Current architecture

The exported runtime is a **stateful streaming** model.

- Input: one STFT frame `spec [B, 1, 1, F, 2]`
- Output: one enhanced STFT frame `enhanced_spec [B, 1, 1, F, 2]`
- Streaming state is stored internally in `tf.Variable`
- Public runtime API is:
  - `forward(spec)`
  - `reset_state()`

For quantized models (`int8`, `int16x8`), the exported TFLite model is **NN-only**:

- inside TFLite: encoder + ERB decoder + DF decoder
- outside TFLite: feature extraction, ERB mask application, deep filtering

This is why wide-range ops like:

- `re^2 + im^2`
- `sqrt(...)`
- `log10(...)`
- ERB mask expansion
- complex deep filtering

stay outside the quantized graph.

### `float32`, `dynamic`, `float16`

These use the **full stateful wrapper**.

- input: raw spectrogram frame
- output: enhanced spectrogram frame

### `int8`, `int16x8`

These use the **NN-only wrapper**.

- input to TFLite: precomputed normalized features
- output from TFLite: mask + DF coefficients
- feature extraction and post-processing stay in Python float32

This is the main reason the current quantized path works much better than the older export path.

---

## Recommended workflow

If you are working on TF conversion today, use this order:

1. edit `tf_modules.py` for layer changes
2. validate with `test_tf_conversion.py`
3. validate streaming/export behavior with `test_tf_stateful_streaming.py`
4. compare TF streaming vs PyTorch with `test_streaming_vs_pytorch.py`
5. export + compare TFLite variants with `test_tflite_vs_pytorch.py`
6. full export with `export_stateful_tflite.py`
7. inspect outputs under `models/stateful/`

---

## Quick command summary

### Run current stateful tests

```bash
python3 -m pytest DeepFilterNet/df/test_tf_stateful_streaming.py -q
```

### Run conversion tests

```bash
python3 -m pytest DeepFilterNet/df/test_tf_conversion.py -q
```

### Export + compare TFLite (float32 / int8 / int16x8) on real audio

```bash
# Default audio (models/streaming_audio_samples/noisy.wav)
python DeepFilterNet/df/test_tflite_vs_pytorch.py

# Custom audio file
python DeepFilterNet/df/test_tflite_vs_pytorch.py --wav wavs/keyboard_steak.wav

# Skip re-export, reuse existing .tflite files
python DeepFilterNet/df/test_tflite_vs_pytorch.py --skip-export
```

### Export all stateful TFLite models

```bash
PYTHONPATH=DeepFilterNet .venv/bin/python DeepFilterNet/df/export_stateful_tflite.py
```

### Inspect generated files

```bash
ls models/stateful
ls models/stateful/audio_samples
```

---

## File-by-file guide

This section explains the Python files that matter for the current TF conversion workflow.

### `tf_modules.py`

TensorFlow reimplementation of the PyTorch building blocks.

- TF versions of conv, transposed-conv replacement, GRU, grouped linear, normalization, and decoder blocks
- weight-loading helpers that map PyTorch tensors into TF/Keras layers
- lowest-level file to edit when you are changing layer behavior

If a conversion mismatch starts at the layer level, this is usually the first file to inspect.

### `tf_stateful_streaming.py`

Current streaming runtime and TFLite wrapper construction.

- builds the stateful streaming model used for export
- contains the reference streaming path used in tests
- implements feature helpers such as ERB projection and normalization
- owns the state variables and the TFLite-friendly wrapper builders

If you are changing state layout, streaming behavior, or export signatures, edit this file.

### `export_stateful_tflite.py`

Main export and real-audio driver.

- loads the PyTorch checkpoint
- builds the TF model
- exports float32 / float16 / dynamic / int8 / int16x8 TFLite files
- runs real audio through exported models
- writes enhanced audio and comparison artifacts under `models/stateful/`

This is the main script to run when you want fresh `.tflite` files.

### `test_tf_conversion.py`

Module-level PyTorch ↔ TensorFlow validation.

- checks individual TF layers against their PyTorch counterparts
- useful after editing `tf_modules.py`
- catches weight-layout and operator mismatches early

### `test_tf_stateful_streaming.py`

End-to-end validation for the current streaming/export path.

- checks state reset behavior
- checks stateful TF vs reference streaming behavior
- checks exported TFLite models
- this is the main regression suite for the cleaned stateful path

### `test_streaming_vs_pytorch.py`

Per-module and full-model comparison of TF streaming vs PyTorch batch.

- uses zero-lookahead config so streaming equals batch output exactly
- tests encoder, ERB decoder, DF decoder, DF op, and full model individually
- includes real WAV processing test (PT batch vs TF streaming)
- expects >120 dB SNR for all float32 comparisons

### `test_tflite_vs_pytorch.py`

Export TFLite models and compare all variants on real audio.

- exports float32, int8, and int16x8 TFLite models (all NN-only wrapper)
- all three share the same I/O interface:
  - inputs: `feat_erb [1,1,32,1]`, `feat_spec [1,1,96,2]`
  - outputs: `mask [1,1,32,1]`, `df_coefs [1,5,1,96,2]`
- runs TF eager streaming as the reference
- runs each TFLite variant with float32 feature extraction + post-processing
- prints SNR, max diff, and timing for each variant
- saves enhanced WAV files to `wavs/`
- generates spectrogram comparison to `models/stateful/`
- supports `--wav` for custom audio and `--skip-export` to reuse existing TFLite files

Typical results on 3 s test audio:

| Model | SNR vs TF eager | Size |
|---|---|---|
| float32 | ~116 dB | 8.9 MB |
| int8 | ~0.4 dB | 2.6 MB |
| int16x8 | ~24 dB | 2.9 MB |

### Supporting source files

These are not the TF conversion entry points, but they define the original PyTorch model and utilities that the TF port mirrors.

- `deepfilternet2.py`: PyTorch DeepFilterNet2 architecture definition
- `modules.py`: PyTorch module building blocks
- `multiframe.py`: PyTorch deep filtering / multiframe logic
- `model.py`: model/config assembly helpers used by training and checkpoint loading
- `checkpoint.py`: checkpoint loading utilities

---

## How to export models

From the repository root, run:

```bash
PYTHONPATH=DeepFilterNet .venv/bin/python DeepFilterNet/df/export_stateful_tflite.py
```

If you are not using the workspace virtual environment, use your Python executable instead:

```bash
PYTHONPATH=DeepFilterNet python3 DeepFilterNet/df/export_stateful_tflite.py
```

This script writes the exported models to:

- `models/stateful/dfnet2_stateful_float32.tflite`
- `models/stateful/dfnet2_stateful_float16.tflite`
- `models/stateful/dfnet2_stateful_dynamic.tflite`
- `models/stateful/dfnet2_stateful_int8.tflite`
- `models/stateful/dfnet2_stateful_int16x8.tflite`

---

## How to run real audio

The same export script also runs real audio inference using the sample files in `models/streaming_audio_samples/`.

Run:

```bash
PYTHONPATH=DeepFilterNet python3 DeepFilterNet/df/export_stateful_tflite.py
```

It uses:

- `models/streaming_audio_samples/noisy.wav`
- `models/streaming_audio_samples/clean.wav`

and writes outputs under `models/stateful/audio_samples/`.

Typical outputs include enhanced audio for the exported variants and any comparison artifacts generated by the script.

### Run only after changing the model

Use this flow when you modify TF code:

1. run `python3 -m pytest DeepFilterNet/df/test_tf_conversion.py -q`
2. run `python3 -m pytest DeepFilterNet/df/test_tf_stateful_streaming.py -q`
3. run `PYTHONPATH=DeepFilterNet python3 DeepFilterNet/df/export_stateful_tflite.py`

### About quantized real-audio runs

For `int8` and `int16x8`, the exported TFLite model is NN-only.

- Python computes the input features
- TFLite predicts mask / DF coefficients
- Python applies post-processing and waveform reconstruction

So if you want to evaluate quantized audio quality, use `export_stateful_tflite.py` rather than calling the quantized TFLite model directly on raw STFT frames.

---

## Outputs you should expect

After a successful export run, check:

- `models/stateful/` for the `.tflite` files
- `models/stateful/audio_samples/` for generated audio

Quick inspection:

```bash
ls models/stateful
ls models/stateful/audio_samples
```

