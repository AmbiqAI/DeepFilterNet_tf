---
applyTo: "**/*.py"
---

# PyTorch → TF → TFLite Conversion Instructions

When converting any PyTorch model to TensorFlow or TFLite, always follow these rules and phases.

## Conversion Phases (follow in order)

1. **Offline TF model**: Re-implement every layer in TF/Keras. Convert one module at a time. Write per-module unit tests.
2. **Weight loader**: Load PyTorch checkpoint into TF model. Transpose all weights NCHW→NHWC. Reorder GRU gates.
3. **Full model test**: Run on real audio/data. Expect >120 dB SNR for float32 PT vs TF.
4. **Streaming TF model** (if applicable): Frame-by-frame with causal caches. Pack all state into single flat tensor.
5. **TFLite float32**: Unroll all RNNs, replace LayerNorm, freeze weights, export stateful model.
6. **Quantized TFLite**: Int8 and int16x8 with representative dataset.
7. **Visualization**: Spectrogram comparisons at each phase.

Never skip a phase. Never move forward until current phase passes tests.

## Data Format

- PyTorch uses NCHW. TensorFlow uses NHWC.
- Conv2d weight: PT `(out, in, kH, kW)` → TF `(kH, kW, in, out)` via `np.transpose(w, (2,3,1,0))`
- DepthwiseConv2D weight: PT `(ch, 1, kH, kW)` → TF `(kH, kW, ch, 1)` via `np.transpose(w, (2,3,0,1))`
- Activation: PT `(B,C,T,F)` → TF `(B,T,F,C)` via `np.transpose(x, (0,2,3,1))`

## GRU/LSTM Weight Loading

- GRU gate order: PT `[reset, update, new]` → TF `[update, reset, new]`. Always reorder.
- LSTM gate order: same in both frameworks, no reorder needed.
- TF GRU uses single bias `(2, 3*H)` = stack of `[bias_ih, bias_hh]`.
- TF LSTM uses single bias = `bias_ih + bias_hh`.

## RNN Unrolling for TFLite

- Never use `tf.keras.layers.GRU` or `tf.keras.layers.LSTM` — they generate WHILE loops that crash in TFLite.
- Always use `GRUCell` / `LSTMCell` + Python for-loop with hardcoded range.
- For bidirectional: forward loop `range(T)` + backward loop `range(T-1,-1,-1)`, concatenate outputs.
- For streaming (1 frame): single cell call, no loop needed.

## LayerNorm for int16x8

- Never use `tf.keras.layers.LayerNormalization` — it uses SQUARED_DIFFERENCE which has no int16 TFLite kernel.
- Always use `LayerNormNoSquaredDiff` which computes variance as `mean(diff * diff)` where `diff = x - mean`.

## Stateful TFLite

- Freeze Keras weight variables to constants using `convert_variables_to_constants_v2`.
- Only streaming state should be a `tf.Variable` (updated via `assign`).
- Export with two signatures: `forward(spec)` → mask, `reset_state()` → zeros.

## Quantization

- Always provide a representative dataset (100-500 real input frames) for calibration.
- Int8: use `TFLITE_BUILTINS_INT8` + `TFLITE_BUILTINS` fallback.
- Int16x8: use `EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8` + `TFLITE_BUILTINS` fallback.
- Keep `inference_input_type=tf.float32` and `inference_output_type=tf.float32`.
- For stateful models, export forward-only SavedModel for quantizer.

## Testing

- Always write per-module unit tests comparing PT vs TF output with `np.testing.assert_allclose(atol=1e-5)`.
- Always test full model on real audio and compute SNR. Expect >120 dB for float32 conversions.
- Always draw spectrogram comparisons at each phase (offline, streaming, TFLite, quantized).
