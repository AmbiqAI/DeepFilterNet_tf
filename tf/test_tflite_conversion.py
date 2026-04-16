"""Unit tests for TFLite conversion of DeepFilterNet2 streaming model.

Tests each conversion step individually to identify failures.
Run with: python -m pytest test_tflite_conversion.py -v --tb=long -s
"""

import os
import sys
import tempfile
import unittest

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import tensorflow as tf

from convert_tflite import (
    build_streaming_model,
    get_concrete_function,
    generate_representative_dataset,
    DEFAULT_PARAMS,
)
from compare_batch_streaming import extract_features, load_wav_mono


def _get_paths():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ckpt = os.path.join(script_dir, "..", "models", "DeepFilterNet2",
                        "checkpoints", "model_96.ckpt.best")
    audio = os.path.join(script_dir, "..", "models",
                         "streaming_audio_samples", "noisy.wav")
    return ckpt, audio


# Module-level fixtures (built once, shared across tests)
_stream_model = None
_concrete_fn = None
_erb_fb_np = None
_feat_erb = None
_feat_spec = None
_p = None
_rep_dataset_samples = None


def _ensure_model():
    global _stream_model, _concrete_fn, _erb_fb_np, _feat_erb, _feat_spec, _p, _rep_dataset_samples
    if _stream_model is not None:
        return

    ckpt, audio = _get_paths()
    _p = DEFAULT_PARAMS.copy()
    _stream_model, _erb_fb_np, _p = build_streaming_model(ckpt, _p)
    _concrete_fn = get_concrete_function(_stream_model, _p).get_concrete_function()

    audio_data, _ = load_wav_mono(audio)
    _, _feat_erb, _feat_spec = extract_features(audio_data, _erb_fb_np, _p)

    # Pre-generate representative dataset samples (reusable)
    gen = generate_representative_dataset(
        _stream_model, _feat_erb, _feat_spec, _p, num_samples=50)
    _rep_dataset_samples = list(gen())


def _make_rep_dataset():
    """Return a fresh generator from cached samples."""
    def gen():
        for s in _rep_dataset_samples:
            yield s
    return gen


class TestBuildModel(unittest.TestCase):
    """Test that the model builds and traces correctly."""

    def test_build_streaming_model(self):
        _ensure_model()
        self.assertIsNotNone(_stream_model)
        self.assertGreater(_stream_model.total_state_size, 0)
        print(f"  state_size={_stream_model.total_state_size}")

    def test_concrete_function(self):
        _ensure_model()
        self.assertIsNotNone(_concrete_fn)
        # Check input signatures
        inp = _concrete_fn.structured_input_signature
        print(f"  inputs: {inp}")

    def test_representative_dataset(self):
        _ensure_model()
        samples = _rep_dataset_samples
        self.assertGreater(len(samples), 0)
        erb, spec, state = samples[0]
        self.assertEqual(erb.shape, (1, 1, 1, _p["nb_erb"]))
        self.assertEqual(spec.shape, (1, 1, 1, _p["nb_df"], 2))
        self.assertEqual(state.shape, (1, _stream_model.total_state_size))
        # Check for NaN/Inf in calibration data
        for i, (e, s, st) in enumerate(samples):
            self.assertFalse(np.any(np.isnan(e)), f"NaN in erb sample {i}")
            self.assertFalse(np.any(np.isnan(s)), f"NaN in spec sample {i}")
            self.assertFalse(np.any(np.isnan(st)), f"NaN in state sample {i}")
            self.assertFalse(np.any(np.isinf(e)), f"Inf in erb sample {i}")
            self.assertFalse(np.any(np.isinf(s)), f"Inf in spec sample {i}")
            self.assertFalse(np.any(np.isinf(st)), f"Inf in state sample {i}")
        # Print value ranges
        all_erb = np.concatenate([s[0] for s in samples])
        all_spec = np.concatenate([s[1] for s in samples])
        all_state = np.concatenate([s[2] for s in samples])
        print(f"  erb range: [{all_erb.min():.4f}, {all_erb.max():.4f}]")
        print(f"  spec range: [{all_spec.min():.4f}, {all_spec.max():.4f}]")
        print(f"  state range: [{all_state.min():.4f}, {all_state.max():.4f}]")


class TestFloat32Conversion(unittest.TestCase):
    """Test float32 TFLite conversion."""

    def test_convert_float32(self):
        _ensure_model()
        converter = tf.lite.TFLiteConverter.from_concrete_functions([_concrete_fn])
        tflite_model = converter.convert()
        self.assertIsNotNone(tflite_model)
        self.assertGreater(len(tflite_model), 0)
        print(f"  float32 size: {len(tflite_model) / 1024 / 1024:.2f} MB")

        # Verify it loads and runs
        with tempfile.NamedTemporaryFile(suffix=".tflite", delete=False) as f:
            f.write(tflite_model)
            tmp_path = f.name
        try:
            interpreter = tf.lite.Interpreter(model_path=tmp_path)
            interpreter.allocate_tensors()
            inp = interpreter.get_input_details()
            out = interpreter.get_output_details()
            print(f"  inputs: {[(d['name'], d['shape'].tolist(), d['dtype']) for d in inp]}")
            print(f"  outputs: {[(d['name'], d['shape'].tolist(), d['dtype']) for d in out]}")

            # Run inference with a calibration sample
            erb, spec, state = _rep_dataset_samples[0]
            for d in inp:
                shape = tuple(d["shape"])
                if shape == (1, 1, 1, _p["nb_erb"]):
                    interpreter.set_tensor(d["index"], erb)
                elif len(shape) == 5:
                    interpreter.set_tensor(d["index"], spec)
                elif shape == (1, _stream_model.total_state_size):
                    interpreter.set_tensor(d["index"], state)
            interpreter.invoke()

            for d in out:
                val = interpreter.get_tensor(d["index"])
                self.assertFalse(np.any(np.isnan(val)),
                                 f"NaN in output {d['name']}")
                print(f"  output {d['name']}: shape={val.shape}, "
                      f"range=[{val.min():.4f}, {val.max():.4f}]")
        finally:
            os.unlink(tmp_path)


class TestInt16x8Conversion(unittest.TestCase):
    """Test int16x8 TFLite conversion step by step."""

    def test_converter_setup(self):
        """Test that converter can be created with int16x8 settings."""
        _ensure_model()
        converter = tf.lite.TFLiteConverter.from_concrete_functions([_concrete_fn])
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = _make_rep_dataset()
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8,
            tf.lite.OpsSet.TFLITE_BUILTINS,
        ]
        converter.inference_input_type = tf.int16
        converter.inference_output_type = tf.int16
        # Just test setup, not conversion
        self.assertIsNotNone(converter)
        print("  Converter setup OK")

    def test_convert_int16x8_no_io_type(self):
        """Test int16x8 conversion WITHOUT setting input/output types (float I/O)."""
        _ensure_model()
        converter = tf.lite.TFLiteConverter.from_concrete_functions([_concrete_fn])
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = _make_rep_dataset()
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8,
            tf.lite.OpsSet.TFLITE_BUILTINS,
        ]
        # Do NOT set inference_input_type/inference_output_type
        print("  Converting int16x8 (float I/O)...", flush=True)
        tflite_model = converter.convert()
        self.assertIsNotNone(tflite_model)
        size_mb = len(tflite_model) / 1024 / 1024
        print(f"  int16x8 (float I/O) size: {size_mb:.2f} MB")

        with tempfile.NamedTemporaryFile(suffix=".tflite", delete=False) as f:
            f.write(tflite_model)
            tmp_path = f.name
        try:
            interpreter = tf.lite.Interpreter(model_path=tmp_path)
            interpreter.allocate_tensors()
            inp = interpreter.get_input_details()
            out = interpreter.get_output_details()
            for d in inp:
                print(f"  input: {d['name']} shape={d['shape'].tolist()} dtype={d['dtype']}")
            for d in out:
                print(f"  output: {d['name']} shape={d['shape'].tolist()} dtype={d['dtype']}")
        finally:
            os.unlink(tmp_path)

    def test_convert_int16x8_with_io_type(self):
        """Test int16x8 conversion WITH int16 input/output types."""
        _ensure_model()
        converter = tf.lite.TFLiteConverter.from_concrete_functions([_concrete_fn])
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = _make_rep_dataset()
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8,
            tf.lite.OpsSet.TFLITE_BUILTINS,
        ]
        converter.inference_input_type = tf.int16
        converter.inference_output_type = tf.int16
        print("  Converting int16x8 (int16 I/O)...", flush=True)
        tflite_model = converter.convert()
        self.assertIsNotNone(tflite_model)
        size_mb = len(tflite_model) / 1024 / 1024
        print(f"  int16x8 (int16 I/O) size: {size_mb:.2f} MB")

        with tempfile.NamedTemporaryFile(suffix=".tflite", delete=False) as f:
            f.write(tflite_model)
            tmp_path = f.name
        try:
            interpreter = tf.lite.Interpreter(model_path=tmp_path)
            interpreter.allocate_tensors()
            inp = interpreter.get_input_details()
            out = interpreter.get_output_details()
            for d in inp:
                self.assertEqual(d["dtype"], np.int16,
                                 f"Input {d['name']} should be int16, got {d['dtype']}")
                print(f"  input: {d['name']} shape={d['shape'].tolist()} dtype={d['dtype']} "
                      f"quant={d['quantization']}")
            for d in out:
                self.assertEqual(d["dtype"], np.int16,
                                 f"Output {d['name']} should be int16, got {d['dtype']}")
                print(f"  output: {d['name']} shape={d['shape'].tolist()} dtype={d['dtype']} "
                      f"quant={d['quantization']}")
        finally:
            os.unlink(tmp_path)

    def test_int16x8_inference(self):
        """Test int16x8 model can run inference and produce reasonable outputs."""
        _ensure_model()
        converter = tf.lite.TFLiteConverter.from_concrete_functions([_concrete_fn])
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = _make_rep_dataset()
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8,
            tf.lite.OpsSet.TFLITE_BUILTINS,
        ]
        converter.inference_input_type = tf.int16
        converter.inference_output_type = tf.int16
        tflite_model = converter.convert()

        with tempfile.NamedTemporaryFile(suffix=".tflite", delete=False) as f:
            f.write(tflite_model)
            tmp_path = f.name
        try:
            interpreter = tf.lite.Interpreter(model_path=tmp_path)
            interpreter.allocate_tensors()
            inp_details = interpreter.get_input_details()
            out_details = interpreter.get_output_details()

            # Run with calibration sample
            erb_f, spec_f, state_f = _rep_dataset_samples[10]

            for d in inp_details:
                shape = tuple(d["shape"])
                scale, zp = d["quantization"]
                if shape == (1, 1, 1, _p["nb_erb"]):
                    val = np.clip(np.round(erb_f / scale + zp), -32768, 32767).astype(np.int16)
                elif len(shape) == 5:
                    val = np.clip(np.round(spec_f / scale + zp), -32768, 32767).astype(np.int16)
                elif shape == (1, _stream_model.total_state_size):
                    val = np.clip(np.round(state_f / scale + zp), -32768, 32767).astype(np.int16)
                else:
                    self.fail(f"Unknown input shape {shape}")
                interpreter.set_tensor(d["index"], val)

            interpreter.invoke()

            for d in out_details:
                raw = interpreter.get_tensor(d["index"])
                scale, zp = d["quantization"]
                val = (raw.astype(np.float32) - zp) * scale
                self.assertFalse(np.any(np.isnan(val)),
                                 f"NaN in output {d['name']}")
                print(f"  output {d['name']}: shape={val.shape}, "
                      f"range=[{val.min():.4f}, {val.max():.4f}]")
        finally:
            os.unlink(tmp_path)


class TestInt8Conversion(unittest.TestCase):
    """Test int8 TFLite conversion step by step."""

    def test_convert_int8(self):
        """Test full int8 conversion."""
        _ensure_model()
        converter = tf.lite.TFLiteConverter.from_concrete_functions([_concrete_fn])
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = _make_rep_dataset()
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8
        ]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        print("  Converting int8...", flush=True)
        tflite_model = converter.convert()
        self.assertIsNotNone(tflite_model)
        size_mb = len(tflite_model) / 1024 / 1024
        print(f"  int8 size: {size_mb:.2f} MB")

        with tempfile.NamedTemporaryFile(suffix=".tflite", delete=False) as f:
            f.write(tflite_model)
            tmp_path = f.name
        try:
            interpreter = tf.lite.Interpreter(model_path=tmp_path)
            interpreter.allocate_tensors()
            inp = interpreter.get_input_details()
            out = interpreter.get_output_details()
            for d in inp:
                self.assertEqual(d["dtype"], np.int8,
                                 f"Input {d['name']} should be int8, got {d['dtype']}")
                print(f"  input: {d['name']} shape={d['shape'].tolist()} dtype={d['dtype']} "
                      f"quant={d['quantization']}")
            for d in out:
                self.assertEqual(d["dtype"], np.int8,
                                 f"Output {d['name']} should be int8, got {d['dtype']}")
                print(f"  output: {d['name']} shape={d['shape'].tolist()} dtype={d['dtype']} "
                      f"quant={d['quantization']}")
        finally:
            os.unlink(tmp_path)

    def test_int8_inference(self):
        """Test int8 model runs inference without NaN."""
        _ensure_model()
        converter = tf.lite.TFLiteConverter.from_concrete_functions([_concrete_fn])
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = _make_rep_dataset()
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8
        ]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        tflite_model = converter.convert()

        with tempfile.NamedTemporaryFile(suffix=".tflite", delete=False) as f:
            f.write(tflite_model)
            tmp_path = f.name
        try:
            interpreter = tf.lite.Interpreter(model_path=tmp_path)
            interpreter.allocate_tensors()
            inp_details = interpreter.get_input_details()
            out_details = interpreter.get_output_details()

            erb_f, spec_f, state_f = _rep_dataset_samples[10]

            for d in inp_details:
                shape = tuple(d["shape"])
                scale, zp = d["quantization"]
                if shape == (1, 1, 1, _p["nb_erb"]):
                    val = np.clip(np.round(erb_f / scale + zp), -128, 127).astype(np.int8)
                elif len(shape) == 5:
                    val = np.clip(np.round(spec_f / scale + zp), -128, 127).astype(np.int8)
                elif shape == (1, _stream_model.total_state_size):
                    val = np.clip(np.round(state_f / scale + zp), -128, 127).astype(np.int8)
                else:
                    self.fail(f"Unknown input shape {shape}")
                interpreter.set_tensor(d["index"], val)

            interpreter.invoke()

            for d in out_details:
                raw = interpreter.get_tensor(d["index"])
                scale, zp = d["quantization"]
                val = (raw.astype(np.float32) - zp) * scale
                self.assertFalse(np.any(np.isnan(val)),
                                 f"NaN in output {d['name']}")
                print(f"  output {d['name']}: shape={val.shape}, "
                      f"range=[{val.min():.4f}, {val.max():.4f}]")
        finally:
            os.unlink(tmp_path)


if __name__ == "__main__":
    unittest.main()
