"""Test int16x8 TFLite conversion module by module to identify crash source.

Each test converts a small sub-model in a subprocess to isolate SIGABRT crashes.
"""
import os, subprocess, sys, json

os.environ["CUDA_VISIBLE_DEVICES"] = ""

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(SCRIPT_DIR, "..", "models", "DeepFilterNet2",
                     "checkpoints", "model_96.ckpt.best")
AUDIO = os.path.join(SCRIPT_DIR, "..", "models", "streaming_audio_samples", "noisy.wav")


def run_test(name, script):
    """Run script in subprocess, return (success, output)."""
    full_script = f'''
import os; os.environ["CUDA_VISIBLE_DEVICES"] = ""
import faulthandler; faulthandler.enable()
import numpy as np
import tensorflow as tf
{script}
'''
    r = subprocess.run(
        [sys.executable, "-c", full_script],
        capture_output=True, text=True, timeout=120, cwd=SCRIPT_DIR)
    stdout = r.stdout.strip()
    if r.returncode == 0:
        print(f"  [PASS] {name}: {stdout[-200:]}")
    else:
        sig = -r.returncode if r.returncode < 0 else r.returncode
        stderr_last = r.stderr.strip().split("\n")[-5:]
        print(f"  [FAIL] {name}: exit={sig}")
        for line in stderr_last:
            print(f"    {line}")
    return r.returncode == 0


def make_int16x8_test(model_fn, input_specs, rep_data_fn, extra_setup=""):
    """Generate script to test int16x8 conversion of a small model."""
    return f'''
{extra_setup}

model = {model_fn}
# Build model
inputs = [{', '.join(f'tf.keras.Input(shape={s[1:]}, dtype=tf.float32)' for s in input_specs)}]
if len(inputs) == 1:
    outputs = model(inputs[0])
else:
    outputs = model(inputs)
func_model = tf.keras.Model(inputs=inputs, outputs=outputs)

# Create concrete function
input_sig = [{', '.join(f'tf.TensorSpec({s}, tf.float32)' for s in input_specs)}]
@tf.function(input_signature=input_sig)
def inference({', '.join(f'x{i}' for i in range(len(input_specs)))}):
    return func_model([{', '.join(f'x{i}' for i in range(len(input_specs)))}], training=False)
cf = inference.get_concrete_function()

# Rep dataset
def rep_ds():
    for _ in range(20):
        yield [{rep_data_fn}]

converter = tf.lite.TFLiteConverter.from_concrete_functions([cf])
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = rep_ds
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8,
    tf.lite.OpsSet.TFLITE_BUILTINS,
]
converter.inference_input_type = tf.int16
converter.inference_output_type = tf.int16
m = converter.convert()
print(f"size={{len(m)}} bytes")
'''


def test_standalone_ops():
    """Test basic ops individually."""
    print("\n=== Standalone Ops ===")

    # Dense
    run_test("Dense(256->64)", '''
layer = tf.keras.layers.Dense(64)
x = tf.keras.Input(shape=(256,))
m = tf.keras.Model(x, layer(x))

@tf.function(input_signature=[tf.TensorSpec([1, 256], tf.float32)])
def f(x): return m(x, training=False)
cf = f.get_concrete_function()

def rep(): 
    for _ in range(20): yield [np.random.randn(1,256).astype(np.float32)]

c = tf.lite.TFLiteConverter.from_concrete_functions([cf])
c.optimizations = [tf.lite.Optimize.DEFAULT]
c.representative_dataset = rep
c.target_spec.supported_ops = [
    tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8,
    tf.lite.OpsSet.TFLITE_BUILTINS]
c.inference_input_type = tf.int16
c.inference_output_type = tf.int16
r = c.convert()
print(f"size={len(r)}")
''')

    # Conv2D
    run_test("Conv2D(1->64, 3x1)", '''
layer = tf.keras.layers.Conv2D(64, (3,1), padding="valid")
x = tf.keras.Input(shape=(3, 32, 1))
m = tf.keras.Model(x, layer(x))

@tf.function(input_signature=[tf.TensorSpec([1, 3, 32, 1], tf.float32)])
def f(x): return m(x, training=False)
cf = f.get_concrete_function()

def rep():
    for _ in range(20): yield [np.random.randn(1,3,32,1).astype(np.float32)]

c = tf.lite.TFLiteConverter.from_concrete_functions([cf])
c.optimizations = [tf.lite.Optimize.DEFAULT]
c.representative_dataset = rep
c.target_spec.supported_ops = [
    tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8,
    tf.lite.OpsSet.TFLITE_BUILTINS]
c.inference_input_type = tf.int16
c.inference_output_type = tf.int16
r = c.convert()
print(f"size={len(r)}")
''')

    # BatchNorm
    run_test("Conv2D+BatchNorm", '''
x = tf.keras.Input(shape=(3, 32, 1))
h = tf.keras.layers.Conv2D(64, (3,1), padding="valid", use_bias=False)(x)
h = tf.keras.layers.BatchNormalization(epsilon=1e-5)(h)
h = tf.keras.layers.ReLU()(h)
m = tf.keras.Model(x, h)

@tf.function(input_signature=[tf.TensorSpec([1, 3, 32, 1], tf.float32)])
def f(x): return m(x, training=False)
cf = f.get_concrete_function()

def rep():
    for _ in range(20): yield [np.random.randn(1,3,32,1).astype(np.float32)]

c = tf.lite.TFLiteConverter.from_concrete_functions([cf])
c.optimizations = [tf.lite.Optimize.DEFAULT]
c.representative_dataset = rep
c.target_spec.supported_ops = [
    tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8,
    tf.lite.OpsSet.TFLITE_BUILTINS]
c.inference_input_type = tf.int16
c.inference_output_type = tf.int16
r = c.convert()
print(f"size={len(r)}")
''')

    # GRU-like matmul (UnrolledGRU pattern)
    run_test("MatMul+Sigmoid (GRU gate)", '''
kernel = tf.Variable(tf.random.normal([256, 768]))
bias = tf.Variable(tf.zeros([768]))
x = tf.keras.Input(shape=(256,))
h = tf.matmul(x, kernel) + bias
h = tf.sigmoid(h)
m = tf.keras.Model(x, h)

@tf.function(input_signature=[tf.TensorSpec([1, 256], tf.float32)])
def f(x): return m(x, training=False)
cf = f.get_concrete_function()

def rep():
    for _ in range(20): yield [np.random.randn(1,256).astype(np.float32)]

c = tf.lite.TFLiteConverter.from_concrete_functions([cf])
c.optimizations = [tf.lite.Optimize.DEFAULT]
c.representative_dataset = rep
c.target_spec.supported_ops = [
    tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8,
    tf.lite.OpsSet.TFLITE_BUILTINS]
c.inference_input_type = tf.int16
c.inference_output_type = tf.int16
r = c.convert()
print(f"size={len(r)}")
''')


def test_dfnet_modules():
    """Test actual DfNet submodules."""
    print("\n=== DfNet Modules ===")

    # Full streaming model (the one that crashes)
    run_test("Full DfNetStreaming", f'''
import sys; sys.path.insert(0, {SCRIPT_DIR!r})
from convert_tflite import build_streaming_model, get_concrete_function, generate_representative_dataset, DEFAULT_PARAMS
from compare_batch_streaming import extract_features, load_wav_mono

p = DEFAULT_PARAMS.copy()
stream_model, erb_fb_np, p = build_streaming_model({CKPT!r}, p)
concrete_fn = get_concrete_function(stream_model, p).get_concrete_function()

audio, _ = load_wav_mono({AUDIO!r})
_, feat_erb, feat_spec = extract_features(audio, erb_fb_np, p)
gen = generate_representative_dataset(stream_model, feat_erb, feat_spec, p, num_samples=10)
samples = list(gen())

def rep_ds():
    for s in samples:
        yield s

converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_fn])
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = rep_ds
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8,
    tf.lite.OpsSet.TFLITE_BUILTINS,
]
converter.inference_input_type = tf.int16
converter.inference_output_type = tf.int16
print("Starting convert()...", flush=True)
m = converter.convert()
print(f"size={{len(m)}} bytes")
''')

    # Same but WITHOUT int16 I/O types
    run_test("Full DfNetStreaming (float I/O)", f'''
import sys; sys.path.insert(0, {SCRIPT_DIR!r})
from convert_tflite import build_streaming_model, get_concrete_function, generate_representative_dataset, DEFAULT_PARAMS
from compare_batch_streaming import extract_features, load_wav_mono

p = DEFAULT_PARAMS.copy()
stream_model, erb_fb_np, p = build_streaming_model({CKPT!r}, p)
concrete_fn = get_concrete_function(stream_model, p).get_concrete_function()

audio, _ = load_wav_mono({AUDIO!r})
_, feat_erb, feat_spec = extract_features(audio, erb_fb_np, p)
gen = generate_representative_dataset(stream_model, feat_erb, feat_spec, p, num_samples=10)
samples = list(gen())

def rep_ds():
    for s in samples:
        yield s

converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_fn])
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = rep_ds
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8,
    tf.lite.OpsSet.TFLITE_BUILTINS,
]
# NO inference_input_type / inference_output_type
print("Starting convert()...", flush=True)
m = converter.convert()
print(f"size={{len(m)}} bytes")
''')

    # Same but dynamic range quantization only (no representative dataset)
    run_test("Full DfNetStreaming (dynamic range only)", f'''
import sys; sys.path.insert(0, {SCRIPT_DIR!r})
from convert_tflite import build_streaming_model, get_concrete_function, DEFAULT_PARAMS

p = DEFAULT_PARAMS.copy()
stream_model, erb_fb_np, p = build_streaming_model({CKPT!r}, p)
concrete_fn = get_concrete_function(stream_model, p).get_concrete_function()

converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_fn])
converter.optimizations = [tf.lite.Optimize.DEFAULT]
# No representative dataset, no supported_ops override = dynamic range quant
print("Starting convert()...", flush=True)
m = converter.convert()
print(f"size={{len(m)}} bytes")
''')


if __name__ == "__main__":
    test_standalone_ops()
    test_dfnet_modules()
    print("\nDone.")
