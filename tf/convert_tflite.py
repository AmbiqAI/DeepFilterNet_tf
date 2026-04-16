"""Convert DeepFilterNet2 streaming model to TFLite (float32, int16x8, int8).

Usage:
    python convert_tflite.py [--output-dir OUTPUT_DIR]

Produces:
    dfnet2_float32.tflite
    dfnet2_int16x8.tflite
    dfnet2_int8.tflite
"""

import argparse
import os
import sys

# Force CPU to avoid Grappler GPU cluster issues
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import tensorflow as tf
import torch

from deepfilternet2_tf import DfNet, DEFAULT_PARAMS
from deepfilternet2_tf_streaming import DfNetStreaming, copy_weights_from_batch_model
from weight_transfer import transfer_weights
from compare_batch_streaming import extract_features, load_wav_mono


def build_streaming_model(ckpt_path, p=None):
    """Build streaming model with pretrained weights."""
    if p is None:
        p = DEFAULT_PARAMS

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    erb_fb_np = ckpt["erb_fb"].numpy()
    erb_inv_fb_np = ckpt["mask.erb_inv_fb"].numpy()

    # Build batch model and transfer weights
    batch_model = DfNet(erb_inv_fb_np, p, run_df=True)
    dummy_erb = tf.zeros([1, 1, 1, p["nb_erb"]])
    dummy_spec = tf.zeros([1, 1, 1, p["nb_df"], 2])
    _ = batch_model(dummy_erb, dummy_spec, training=False)
    transfer_weights(ckpt, batch_model)

    # Build streaming model and copy weights
    stream_model = DfNetStreaming(p)
    state = stream_model.get_initial_state(batch_size=1)
    _ = stream_model(dummy_erb, dummy_spec, state)
    copy_weights_from_batch_model(batch_model, stream_model)

    return stream_model, erb_fb_np, p


def get_concrete_function(stream_model, p):
    """Create a concrete function with fixed input shapes for TFLite."""
    nb_erb = p["nb_erb"]
    nb_df = p["nb_df"]
    state_size = stream_model.total_state_size

    @tf.function(input_signature=[
        tf.TensorSpec([1, 1, 1, nb_erb], tf.float32, name="feat_erb"),
        tf.TensorSpec([1, 1, 1, nb_df, 2], tf.float32, name="feat_spec"),
        tf.TensorSpec([1, state_size], tf.float32, name="state"),
    ])
    def inference(feat_erb, feat_spec, state):
        m, lsnr, df_coefs, alpha, new_state = stream_model(
            feat_erb, feat_spec, state)
        return {
            "mask": m,
            "lsnr": lsnr,
            "df_coefs": df_coefs,
            "alpha": alpha,
            "new_state": new_state,
        }

    return inference


def generate_representative_dataset(stream_model, feat_erb, feat_spec, p, num_samples=200):
    """Generate representative dataset for quantization calibration.

    Uses the actual model with real weights to evolve states realistically.
    """
    T = feat_erb.shape[2]

    # Select frames evenly across time
    if T <= num_samples:
        frames = list(range(T))
    else:
        frames = np.linspace(0, T - 1, num_samples, dtype=int).tolist()

    # Pre-compute all (input, state) pairs by running the real model
    print(f"  Running model on {len(frames)} frames to collect calibration data...", flush=True)
    samples = []
    state = stream_model.get_initial_state(batch_size=1)
    frame_set = set(frames)
    for t in range(T):
        erb_frame = feat_erb[:, :, t:t+1, :]
        spec_frame = feat_spec[:, :, t:t+1, :, :]
        if t in frame_set:
            state_np = state.numpy() if hasattr(state, 'numpy') else state
            samples.append((
                erb_frame.numpy().astype(np.float32),
                spec_frame.numpy().astype(np.float32),
                state_np.astype(np.float32),
            ))
        mask, lsnr, df_coefs, alpha, state = stream_model(erb_frame, spec_frame, state)
        state = state.numpy() if hasattr(state, 'numpy') else state  # detach from graph
    print(f"  Collected {len(samples)} calibration samples.", flush=True)

    def dataset_gen():
        for erb_np, spec_np, state_np in samples:
            yield [erb_np, spec_np, state_np]

    return dataset_gen


def convert_float32(concrete_fn, output_path):
    """Convert to float32 TFLite."""
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_fn])
    tflite_model = converter.convert()
    with open(output_path, "wb") as f:
        f.write(tflite_model)
    size_mb = len(tflite_model) / 1024 / 1024
    print(f"  float32: {output_path} ({size_mb:.2f} MB)")
    return tflite_model


def convert_int16x8(concrete_fn, representative_dataset, output_path):
    """Convert to int16x8 TFLite (16-bit activations, 8-bit weights)."""
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_fn])
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8,
        tf.lite.OpsSet.TFLITE_BUILTINS,
    ]
    converter.inference_input_type = tf.int16
    converter.inference_output_type = tf.int16
    tflite_model = converter.convert()
    with open(output_path, "wb") as f:
        f.write(tflite_model)
    size_mb = len(tflite_model) / 1024 / 1024
    print(f"  int16x8: {output_path} ({size_mb:.2f} MB)")
    return tflite_model


def convert_int8(concrete_fn, representative_dataset, output_path):
    """Convert to full int8 TFLite."""
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_fn])
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8
    ]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_model = converter.convert()
    with open(output_path, "wb") as f:
        f.write(tflite_model)
    size_mb = len(tflite_model) / 1024 / 1024
    print(f"  int8:    {output_path} ({size_mb:.2f} MB)")
    return tflite_model


def _run_quantized_conversion(saved_model_dir, cal_path, output_path, quant_type):
    """Run quantized conversion in a subprocess to isolate potential crashes."""
    import subprocess
    script = f'''
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
import tensorflow as tf

saved_model_dir = {saved_model_dir!r}
cal_path = {cal_path!r}
output_path = {output_path!r}
quant_type = {quant_type!r}

# Load calibration data
cal = np.load(cal_path)
erb_data = cal["erb"]
spec_data = cal["spec"]
state_data = cal["state"]

def representative_dataset():
    for i in range(len(erb_data)):
        yield [erb_data[i], spec_data[i], state_data[i]]

converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset

if quant_type == "int16x8":
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8,
        tf.lite.OpsSet.TFLITE_BUILTINS,
    ]
elif quant_type == "int8":
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8
    ]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

tflite_model = converter.convert()
with open(output_path, "wb") as f:
    f.write(tflite_model)
size_mb = len(tflite_model) / 1024 / 1024
print(f"  {{quant_type}}: {{output_path}} ({{size_mb:.2f}} MB)")
'''
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, timeout=300)
    print(result.stdout, end="")
    if result.returncode != 0:
        print(f"  {quant_type} subprocess failed (exit={result.returncode})")
        if result.stderr:
            # Only print last few lines of stderr
            lines = result.stderr.strip().split("\n")
            for line in lines[-10:]:
                print(f"    {line}")
    return result.returncode == 0


def validate_tflite(tflite_path, stream_model, feat_erb, feat_spec, p,
                    num_frames=10, label=""):
    """Validate TFLite model against Keras streaming model."""
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Build name->index maps
    inp_map = {d["name"]: d for d in input_details}

    # Match outputs by shape since TFLite uses generic "Identity" names
    state_size = stream_model.total_state_size
    nb_erb = p["nb_erb"]
    mask_det = None
    new_state_det = None
    for d in output_details:
        shape = tuple(d["shape"])
        if shape == (1, 1, 1, nb_erb):
            mask_det = d
        elif shape == (1, state_size):
            new_state_det = d

    erb_det = inp_map.get("feat_erb") or inp_map.get("serving_default_feat_erb:0")
    spec_det = inp_map.get("feat_spec") or inp_map.get("serving_default_feat_spec:0")
    state_det = inp_map.get("state") or inp_map.get("serving_default_state:0")

    if erb_det is None or spec_det is None or state_det is None:
        for d in input_details:
            shape = tuple(d["shape"])
            if shape == (1, 1, 1, nb_erb):
                erb_det = d
            elif len(shape) == 5:
                spec_det = d
            elif shape == (1, state_size):
                state_det = d

    if erb_det is None or spec_det is None or state_det is None or mask_det is None:
        print(f"  Warning: Could not identify all tensors for {label}")
        print(f"  Inputs: {[(d['name'], d['shape']) for d in input_details]}")
        print(f"  Outputs: {[(d['name'], d['shape']) for d in output_details]}")
        return

    def quantize_input(value, detail):
        """Quantize float input for int8 models."""
        if detail["dtype"] == np.int8:
            scale, zero_point = detail["quantization"]
            return np.clip(np.round(value / scale + zero_point), -128, 127).astype(np.int8)
        elif detail["dtype"] == np.int16:
            scale, zero_point = detail["quantization"]
            return np.clip(np.round(value / scale + zero_point), -32768, 32767).astype(np.int16)
        return value.astype(np.float32)

    def dequantize_output(value, detail):
        """Dequantize output for quantized models."""
        if detail["dtype"] in (np.int8, np.int16):
            scale, zero_point = detail["quantization"]
            return (value.astype(np.float32) - zero_point) * scale
        return value.astype(np.float32)

    # Run Keras model
    state_keras = stream_model.get_initial_state(batch_size=1)
    state_tflite = np.zeros([1, stream_model.total_state_size], dtype=np.float32)

    T = min(num_frames, feat_erb.shape[2])
    mask_diffs = []

    for t in range(T):
        erb_frame = feat_erb[:, :, t:t+1, :].numpy().astype(np.float32)
        spec_frame = feat_spec[:, :, t:t+1, :, :].numpy().astype(np.float32)

        # Keras
        m_k, lsnr_k, _, _, state_keras = stream_model(
            tf.constant(erb_frame), tf.constant(spec_frame), state_keras)

        # TFLite
        interpreter.set_tensor(erb_det["index"], quantize_input(erb_frame, erb_det))
        interpreter.set_tensor(spec_det["index"], quantize_input(spec_frame, spec_det))
        interpreter.set_tensor(state_det["index"], quantize_input(state_tflite, state_det))
        interpreter.invoke()

        m_tfl = dequantize_output(
            interpreter.get_tensor(mask_det["index"]), mask_det)
        if new_state_det is not None:
            state_tflite = dequantize_output(
                interpreter.get_tensor(new_state_det["index"]), new_state_det)

        mask_diff = np.abs(m_k.numpy() - m_tfl).max()
        mask_diffs.append(mask_diff)

    max_diff = max(mask_diffs)
    mean_diff = np.mean(mask_diffs)
    print(f"  {label} validation ({T} frames): mask max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")


def main():
    parser = argparse.ArgumentParser(description="Convert DeepFilterNet2 to TFLite")
    # Resolve paths relative to this script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_output = os.path.join(script_dir, "..", "models", "tflite")
    default_ckpt = os.path.join(script_dir, "..", "models", "DeepFilterNet2",
                                "checkpoints", "model_96.ckpt.best")
    default_audio = os.path.join(script_dir, "..", "models",
                                 "streaming_audio_samples", "noisy.wav")

    parser.add_argument("--output-dir", default=default_output,
                        help="Output directory for TFLite files")
    parser.add_argument("--ckpt", default=default_ckpt,
                        help="PyTorch checkpoint path")
    parser.add_argument("--audio", default=default_audio,
                        help="Audio file for calibration and validation")
    parser.add_argument("--num-cal", type=int, default=200,
                        help="Number of calibration samples for quantization")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    p = DEFAULT_PARAMS

    print("Building streaming model...", flush=True)
    stream_model, erb_fb_np, p = build_streaming_model(args.ckpt, p)

    print("Creating concrete function...", flush=True)
    concrete_fn = get_concrete_function(stream_model, p)
    # Trace to get the concrete function
    concrete_fn = concrete_fn.get_concrete_function()

    # Load audio for calibration and validation
    audio, _ = load_wav_mono(args.audio)
    _, feat_erb, feat_spec = extract_features(audio, erb_fb_np, p)

    # --- Float32 ---
    print("\nConverting float32...", flush=True)
    float_path = os.path.join(args.output_dir, "dfnet2_float32.tflite")
    convert_float32(concrete_fn, float_path)
    sys.stdout.flush()
    validate_tflite(float_path, stream_model, feat_erb, feat_spec, p,
                    num_frames=20, label="float32")
    sys.stdout.flush()

    # --- Representative dataset for quantization ---
    print("\nGenerating representative dataset for quantization...", flush=True)
    rep_dataset = generate_representative_dataset(
        stream_model, feat_erb, feat_spec, p, num_samples=args.num_cal)

    # --- Int16x8 ---
    print("\nConverting int16x8...", flush=True)
    int16x8_path = os.path.join(args.output_dir, "dfnet2_int16x8.tflite")
    try:
        convert_int16x8(concrete_fn, rep_dataset, int16x8_path)
        sys.stdout.flush()
        validate_tflite(int16x8_path, stream_model, feat_erb, feat_spec, p,
                        num_frames=20, label="int16x8")
    except Exception as e:
        print(f"  int16x8 conversion failed: {e}")
    sys.stdout.flush()

    # --- Int8 ---
    print("\nConverting int8...", flush=True)
    int8_path = os.path.join(args.output_dir, "dfnet2_int8.tflite")
    try:
        # Need fresh representative dataset generator
        rep_dataset_int8 = generate_representative_dataset(
            stream_model, feat_erb, feat_spec, p, num_samples=args.num_cal)
        convert_int8(concrete_fn, rep_dataset_int8, int8_path)
        sys.stdout.flush()
        validate_tflite(int8_path, stream_model, feat_erb, feat_spec, p,
                        num_frames=20, label="int8")
    except Exception as e:
        print(f"  int8 conversion failed: {e}")
    sys.stdout.flush()

    print("\nDone!", flush=True)


if __name__ == "__main__":
    main()
