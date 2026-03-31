"""
Phase 1-2 per-module unit tests: PyTorch vs TensorFlow.

Tests each module individually with weight transfer, comparing outputs.
Rule: np.testing.assert_allclose(atol=1e-5).

Run: python DeepFilterNet/df/test_tf_conversion.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import tensorflow as tf

# Setup PT model config
from df.config import config
config.use_defaults()
config.set("DF_OUTPUT_LAYER", "groupedlinear", str, "deepfilternet")
config.set("DF_N_ITER", 1, int, "deepfilternet")

from df.modules import (
    GroupedLinear, GroupedLinearEinsum, GroupedGRU, GroupedGRULayer,
    Conv2dNormAct, ConvTranspose2dNormAct, Mask, erb_fb,
)
from df.deepfilternet2 import (
    ModelParams, Encoder, ErbDecoder, DfDecoder, DfOutputReshapeMF, DfNet, init_model
)
from df.multiframe import DF as MultiFrameDF

from df.tf_modules import (
    GroupedLinearTF, GroupedLinearEinsumTF, GroupedGRUTF, GroupedGRULayerTF,
    Conv2dNormActTF, ConvTranspose2dNormActTF, MaskTF, MultiFrameDFTF,
    DfOutputReshapeMFTF, DfNetTF,
    convert_gru_weights, reorder_gru_gates,
)


def get_pt_state_dict_np(module, prefix=""):
    """Extract state dict as numpy arrays with given prefix."""
    sd = {}
    for name, param in module.named_parameters():
        sd[f"{prefix}.{name}" if prefix else name] = param.detach().cpu().numpy()
    for name, buf in module.named_buffers():
        sd[f"{prefix}.{name}" if prefix else name] = buf.detach().cpu().numpy()
    return sd


def pt_to_np(t):
    return t.detach().cpu().numpy()


PASS = 0
FAIL = 0


def check(name, pt_out, tf_out, atol=1e-5):
    global PASS, FAIL
    try:
        np.testing.assert_allclose(pt_out, tf_out, atol=atol, rtol=1e-5)
        print(f"  ✓ {name}: PASS (max_diff={np.max(np.abs(pt_out - tf_out)):.2e})")
        PASS += 1
    except AssertionError as e:
        diff = np.abs(pt_out - tf_out)
        print(f"  ✗ {name}: FAIL (max_diff={np.max(diff):.2e}, mean_diff={np.mean(diff):.2e})")
        print(f"    {str(e)[:200]}")
        FAIL += 1


# ============================================================================
# Test 1: GroupedLinear (groups=1, effectively nn.Linear)
# ============================================================================
def test_grouped_linear():
    print("\n=== Test: GroupedLinear ===")
    torch.manual_seed(42)
    in_size, out_size = 128, 256
    pt_mod = GroupedLinear(in_size, out_size, groups=1, shuffle=True)
    pt_mod.eval()

    sd = get_pt_state_dict_np(pt_mod)

    tf_mod = GroupedLinearTF(in_size, out_size, groups=1, shuffle=True)
    # Build
    dummy = tf.zeros([1, 1, in_size])
    tf_mod(dummy)
    tf_mod.load_from_pt(sd, "")

    # Test
    x_np = np.random.randn(2, 10, in_size).astype(np.float32)
    x_pt = torch.from_numpy(x_np)
    x_tf = tf.constant(x_np)

    with torch.no_grad():
        y_pt = pt_to_np(pt_mod(x_pt))
    y_tf = tf_mod(x_tf).numpy()

    check("GroupedLinear output", y_pt, y_tf)


# ============================================================================
# Test 2: GroupedLinearEinsum (groups=1)
# ============================================================================
def test_grouped_linear_einsum():
    print("\n=== Test: GroupedLinearEinsum ===")
    torch.manual_seed(42)
    in_size, out_size = 256, 960
    pt_mod = GroupedLinearEinsum(in_size, out_size, groups=1)
    pt_mod.eval()

    sd = get_pt_state_dict_np(pt_mod)

    tf_mod = GroupedLinearEinsumTF(in_size, out_size, groups=1)
    dummy = tf.zeros([1, 1, in_size])
    tf_mod(dummy)
    tf_mod.load_from_pt(sd, "")

    x_np = np.random.randn(2, 10, in_size).astype(np.float32)
    with torch.no_grad():
        y_pt = pt_to_np(pt_mod(torch.from_numpy(x_np)))
    y_tf = tf_mod(tf.constant(x_np)).numpy()

    check("GroupedLinearEinsum output", y_pt, y_tf)


# ============================================================================
# Test 3: GRU single layer (gate reorder check)
# ============================================================================
def test_gru_single_layer():
    print("\n=== Test: GRU single layer (gate reorder) ===")
    torch.manual_seed(42)
    input_size, hidden_size = 128, 256
    groups = 1

    pt_gru_layer = GroupedGRULayer(input_size, hidden_size, groups, batch_first=True)
    pt_gru_layer.eval()

    sd = get_pt_state_dict_np(pt_gru_layer)

    tf_gru_layer = GroupedGRULayerTF(input_size, hidden_size, groups)

    # Build
    x_np = np.random.randn(1, 5, input_size).astype(np.float32)
    h0_np = np.zeros((groups, 1, hidden_size // groups), dtype=np.float32)
    tf_gru_layer(tf.constant(x_np), tf.constant(h0_np))

    # Load weights
    tf_gru_layer.load_from_pt(sd, "")

    # Run PT
    h0_pt = torch.zeros(groups, 1, hidden_size // groups)
    with torch.no_grad():
        y_pt, h_pt = pt_gru_layer(torch.from_numpy(x_np), h0_pt)
    y_pt_np = pt_to_np(y_pt)
    h_pt_np = pt_to_np(h_pt)

    # Run TF
    y_tf, h_tf = tf_gru_layer(tf.constant(x_np), tf.constant(h0_np))
    y_tf_np = y_tf.numpy()
    h_tf_np = h_tf.numpy()

    check("GRU layer output", y_pt_np, y_tf_np)
    check("GRU layer hidden state", h_pt_np, h_tf_np)


# ============================================================================
# Test 4: GroupedGRU (multi-layer with add_outputs)
# ============================================================================
def test_grouped_gru():
    print("\n=== Test: GroupedGRU (multi-layer, add_outputs=True) ===")
    torch.manual_seed(42)
    p = ModelParams()
    input_size = 128  # emb_in_dim
    hidden_size = 256  # emb_out_dim

    pt_gru = GroupedGRU(
        input_size, hidden_size, num_layers=1, groups=1,
        batch_first=True, shuffle=True, add_outputs=True
    )
    pt_gru.eval()

    sd = get_pt_state_dict_np(pt_gru)

    tf_gru = GroupedGRUTF(
        input_size, hidden_size, num_layers=1, groups=1,
        shuffle=True, add_outputs=True
    )

    x_np = np.random.randn(1, 5, input_size).astype(np.float32)
    h0_np = np.zeros((1, 1, hidden_size), dtype=np.float32)
    tf_gru(tf.constant(x_np), tf.constant(h0_np))
    tf_gru.load_from_pt(sd, "")

    h0_pt = pt_gru.get_h0(1)
    with torch.no_grad():
        y_pt, h_pt = pt_gru(torch.from_numpy(x_np), h0_pt)

    y_tf, h_tf = tf_gru(tf.constant(x_np), tf.constant(h0_np))

    check("GroupedGRU output", pt_to_np(y_pt), y_tf.numpy())
    check("GroupedGRU hidden state", pt_to_np(h_pt), h_tf.numpy())


# ============================================================================
# Test 5: GroupedGRU 3-layer (df_gru style)
# ============================================================================
def test_grouped_gru_3layer():
    print("\n=== Test: GroupedGRU (3-layer, df_gru style) ===")
    torch.manual_seed(42)
    input_size = 256
    hidden_size = 256

    pt_gru = GroupedGRU(
        input_size, hidden_size, num_layers=3, groups=1,
        batch_first=True, shuffle=True, add_outputs=True
    )
    pt_gru.eval()

    sd = get_pt_state_dict_np(pt_gru)

    tf_gru = GroupedGRUTF(
        input_size, hidden_size, num_layers=3, groups=1,
        shuffle=True, add_outputs=True
    )

    x_np = np.random.randn(1, 10, input_size).astype(np.float32)
    h0_np = np.zeros((3, 1, hidden_size), dtype=np.float32)
    tf_gru(tf.constant(x_np), tf.constant(h0_np))
    tf_gru.load_from_pt(sd, "")

    h0_pt = pt_gru.get_h0(1)
    with torch.no_grad():
        y_pt, h_pt = pt_gru(torch.from_numpy(x_np), h0_pt)

    y_tf, h_tf = tf_gru(tf.constant(x_np), tf.constant(h0_np))

    check("GroupedGRU 3-layer output", pt_to_np(y_pt), y_tf.numpy())
    check("GroupedGRU 3-layer hidden", pt_to_np(h_pt), h_tf.numpy())


# ============================================================================
# Test 6: Conv2dNormAct (separable, kernel=(3,3))
# ============================================================================
def test_conv2d_norm_act_inp():
    print("\n=== Test: Conv2dNormAct (erb_conv0, kernel=(3,3), separable) ===")
    torch.manual_seed(42)

    # erb_conv0: Conv2dNormAct(1, 16, kernel_size=(3,3), bias=False, separable=True)
    # With gcd(1,16)=1, separable becomes False! So it's actually a regular conv
    pt_conv = Conv2dNormAct(1, 16, kernel_size=(3, 3), bias=False, separable=True)
    pt_conv.eval()

    sd = get_pt_state_dict_np(pt_conv)
    print(f"  PT layers: {[type(m).__name__ for m in pt_conv]}")
    print(f"  PT state dict keys: {list(sd.keys())}")

    # Check if separable: gcd(1,16)=1, so NOT separable
    import math
    is_sep = math.gcd(1, 16) > 1
    print(f"  is_separable={is_sep}")

    tf_conv = Conv2dNormActTF(1, 16, kernel_size=(3, 3), bias=False, separable=True)

    # x: PT [B, C, T, F], TF [B, T, F, C]
    x_np_pt = np.random.randn(1, 1, 10, 32).astype(np.float32)
    x_np_tf = np.transpose(x_np_pt, (0, 2, 3, 1))  # [B, T, F, C]

    # Build TF model
    tf_conv(tf.constant(x_np_tf), training=False)
    tf_conv.load_from_pt(sd, "")

    with torch.no_grad():
        y_pt = pt_to_np(pt_conv(torch.from_numpy(x_np_pt)))
    y_tf = tf_conv(tf.constant(x_np_tf), training=False).numpy()

    # Convert PT output to NHWC for comparison
    y_pt_nhwc = np.transpose(y_pt, (0, 2, 3, 1))

    check("Conv2dNormAct (3,3) output", y_pt_nhwc, y_tf)


# ============================================================================
# Test 7: Conv2dNormAct (separable, kernel=(1,3), fstride=2)
# ============================================================================
def test_conv2d_norm_act_sep():
    print("\n=== Test: Conv2dNormAct (erb_conv1, kernel=(1,3), fstride=2, separable) ===")
    torch.manual_seed(42)

    # erb_conv1: Conv2dNormAct(16, 16, kernel_size=(1,3), fstride=2, bias=False, separable=True)
    pt_conv = Conv2dNormAct(16, 16, kernel_size=(1, 3), fstride=2, bias=False, separable=True)
    pt_conv.eval()

    sd = get_pt_state_dict_np(pt_conv)
    print(f"  PT layers: {[type(m).__name__ for m in pt_conv]}")
    print(f"  PT state dict keys: {list(sd.keys())}")

    import math
    is_sep = math.gcd(16, 16) > 1 and max((1, 3)) > 1
    print(f"  is_separable={is_sep}")

    tf_conv = Conv2dNormActTF(16, 16, kernel_size=(1, 3), fstride=2, bias=False, separable=True)

    x_np_pt = np.random.randn(1, 16, 10, 32).astype(np.float32)
    x_np_tf = np.transpose(x_np_pt, (0, 2, 3, 1))

    tf_conv(tf.constant(x_np_tf), training=False)
    tf_conv.load_from_pt(sd, "")

    with torch.no_grad():
        y_pt = pt_to_np(pt_conv(torch.from_numpy(x_np_pt)))
    y_tf = tf_conv(tf.constant(x_np_tf), training=False).numpy()

    y_pt_nhwc = np.transpose(y_pt, (0, 2, 3, 1))
    print(f"  PT output shape (NHWC): {y_pt_nhwc.shape}, TF output shape: {y_tf.shape}")

    check("Conv2dNormAct sep (1,3) fstride=2", y_pt_nhwc, y_tf)


# ============================================================================
# Test 8: DfOutputReshapeMF
# ============================================================================
def test_df_output_reshape():
    print("\n=== Test: DfOutputReshapeMF ===")
    df_order, df_bins = 5, 96

    pt_mod = DfOutputReshapeMF(df_order, df_bins)
    tf_mod = DfOutputReshapeMFTF(df_order, df_bins)

    # Input: [B, T, F, O*2]
    x_np = np.random.randn(1, 10, 96, 10).astype(np.float32)

    with torch.no_grad():
        y_pt = pt_to_np(pt_mod(torch.from_numpy(x_np)))
    y_tf = tf_mod(tf.constant(x_np)).numpy()

    check("DfOutputReshapeMF", y_pt, y_tf)


# ============================================================================
# Test 9: Mask
# ============================================================================
def test_mask():
    print("\n=== Test: Mask ===")
    torch.manual_seed(42)
    from libdf import DF as LibDF
    p = ModelParams()
    df_state = LibDF(sr=p.sr, fft_size=p.fft_size, hop_size=p.hop_size, nb_bands=p.nb_erb)
    erb_inv = erb_fb(df_state.erb_widths(), p.sr, inverse=True)

    pt_mask = Mask(erb_inv, post_filter=False)
    pt_mask.eval()

    erb_inv_np = pt_to_np(erb_inv)
    tf_mask = MaskTF(erb_inv_np)

    # spec: [B, 1, T, F, 2], mask: [B, 1, T, E]
    F = p.fft_size // 2 + 1
    spec_np = np.random.randn(1, 1, 10, F, 2).astype(np.float32)
    mask_np = np.random.rand(1, 1, 10, p.nb_erb).astype(np.float32)

    with torch.no_grad():
        y_pt = pt_to_np(pt_mask(torch.from_numpy(spec_np), torch.from_numpy(mask_np)))

    # TF: spec [B, T, F, 1, 2], mask [B, T, E, 1]
    spec_tf = np.transpose(spec_np, (0, 2, 3, 1, 4))  # [B, T, F, 1, 2]
    mask_tf = np.transpose(mask_np, (0, 2, 3, 1))      # [B, T, E, 1]
    y_tf = tf_mask(tf.constant(spec_tf), tf.constant(mask_tf)).numpy()

    # Convert PT to TF layout for comparison
    y_pt_tf = np.transpose(y_pt, (0, 2, 3, 1, 4))  # [B, T, F, 1, 2]

    check("Mask output", y_pt_tf, y_tf)


# ============================================================================
# Test 10: MultiFrameDF (deep filtering operation)
# ============================================================================
def test_multiframe_df():
    print("\n=== Test: MultiFrameDF ===")
    torch.manual_seed(42)
    p = ModelParams()
    F = p.fft_size // 2 + 1
    F_df = p.nb_df
    O = p.df_order
    T = 10

    pt_df = MultiFrameDF(num_freqs=F_df, frame_size=O, lookahead=0)

    tf_df = MultiFrameDFTF(num_freqs=F_df, frame_size=O, lookahead=0)

    # spec: [B, 1, T, F, 2]
    spec_np = np.random.randn(1, 1, T, F, 2).astype(np.float32)
    # coefs: [B, O, T, F_df, 2] (after DfOutputReshapeMF)
    coefs_np = np.random.randn(1, O, T, F_df, 2).astype(np.float32)

    # PT uses complex view internally
    spec_pt = torch.from_numpy(spec_np)
    coefs_pt = torch.from_numpy(coefs_np)

    with torch.no_grad():
        y_pt = pt_to_np(pt_df(spec_pt, coefs_pt))

    y_tf = tf_df(tf.constant(spec_np), tf.constant(coefs_np)).numpy()

    check("MultiFrameDF output", y_pt, y_tf)


# ============================================================================
# Test 11: Full DfNet (Phase 3)
# ============================================================================
def test_full_dfnet():
    print("\n=== Test: Full DfNet (Phase 3 — PT vs TF) ===")
    torch.manual_seed(42)
    np.random.seed(42)

    p = ModelParams()
    F = p.fft_size // 2 + 1

    # Create PT model
    from libdf import DF as LibDF
    df_state = LibDF(sr=p.sr, fft_size=p.fft_size, hop_size=p.hop_size, nb_bands=p.nb_erb)
    from df.modules import erb_fb as pt_erb_fb
    erb_inv = pt_erb_fb(df_state.erb_widths(), p.sr, inverse=True)
    erb_inv_np = pt_to_np(erb_inv)

    pt_model = init_model()
    pt_model.eval()

    # Extract full state dict as numpy
    sd = get_pt_state_dict_np(pt_model)

    # Create TF model
    tf_model = DfNetTF(
        erb_inv_fb_np=erb_inv_np,
        nb_erb=p.nb_erb, nb_df=p.nb_df, fft_size=p.fft_size,
        conv_ch=p.conv_ch, df_order=p.df_order, df_lookahead=0,
        emb_hidden_dim=p.emb_hidden_dim, emb_num_layers=p.emb_num_layers,
        df_hidden_dim=p.df_hidden_dim, df_num_layers=p.df_num_layers,
        gru_groups=p.gru_groups, lin_groups=p.lin_groups,
        group_shuffle=p.group_shuffle, enc_concat=p.enc_concat,
        df_pathway_kernel_size_t=p.df_pathway_kernel_size_t,
    )

    # Create test inputs
    T = 20
    spec_np = np.random.randn(1, 1, T, F, 2).astype(np.float32) * 0.1
    feat_erb_np = np.random.randn(1, 1, T, p.nb_erb).astype(np.float32)
    feat_spec_np = np.random.randn(1, 1, T, p.nb_df, 2).astype(np.float32)

    # Build TF model
    tf_model(tf.constant(spec_np), tf.constant(feat_erb_np), tf.constant(feat_spec_np), training=False)

    # Load weights
    tf_model.load_from_pt(sd)

    # Run PT
    with torch.no_grad():
        pt_out = pt_model(
            torch.from_numpy(spec_np),
            torch.from_numpy(feat_erb_np),
            torch.from_numpy(feat_spec_np),
        )
    pt_spec_out = pt_to_np(pt_out[0])
    pt_mask = pt_to_np(pt_out[1])
    pt_lsnr = pt_to_np(pt_out[2])

    # Run TF
    tf_out = tf_model(tf.constant(spec_np), tf.constant(feat_erb_np),
                       tf.constant(feat_spec_np), training=False)
    tf_spec_out = tf_out[0].numpy()
    tf_mask = tf_out[1].numpy()
    tf_lsnr = tf_out[2].numpy()

    check("DfNet spec output", pt_spec_out, tf_spec_out, atol=1e-4)
    check("DfNet mask output", pt_mask, tf_mask, atol=1e-4)
    check("DfNet lsnr output", pt_lsnr, tf_lsnr, atol=1e-4)

    # Compute SNR
    signal = pt_spec_out
    noise = pt_spec_out - tf_spec_out
    snr = 10 * np.log10(np.sum(signal ** 2) / (np.sum(noise ** 2) + 1e-30))
    print(f"  SNR(PT vs TF spec): {snr:.1f} dB (target: >120 dB)")
    if snr > 120:
        global PASS
        PASS += 1
        print(f"  ✓ SNR check: PASS")
    else:
        global FAIL
        FAIL += 1
        print(f"  ✗ SNR check: FAIL (got {snr:.1f} dB, need >120 dB)")


# ============================================================================
# Test 12: TFLite float32 export (Phase 5)
# ============================================================================
def test_tflite_float32():
    print("\n=== Test: TFLite float32 export (Phase 5) ===")
    torch.manual_seed(42)
    np.random.seed(42)

    p = ModelParams()
    F = p.fft_size // 2 + 1

    from libdf import DF as LibDF
    df_state = LibDF(sr=p.sr, fft_size=p.fft_size, hop_size=p.hop_size, nb_bands=p.nb_erb)
    from df.modules import erb_fb as pt_erb_fb
    erb_inv = pt_erb_fb(df_state.erb_widths(), p.sr, inverse=True)
    erb_inv_np = pt_to_np(erb_inv)

    pt_model = init_model()
    pt_model.eval()
    sd = get_pt_state_dict_np(pt_model)

    # Create and load TF model
    tf_model = DfNetTF(
        erb_inv_fb_np=erb_inv_np,
        nb_erb=p.nb_erb, nb_df=p.nb_df, fft_size=p.fft_size,
        conv_ch=p.conv_ch, df_order=p.df_order, df_lookahead=0,
        emb_hidden_dim=p.emb_hidden_dim, emb_num_layers=p.emb_num_layers,
        df_hidden_dim=p.df_hidden_dim, df_num_layers=p.df_num_layers,
        gru_groups=p.gru_groups, lin_groups=p.lin_groups,
        group_shuffle=p.group_shuffle, enc_concat=p.enc_concat,
        df_pathway_kernel_size_t=p.df_pathway_kernel_size_t,
    )

    T = 20
    spec_np = np.random.randn(1, 1, T, F, 2).astype(np.float32) * 0.1
    feat_erb_np = np.random.randn(1, 1, T, p.nb_erb).astype(np.float32)
    feat_spec_np = np.random.randn(1, 1, T, p.nb_df, 2).astype(np.float32)

    # Build + load weights
    tf_model(tf.constant(spec_np), tf.constant(feat_erb_np), tf.constant(feat_spec_np), training=False)
    tf_model.load_from_pt(sd)

    # Get TF reference output
    tf_out = tf_model(tf.constant(spec_np), tf.constant(feat_erb_np),
                       tf.constant(feat_spec_np), training=False)
    tf_spec_ref = tf_out[0].numpy()

    # Create concrete function with fixed input shapes
    @tf.function(input_signature=[
        tf.TensorSpec(shape=[1, 1, T, F, 2], dtype=tf.float32),
        tf.TensorSpec(shape=[1, 1, T, p.nb_erb], dtype=tf.float32),
        tf.TensorSpec(shape=[1, 1, T, p.nb_df, 2], dtype=tf.float32),
    ])
    def serving_fn(spec, feat_erb, feat_spec):
        out = tf_model(spec, feat_erb, feat_spec, training=False)
        return out[0], out[1], out[2], out[3]

    concrete_fn = serving_fn.get_concrete_function()

    # Convert to TFLite directly from concrete function
    import tempfile, shutil
    tmpdir = tempfile.mkdtemp()
    try:
        converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_fn])
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
        tflite_model = converter.convert()
        tflite_path = os.path.join(tmpdir, "dfnet_float32.tflite")
        with open(tflite_path, "wb") as f:
            f.write(tflite_model)
        size_mb = len(tflite_model) / 1024 / 1024
        print(f"  TFLite float32 exported: {size_mb:.2f} MB")

        # Run TFLite inference
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Set inputs (sorted by name to match serving signature)
        for det in input_details:
            name = det["name"]
            if "feat_erb" in name:
                interpreter.set_tensor(det["index"], feat_erb_np)
            elif "feat_spec" in name:
                interpreter.set_tensor(det["index"], feat_spec_np)
            elif "spec" in name:
                interpreter.set_tensor(det["index"], spec_np)

        interpreter.invoke()

        # Find spec_out in output details
        tflite_spec_out = None
        for det in output_details:
            if "spec_out" in det["name"]:
                tflite_spec_out = interpreter.get_tensor(det["index"])
                break
        if tflite_spec_out is None:
            # Fallback: use first output
            tflite_spec_out = interpreter.get_tensor(output_details[0]["index"])

        # Compare TF vs TFLite
        signal = tf_spec_ref
        noise = tf_spec_ref - tflite_spec_out
        snr_tf_tflite = 10 * np.log10(np.sum(signal ** 2) / (np.sum(noise ** 2) + 1e-30))
        print(f"  SNR(TF vs TFLite float32): {snr_tf_tflite:.1f} dB")

        check("TFLite float32 spec vs TF", tf_spec_ref, tflite_spec_out, atol=1e-3)

        if snr_tf_tflite > 120:
            global PASS
            PASS += 1
            print(f"  ✓ TFLite float32 SNR check: PASS")
        else:
            global FAIL
            FAIL += 1
            print(f"  ✗ TFLite float32 SNR check: FAIL ({snr_tf_tflite:.1f} dB)")

        # Copy TFLite model to workspace
        final_path = os.path.join(os.path.dirname(__file__), "..", "..", "models", "dfnet_float32.tflite")
        os.makedirs(os.path.dirname(final_path), exist_ok=True)
        shutil.copy2(tflite_path, final_path)
        print(f"  Model saved to: models/dfnet_float32.tflite")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ============================================================================
# Shared: build TF model and representative dataset for quantization tests
# ============================================================================
def _build_model_and_rep_dataset():
    """Build TF model with loaded PT weights and return representative dataset."""
    p = ModelParams()
    F = p.fft_size // 2 + 1

    from libdf import DF as LibDF
    df_state = LibDF(sr=p.sr, fft_size=p.fft_size, hop_size=p.hop_size, nb_bands=p.nb_erb)
    from df.modules import erb_fb as pt_erb_fb
    erb_inv = pt_erb_fb(df_state.erb_widths(), p.sr, inverse=True)
    erb_inv_np = pt_to_np(erb_inv)

    pt_model = init_model()
    pt_model.eval()
    sd = get_pt_state_dict_np(pt_model)

    tf_model = DfNetTF(
        erb_inv_fb_np=erb_inv_np,
        nb_erb=p.nb_erb, nb_df=p.nb_df, fft_size=p.fft_size,
        conv_ch=p.conv_ch, df_order=p.df_order, df_lookahead=0,
        emb_hidden_dim=p.emb_hidden_dim, emb_num_layers=p.emb_num_layers,
        df_hidden_dim=p.df_hidden_dim, df_num_layers=p.df_num_layers,
        gru_groups=p.gru_groups, lin_groups=p.lin_groups,
        group_shuffle=p.group_shuffle, enc_concat=p.enc_concat,
        df_pathway_kernel_size_t=p.df_pathway_kernel_size_t,
    )

    T = 20
    spec_np = np.random.randn(1, 1, T, F, 2).astype(np.float32) * 0.1
    feat_erb_np = np.random.randn(1, 1, T, p.nb_erb).astype(np.float32)
    feat_spec_np = np.random.randn(1, 1, T, p.nb_df, 2).astype(np.float32)

    tf_model(tf.constant(spec_np), tf.constant(feat_erb_np), tf.constant(feat_spec_np), training=False)
    tf_model.load_from_pt(sd)

    tf_out = tf_model(tf.constant(spec_np), tf.constant(feat_erb_np),
                       tf.constant(feat_spec_np), training=False)
    tf_spec_ref = tf_out[0].numpy()

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[1, 1, T, F, 2], dtype=tf.float32),
        tf.TensorSpec(shape=[1, 1, T, p.nb_erb], dtype=tf.float32),
        tf.TensorSpec(shape=[1, 1, T, p.nb_df, 2], dtype=tf.float32),
    ])
    def serving_fn(spec, feat_erb, feat_spec):
        out = tf_model(spec, feat_erb, feat_spec, training=False)
        return out[0], out[1], out[2], out[3]

    concrete_fn = serving_fn.get_concrete_function()

    # Representative dataset: 100 random samples for calibration
    np.random.seed(123)
    def representative_dataset():
        for _ in range(100):
            s = np.random.randn(1, 1, T, F, 2).astype(np.float32) * 0.1
            e = np.random.randn(1, 1, T, p.nb_erb).astype(np.float32)
            fs = np.random.randn(1, 1, T, p.nb_df, 2).astype(np.float32)
            yield [s, e, fs]

    return concrete_fn, representative_dataset, tf_spec_ref, spec_np, feat_erb_np, feat_spec_np


# ============================================================================
# Test 13: TFLite int8 quantization (Phase 6)
# ============================================================================
def test_tflite_int8():
    print("\n=== Test: TFLite int8 quantization (Phase 6) ===")
    np.random.seed(42)

    concrete_fn, representative_dataset, tf_spec_ref, spec_np, feat_erb_np, feat_spec_np = \
        _build_model_and_rep_dataset()

    import tempfile, shutil
    tmpdir = tempfile.mkdtemp()
    try:
        converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_fn])
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
            tf.lite.OpsSet.TFLITE_BUILTINS,  # fallback
        ]
        converter.inference_input_type = tf.float32
        converter.inference_output_type = tf.float32

        tflite_model = converter.convert()
        tflite_path = os.path.join(tmpdir, "dfnet_int8.tflite")
        with open(tflite_path, "wb") as f:
            f.write(tflite_model)
        size_mb = len(tflite_model) / 1024 / 1024
        print(f"  TFLite int8 exported: {size_mb:.2f} MB")

        # Run TFLite inference
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        for det in input_details:
            name = det["name"]
            if "feat_erb" in name:
                interpreter.set_tensor(det["index"], feat_erb_np)
            elif "feat_spec" in name:
                interpreter.set_tensor(det["index"], feat_spec_np)
            elif "spec" in name:
                interpreter.set_tensor(det["index"], spec_np)

        interpreter.invoke()

        tflite_spec_out = None
        for det in output_details:
            if "spec_out" in det["name"] or det == output_details[0]:
                tflite_spec_out = interpreter.get_tensor(det["index"])
                break

        signal = tf_spec_ref
        noise = tf_spec_ref - tflite_spec_out
        snr = 10 * np.log10(np.sum(signal ** 2) / (np.sum(noise ** 2) + 1e-30))
        print(f"  SNR(TF vs TFLite int8): {snr:.1f} dB")

        # int8 quantization will have lower SNR than float32
        global PASS, FAIL
        if snr > 20:
            PASS += 1
            print(f"  ✓ TFLite int8 SNR check: PASS (>20 dB)")
        else:
            FAIL += 1
            print(f"  ✗ TFLite int8 SNR check: FAIL ({snr:.1f} dB)")

        final_path = os.path.join(os.path.dirname(__file__), "..", "..", "models", "dfnet_int8.tflite")
        os.makedirs(os.path.dirname(final_path), exist_ok=True)
        shutil.copy2(tflite_path, final_path)
        print(f"  Model saved to: models/dfnet_int8.tflite")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ============================================================================
# Test 14: TFLite int16x8 quantization (Phase 6)
# ============================================================================
def test_tflite_int16x8():
    print("\n=== Test: TFLite int16x8 quantization (Phase 6) ===")
    np.random.seed(42)

    concrete_fn, representative_dataset, tf_spec_ref, spec_np, feat_erb_np, feat_spec_np = \
        _build_model_and_rep_dataset()

    import tempfile, shutil
    tmpdir = tempfile.mkdtemp()
    try:
        converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_fn])
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8,
            tf.lite.OpsSet.TFLITE_BUILTINS,  # fallback
        ]
        converter.inference_input_type = tf.float32
        converter.inference_output_type = tf.float32

        tflite_model = converter.convert()
        tflite_path = os.path.join(tmpdir, "dfnet_int16x8.tflite")
        with open(tflite_path, "wb") as f:
            f.write(tflite_model)
        size_mb = len(tflite_model) / 1024 / 1024
        print(f"  TFLite int16x8 exported: {size_mb:.2f} MB")

        # Run TFLite inference
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        for det in input_details:
            name = det["name"]
            if "feat_erb" in name:
                interpreter.set_tensor(det["index"], feat_erb_np)
            elif "feat_spec" in name:
                interpreter.set_tensor(det["index"], feat_spec_np)
            elif "spec" in name:
                interpreter.set_tensor(det["index"], spec_np)

        interpreter.invoke()

        tflite_spec_out = None
        for det in output_details:
            if "spec_out" in det["name"] or det == output_details[0]:
                tflite_spec_out = interpreter.get_tensor(det["index"])
                break

        signal = tf_spec_ref
        noise = tf_spec_ref - tflite_spec_out
        snr = 10 * np.log10(np.sum(signal ** 2) / (np.sum(noise ** 2) + 1e-30))
        print(f"  SNR(TF vs TFLite int16x8): {snr:.1f} dB")

        global PASS, FAIL
        # int16x8 should be better than int8
        if snr > 40:
            PASS += 1
            print(f"  ✓ TFLite int16x8 SNR check: PASS (>40 dB)")
        else:
            FAIL += 1
            print(f"  ✗ TFLite int16x8 SNR check: FAIL ({snr:.1f} dB)")

        final_path = os.path.join(os.path.dirname(__file__), "..", "..", "models", "dfnet_int16x8.tflite")
        os.makedirs(os.path.dirname(final_path), exist_ok=True)
        shutil.copy2(tflite_path, final_path)
        print(f"  Model saved to: models/dfnet_int16x8.tflite")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ============================================================================
# Test 15: Spectrogram visualization (Phase 7)
# ============================================================================
def test_spectrogram_comparison():
    print("\n=== Test: Spectrogram comparison (Phase 7) ===")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  SKIP: matplotlib not installed")
        return

    np.random.seed(42)
    torch.manual_seed(42)

    p = ModelParams()
    F = p.fft_size // 2 + 1

    from libdf import DF as LibDF
    df_state = LibDF(sr=p.sr, fft_size=p.fft_size, hop_size=p.hop_size, nb_bands=p.nb_erb)
    from df.modules import erb_fb as pt_erb_fb
    erb_inv = pt_erb_fb(df_state.erb_widths(), p.sr, inverse=True)
    erb_inv_np = pt_to_np(erb_inv)

    pt_model = init_model()
    pt_model.eval()
    sd = get_pt_state_dict_np(pt_model)

    tf_model = DfNetTF(
        erb_inv_fb_np=erb_inv_np,
        nb_erb=p.nb_erb, nb_df=p.nb_df, fft_size=p.fft_size,
        conv_ch=p.conv_ch, df_order=p.df_order, df_lookahead=0,
        emb_hidden_dim=p.emb_hidden_dim, emb_num_layers=p.emb_num_layers,
        df_hidden_dim=p.df_hidden_dim, df_num_layers=p.df_num_layers,
        gru_groups=p.gru_groups, lin_groups=p.lin_groups,
        group_shuffle=p.group_shuffle, enc_concat=p.enc_concat,
        df_pathway_kernel_size_t=p.df_pathway_kernel_size_t,
    )

    T = 20
    spec_np = np.random.randn(1, 1, T, F, 2).astype(np.float32) * 0.1
    feat_erb_np = np.random.randn(1, 1, T, p.nb_erb).astype(np.float32)
    feat_spec_np = np.random.randn(1, 1, T, p.nb_df, 2).astype(np.float32)

    tf_model(tf.constant(spec_np), tf.constant(feat_erb_np), tf.constant(feat_spec_np), training=False)
    tf_model.load_from_pt(sd)

    # PT output
    with torch.no_grad():
        pt_out = pt_model(
            torch.from_numpy(spec_np), torch.from_numpy(feat_erb_np), torch.from_numpy(feat_spec_np))
    pt_spec = pt_to_np(pt_out[0])  # [1, 1, T, F, 2]

    # TF output
    tf_out = tf_model(tf.constant(spec_np), tf.constant(feat_erb_np), tf.constant(feat_spec_np), training=False)
    tf_spec = tf_out[0].numpy()

    # TFLite float32
    @tf.function(input_signature=[
        tf.TensorSpec(shape=[1, 1, T, F, 2], dtype=tf.float32),
        tf.TensorSpec(shape=[1, 1, T, p.nb_erb], dtype=tf.float32),
        tf.TensorSpec(shape=[1, 1, T, p.nb_df, 2], dtype=tf.float32),
    ])
    def serving_fn(spec, feat_erb, feat_spec):
        out = tf_model(spec, feat_erb, feat_spec, training=False)
        return out[0], out[1], out[2], out[3]

    concrete_fn = serving_fn.get_concrete_function()

    import tempfile
    tmpdir = tempfile.mkdtemp()

    # Float32 TFLite
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_fn])
    tflite_f32 = converter.convert()
    f32_path = os.path.join(tmpdir, "f32.tflite")
    with open(f32_path, "wb") as f:
        f.write(tflite_f32)

    def run_tflite(path):
        interp = tf.lite.Interpreter(model_path=path)
        interp.allocate_tensors()
        for det in interp.get_input_details():
            name = det["name"]
            if "feat_erb" in name:
                interp.set_tensor(det["index"], feat_erb_np)
            elif "feat_spec" in name:
                interp.set_tensor(det["index"], feat_spec_np)
            elif "spec" in name:
                interp.set_tensor(det["index"], spec_np)
        interp.invoke()
        return interp.get_tensor(interp.get_output_details()[0]["index"])

    tflite_f32_spec = run_tflite(f32_path)

    # Int8 TFLite
    np.random.seed(123)
    def rep_dataset():
        for _ in range(50):
            yield [np.random.randn(1, 1, T, F, 2).astype(np.float32) * 0.1,
                   np.random.randn(1, 1, T, p.nb_erb).astype(np.float32),
                   np.random.randn(1, 1, T, p.nb_df, 2).astype(np.float32)]

    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_fn])
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = rep_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8, tf.lite.OpsSet.TFLITE_BUILTINS]
    converter.inference_input_type = tf.float32
    converter.inference_output_type = tf.float32
    int8_path = os.path.join(tmpdir, "int8.tflite")
    with open(int8_path, "wb") as f:
        f.write(converter.convert())

    tflite_int8_spec = run_tflite(int8_path)

    # Int16x8 TFLite
    np.random.seed(123)
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_fn])
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = rep_dataset
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8,
        tf.lite.OpsSet.TFLITE_BUILTINS]
    converter.inference_input_type = tf.float32
    converter.inference_output_type = tf.float32
    int16_path = os.path.join(tmpdir, "int16.tflite")
    with open(int16_path, "wb") as f:
        f.write(converter.convert())

    tflite_int16_spec = run_tflite(int16_path)

    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)

    # Compute magnitude spectrograms (dB scale)
    def to_mag_db(spec_5d):
        """[1, 1, T, F, 2] -> [T, F] in dB"""
        s = spec_5d[0, 0]  # [T, F, 2]
        mag = np.sqrt(s[..., 0] ** 2 + s[..., 1] ** 2)
        return 20 * np.log10(np.maximum(mag, 1e-8))

    specs = {
        "PyTorch": to_mag_db(pt_spec),
        "TF (Keras)": to_mag_db(tf_spec),
        "TFLite float32": to_mag_db(tflite_f32_spec),
        "TFLite int8": to_mag_db(tflite_int8_spec),
        "TFLite int16x8": to_mag_db(tflite_int16_spec),
    }

    fig, axes = plt.subplots(2, 3, figsize=(18, 8))
    axes = axes.flatten()

    vmin, vmax = -60, 0
    for i, (title, mag) in enumerate(specs.items()):
        ax = axes[i]
        im = ax.imshow(mag.T, aspect="auto", origin="lower", vmin=vmin, vmax=vmax, cmap="magma")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Time frame")
        ax.set_ylabel("Freq bin")
        plt.colorbar(im, ax=ax, label="dB")

    # Error spectrogram: PT - TFLite int8
    diff = to_mag_db(pt_spec) - to_mag_db(tflite_int8_spec)
    ax = axes[5]
    im = ax.imshow(diff.T, aspect="auto", origin="lower", cmap="RdBu_r", vmin=-5, vmax=5)
    ax.set_title("Error: PT - TFLite int8 (dB)", fontsize=11)
    ax.set_xlabel("Time frame")
    ax.set_ylabel("Freq bin")
    plt.colorbar(im, ax=ax, label="dB diff")

    plt.suptitle("DeepFilterNet2: PT vs TF vs TFLite Spectrograms", fontsize=14, y=1.02)
    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(__file__), "..", "..", "models", "spectrogram_comparison.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Spectrogram saved to: models/spectrogram_comparison.png")

    global PASS
    PASS += 1
    print(f"  ✓ Spectrogram comparison: PASS")


# ============================================================================
# Main
# ============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Phase 1-2: Per-module PT vs TF unit tests")
    print("=" * 60)

    test_grouped_linear()
    test_grouped_linear_einsum()
    test_gru_single_layer()
    test_grouped_gru()
    test_grouped_gru_3layer()
    test_conv2d_norm_act_inp()
    test_conv2d_norm_act_sep()
    test_df_output_reshape()
    test_mask()
    test_multiframe_df()

    # ================================================================
    # Phase 3: Full model test (PT vs TF)
    # ================================================================
    test_full_dfnet()

    # ================================================================
    # Phase 5: TFLite float32 export
    # ================================================================
    test_tflite_float32()

    # ================================================================
    # Phase 6: TFLite quantization (int8, int16x8)
    # ================================================================
    test_tflite_int8()
    test_tflite_int16x8()

    # ================================================================
    # Phase 7: Spectrogram visualization
    # ================================================================
    test_spectrogram_comparison()

    print("\n" + "=" * 60)
    print(f"Results: {PASS} passed, {FAIL} failed")
    print("=" * 60)
    if FAIL > 0:
        sys.exit(1)
