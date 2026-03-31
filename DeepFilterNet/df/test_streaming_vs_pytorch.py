"""
Compare TF streaming model vs PyTorch batch model with the SAME input.

Goal: Feed identical pre-computed features (feat_erb, feat_spec) and
      raw spec to both models, then assert the outputs match (>120 dB SNR).

Both models use conv_lookahead=0, df_lookahead=0 so everything is
purely causal and the streaming frame-by-frame output should be
*exactly* equal to the batch output (no lookahead ambiguity).

Usage:
    python DeepFilterNet/df/test_streaming_vs_pytorch.py
"""

import os, sys, math
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import tensorflow as tf

# ── Configure PT model with zero lookahead ──
from df.config import config
config.use_defaults()
config.set("CONV_LOOKAHEAD", 0, int, "deepfilternet")
config.set("DF_LOOKAHEAD", 0, int, "DF")
config.set("PAD_MODE", "none", str, "DF")
config.set("DF_OUTPUT_LAYER", "groupedlinear", str, "deepfilternet")
config.set("DF_N_ITER", 1, int, "deepfilternet")

from df.deepfilternet2 import ModelParams, Encoder, ErbDecoder, DfDecoder, DfNet, init_model
from df.modules import erb_fb as pt_erb_fb
from df.multiframe import DF as MultiFrameDF
from libdf import DF as LibDF

from df.tf_stateful_streaming import (
    DfNetStreamingTF,
    compute_erb_fb,
    compute_erb_inv_fb,
)


def snr_db(ref, est):
    noise = ref - est
    ref_pow = np.mean(ref ** 2)
    noise_pow = np.mean(noise ** 2)
    if noise_pow < 1e-30:
        return 300.0
    return 10 * np.log10(ref_pow / noise_pow)


def pt_to_np(t):
    return t.detach().cpu().numpy()


def get_pt_state_dict_np(module, prefix=""):
    sd = {}
    pfx = f"{prefix}." if prefix else ""
    for name, param in module.named_parameters():
        sd[f"{pfx}{name}"] = param.detach().cpu().numpy()
    for name, buf in module.named_buffers():
        sd[f"{pfx}{name}"] = buf.detach().cpu().numpy()
    return sd


def build_pt_model():
    """Build full PT model and return model + full state dict with correct prefixes."""
    p = ModelParams()
    df_state = LibDF(sr=p.sr, fft_size=p.fft_size, hop_size=p.hop_size, nb_bands=p.nb_erb)
    erb = pt_erb_fb(df_state.erb_widths(), p.sr, inverse=False)
    erb_inv = pt_erb_fb(df_state.erb_widths(), p.sr, inverse=True)
    pt_model = DfNet(erb, erb_inv, run_df=True, train_mask=True)
    pt_model.eval()
    # Full state dict with proper prefixes (enc.xxx, erb_dec.xxx, etc.)
    sd = {}
    for name, param in pt_model.named_parameters():
        sd[name] = param.detach().cpu().numpy()
    for name, buf in pt_model.named_buffers():
        sd[name] = buf.detach().cpu().numpy()
    return pt_model, sd, p


# ============================================================================
# Test 1: Encoder — PT batch vs TF streaming frame-by-frame
# ============================================================================
def test_encoder():
    print("\n" + "=" * 60)
    print("Test 1: Encoder — PT batch vs TF streaming")
    print("=" * 60)
    torch.manual_seed(42)
    np.random.seed(42)

    pt_model, sd, p = build_pt_model()
    pt_enc = pt_model.enc
    print(f"  conv_lookahead={p.conv_lookahead}, gru_type={p.gru_type}")
    print(f"  nb_erb={p.nb_erb}, nb_df={p.nb_df}, conv_ch={p.conv_ch}")

    # TF streaming encoder
    from df.tf_stateful_streaming import EncoderStreamingTF
    tf_enc = EncoderStreamingTF(
        nb_erb=p.nb_erb, nb_df=p.nb_df, conv_ch=p.conv_ch,
        emb_hidden_dim=p.emb_hidden_dim, lin_groups=p.lin_groups,
        gru_groups=p.gru_groups, group_shuffle=p.group_shuffle,
        enc_concat=p.enc_concat, gru_type=p.gru_type,
    )

    # Build TF encoder with dummy input
    erb_buf = tf.zeros([1, 2, p.nb_erb, 1])
    df_buf = tf.zeros([1, 2, p.nb_df, 2])
    h_per_group = p.emb_hidden_dim // p.gru_groups
    h_enc = tf.zeros([1 * p.gru_groups, 1, h_per_group])

    dummy_erb = tf.zeros([1, 1, p.nb_erb, 1])
    dummy_spec = tf.zeros([1, 1, p.nb_df, 2])
    tf_enc(dummy_erb, dummy_spec, erb_buf, df_buf, h_enc)

    # Load weights using full model state dict
    tf_enc.load_from_pt(sd, "enc")

    # Generate input: T frames
    T = 10
    feat_erb_np = np.random.randn(1, 1, T, p.nb_erb).astype(np.float32) * 0.5
    feat_spec_np = np.random.randn(1, 1, T, p.nb_df, 2).astype(np.float32) * 0.5

    # === PT batch forward ===
    # PT Encoder expects: feat_erb [B,1,T,E], feat_spec [B,2,T,F_df]
    feat_erb_pt = torch.from_numpy(feat_erb_np)  # [1,1,T,E]
    # feat_spec in PT: the DfNet does squeeze(1).permute(0,3,1,2)
    # [B,1,T,F,2] -> squeeze -> [B,T,F,2] -> permute(0,3,1,2) -> [B,2,T,F]
    feat_spec_pt = torch.from_numpy(feat_spec_np).squeeze(1).permute(0, 3, 1, 2)  # [1,2,T,F_df]

    with torch.no_grad():
        e0_pt, e1_pt, e2_pt, e3_pt, emb_pt, c0_pt, lsnr_pt = pt_enc(feat_erb_pt, feat_spec_pt)

    # Convert PT outputs to numpy (NCHW -> NHWC for comparison)
    e0_np = pt_to_np(e0_pt).transpose(0, 2, 3, 1)  # [B,T,F,C]
    e1_np = pt_to_np(e1_pt).transpose(0, 2, 3, 1)
    e2_np = pt_to_np(e2_pt).transpose(0, 2, 3, 1)
    e3_np = pt_to_np(e3_pt).transpose(0, 2, 3, 1)
    emb_np = pt_to_np(emb_pt)  # [B,T,H]
    c0_np = pt_to_np(c0_pt).transpose(0, 2, 3, 1)  # [B,T,F,C]
    lsnr_np = pt_to_np(lsnr_pt)

    # === TF streaming frame-by-frame ===
    erb_buf = tf.zeros([1, 2, p.nb_erb, 1])
    df_buf = tf.zeros([1, 2, p.nb_df, 2])
    h_enc = tf.zeros([1 * p.gru_groups, 1, h_per_group])

    tf_e0_list, tf_e1_list, tf_e2_list, tf_e3_list = [], [], [], []
    tf_emb_list, tf_c0_list, tf_lsnr_list = [], [], []

    for t in range(T):
        # feat_erb: [B,1,T,E] -> take frame t -> [B,1,E] -> reshape [B,1,E,1]
        erb_frame_nhwc = feat_erb_np[:, :, t:t+1, :].transpose(0, 2, 3, 1)  # [B,1,E,1]
        # feat_spec: [B,1,T,F,2] -> take frame t -> [B,1,F,2]
        spec_frame_nhwc = feat_spec_np[:, 0, t:t+1, :, :]  # [B,1,F,2]

        e0, e1, e2, e3, emb, c0, lsnr, erb_buf, df_buf, h_enc = tf_enc(
            tf.constant(erb_frame_nhwc),
            tf.constant(spec_frame_nhwc),
            erb_buf, df_buf, h_enc,
            training=False,
        )

        tf_e0_list.append(e0.numpy())
        tf_e1_list.append(e1.numpy())
        tf_e2_list.append(e2.numpy())
        tf_e3_list.append(e3.numpy())
        tf_emb_list.append(emb.numpy())
        tf_c0_list.append(c0.numpy())
        tf_lsnr_list.append(lsnr.numpy())

    # Concatenate along time axis
    tf_e0 = np.concatenate(tf_e0_list, axis=1)
    tf_e1 = np.concatenate(tf_e1_list, axis=1)
    tf_e2 = np.concatenate(tf_e2_list, axis=1)
    tf_e3 = np.concatenate(tf_e3_list, axis=1)
    tf_emb = np.concatenate(tf_emb_list, axis=1)
    tf_c0 = np.concatenate(tf_c0_list, axis=1)
    tf_lsnr = np.concatenate(tf_lsnr_list, axis=1)

    # Compare each output
    results = {}
    for name, pt_val, tf_val in [
        ("e0", e0_np, tf_e0),
        ("e1", e1_np, tf_e1),
        ("e2", e2_np, tf_e2),
        ("e3", e3_np, tf_e3),
        ("emb", emb_np, tf_emb),
        ("c0", c0_np, tf_c0),
        ("lsnr", lsnr_np, tf_lsnr),
    ]:
        s = snr_db(pt_val, tf_val)
        max_diff = np.max(np.abs(pt_val - tf_val))
        status = "✓ PASS" if s > 120 else "✗ FAIL"
        print(f"  {status} {name}: SNR={s:.1f} dB, max_diff={max_diff:.2e}")
        results[name] = s

    return results


# ============================================================================
# Test 2: ERB Decoder — PT batch vs TF streaming frame-by-frame
# ============================================================================
def test_erb_decoder():
    print("\n" + "=" * 60)
    print("Test 2: ERB Decoder — PT batch vs TF streaming")
    print("=" * 60)
    torch.manual_seed(42)
    np.random.seed(42)

    pt_model, sd, p = build_pt_model()
    pt_dec = pt_model.erb_dec

    from df.tf_stateful_streaming import ErbDecoderStreamingTF
    tf_dec = ErbDecoderStreamingTF(
        nb_erb=p.nb_erb, conv_ch=p.conv_ch,
        emb_hidden_dim=p.emb_hidden_dim,
        emb_num_layers=p.emb_num_layers,
        lin_groups=p.lin_groups, gru_groups=p.gru_groups,
        group_shuffle=p.group_shuffle, gru_type=p.gru_type,
    )

    # Build
    T = 10
    f8 = p.nb_erb // 4
    emb_dummy = tf.zeros([1, 1, p.emb_hidden_dim])
    e3_dummy = tf.zeros([1, 1, f8, p.conv_ch])
    e2_dummy = tf.zeros([1, 1, f8, p.conv_ch])
    e1_dummy = tf.zeros([1, 1, p.nb_erb // 2, p.conv_ch])
    e0_dummy = tf.zeros([1, 1, p.nb_erb, p.conv_ch])
    h_per_group = p.emb_hidden_dim // p.gru_groups
    h_erb = tf.zeros([(p.emb_num_layers - 1) * p.gru_groups, 1, h_per_group])
    tf_dec(emb_dummy, e3_dummy, e2_dummy, e1_dummy, e0_dummy, h_erb)

    tf_dec.load_from_pt(sd, "erb_dec")

    # Generate inputs (random encoder outputs)
    emb_np = np.random.randn(1, T, p.emb_hidden_dim).astype(np.float32) * 0.1
    e3_nchw = np.random.randn(1, p.conv_ch, T, f8).astype(np.float32) * 0.1
    e2_nchw = np.random.randn(1, p.conv_ch, T, f8).astype(np.float32) * 0.1
    e1_nchw = np.random.randn(1, p.conv_ch, T, p.nb_erb // 2).astype(np.float32) * 0.1
    e0_nchw = np.random.randn(1, p.conv_ch, T, p.nb_erb).astype(np.float32) * 0.1

    # PT batch
    with torch.no_grad():
        m_pt = pt_dec(
            torch.from_numpy(emb_np),
            torch.from_numpy(e3_nchw),
            torch.from_numpy(e2_nchw),
            torch.from_numpy(e1_nchw),
            torch.from_numpy(e0_nchw),
        )
    m_pt_np = pt_to_np(m_pt).transpose(0, 2, 3, 1)  # NCHW -> NHWC [B,T,F,1]

    # TF streaming frame by frame
    h_erb = tf.zeros([(p.emb_num_layers - 1) * p.gru_groups, 1, h_per_group])
    tf_m_list = []

    # Convert encoder outputs NCHW -> NHWC
    e3_nhwc = e3_nchw.transpose(0, 2, 3, 1)
    e2_nhwc = e2_nchw.transpose(0, 2, 3, 1)
    e1_nhwc = e1_nchw.transpose(0, 2, 3, 1)
    e0_nhwc = e0_nchw.transpose(0, 2, 3, 1)

    for t in range(T):
        m, h_erb = tf_dec(
            tf.constant(emb_np[:, t:t+1, :]),
            tf.constant(e3_nhwc[:, t:t+1, :, :]),
            tf.constant(e2_nhwc[:, t:t+1, :, :]),
            tf.constant(e1_nhwc[:, t:t+1, :, :]),
            tf.constant(e0_nhwc[:, t:t+1, :, :]),
            h_erb,
            training=False,
        )
        tf_m_list.append(m.numpy())

    tf_m = np.concatenate(tf_m_list, axis=1)

    s = snr_db(m_pt_np, tf_m)
    max_diff = np.max(np.abs(m_pt_np - tf_m))
    status = "✓ PASS" if s > 120 else "✗ FAIL"
    print(f"  {status} ERB mask: SNR={s:.1f} dB, max_diff={max_diff:.2e}")
    return {"mask": s}


# ============================================================================
# Test 3: DF Decoder — PT batch vs TF streaming frame-by-frame
# ============================================================================
def test_df_decoder():
    print("\n" + "=" * 60)
    print("Test 3: DF Decoder — PT batch vs TF streaming")
    print("=" * 60)
    torch.manual_seed(42)
    np.random.seed(42)

    pt_model, sd, p = build_pt_model()
    pt_dec = pt_model.df_dec

    from df.tf_stateful_streaming import DfDecoderStreamingTF
    tf_dec = DfDecoderStreamingTF(
        nb_df=p.nb_df, df_order=p.df_order, conv_ch=p.conv_ch,
        emb_hidden_dim=p.emb_hidden_dim,
        df_hidden_dim=p.df_hidden_dim,
        df_num_layers=p.df_num_layers,
        gru_groups=p.gru_groups,
        group_shuffle=p.group_shuffle,
        lin_groups=p.lin_groups,
        df_pathway_kernel_size_t=p.df_pathway_kernel_size_t,
        gru_type=p.gru_type,
        df_gru_skip=p.df_gru_skip,
    )

    # Build
    T = 10
    emb_dummy = tf.zeros([1, 1, p.emb_hidden_dim])
    c0_dummy = tf.zeros([1, 1, p.nb_df, p.conv_ch])
    h_per_group = p.df_hidden_dim // p.gru_groups
    h_df = tf.zeros([p.df_num_layers * p.gru_groups, 1, h_per_group])
    df_convp_buf = None
    if p.df_pathway_kernel_size_t > 1:
        df_convp_buf = tf.zeros([1, p.df_pathway_kernel_size_t - 1, p.nb_df, p.conv_ch])
    tf_dec(emb_dummy, c0_dummy, h_df, df_convp_buf)

    tf_dec.load_from_pt(sd, "df_dec")

    # Generate random inputs
    emb_np = np.random.randn(1, T, p.emb_hidden_dim).astype(np.float32) * 0.1
    c0_nchw = np.random.randn(1, p.conv_ch, T, p.nb_df).astype(np.float32) * 0.1

    # PT batch
    with torch.no_grad():
        coefs_pt, alpha_pt = pt_dec(
            torch.from_numpy(emb_np),
            torch.from_numpy(c0_nchw),
        )
    coefs_pt_np = pt_to_np(coefs_pt)  # [B, O, T, F, 2]
    alpha_pt_np = pt_to_np(alpha_pt)  # [B, T, 1]

    # TF streaming frame by frame
    h_df = tf.zeros([p.df_num_layers * p.gru_groups, 1, h_per_group])
    df_convp_buf = None
    if p.df_pathway_kernel_size_t > 1:
        df_convp_buf = tf.zeros([1, p.df_pathway_kernel_size_t - 1, p.nb_df, p.conv_ch])

    c0_nhwc = c0_nchw.transpose(0, 2, 3, 1)  # NCHW -> NHWC
    tf_coefs_list, tf_alpha_list = [], []

    for t in range(T):
        coefs, alpha, h_df, df_convp_buf = tf_dec(
            tf.constant(emb_np[:, t:t+1, :]),
            tf.constant(c0_nhwc[:, t:t+1, :, :]),
            h_df, df_convp_buf,
            training=False,
        )
        tf_coefs_list.append(coefs.numpy())
        tf_alpha_list.append(alpha.numpy())

    # coefs: each frame is [B, O, 1, F, 2], concat on T dim
    tf_coefs = np.concatenate(tf_coefs_list, axis=2)  # [B, O, T, F, 2]
    tf_alpha = np.concatenate(tf_alpha_list, axis=1)   # [B, T, 1]

    s_coefs = snr_db(coefs_pt_np, tf_coefs)
    s_alpha = snr_db(alpha_pt_np, tf_alpha)
    max_diff_c = np.max(np.abs(coefs_pt_np - tf_coefs))
    max_diff_a = np.max(np.abs(alpha_pt_np - tf_alpha))

    for name, s, md in [("df_coefs", s_coefs, max_diff_c), ("df_alpha", s_alpha, max_diff_a)]:
        status = "✓ PASS" if s > 120 else "✗ FAIL"
        print(f"  {status} {name}: SNR={s:.1f} dB, max_diff={md:.2e}")

    return {"coefs": s_coefs, "alpha": s_alpha}


# ============================================================================
# Test 4: Full model — PT batch vs TF streaming frame-by-frame
# ============================================================================
def test_full_model():
    print("\n" + "=" * 60)
    print("Test 4: Full model — PT batch vs TF streaming")
    print("=" * 60)
    torch.manual_seed(42)
    np.random.seed(42)

    p = ModelParams()
    print(f"  Config: conv_lookahead={p.conv_lookahead}, "
          f"df_lookahead={p.df_lookahead}, pad_mode={p.pad_mode}")
    print(f"  gru_type={p.gru_type}, gru_groups={p.gru_groups}")

    # Build PT model
    df_state = LibDF(sr=p.sr, fft_size=p.fft_size, hop_size=p.hop_size, nb_bands=p.nb_erb)
    erb = pt_erb_fb(df_state.erb_widths(), p.sr, inverse=False)
    erb_inv = pt_erb_fb(df_state.erb_widths(), p.sr, inverse=True)
    pt_model = DfNet(erb, erb_inv, run_df=True, train_mask=True)
    pt_model.eval()
    sd = get_pt_state_dict_np(pt_model)

    # Build TF streaming model — use erb_inv_fb from PT model directly
    # (compute_erb_fb uses min_nb_freqs=2 but libDF defaults to 1)
    erb_inv_fb_np = sd["mask.erb_inv_fb"]  # exact same filterbank as PT

    F_bins = p.fft_size // 2 + 1

    tf_model = DfNetStreamingTF(
        erb_inv_fb_np=erb_inv_fb_np,
        nb_erb=p.nb_erb, nb_df=p.nb_df,
        fft_size=p.fft_size,
        conv_ch=p.conv_ch, df_order=p.df_order,
        df_lookahead=0,
        emb_hidden_dim=p.emb_hidden_dim, emb_num_layers=p.emb_num_layers,
        df_hidden_dim=p.df_hidden_dim, df_num_layers=p.df_num_layers,
        gru_groups=p.gru_groups, lin_groups=p.lin_groups,
        group_shuffle=p.group_shuffle,
        enc_concat=p.enc_concat,
        df_pathway_kernel_size_t=p.df_pathway_kernel_size_t,
        gru_type=p.gru_type,
        df_gru_skip=p.df_gru_skip,
    )

    # Build & load TF model
    state = tf_model.get_initial_state(batch_size=1)
    _ = tf_model(
        tf.zeros([1, 1, 1, F_bins, 2]),
        tf.zeros([1, 1, 1, p.nb_erb]),
        tf.zeros([1, 1, 1, p.nb_df, 2]),
        state,
    )
    tf_model.load_from_pt(sd)

    # ── Verify erb_inv_fb matrices match ──
    pt_erb_inv = sd.get("mask.erb_inv_fb", None)
    if pt_erb_inv is not None:
        tf_erb_inv = tf_model.mask.erb_inv_fb.numpy()
        erb_inv_match = snr_db(pt_erb_inv, tf_erb_inv)
        max_diff_inv = np.max(np.abs(pt_erb_inv - tf_erb_inv))
        print(f"  erb_inv_fb check: SNR={erb_inv_match:.1f} dB, max_diff={max_diff_inv:.2e}")
        print(f"    PT shape={pt_erb_inv.shape}, TF shape={tf_erb_inv.shape}")
        print(f"    PT sum={pt_erb_inv.sum():.4f}, TF sum={tf_erb_inv.sum():.4f}")
        if erb_inv_match < 120:
            # Show where they differ
            diff = np.abs(pt_erb_inv - tf_erb_inv)
            max_idx = np.unravel_index(np.argmax(diff), diff.shape)
            print(f"    Max diff at {max_idx}: PT={pt_erb_inv[max_idx]:.6f}, TF={tf_erb_inv[max_idx]:.6f}")
    else:
        print(f"  WARNING: mask.erb_inv_fb not in PT state dict")

    # Generate random input
    T = 15
    spec_np = np.random.randn(1, 1, T, F_bins, 2).astype(np.float32) * 0.1
    feat_erb_np = np.random.randn(1, 1, T, p.nb_erb).astype(np.float32) * 0.5
    feat_spec_np = np.random.randn(1, 1, T, p.nb_df, 2).astype(np.float32) * 0.5

    # === PT batch forward ===
    # First get the intermediates by running encoder, decoder, mask, DF separately
    with torch.no_grad():
        # Step 1: Encoder
        feat_spec_pt = torch.from_numpy(feat_spec_np.copy()).squeeze(1).permute(0, 3, 1, 2)
        feat_erb_pt = torch.from_numpy(feat_erb_np.copy())
        e0_pt, e1_pt, e2_pt, e3_pt, emb_pt, c0_pt, lsnr_pt = pt_model.enc(feat_erb_pt, feat_spec_pt)

        # Step 2: ERB Decoder -> mask
        m_pt = pt_model.erb_dec(emb_pt, e3_pt, e2_pt, e1_pt, e0_pt)  # [B,1,T,E]

        # Step 3: Apply mask
        spec_pt_in = torch.from_numpy(spec_np.copy())
        spec_masked_pt = pt_model.mask(spec_pt_in, m_pt)  # [B,1,T,F,2]

        # Step 4: DF Decoder -> coefs
        df_coefs_pt, df_alpha_pt = pt_model.df_dec(emb_pt, c0_pt)  # [B,O,T,F,2]

        # Step 5: DF op
        spec_out_pt = pt_model.df_op(spec_masked_pt, df_coefs_pt)  # [B,1,T,F,2]

    pt_spec_np = pt_to_np(spec_out_pt)
    pt_m_np = pt_to_np(m_pt)
    pt_lsnr_np = pt_to_np(lsnr_pt)
    pt_masked_np = pt_to_np(spec_masked_pt)  # [B,1,T,F,2]
    pt_coefs_np = pt_to_np(df_coefs_pt)  # [B,O,T,F,2]

    # Also get the full model output to verify
    with torch.no_grad():
        pt_full_out, _, _, _ = pt_model(
            torch.from_numpy(spec_np.copy()),
            torch.from_numpy(feat_erb_np.copy()),
            torch.from_numpy(feat_spec_np.copy()),
        )
    pt_full_np = pt_to_np(pt_full_out)
    s_full_vs_step = snr_db(pt_full_np, pt_spec_np)
    print(f"  PT sanity check (full vs step-by-step): SNR={s_full_vs_step:.1f} dB")

    # === TF streaming frame-by-frame ===
    state = {k: v.numpy() for k, v in tf_model.get_initial_state(batch_size=1).items()}

    tf_spec_list, tf_m_list, tf_lsnr_list = [], [], []
    tf_masked_list, tf_coefs_list = [], []

    for t in range(T):
        spec_frame = spec_np[:, :, t:t+1, :, :]
        erb_frame = feat_erb_np[:, :, t:t+1, :]
        feat_frame = feat_spec_np[:, :, t:t+1, :, :]

        tf_state = {k: tf.constant(v) for k, v in state.items()}

        # Run the streaming model's internal steps manually for debugging
        # 1. Get intermediates from encoder
        feat_spec_nhwc = tf.squeeze(tf.constant(feat_frame), axis=1)
        feat_erb_nhwc = tf.transpose(tf.constant(erb_frame), [0, 2, 3, 1])

        e0, e1, e2, e3, emb, c0, lsnr, erb_buf_new, df_buf_new, h_enc_new = \
            tf_model.enc(feat_erb_nhwc, feat_spec_nhwc,
                         tf_state["erb_buf"], tf_state["df_buf"], tf_state["h_enc"],
                         training=False)

        # 2. ERB decoder
        m_nhwc, h_erb_new = tf_model.erb_dec(emb, e3, e2, e1, e0,
                                              tf_state["h_erb"], training=False)

        # 3. Apply mask
        spec_tf = tf.transpose(tf.constant(spec_frame), [0, 2, 3, 1, 4])
        spec_masked_tf = tf_model.mask(spec_tf, m_nhwc)
        spec_masked = tf.transpose(spec_masked_tf, [0, 3, 1, 2, 4])
        tf_masked_list.append(spec_masked.numpy())

        # 4. DF decoder
        df_convp_buf = tf_state.get("df_convp_buf", None)
        if df_convp_buf is not None:
            df_convp_buf = tf.constant(df_convp_buf) if isinstance(df_convp_buf, np.ndarray) else df_convp_buf
        df_coefs, df_alpha, h_df_new, df_convp_buf_new = \
            tf_model.df_dec(emb, c0, tf_state["h_df"], df_convp_buf, training=False)
        tf_coefs_list.append(df_coefs.numpy())

        # 5. DF op
        spec_out, spec_buf_new = tf_model.df_op(spec_masked, df_coefs, tf_state["spec_buf"])

        m_pt_format = tf.transpose(m_nhwc, [0, 3, 1, 2])
        tf_spec_list.append(spec_out.numpy())
        tf_m_list.append(m_pt_format.numpy())
        tf_lsnr_list.append(lsnr.numpy())

        # Update state
        state = {
            "erb_buf": erb_buf_new.numpy(),
            "df_buf": df_buf_new.numpy(),
            "h_enc": h_enc_new.numpy(),
            "h_erb": h_erb_new.numpy(),
            "h_df": h_df_new.numpy(),
            "spec_buf": spec_buf_new.numpy(),
        }
        if df_convp_buf_new is not None:
            state["df_convp_buf"] = df_convp_buf_new.numpy()

    tf_spec_out = np.concatenate(tf_spec_list, axis=2)
    tf_m_out = np.concatenate(tf_m_list, axis=2)
    tf_lsnr_out = np.concatenate(tf_lsnr_list, axis=1)
    tf_masked_out = np.concatenate(tf_masked_list, axis=2)
    tf_coefs_out = np.concatenate(tf_coefs_list, axis=2)

    # === Compare at each stage ===
    results = {}
    for name, pt_val, tf_val in [
        ("mask", pt_m_np, tf_m_out),
        ("lsnr", pt_lsnr_np, tf_lsnr_out),
        ("df_coefs", pt_coefs_np, tf_coefs_out),
        ("spec_out", pt_spec_np, tf_spec_out),
    ]:
        s = snr_db(pt_val, tf_val)
        max_diff = np.max(np.abs(pt_val - tf_val))
        status = "✓ PASS" if s > 120 else "✗ FAIL"
        print(f"  {status} {name}: SNR={s:.1f} dB, max_diff={max_diff:.2e}")
        results[name] = s

    # masked_spec comparison is informational only — differences are expected
    # because PT Mask.forward modifies spec in-place during batch processing,
    # so comparing frame-by-frame masked spec is not a valid check.
    s = snr_db(pt_masked_np, tf_masked_out)
    max_diff = np.max(np.abs(pt_masked_np - tf_masked_out))
    print(f"  (info) masked_spec: SNR={s:.1f} dB, max_diff={max_diff:.2e} "
          f"(not in pass/fail — PT modifies spec in-place)")

    # Also show per-frame SNR for spec_out to localize issues
    print("\n  Per-frame spec_out SNR:")
    for t in range(T):
        s = snr_db(pt_spec_np[:, :, t:t+1, :, :], tf_spec_out[:, :, t:t+1, :, :])
        print(f"    frame {t}: {s:.1f} dB")

    return results


# ============================================================================
# Test 5: DF operation — PT batch unfold vs TF streaming ring buffer
# ============================================================================
def test_df_op():
    print("\n" + "=" * 60)
    print("Test 5: DF operation — PT unfold vs TF streaming")
    print("=" * 60)
    np.random.seed(42)

    p = ModelParams()
    F_df = p.nb_df
    O = p.df_order

    # PT DF op
    pt_df = MultiFrameDF(num_freqs=F_df, frame_size=O, lookahead=0, conj=False)

    # Generate random data
    T = 10
    spec_np = np.random.randn(1, 1, T, F_df, 2).astype(np.float32) * 0.1
    coefs_np = np.random.randn(1, O, T, F_df, 2).astype(np.float32) * 0.1

    # PT batch forward
    with torch.no_grad():
        spec_pt_out = pt_df(
            torch.from_numpy(spec_np.copy()),
            torch.from_numpy(coefs_np),
        )
    spec_pt_np = pt_to_np(spec_pt_out)  # [B, 1, T, F_df, 2]

    # TF streaming frame by frame
    from df.tf_stateful_streaming import MultiFrameDFStreamingTF
    tf_df = MultiFrameDFStreamingTF(num_freqs=F_df, frame_size=O, lookahead=0)
    spec_buf = tf.zeros([1, O - 1, F_df, 2])

    tf_spec_list = []
    for t in range(T):
        spec_frame = spec_np[:, :, t:t+1, :, :]  # [B,1,1,F,2]
        coefs_frame = coefs_np[:, :, t:t+1, :, :]  # [B,O,1,F,2]
        spec_out, spec_buf = tf_df(
            tf.constant(spec_frame),
            tf.constant(coefs_frame),
            spec_buf,
        )
        tf_spec_list.append(spec_out.numpy())

    tf_spec_out = np.concatenate(tf_spec_list, axis=2)

    # Only compare the DF bins (0:F_df) — the rest is pass-through
    pt_df_bins = spec_pt_np[:, :, :, :F_df, :]
    tf_df_bins = tf_spec_out[:, :, :, :F_df, :]

    s = snr_db(pt_df_bins, tf_df_bins)
    max_diff = np.max(np.abs(pt_df_bins - tf_df_bins))
    status = "✓ PASS" if s > 120 else "✗ FAIL"
    print(f"  {status} DF output (bins 0:{F_df}): SNR={s:.1f} dB, max_diff={max_diff:.2e}")

    # Per-frame
    print("\n  Per-frame DF SNR:")
    for t in range(T):
        s_t = snr_db(pt_df_bins[:, :, t:t+1, :, :], tf_df_bins[:, :, t:t+1, :, :])
        print(f"    frame {t}: {s_t:.1f} dB")

    return {"df_op": s}


# ============================================================================
# Test 6: Real WAV — PT batch vs TF streaming on actual audio
# ============================================================================
def test_real_wav(noisy_wav_path=None):
    """Process a real WAV file through both PyTorch and TF streaming.

    Both use the same STFT (numpy rfft + sqrt-hann window) and the same
    features (libdf erb/erb_norm/unit_norm), so outputs should match
    up to float32 precision.
    """
    print("\n" + "=" * 60)
    print("Test 6: Real WAV — PT batch vs TF streaming")
    print("=" * 60)

    # Find a noisy WAV
    if noisy_wav_path is None:
        candidates = [
            os.path.join(os.path.dirname(__file__), "..", "..", "models",
                         "streaming_audio_samples", "noisy.wav"),
            os.path.join(os.path.dirname(__file__), "..", "..", "wavs",
                         "keyboard_steak.wav"),
        ]
        for c in candidates:
            if os.path.exists(c):
                noisy_wav_path = c
                break
    if noisy_wav_path is None or not os.path.exists(noisy_wav_path):
        print("  SKIP: no WAV file found")
        return {}

    import soundfile as sf
    audio, sr = sf.read(noisy_wav_path)
    assert sr == 48000, f"Expected 48kHz, got {sr}"
    if audio.ndim > 1:
        audio = audio[:, 0]
    print(f"  Input: {noisy_wav_path}")
    print(f"  Length: {len(audio)/sr:.2f}s ({len(audio)} samples)")

    pt_model, sd, p = build_pt_model()
    fft_size = p.fft_size
    hop_size = p.hop_size
    F = fft_size // 2 + 1
    window = np.sqrt(np.hanning(fft_size + 1)[:fft_size]).astype(np.float32)

    n_samples = len(audio)
    n_frames = (n_samples - fft_size) // hop_size + 1
    print(f"  Frames: {n_frames}, fft={fft_size}, hop={hop_size}")

    # ── STFT all frames ──
    spec_list = []
    for t in range(n_frames):
        start = t * hop_size
        frame = audio[start:start + fft_size].astype(np.float32)
        windowed = frame * window
        stft_frame = np.fft.rfft(windowed).astype(np.complex64)
        spec_ri = np.stack([stft_frame.real, stft_frame.imag], axis=-1)
        spec_list.append(spec_ri)
    spec_all = np.array(spec_list)  # [T, F, 2]
    spec_5d = spec_all[np.newaxis, np.newaxis, :, :, :]  # [1,1,T,F,2]

    # ── Compute features with libdf ──
    from df.utils import get_norm_alpha
    from libdf import erb, erb_norm, unit_norm
    import warnings

    alpha = get_norm_alpha(False)
    df_state = LibDF(sr=p.sr, fft_size=p.fft_size, hop_size=p.hop_size,
                     nb_bands=p.nb_erb)
    erb_widths_np = df_state.erb_widths()

    spec_complex = (spec_all[:, :, 0] + 1j * spec_all[:, :, 1]).astype(np.complex64)
    spec_complex = spec_complex[np.newaxis, :, :]  # [C=1, T, F]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        erb_raw = erb(spec_complex.copy(), erb_widths_np, db=True)
        feat_erb_np = erb_norm(erb_raw.copy(), alpha)
    feat_erb_5d = feat_erb_np[np.newaxis, :, :, :]  # [B=1,C=1,T,E]

    feat_spec_complex = unit_norm(spec_complex[:, :, :p.nb_df].copy(), alpha)
    feat_spec_5d = np.stack([
        feat_spec_complex.real.astype(np.float32),
        feat_spec_complex.imag.astype(np.float32),
    ], axis=-1)[:, np.newaxis, :, :, :]  # [B=1,C=1,T,F_df,2]

    # ── PyTorch batch forward ──
    print("  Running PyTorch batch...")
    with torch.no_grad():
        pt_out = pt_model(
            torch.from_numpy(spec_5d.copy()),
            torch.from_numpy(feat_erb_5d.astype(np.float32)),
            torch.from_numpy(feat_spec_5d.astype(np.float32)),
        )
    pt_spec = pt_to_np(pt_out[0])  # [1,1,T,F,2]

    # ── TF streaming frame-by-frame ──
    print("  Running TF streaming...")
    erb_inv_fb_np = sd["mask.erb_inv_fb"]
    tf_model = DfNetStreamingTF(
        erb_inv_fb_np=erb_inv_fb_np,
        nb_erb=p.nb_erb, nb_df=p.nb_df, fft_size=p.fft_size,
        conv_ch=p.conv_ch, df_order=p.df_order, df_lookahead=0,
        emb_hidden_dim=p.emb_hidden_dim, emb_num_layers=p.emb_num_layers,
        df_hidden_dim=p.df_hidden_dim, df_num_layers=p.df_num_layers,
        gru_groups=p.gru_groups, lin_groups=p.lin_groups,
        group_shuffle=p.group_shuffle, enc_concat=p.enc_concat,
        df_pathway_kernel_size_t=p.df_pathway_kernel_size_t,
        gru_type=p.gru_type, df_gru_skip=p.df_gru_skip,
    )
    state = tf_model.get_initial_state(batch_size=1)
    _ = tf_model(
        tf.zeros([1, 1, 1, F, 2]),
        tf.zeros([1, 1, 1, p.nb_erb]),
        tf.zeros([1, 1, 1, p.nb_df, 2]),
        state,
    )
    tf_model.load_from_pt(sd)

    state = {k: v.numpy() for k, v in tf_model.get_initial_state(batch_size=1).items()}
    tf_spec_list = []
    for t in range(n_frames):
        spec_frame = spec_5d[:, :, t:t+1, :, :]
        erb_frame = feat_erb_5d[:, :, t:t+1, :]
        feat_frame = feat_spec_5d[:, :, t:t+1, :, :]
        tf_state = {k: tf.constant(v) for k, v in state.items()}
        spec_out, m, lsnr, alpha_out, new_state = tf_model(
            tf.constant(spec_frame), tf.constant(erb_frame),
            tf.constant(feat_frame), tf_state, training=False,
        )
        state = {k: v.numpy() for k, v in new_state.items()}
        tf_spec_list.append(spec_out.numpy())

    tf_spec = np.concatenate(tf_spec_list, axis=2)  # [1,1,T,F,2]

    # ── Compare spectra ──
    s_spec = snr_db(pt_spec, tf_spec)
    max_diff = np.max(np.abs(pt_spec - tf_spec))
    status = "✓ PASS" if s_spec > 120 else "✗ FAIL"
    print(f"  {status} Spectrum SNR: {s_spec:.1f} dB, max_diff={max_diff:.2e}")

    # ── ISTFT + overlap-add for both ──
    def istft_ola(spec_5d_data):
        frames = []
        for t in range(n_frames):
            enh_re = spec_5d_data[0, 0, t, :, 0]
            enh_im = spec_5d_data[0, 0, t, :, 1]
            enh_complex = enh_re + 1j * enh_im
            out_frame = np.fft.irfft(enh_complex, n=fft_size).astype(np.float32)
            out_frame *= window
            frames.append(out_frame)
        output_len = (n_frames - 1) * hop_size + fft_size
        output = np.zeros(output_len, dtype=np.float32)
        for t, frame in enumerate(frames):
            start = t * hop_size
            output[start:start + fft_size] += frame
        return output[:n_samples]

    pt_audio = istft_ola(pt_spec)
    tf_audio = istft_ola(tf_spec)

    s_audio = snr_db(pt_audio, tf_audio)
    status = "✓ PASS" if s_audio > 120 else "✗ FAIL"
    print(f"  {status} Audio SNR: {s_audio:.1f} dB")

    # ── Save outputs ──
    out_dir = os.path.join(os.path.dirname(__file__), "..", "..", "wavs")
    os.makedirs(out_dir, exist_ok=True)
    pt_path = os.path.join(out_dir, "pytorch_original", "enhanced_pytorch.wav")
    tf_path = os.path.join(out_dir, "enhanced_tf_streaming.wav")
    os.makedirs(os.path.dirname(pt_path), exist_ok=True)
    sf.write(pt_path, pt_audio, 48000)
    sf.write(tf_path, tf_audio, 48000)
    print(f"\n  Saved PyTorch output:  {pt_path}")
    print(f"  Saved TF output:       {tf_path}")

    return {"wav_spec": s_spec, "wav_audio": s_audio}


# ============================================================================
# Main
# ============================================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Compare TF streaming vs PyTorch")
    parser.add_argument("--wav", type=str, default=None,
                        help="Path to noisy WAV file (48kHz). "
                             "If not given, uses models/streaming_audio_samples/noisy.wav")
    parser.add_argument("--skip-modules", action="store_true",
                        help="Skip per-module tests, only run full model + wav")
    args = parser.parse_args()

    print("=" * 60)
    print("Streaming TF vs PyTorch comparison")
    print("(conv_lookahead=0, df_lookahead=0, pad_mode=none)")
    print("=" * 60)

    all_results = {}

    if not args.skip_modules:
        r = test_encoder()
        all_results.update(r)

        r = test_erb_decoder()
        all_results.update(r)

        r = test_df_decoder()
        all_results.update(r)

        r = test_df_op()
        all_results.update(r)

    r = test_full_model()
    all_results.update(r)

    r = test_real_wav(args.wav)
    all_results.update(r)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    all_pass = True
    for name, s in all_results.items():
        status = "✓ PASS" if s > 120 else "✗ FAIL"
        if s <= 120:
            all_pass = False
        print(f"  {status} {name}: {s:.1f} dB")

    if all_pass:
        print("\nAll tests PASSED (>120 dB SNR)")
    else:
        print("\nSome tests FAILED — see details above")
        sys.exit(1)
