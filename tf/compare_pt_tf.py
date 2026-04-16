"""Compare PyTorch vs TF model outputs on identical features.

Patches torch._six for newer PyTorch versions, loads the pretrained
checkpoint into both PT and TF models, runs them on the same features,
and prints per-output max/mean differences.
"""
import types
import sys

# Patch torch._six for PyTorch >= 2.0
import torch
if not hasattr(torch, '_six'):
    m = types.ModuleType('torch._six')
    m.string_classes = (str,)
    sys.modules['torch._six'] = m

import numpy as np
import tensorflow as tf

sys.path.insert(0, '../DeepFilterNet')

from df.config import config
config.load('../models/DeepFilterNet2/config.ini', allow_defaults=True)

from df.deepfilternet2 import ModelParams, DfNet as DfNetPT, Encoder as EncPT, ErbDecoder as ErbDecPT, DfDecoder as DfDecPT

from deepfilternet2_tf import DfNet as DfNetTF, DEFAULT_PARAMS
from weight_transfer import transfer_weights
from compare_batch_streaming import extract_features, load_wav_mono


def main():
    p = DEFAULT_PARAMS
    ckpt = torch.load(
        '../models/DeepFilterNet2/checkpoints/model_96.ckpt.best',
        map_location='cpu', weights_only=False
    )
    erb_fb = ckpt['erb_fb']  # Tensor [F, E]
    erb_inv_fb = ckpt['mask.erb_inv_fb']  # Tensor [E, F]

    # Load audio
    audio, sr = load_wav_mono('../models/streaming_audio_samples/noisy.wav')
    erb_fb_np = erb_fb.numpy()
    erb_inv_fb_np = erb_inv_fb.numpy()

    # Feature extraction (same for both)
    spec_tf, feat_erb_tf, feat_spec_tf = extract_features(audio, erb_fb_np, p)
    T = feat_erb_tf.shape[2]

    # =========================================================================
    # PyTorch model
    # =========================================================================
    pt_model = DfNetPT(erb_fb, erb_inv_fb, run_df=True, train_mask=True)
    # Load state_dict
    pt_sd = {k: v for k, v in ckpt.items() if k not in ('erb_fb', 'mask.erb_inv_fb')}
    pt_model.load_state_dict(pt_sd, strict=False)
    pt_model.eval()

    # Convert TF tensors to PT
    feat_erb_pt = torch.from_numpy(feat_erb_tf.numpy())  # [1,1,T,E]
    feat_spec_pt_raw = torch.from_numpy(feat_spec_tf.numpy())  # [1,1,T,Fc,2]
    spec_pt = torch.from_numpy(spec_tf.numpy())  # [1,1,T,F,2]

    with torch.no_grad():
        spec_out_pt, m_pt, lsnr_pt, alpha_pt = pt_model(spec_pt, feat_erb_pt, feat_spec_pt_raw)

    m_pt_np = m_pt.numpy()  # [1,1,T,E]
    lsnr_pt_np = lsnr_pt.numpy()  # [1,T,1]

    print("=== PyTorch ===")
    print(f"  mask shape: {m_pt_np.shape}, range: [{m_pt_np.min():.4f}, {m_pt_np.max():.4f}], mean: {m_pt_np.mean():.4f}")
    print(f"  lsnr range: [{lsnr_pt_np.min():.2f}, {lsnr_pt_np.max():.2f}]")

    # =========================================================================
    # TF model
    # =========================================================================
    tf_model = DfNetTF(erb_inv_fb_np, p, run_df=True)
    _ = tf_model(feat_erb_tf, feat_spec_tf, training=False)
    transfer_weights(ckpt, tf_model)
    m_tf, lsnr_tf, coefs_tf, alpha_tf = tf_model(feat_erb_tf, feat_spec_tf, training=False)

    m_tf_np = m_tf.numpy()  # [1,1,T,E]
    lsnr_tf_np = lsnr_tf.numpy()  # [1,T,1]

    print("\n=== TensorFlow ===")
    print(f"  mask shape: {m_tf_np.shape}, range: [{m_tf_np.min():.4f}, {m_tf_np.max():.4f}], mean: {m_tf_np.mean():.4f}")
    print(f"  lsnr range: [{lsnr_tf_np.min():.2f}, {lsnr_tf_np.max():.2f}]")

    # =========================================================================
    # Compare
    # =========================================================================
    print("\n=== Comparison (raw, no lookahead alignment) ===")
    mask_diff = np.abs(m_pt_np - m_tf_np)
    print(f"  Mask max diff: {mask_diff.max():.6f}")
    print(f"  Mask mean diff: {mask_diff.mean():.6f}")

    lsnr_diff = np.abs(lsnr_pt_np - lsnr_tf_np)
    print(f"  LSNR max diff: {lsnr_diff.max():.6f}")
    print(f"  LSNR mean diff: {lsnr_diff.mean():.6f}")

    # Try with lookahead alignment: PT output is shifted by conv_lookahead=2
    # PT's pad_feat shifts features: removes first 2 frames, adds 2 zeros at end
    # So PT mask at frame t corresponds to TF mask at frame t+2
    la = p.get("conv_lookahead", 2)
    if la > 0:
        print(f"\n=== Comparison (aligned, lookahead={la}) ===")
        m_pt_aligned = m_pt_np[:, :, la:, :]
        m_tf_aligned = m_tf_np[:, :, :-la, :]
        mask_diff_a = np.abs(m_pt_aligned - m_tf_aligned)
        print(f"  Mask max diff: {mask_diff_a.max():.6f}")
        print(f"  Mask mean diff: {mask_diff_a.mean():.6f}")

        lsnr_pt_a = lsnr_pt_np[:, la:, :]
        lsnr_tf_a = lsnr_tf_np[:, :-la, :]
        lsnr_diff_a = np.abs(lsnr_pt_a - lsnr_tf_a)
        print(f"  LSNR max diff: {lsnr_diff_a.max():.6f}")
        print(f"  LSNR mean diff: {lsnr_diff_a.mean():.6f}")

    # Also test: what if we manually apply pad_feat to TF features?
    print(f"\n=== TF with manual pad_feat (lookahead={la}) ===")
    feat_erb_shifted = tf.pad(feat_erb_tf[:, :, la:, :], [[0,0],[0,0],[0,la],[0,0]])
    feat_spec_shifted = tf.pad(feat_spec_tf[:, :, la:, :, :], [[0,0],[0,0],[0,la],[0,0],[0,0]])
    m_tf2, lsnr_tf2, _, _ = tf_model(feat_erb_shifted, feat_spec_shifted, training=False)
    m_tf2_np = m_tf2.numpy()
    lsnr_tf2_np = lsnr_tf2.numpy()
    mask_diff2 = np.abs(m_pt_np - m_tf2_np)
    print(f"  Mask max diff: {mask_diff2.max():.6f}")
    print(f"  Mask mean diff: {mask_diff2.mean():.6f}")
    lsnr_diff2 = np.abs(lsnr_pt_np - lsnr_tf2_np)
    print(f"  LSNR max diff: {lsnr_diff2.max():.6f}")
    print(f"  LSNR mean diff: {lsnr_diff2.mean():.6f}")


if __name__ == "__main__":
    main()
