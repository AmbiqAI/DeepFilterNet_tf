"""Compare batch vs streaming model outputs using real audio + pretrained weights.

Implements the full feature extraction pipeline (STFT, ERB, normalization)
in pure NumPy/TF, then runs both batch and streaming models and plots
spectrograms side-by-side.
"""

import numpy as np
import tensorflow as tf
import wave
import struct
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from deepfilternet2_tf import DfNet, DEFAULT_PARAMS
from deepfilternet2_tf_streaming import DfNetStreaming, copy_weights_from_batch_model
from weight_transfer import transfer_weights

# =============================================================================
# Feature extraction (pure NumPy)
# =============================================================================

def vorbis_window(n):
    """Vorbis/sine-squared window used by DeepFilterNet."""
    i = np.arange(n, dtype=np.float64)
    return np.sin(0.5 * np.pi * np.sin(0.5 * np.pi * (i + 0.5) / (n / 2)) ** 2).astype(np.float32)


def load_wav_mono(path):
    """Load wav as float32 mono, return (samples, sr)."""
    with wave.open(path, "r") as w:
        sr = w.getframerate()
        n = w.getnframes()
        ch = w.getnchannels()
        raw = w.readframes(n)
        sw = w.getsampwidth()
    if sw == 2:
        samples = np.array(struct.unpack(f"<{n * ch}h", raw), dtype=np.float32) / 32768.0
    elif sw == 4:
        samples = np.array(struct.unpack(f"<{n * ch}i", raw), dtype=np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported sample width: {sw}")
    if ch > 1:
        samples = samples.reshape(-1, ch).mean(axis=1)
    return samples, sr


def stft_analysis(audio, fft_size=960, hop_size=480):
    """Frame-by-frame STFT matching DeepFilterNet's Rust implementation.

    Returns: complex spectrum [T, F] where F = fft_size//2 + 1
    """
    win = vorbis_window(fft_size)
    wnorm = hop_size / (fft_size * np.sum(win ** 2))

    freq_size = fft_size // 2 + 1
    # Pad audio to fill last frame
    pad_len = (fft_size - hop_size)  # initial analysis memory = zeros
    audio_padded = np.concatenate([np.zeros(pad_len, dtype=np.float32), audio])
    n_frames = (len(audio_padded) - fft_size) // hop_size + 1

    specs = []
    for t in range(n_frames):
        start = t * hop_size
        frame = audio_padded[start:start + fft_size]
        windowed = frame * win
        spectrum = np.fft.rfft(windowed) * wnorm
        specs.append(spectrum)

    return np.array(specs, dtype=np.complex64)  # [T, F]


def compute_erb(spec, erb_fb):
    """Compute ERB band energies.

    spec: [T, F] complex
    erb_fb: [F, E] normalized filterbank (columns sum to 1)
    Returns: [T, E] real (mean power per band)
    """
    power = np.abs(spec) ** 2  # [T, F]
    return (power @ erb_fb).astype(np.float32)  # [T, E]


def erb_norm(erb_feat, alpha, nb_erb=32):
    """Exponential mean normalization for ERB features.

    erb_feat: [T, E]
    alpha: smoothing factor (~0.99)
    Returns: [T, E] normalized
    """
    state = np.linspace(-60.0, -90.0, nb_erb).astype(np.float32)
    out = np.empty_like(erb_feat)
    for t in range(erb_feat.shape[0]):
        state = erb_feat[t] * (1 - alpha) + state * alpha
        out[t] = (erb_feat[t] - state) / 40.0
    return out


def unit_norm(spec, alpha, nb_df=96):
    """Unit normalization for complex spectral features.

    spec: [T, F] complex (F <= nb_df)
    alpha: smoothing factor
    Returns: [T, F] complex normalized
    """
    state = np.linspace(0.001, 0.0001, nb_df).astype(np.float32)
    out = np.empty_like(spec)
    for t in range(spec.shape[0]):
        mag = np.abs(spec[t])
        state = mag * (1 - alpha) + state * alpha
        out[t] = spec[t] / np.sqrt(np.maximum(state, 1e-10))
    return out


def extract_features(audio, erb_fb, p):
    """Full feature extraction pipeline.

    Returns:
        spec:      [1, 1, T, F, 2] full spectrum (2-ch real)
        feat_erb:  [1, 1, T, E]    ERB features
        feat_spec: [1, 1, T, Fc, 2] spectral features (2-ch real)
    """
    fft_size = p["fft_size"]
    hop_size = p["hop_size"]
    nb_df = p["nb_df"]
    nb_erb = p["nb_erb"]

    alpha = np.exp(-hop_size / (p["sr"] * 1.0)).astype(np.float32)  # norm_tau=1.0

    # STFT
    spec_complex = stft_analysis(audio, fft_size, hop_size)  # [T, F]

    # ERB features: compute power, convert to dB, then normalize
    erb_feat = compute_erb(spec_complex, erb_fb)    # [T, E] raw power
    erb_feat = 10.0 * np.log10(np.maximum(erb_feat, 1e-10)).astype(np.float32)  # dB
    erb_feat = erb_norm(erb_feat, alpha, nb_erb)    # [T, E] normalized

    # Spectral features (first nb_df bins)
    spec_df = spec_complex[:, :nb_df]               # [T, Fc]
    spec_df_normed = unit_norm(spec_df, alpha, nb_df)  # [T, Fc]

    # Convert to 2-channel real tensors
    T = spec_complex.shape[0]
    F = spec_complex.shape[1]

    spec_2ch = np.stack([spec_complex.real, spec_complex.imag], axis=-1)  # [T, F, 2]
    feat_spec_2ch = np.stack([spec_df_normed.real, spec_df_normed.imag], axis=-1)  # [T, Fc, 2]

    # Add batch and channel dims
    spec_tf = tf.constant(spec_2ch[np.newaxis, np.newaxis], dtype=tf.float32)       # [1,1,T,F,2]
    feat_erb_tf = tf.constant(erb_feat[np.newaxis, np.newaxis], dtype=tf.float32)   # [1,1,T,E]
    feat_spec_tf = tf.constant(feat_spec_2ch[np.newaxis, np.newaxis], dtype=tf.float32)  # [1,1,T,Fc,2]

    return spec_tf, feat_erb_tf, feat_spec_tf


# =============================================================================
# Main comparison
# =============================================================================

def main():
    import torch

    p = DEFAULT_PARAMS
    audio_path = "../models/streaming_audio_samples/noisy.wav"
    audio, sr = load_wav_mono(audio_path)
    assert sr == p["sr"], f"Expected sr={p['sr']}, got {sr}"
    print(f"Loaded audio: {len(audio)} samples, {len(audio)/sr:.2f}s")

    # Load checkpoint
    ckpt = torch.load(
        "../models/DeepFilterNet2/checkpoints/model_96.ckpt.best",
        map_location="cpu", weights_only=False
    )
    erb_fb_np = ckpt["erb_fb"].numpy()  # [481, 32]
    erb_inv_fb_np = ckpt["mask.erb_inv_fb"].numpy()  # [32, 481]

    # Extract features
    spec, feat_erb, feat_spec = extract_features(audio, erb_fb_np, p)
    T = feat_erb.shape[2]
    print(f"Extracted features: T={T} frames")
    print(f"  spec: {spec.shape}, feat_erb: {feat_erb.shape}, feat_spec: {feat_spec.shape}")

    # =========================================================================
    # Batch model
    # =========================================================================
    print("\n--- Batch Model ---")
    batch_model = DfNet(erb_inv_fb_np, p, run_df=True)
    _ = batch_model(feat_erb, feat_spec, training=False)
    transfer_weights(ckpt, batch_model)

    m_batch, lsnr_batch, coefs_batch, alpha_batch = batch_model(
        feat_erb, feat_spec, training=False
    )
    print(f"  mask: {m_batch.shape}, lsnr: {lsnr_batch.shape}")
    print(f"  df_coefs: {coefs_batch.shape}, alpha: {alpha_batch.shape}")

    # =========================================================================
    # Streaming model
    # =========================================================================
    print("\n--- Streaming Model ---")
    stream_model = DfNetStreaming(p)
    state = stream_model.get_initial_state(batch_size=1)
    # Build with dummy
    _ = stream_model(
        tf.zeros([1, 1, 1, p["nb_erb"]]),
        tf.zeros([1, 1, 1, p["nb_df"], 2]),
        state,
    )
    copy_weights_from_batch_model(batch_model, stream_model)

    # Run frame-by-frame
    m_frames = []
    lsnr_frames = []
    coefs_frames = []
    alpha_frames = []
    state = stream_model.get_initial_state(batch_size=1)

    for t in range(T):
        erb_frame = feat_erb[:, :, t:t+1, :]         # [1,1,1,E]
        spec_frame = feat_spec[:, :, t:t+1, :, :]    # [1,1,1,Fc,2]

        m_t, lsnr_t, coefs_t, alpha_t, state = stream_model(
            erb_frame, spec_frame, state
        )
        m_frames.append(m_t.numpy())
        lsnr_frames.append(lsnr_t.numpy())
        coefs_frames.append(coefs_t.numpy())
        alpha_frames.append(alpha_t.numpy())

    # Stack streaming results: [1,1,T,E] etc
    m_stream = np.concatenate(m_frames, axis=2)        # [1,1,T,E]
    lsnr_stream = np.concatenate(lsnr_frames, axis=1)  # [1,T,1]... actually [1,1] per frame
    # lsnr is [B,1] per frame -> stack along dim1
    lsnr_stream = np.stack([l[0] for l in lsnr_frames], axis=0)[np.newaxis]  # [1,T,1]
    alpha_stream = np.stack([a[0] for a in alpha_frames], axis=0)[np.newaxis]  # [1,T,1]

    print(f"  mask: {m_stream.shape}, lsnr: {lsnr_stream.shape}")

    # Stack DF coefs for streaming
    # coefs_frames: list of [1, O, 1, Fc, 2] -> concat on axis=2
    coefs_stream = np.concatenate(coefs_frames, axis=2)  # [1, O, T, Fc, 2]

    # =========================================================================
    # Compare numeric outputs
    # =========================================================================
    print("\n--- Comparison ---")
    m_b = m_batch.numpy()
    m_s = m_stream

    mask_diff = np.abs(m_b - m_s)
    print(f"  Mask max diff: {mask_diff.max():.6f}")
    print(f"  Mask mean diff: {mask_diff.mean():.6f}")

    lsnr_b = lsnr_batch.numpy()
    lsnr_s = lsnr_stream
    lsnr_diff = np.abs(lsnr_b - lsnr_s)
    print(f"  LSNR max diff: {lsnr_diff.max():.6f}")
    print(f"  LSNR mean diff: {lsnr_diff.mean():.6f}")

    coefs_b = coefs_batch.numpy()  # [1, O, T, Fc, 2]
    coefs_s = coefs_stream
    coefs_diff = np.abs(coefs_b - coefs_s)
    print(f"  DF coefs max diff: {coefs_diff.max():.6f}")
    print(f"  DF coefs mean diff: {coefs_diff.mean():.6f}")

    # =========================================================================
    # Apply full enhancement: ERB mask + DF operation (in NumPy)
    # =========================================================================
    spec_np = spec.numpy()[0, 0]  # [T, F, 2]

    def apply_erb_mask(spec_2ch, mask_erb, erb_inv_fb):
        """Apply ERB mask to full spectrum. spec_2ch: [T,F,2], mask: [T,E]"""
        gain = mask_erb @ erb_inv_fb  # [T, F]
        out = spec_2ch.copy()
        out[..., 0] *= gain
        out[..., 1] *= gain
        return out

    def apply_df(spec_2ch, coefs, df_order, nb_df):
        """Apply deep filtering to the first nb_df bins.

        spec_2ch: [T, F, 2]
        coefs: [O, T, Fc, 2]
        """
        T_s, F_s = spec_2ch.shape[:2]
        O = coefs.shape[0]
        out = spec_2ch.copy()

        # Pad time for lookback
        pad = np.zeros((O - 1, nb_df, 2), dtype=spec_2ch.dtype)
        spec_df_padded = np.concatenate([pad, spec_2ch[:, :nb_df, :]], axis=0)  # [T+O-1, Fc, 2]

        for t in range(T_s):
            for f in range(nb_df):
                re_acc = 0.0
                im_acc = 0.0
                for o in range(O):
                    s_re = spec_df_padded[t + o, f, 0]
                    s_im = spec_df_padded[t + o, f, 1]
                    c_re = coefs[o, t, f, 0]
                    c_im = coefs[o, t, f, 1]
                    re_acc += s_re * c_re - s_im * c_im
                    im_acc += s_re * c_im + s_im * c_re
                out[t, f, 0] = re_acc
                out[t, f, 1] = im_acc
        return out

    df_order = p["df_order"]
    nb_df = p["nb_df"]

    # Batch: mask then DF
    enhanced_batch = apply_erb_mask(spec_np, m_b[0, 0], erb_inv_fb_np)
    enhanced_batch = apply_df(enhanced_batch, coefs_b[0], df_order, nb_df)

    # Streaming: mask then DF
    enhanced_stream = apply_erb_mask(spec_np, m_s[0, 0], erb_inv_fb_np)
    enhanced_stream = apply_df(enhanced_stream, coefs_s[0], df_order, nb_df)

    # Magnitude spectrograms
    spec_mag = np.sqrt(spec_np[..., 0]**2 + spec_np[..., 1]**2)
    spec_db = 20 * np.log10(np.maximum(spec_mag, 1e-8))

    enh_mag_batch = np.sqrt(enhanced_batch[..., 0]**2 + enhanced_batch[..., 1]**2)
    enh_db_batch = 20 * np.log10(np.maximum(enh_mag_batch, 1e-8))

    enh_mag_stream = np.sqrt(enhanced_stream[..., 0]**2 + enhanced_stream[..., 1]**2)
    enh_db_stream = 20 * np.log10(np.maximum(enh_mag_stream, 1e-8))

    enh_diff_db = np.abs(enh_db_batch - enh_db_stream)

    print(f"\n  Enhanced spec max diff (dB): {enh_diff_db.max():.4f}")
    print(f"  Enhanced spec mean diff (dB): {enh_diff_db.mean():.4f}")

    # =========================================================================
    # Plot
    # =========================================================================
    print("\n--- Plotting ---")

    hop_ms = p["hop_size"] / p["sr"] * 1000
    time_axis = np.arange(T) * hop_ms / 1000  # seconds
    freq_khz = p["sr"] // 2 / 1000

    fig, axes = plt.subplots(5, 2, figsize=(16, 18))
    fig.suptitle("DeepFilterNet2: Batch vs Streaming Comparison", fontsize=15, y=0.995)

    vmin, vmax = -80, 0

    # Row 0: Input noisy | ERB features
    ax = axes[0, 0]
    im = ax.imshow(spec_db.T, aspect="auto", origin="lower",
                   extent=[0, time_axis[-1], 0, freq_khz],
                   cmap="magma", vmin=vmin, vmax=vmax)
    ax.set_title("Input Noisy Spectrogram")
    ax.set_ylabel("Freq (kHz)")
    plt.colorbar(im, ax=ax, label="dB")

    ax = axes[0, 1]
    erb_np = feat_erb.numpy()[0, 0]
    im = ax.imshow(erb_np.T, aspect="auto", origin="lower",
                   extent=[0, time_axis[-1], 0, 32], cmap="viridis")
    ax.set_title("ERB Features (normalized)")
    ax.set_ylabel("ERB Band")
    plt.colorbar(im, ax=ax)

    # Row 1: Enhanced (Mask+DF) — Batch | Streaming
    ax = axes[1, 0]
    im = ax.imshow(enh_db_batch.T, aspect="auto", origin="lower",
                   extent=[0, time_axis[-1], 0, freq_khz],
                   cmap="magma", vmin=vmin, vmax=vmax)
    ax.set_title("Enhanced (Mask+DF) — Batch")
    ax.set_ylabel("Freq (kHz)")
    plt.colorbar(im, ax=ax, label="dB")

    ax = axes[1, 1]
    im = ax.imshow(enh_db_stream.T, aspect="auto", origin="lower",
                   extent=[0, time_axis[-1], 0, freq_khz],
                   cmap="magma", vmin=vmin, vmax=vmax)
    ax.set_title("Enhanced (Mask+DF) — Streaming")
    ax.set_ylabel("Freq (kHz)")
    plt.colorbar(im, ax=ax, label="dB")

    # Row 2: Enhanced difference | LSNR comparison
    ax = axes[2, 0]
    diff_max = max(enh_diff_db.max(), 0.01)
    im = ax.imshow(enh_diff_db.T, aspect="auto", origin="lower",
                   extent=[0, time_axis[-1], 0, freq_khz],
                   cmap="hot", vmin=0, vmax=diff_max)
    ax.set_title(f"Enhanced Difference |Batch−Stream| (max={enh_diff_db.max():.2f} dB)")
    ax.set_ylabel("Freq (kHz)")
    plt.colorbar(im, ax=ax, label="dB")

    ax = axes[2, 1]
    ax.plot(time_axis, lsnr_b[0, :, 0], label="Batch", alpha=0.8, linewidth=1.5)
    ax.plot(time_axis, lsnr_s[0, :, 0], label="Streaming", alpha=0.8, linestyle="--", linewidth=1.5)
    ax.set_title("Local SNR Estimate")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("LSNR (dB)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Row 3: ERB Mask — Batch | Streaming
    ax = axes[3, 0]
    im = ax.imshow(m_b[0, 0].T, aspect="auto", origin="lower",
                   extent=[0, time_axis[-1], 0, 32],
                   cmap="inferno", vmin=0, vmax=1)
    ax.set_title("ERB Mask — Batch")
    ax.set_ylabel("ERB Band")
    plt.colorbar(im, ax=ax)

    ax = axes[3, 1]
    im = ax.imshow(m_s[0, 0].T, aspect="auto", origin="lower",
                   extent=[0, time_axis[-1], 0, 32],
                   cmap="inferno", vmin=0, vmax=1)
    ax.set_title("ERB Mask — Streaming")
    ax.set_ylabel("ERB Band")
    plt.colorbar(im, ax=ax)

    # Row 4: Mask difference | DF coefs magnitude (batch, order=0)
    ax = axes[4, 0]
    im = ax.imshow(mask_diff[0, 0].T, aspect="auto", origin="lower",
                   extent=[0, time_axis[-1], 0, 32],
                   cmap="hot", vmin=0, vmax=max(mask_diff.max(), 1e-4))
    ax.set_title(f"Mask Difference (max={mask_diff.max():.4f})")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("ERB Band")
    plt.colorbar(im, ax=ax)

    ax = axes[4, 1]
    coef0_mag = np.sqrt(coefs_b[0, 0, :, :, 0]**2 + coefs_b[0, 0, :, :, 1]**2)  # [T, Fc]
    im = ax.imshow(coef0_mag.T, aspect="auto", origin="lower",
                   extent=[0, time_axis[-1], 0, nb_df], cmap="viridis")
    ax.set_title("DF Coefficients Magnitude (order=0, Batch)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("DF Freq Bin")
    plt.colorbar(im, ax=ax)

    plt.tight_layout()
    out_path = "batch_vs_streaming.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close()


if __name__ == "__main__":
    main()
