"""
Export TFLite models (float32 / int8 / int16x8) and compare on real audio.

Comparisons:
  - float32 TFLite vs TF eager   (same streaming pipeline, expect >115 dB)
  - int8    TFLite vs TF eager   (quantization loss, expect ~25-45 dB)
  - int16x8 TFLite vs TF eager   (quantization loss, expect ~40-60 dB)

Usage:
    python DeepFilterNet/df/test_tflite_vs_pytorch.py
    python DeepFilterNet/df/test_tflite_vs_pytorch.py --wav wavs/keyboard_steak.wav
    python DeepFilterNet/df/test_tflite_vs_pytorch.py --skip-export
"""

import os, sys, time
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
import tensorflow as tf
import soundfile as sf

# ── Load pretrained config (with real lookahead=2) ──
PRETRAINED_DIR = "/tmp/dfnet2_pretrained/DeepFilterNet2"
PRETRAINED_CONFIG = os.path.join(PRETRAINED_DIR, "config.ini")
PRETRAINED_CKPT = os.path.join(PRETRAINED_DIR, "checkpoints", "model_96.ckpt.best")

from df.config import config
if not os.path.exists(PRETRAINED_CONFIG):
    import zipfile
    zip_path = os.path.join(os.path.dirname(__file__), "..", "..", "models",
                            "DeepFilterNet2.zip")
    os.makedirs(PRETRAINED_DIR, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall("/tmp/dfnet2_pretrained")

config.load(PRETRAINED_CONFIG, allow_defaults=True, allow_reload=True)

from df.deepfilternet2 import ModelParams, DfNet
from df.modules import erb_fb as pt_erb_fb
from df.tf_stateful_streaming import (
    DfNetStatefulStreamingTF,
    compute_erb_fb,
    build_tflite_nn_module,
)
from df.export_stateful_tflite import (
    export_tflite,
    export_nn_tflite,
    run_tf_on_audio,
    run_nn_tflite_on_audio,
    plot_spectrograms,
)
from libdf import DF as LibDF


def snr_db(ref, est):
    noise = ref - est
    ref_pow = np.mean(ref ** 2)
    noise_pow = np.mean(noise ** 2)
    if noise_pow < 1e-30:
        return 300.0
    return 10 * np.log10(ref_pow / noise_pow)


def build_models():
    """Build PyTorch + TF stateful model with pretrained weights."""
    p = ModelParams()
    print(f"  conv_lookahead={p.conv_lookahead}, df_lookahead={p.df_lookahead}")
    print(f"  pad_mode={p.pad_mode}, nb_erb={p.nb_erb}, nb_df={p.nb_df}")

    # ── PyTorch model ──
    df_state = LibDF(sr=p.sr, fft_size=p.fft_size, hop_size=p.hop_size,
                     nb_bands=p.nb_erb)
    erb = pt_erb_fb(df_state.erb_widths(), p.sr, inverse=False)
    erb_inv = pt_erb_fb(df_state.erb_widths(), p.sr, inverse=True)
    pt_model = DfNet(erb, erb_inv, run_df=True, train_mask=True)

    ckpt = torch.load(PRETRAINED_CKPT, map_location="cpu")
    model_sd = ckpt.get("model", ckpt)
    missing, unexpected = pt_model.load_state_dict(model_sd, strict=False)
    print(f"  PT: {len(missing)} missing, {len(unexpected)} unexpected keys")
    pt_model.eval()

    sd = {k: v.cpu().numpy() for k, v in pt_model.state_dict().items()}

    # ── TF stateful streaming model ──
    erb_inv_fb = sd["mask.erb_inv_fb"]
    erb_widths = compute_erb_fb(sr=48000, fft_size=p.fft_size,
                                nb_bands=p.nb_erb, min_nb_freqs=p.min_nb_freqs)
    F_bins = p.fft_size // 2 + 1

    tf_model = DfNetStatefulStreamingTF(
        erb_widths_np=erb_widths, erb_inv_fb_np=erb_inv_fb,
        nb_erb=p.nb_erb, nb_df=p.nb_df,
        fft_size=p.fft_size, sr=48000, hop_size=480, norm_tau=1.0,
        conv_ch=p.conv_ch, df_order=p.df_order, df_lookahead=p.df_lookahead,
        emb_hidden_dim=p.emb_hidden_dim, emb_num_layers=p.emb_num_layers,
        df_hidden_dim=p.df_hidden_dim, df_num_layers=p.df_num_layers,
        gru_groups=p.gru_groups, lin_groups=p.lin_groups,
        group_shuffle=p.group_shuffle, enc_concat=p.enc_concat,
        df_pathway_kernel_size_t=p.df_pathway_kernel_size_t,
        gru_type=p.gru_type, df_gru_skip=p.df_gru_skip,
        conv_lookahead=p.conv_lookahead, pad_mode=p.pad_mode,
        batch_size=1,
    )
    dummy = tf.zeros([1, 1, 1, F_bins, 2])
    tf_model.forward(dummy)
    tf_model.load_from_pt(sd)
    tf_model.reset_state()

    return pt_model, tf_model, sd, p, df_state


def print_tflite_io(tflite_path, label):
    """Print TFLite model input/output shapes, dtypes, quantization params."""
    interp = tf.lite.Interpreter(model_path=tflite_path)
    interp.allocate_tensors()
    inp = interp.get_input_details()
    out = interp.get_output_details()
    size_mb = os.path.getsize(tflite_path) / 1024 / 1024
    print(f"\n  ── {label} ({size_mb:.2f} MB) ──")
    print(f"  INPUTS ({len(inp)}):")
    for d in inp:
        shape = [int(x) for x in d['shape']]
        q_scale, q_zp = d.get("quantization", (0, 0))
        q_str = ""
        if d["dtype"] != np.float32:
            q_str = f", scale={q_scale:.6e}, zero_point={q_zp}"
        print(f"    {d['name']}: shape={shape}, "
              f"dtype={d['dtype'].__name__}{q_str}")
    print(f"  OUTPUTS ({len(out)}):")
    for d in out:
        shape = [int(x) for x in d['shape']]
        q_scale, q_zp = d.get("quantization", (0, 0))
        q_str = ""
        if d["dtype"] != np.float32:
            q_str = f", scale={q_scale:.6e}, zero_point={q_zp}"
        print(f"    {d['name']}: shape={shape}, "
              f"dtype={d['dtype'].__name__}{q_str}")


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Export TFLite (float32/int8/int16x8) and compare on real audio")
    parser.add_argument("--wav", type=str, default=None,
                        help="Path to noisy WAV (48kHz)")
    parser.add_argument("--skip-export", action="store_true",
                        help="Skip export, re-use existing TFLite files")
    args = parser.parse_args()

    print("=" * 60)
    print("TFLite Export + Real Audio Comparison")
    print("  float32 / int8 / int16x8")
    print("=" * 60)

    # Find WAV file
    if args.wav:
        noisy_wav = args.wav
    else:
        noisy_wav = os.path.join(os.path.dirname(__file__), "..", "..",
                                 "models", "streaming_audio_samples", "noisy.wav")
    if not os.path.exists(noisy_wav):
        print(f"ERROR: WAV not found: {noisy_wav}")
        sys.exit(1)

    audio, sr = sf.read(noisy_wav)
    assert sr == 48000, f"Expected 48kHz, got {sr}"
    if audio.ndim > 1:
        audio = audio[:, 0]
    print(f"  Input: {noisy_wav}")
    print(f"  Length: {len(audio)/sr:.2f}s ({len(audio)} samples)")

    # ── Step 1: Build models ──
    print("\n--- Building models ---")
    t0 = time.time()
    pt_model, tf_model, sd, p, df_state = build_models()
    print(f"  Build time: {time.time()-t0:.1f}s")

    out_dir = os.path.join(os.path.dirname(__file__), "..", "..", "models", "stateful")
    os.makedirs(out_dir, exist_ok=True)
    wav_dir = os.path.join(os.path.dirname(__file__), "..", "..", "wavs")
    os.makedirs(os.path.join(wav_dir, "pytorch_original"), exist_ok=True)

    f32_path = os.path.join(out_dir, "dfnet2_stateful_float32.tflite")
    i8_path = os.path.join(out_dir, "dfnet2_stateful_int8.tflite")
    i16_path = os.path.join(out_dir, "dfnet2_stateful_int16x8.tflite")

    # ── Step 2: Export TFLite models ──
    # All three variants use the NN-only wrapper so they share the same I/O:
    #   Input:  feat_erb [1,1,32,1], feat_spec [1,1,96,2]
    #   Output: mask [1,1,32,1], df_coefs [1,5,1,96,2]
    # Feature extraction + mask application + deep filtering stay in float32 Python.
    if args.skip_export and all(os.path.exists(p_) for p_ in [f32_path, i8_path, i16_path]):
        print("\n--- Skipping export, reusing existing TFLite files ---")
    else:
        print("\n--- Exporting TFLite models (all NN-only) ---")
        t0 = time.time()
        nn_wrapper = build_tflite_nn_module(tf_model)

        size = export_tflite(nn_wrapper, f32_path)
        print(f"  float32 (nn-only): {size:.2f} MB")

        size = export_nn_tflite(nn_wrapper, tf_model, i8_path, "int8")
        print(f"  int8 (nn-only):    {size:.2f} MB")

        size = export_nn_tflite(nn_wrapper, tf_model, i16_path, "int16x8")
        print(f"  int16x8 (nn-only): {size:.2f} MB")

        print(f"  Export time: {time.time()-t0:.1f}s")

    # ── Step 3: Print I/O shapes for all models ──
    print("\n" + "=" * 60)
    print("TFLite I/O Shapes & Quantization")
    print("=" * 60)
    print_tflite_io(f32_path, "float32 (NN-only: features → mask + df_coefs)")
    print_tflite_io(i8_path, "int8 (NN-only: features → mask + df_coefs)")
    print_tflite_io(i16_path, "int16x8 (NN-only: features → mask + df_coefs)")

    # ── Step 4: Run TF eager reference ──
    print(f"\n{'='*60}")
    print("Running inference on real audio")
    print(f"{'='*60}")
    print("\n  Running TF eager streaming (reference)...")
    t0 = time.time()
    tf_audio = run_tf_on_audio(tf_model, noisy_wav, lookahead=p.conv_lookahead)
    tf_time = time.time() - t0
    print(f"  TF eager time: {tf_time:.2f}s")

    # ── Step 5: Run TFLite float32 (NN-only + float32 pre/post) ──
    print("\n  Running TFLite float32 (NN-only + float32 pre/post)...")
    t0 = time.time()
    f32_audio = run_nn_tflite_on_audio(f32_path, tf_model, noisy_wav)
    f32_time = time.time() - t0
    print(f"  TFLite float32 time: {f32_time:.2f}s")

    # ── Step 6: Run TFLite int8 ──
    print("\n  Running TFLite int8 (NN-only + float32 pre/post)...")
    t0 = time.time()
    i8_audio = run_nn_tflite_on_audio(i8_path, tf_model, noisy_wav)
    i8_time = time.time() - t0
    print(f"  TFLite int8 time: {i8_time:.2f}s")

    # ── Step 7: Run TFLite int16x8 ──
    print("\n  Running TFLite int16x8 (NN-only + float32 pre/post)...")
    t0 = time.time()
    i16_audio = run_nn_tflite_on_audio(i16_path, tf_model, noisy_wav)
    i16_time = time.time() - t0
    print(f"  TFLite int16x8 time: {i16_time:.2f}s")

    # ── Step 8: Compare all vs TF eager ──
    min_len = min(len(tf_audio), len(f32_audio), len(i8_audio), len(i16_audio))
    tf_audio = tf_audio[:min_len]
    f32_audio = f32_audio[:min_len]
    i8_audio = i8_audio[:min_len]
    i16_audio = i16_audio[:min_len]

    s_f32 = snr_db(tf_audio, f32_audio)
    s_i8 = snr_db(tf_audio, i8_audio)
    s_i16 = snr_db(tf_audio, i16_audio)

    print(f"\n{'='*60}")
    print("RESULTS — SNR vs TF eager streaming reference")
    print(f"{'='*60}")
    print(f"  {'Model':<22} {'SNR (dB)':>10} {'Max Diff':>12} {'Time':>8}")
    print(f"  {'-'*22} {'-'*10} {'-'*12} {'-'*8}")

    for label, audio_out, s, t_ in [
        ("TFLite float32", f32_audio, s_f32, f32_time),
        ("TFLite int8", i8_audio, s_i8, i8_time),
        ("TFLite int16x8", i16_audio, s_i16, i16_time),
    ]:
        md = np.max(np.abs(tf_audio - audio_out))
        print(f"  {label:<22} {s:>10.1f} {md:>12.2e} {t_:>7.2f}s")

    # ── Step 9: Save WAV outputs ──
    sf.write(os.path.join(wav_dir, "enhanced_tf_stateful.wav"), tf_audio, 48000)
    sf.write(os.path.join(wav_dir, "enhanced_tflite_float32.wav"), f32_audio, 48000)
    sf.write(os.path.join(wav_dir, "enhanced_tflite_int8.wav"), i8_audio, 48000)
    sf.write(os.path.join(wav_dir, "enhanced_tflite_int16x8.wav"), i16_audio, 48000)
    print(f"\n  WAV outputs saved to {wav_dir}/")

    # ── Step 10: Spectrogram comparison ──
    print("\n--- Generating spectrograms ---")
    noisy_trimmed = audio[:min_len]
    audio_dict = {"Noisy": noisy_trimmed}
    clean_path = os.path.join(os.path.dirname(__file__), "..", "..",
                              "models", "streaming_audio_samples", "clean.wav")
    if os.path.exists(clean_path):
        clean_audio, _ = sf.read(clean_path)
        if clean_audio.ndim > 1:
            clean_audio = clean_audio[:, 0]
        audio_dict["Clean"] = clean_audio[:min_len]

    audio_dict["Enhanced (TF eager ref)"] = tf_audio
    audio_dict["Enhanced (TFLite float32)"] = f32_audio
    audio_dict["Enhanced (TFLite int8)"] = i8_audio
    audio_dict["Enhanced (TFLite int16x8)"] = i16_audio

    spec_path = os.path.join(out_dir, "tflite_all_variants_spectrograms.png")
    plot_spectrograms(
        audio_dict,
        title="TFLite Variants — Spectrogram Comparison",
        save_path=spec_path,
    )

    print(f"\n{'='*60}")
    print("Done!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
