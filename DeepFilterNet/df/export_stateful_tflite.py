"""
Export stateful streaming DeepFilterNet2 to TFLite and run on real audio.

Export split:
    - float32 / dynamic / float16 use the full stateful wrapper
        (spec -> enhanced_spec).
    - int8 / int16x8 use the NN-only wrapper
        (feat_erb, feat_spec -> mask, df_coefs).

This is intentional: for int8/int16x8, all feature extraction must stay
outside the exported neural network. That means no a^2+b^2, no sqrt, no log10,
no mask expansion, and no deep filtering inside the quantized TFLite model.

Produces:
    models/stateful/dfnet2_stateful_float32.tflite
    models/stateful/dfnet2_stateful_dynamic.tflite
    models/stateful/dfnet2_stateful_float16.tflite
    models/stateful/dfnet2_stateful_int8.tflite       (NN-only quantized model)
    models/stateful/dfnet2_stateful_int16x8.tflite    (NN-only quantized model)
    models/stateful/audio_samples/{noisy,clean,enhanced_*.wav}
    models/stateful/spectrogram_comparison.png
"""

import os
import sys
import tempfile
import shutil

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf
import torch
import soundfile as sf

# ── Load pretrained config ──
PRETRAINED_DIR = "/tmp/dfnet2_pretrained/DeepFilterNet2"
PRETRAINED_CONFIG = os.path.join(PRETRAINED_DIR, "config.ini")
PRETRAINED_CKPT = os.path.join(PRETRAINED_DIR, "checkpoints", "model_96.ckpt.best")

from df.config import config
if not os.path.exists(PRETRAINED_CONFIG):
    import zipfile
    zip_path = os.path.join(os.path.dirname(__file__), "..", "..", "models", "DeepFilterNet2.zip")
    os.makedirs(PRETRAINED_DIR, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall("/tmp/dfnet2_pretrained")

config.load(PRETRAINED_CONFIG, allow_defaults=True, allow_reload=True)

from df.deepfilternet2 import ModelParams, init_model
from df.tf_stateful_streaming import (
    DfNetStatefulStreamingTF,
    compute_erb_fb,
    compute_norm_alpha,
    build_tflite_module,
    build_tflite_nn_module,
    MEAN_NORM_INIT,
    UNIT_NORM_INIT,
)

# ── Output directories ──
BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models", "stateful")
AUDIO_DIR = os.path.join(BASE_DIR, "audio_samples")
os.makedirs(AUDIO_DIR, exist_ok=True)

# ── Source audio ──
AUDIO_SRC = os.path.join(os.path.dirname(__file__), "..", "..", "models",
                         "streaming_audio_samples")


def build_model():
    """Build and return stateful model + wrapper + PT model."""
    p = ModelParams()
    pt_model = init_model()
    # Load pretrained checkpoint
    ckpt = torch.load(PRETRAINED_CKPT, map_location="cpu")
    model_sd = ckpt.get("model", ckpt)
    missing, unexpected = pt_model.load_state_dict(model_sd, strict=False)
    print(f"  PT model loaded: {len(missing)} missing, {len(unexpected)} unexpected keys")
    pt_model.eval()
    sd = {k: v.cpu().numpy() for k, v in pt_model.state_dict().items()}
    # Use the erb_inv_fb directly from PT state dict to ensure exact match.
    # compute_erb_fb min_nb_freqs must match the Rust DF() constructor's
    # min_nb_erb_freqs (default=1, but config may override via MIN_NB_ERB_FREQS).
    erb_inv_fb = sd["mask.erb_inv_fb"]
    erb_widths = compute_erb_fb(sr=48000, fft_size=p.fft_size,
                                 nb_bands=p.nb_erb, min_nb_freqs=p.min_nb_freqs)
    F_bins = p.fft_size // 2 + 1

    m = DfNetStatefulStreamingTF(
        erb_widths_np=erb_widths, erb_inv_fb_np=erb_inv_fb,
        nb_erb=p.nb_erb, nb_df=p.nb_df,
        fft_size=p.fft_size, sr=48000, hop_size=480, norm_tau=1.0,
        conv_ch=p.conv_ch, df_order=p.df_order, df_lookahead=p.df_lookahead,
        emb_hidden_dim=p.emb_hidden_dim, emb_num_layers=p.emb_num_layers,
        df_hidden_dim=p.df_hidden_dim, df_num_layers=p.df_num_layers,
        gru_groups=p.gru_groups, lin_groups=p.lin_groups,
        group_shuffle=p.group_shuffle,
        enc_concat=p.enc_concat,
        df_pathway_kernel_size_t=p.df_pathway_kernel_size_t,
        gru_type=p.gru_type,
        df_gru_skip=p.df_gru_skip,
        conv_lookahead=p.conv_lookahead,
        pad_mode=p.pad_mode,
        batch_size=1,
    )
    dummy = tf.zeros([1, 1, 1, F_bins, 2])
    m.forward(dummy)
    m.load_from_pt(sd)
    m.reset_state()

    print("Building TFLite-ready wrapper (freezing NN weights)...")
    wrapper = build_tflite_module(m)

    print("Building NN-only wrapper for quantized export...")
    nn_wrapper = build_tflite_nn_module(m)

    return m, wrapper, nn_wrapper, pt_model, p


def generate_speech_rep_data(num_sentences=200, sr=48000, fft_size=960,
                            hop_size=480, seed=42):
    """Generate representative dataset from real speech + noise audio.

    Loads real speech from assets/clean.hdf5 and noise from assets/noise.hdf5,
    then creates `num_sentences` random noisy mixtures by:
      - Slicing a random segment (0.5–3.0 s) from the speech recording
      - Slicing a random noise segment and mixing at random SNR (-9 to 40 dB)
      - Scaling to a random max amplitude (0.01–0.95)

    Yields STFT spec frames [1,1,1,F,2] sequentially per sentence.
    The stateful model’s running norms converge within ~10 frames, so
    state naturally stays in a realistic range without explicit reset.

    Calibration pattern:
        for i in range(num_sentences):
            noisy_i = mix_speech_noise(...)  # random params
            frames_i = stft(noisy_i)
            for j in range(len(frames_i)):
                yield frames_i[j]
    """
    import h5py

    rng = np.random.RandomState(seed)
    window = np.sqrt(np.hanning(fft_size + 1)[:fft_size]).astype(np.float32)

    # ── Load real audio from HDF5 ──
    assets_dir = os.path.join(os.path.dirname(__file__), "..", "..", "assets")
    speech_path = os.path.join(assets_dir, "clean.hdf5")
    noise_path = os.path.join(assets_dir, "noise.hdf5")

    # Read all speech data (int16 -> float32)
    with h5py.File(speech_path, "r") as f:
        speech_all = []
        for k in f["speech"]:
            data = f["speech"][k][:].astype(np.float32) / 32768.0
            speech_all.append(data[0])
    speech_audio = np.concatenate(speech_all)

    # Read all noise data (int16 -> float32)
    with h5py.File(noise_path, "r") as f:
        noise_all = []
        for k in f["noise"]:
            data = f["noise"][k][:].astype(np.float32) / 32768.0
            noise_all.append(data[0])
    noise_audio = np.concatenate(noise_all)

    speech_len = len(speech_audio)
    noise_len = len(noise_audio)

    total_frames = 0
    for i in range(num_sentences):
        duration = rng.uniform(0.5, 3.0)
        n_samples = int(sr * duration)

        sp_start = rng.randint(0, speech_len)
        if sp_start + n_samples <= speech_len:
            speech = speech_audio[sp_start:sp_start + n_samples].copy()
        else:
            speech = np.concatenate([
                speech_audio[sp_start:],
                speech_audio[:n_samples - (speech_len - sp_start)]
            ])

        ns_start = rng.randint(0, noise_len)
        if ns_start + n_samples <= noise_len:
            noise = noise_audio[ns_start:ns_start + n_samples].copy()
        else:
            noise = np.concatenate([
                noise_audio[ns_start:],
                noise_audio[:n_samples - (noise_len - ns_start)]
            ])

        target_snr_db = rng.uniform(-9, 40)
        speech_pow = np.mean(speech ** 2) + 1e-12
        noise_pow = np.mean(noise ** 2) + 1e-12
        snr_linear = 10 ** (target_snr_db / 10)
        noise_scale = np.sqrt(speech_pow / (noise_pow * snr_linear))
        noisy = speech + noise_scale * noise

        target_max = rng.uniform(0.01, 0.95)
        current_max = np.max(np.abs(noisy)) + 1e-12
        noisy = (noisy * target_max / current_max).astype(np.float32)

        n_frames = (len(noisy) - fft_size) // hop_size + 1
        for j in range(n_frames):
            start = j * hop_size
            frame = noisy[start:start + fft_size] * window
            spec = np.fft.rfft(frame).astype(np.complex64)
            spec_ri = np.stack([spec.real, spec.imag], axis=-1)  # [F, 2]
            yield spec_ri[np.newaxis, np.newaxis, np.newaxis, :, :].astype(
                np.float32)  # [1,1,1,F,2]
            total_frames += 1

    print(f"  Rep data: {num_sentences} sentences, {total_frames} total frames")


def generate_nn_rep_data(model, num_sentences=200, sr=48000, fft_size=960,
                         hop_size=480, seed=42):
    """Generate representative dataset of pre-computed NN features.

    Uses real speech from assets/clean.hdf5 mixed with white noise at random
    SNR (-10 to 40 dB) and random max amplitude (0.01 to 0.95).

    Runs feature extraction (a^2+b^2, ERB, log10, sqrt, norms) in float32
    to produce the actual NN inputs: feat_erb [1,1,E,1] and feat_spec [1,1,F_df,2].

    These narrow-range normalized features (~+-3) give the quantizer accurate
    calibration ranges for the NN's internal activations.

    Yields:
        dict with "feat_erb" and "feat_spec" arrays
    """
    import h5py

    rng = np.random.RandomState(seed)
    window = np.sqrt(np.hanning(fft_size + 1)[:fft_size]).astype(np.float32)
    F_bins = fft_size // 2 + 1
    nb_erb = model.nb_erb
    nb_df = model.nb_df
    norm_alpha = compute_norm_alpha(sr, hop_size, 1.0)
    erb_band_matrix = model.erb_band_matrix.numpy()  # [F, E]

    # Load real speech from HDF5
    assets_dir = os.path.join(os.path.dirname(__file__), "..", "..", "assets")
    with h5py.File(os.path.join(assets_dir, "clean.hdf5"), "r") as f:
        speech_all = []
        for k in f["speech"]:
            data = f["speech"][k][:].astype(np.float32) / 32768.0
            speech_all.append(data[0])
    speech_audio = np.concatenate(speech_all)

    speech_len = len(speech_audio)
    total_frames = 0

    for i in range(num_sentences):
        duration = rng.uniform(0.5, 3.0)
        n_samples = int(sr * duration)

        sp_start = rng.randint(0, speech_len)
        if sp_start + n_samples <= speech_len:
            speech = speech_audio[sp_start:sp_start + n_samples].copy()
        else:
            speech = np.concatenate([
                speech_audio[sp_start:],
                speech_audio[:n_samples - (speech_len - sp_start)]
            ])

        # White noise
        noise = rng.randn(n_samples).astype(np.float32)

        target_snr_db = rng.uniform(-10, 40)
        speech_pow = np.mean(speech ** 2) + 1e-12
        noise_pow = np.mean(noise ** 2) + 1e-12
        snr_linear = 10 ** (target_snr_db / 10)
        noise_scale = np.sqrt(speech_pow / (noise_pow * snr_linear))
        noisy = speech + noise_scale * noise

        target_max = rng.uniform(0.01, 0.95)
        current_max = np.max(np.abs(noisy)) + 1e-12
        noisy = (noisy * target_max / current_max).astype(np.float32)

        # Initialize feature extraction state for this sentence
        erb_state = np.linspace(MEAN_NORM_INIT[0], MEAN_NORM_INIT[1],
                                nb_erb).astype(np.float32)[np.newaxis, :]  # [1, E]
        un_state = np.linspace(UNIT_NORM_INIT[0], UNIT_NORM_INIT[1],
                               nb_df).astype(np.float32)[np.newaxis, :]  # [1, F_df]

        n_frames = (len(noisy) - fft_size) // hop_size + 1
        for j in range(n_frames):
            start = j * hop_size
            frame = noisy[start:start + fft_size] * window
            spec = np.fft.rfft(frame).astype(np.complex64)
            spec_re = spec.real[np.newaxis, :]  # [1, F]
            spec_im = spec.imag[np.newaxis, :]

            # ERB features: power -> band avg -> dB -> erb_norm
            power = spec_re * spec_re + spec_im * spec_im
            erb_energy = power @ erb_band_matrix  # [1, E]
            erb_feat = 10.0 * np.log10(erb_energy + 1e-10)
            new_erb_state = erb_feat * (1.0 - norm_alpha) + erb_state * norm_alpha
            feat_erb = (erb_feat - new_erb_state) / 40.0
            erb_state = new_erb_state

            # Unit norm: magnitude -> running norm -> normalized spec
            sdf_re = spec_re[:, :nb_df]
            sdf_im = spec_im[:, :nb_df]
            x_abs = np.sqrt(sdf_re * sdf_re + sdf_im * sdf_im + 1e-14)
            new_un_state = x_abs * (1.0 - norm_alpha) + un_state * norm_alpha
            denom = np.sqrt(new_un_state)
            feat_spec_re = sdf_re / denom
            feat_spec_im = sdf_im / denom
            un_state = new_un_state

            # Shape for NN inputs
            feat_erb_nhwc = feat_erb.reshape(1, 1, nb_erb, 1).astype(np.float32)
            feat_spec_nhwc = np.stack(
                [feat_spec_re, feat_spec_im], axis=-1
            ).reshape(1, 1, nb_df, 2).astype(np.float32)

            yield {"feat_erb": feat_erb_nhwc, "feat_spec": feat_spec_nhwc}
            total_frames += 1

    print(f"  NN rep data: {num_sentences} sentences, {total_frames} total frames")


def export_nn_tflite(nn_wrapper, model, path, quantization, F_bins=481):
    """Export NN-only wrapper to TFLite with int8 or int16x8 quantization.

    The NN-only model takes pre-computed features (feat_erb, feat_spec) and
    returns (mask, df_coefs). All wide-range ops (a^2+b^2, sqrt, ERB, log10,
    norms, mask application, deep filtering) stay in float32 caller code.
    """
    saved_model_dir = tempfile.mkdtemp()
    tf.saved_model.save(
        nn_wrapper, saved_model_dir,
        signatures={
            "forward": nn_wrapper.forward,
            "reset_state": nn_wrapper.reset_state,
        },
    )

    converter = tf.lite.TFLiteConverter.from_saved_model(
        saved_model_dir, signature_keys=["forward"])

    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    if quantization == "int8":
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
            tf.lite.OpsSet.TFLITE_BUILTINS,
        ]
    else:  # int16x8
        converter.inference_input_type = tf.int16
        converter.inference_output_type = tf.int16
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8,
            tf.lite.OpsSet.TFLITE_BUILTINS,
        ]

    def rep_data():
        for feats in generate_nn_rep_data(
                model, num_sentences=200, sr=48000,
                fft_size=960, hop_size=480, seed=42):
            yield feats
    converter.representative_dataset = rep_data

    tflite_model = converter.convert()
    with open(path, "wb") as f:
        f.write(tflite_model)
    size_mb = len(tflite_model) / 1024 / 1024

    shutil.rmtree(saved_model_dir, ignore_errors=True)
    return size_mb


def run_nn_tflite_on_audio(tflite_path, model, noisy_wav_path,
                           fft_size=960, hop_size=480):
    """Run NN-only TFLite model with feature extraction + post-processing in Python.

    The TFLite model takes (feat_erb, feat_spec) and returns (mask, df_coefs).
    Feature extraction, ERB mask application, and deep filtering are done here
    in float32 NumPy.

    Handles conv_lookahead / pad_mode properly:
      - input_specf: mask applied to delayed spec (conv_lookahead frames behind),
                     DF uses current masked spec for low-freq, delayed for high-freq.
      - none:        mask/DF applied to current spec (no delay).
    """
    audio, sr = sf.read(noisy_wav_path)
    assert sr == 48000
    if audio.ndim > 1:
        audio = audio[:, 0]

    F = fft_size // 2 + 1
    nb_erb = model.nb_erb
    nb_df = model.nb_df
    df_order = model.df_order
    norm_alpha = model.norm_alpha
    erb_band_matrix = model.erb_band_matrix.numpy()  # [F, E]
    erb_inv_fb = model.mask.erb_inv_fb.numpy()       # [E, F]
    window = np.sqrt(np.hanning(fft_size + 1)[:fft_size]).astype(np.float32)

    # Lookahead settings
    conv_lookahead = getattr(model, 'conv_lookahead', 0)
    pad_mode = getattr(model, 'pad_mode', 'none')
    pad_specf = pad_mode.endswith("specf")
    stream_lookahead = conv_lookahead if pad_mode.startswith("input") else 0

    n_samples = len(audio)
    n_frames = (n_samples - fft_size) // hop_size + 1

    # Initialize feature extraction state
    erb_state = np.linspace(MEAN_NORM_INIT[0], MEAN_NORM_INIT[1],
                            nb_erb).astype(np.float32)[np.newaxis, :]
    un_state = np.linspace(UNIT_NORM_INIT[0], UNIT_NORM_INIT[1],
                           nb_df).astype(np.float32)[np.newaxis, :]
    # Spec ring buffer for deep filtering
    spec_buf = np.zeros([1, df_order - 1, nb_df, 2], dtype=np.float32)

    # Spec lookahead delay buffer for input_specf mode
    if stream_lookahead > 0:
        spec_lookahead_buf = np.zeros([stream_lookahead, F, 2], dtype=np.float32)
    else:
        spec_lookahead_buf = None

    interp = tf.lite.Interpreter(model_path=tflite_path)
    interp.allocate_tensors()
    inp_details = interp.get_input_details()
    out_details = interp.get_output_details()

    # Map input names to indices
    inp_map = {d["name"].split(":")[0]: d["index"] for d in inp_details}
    # Handle both "serving_default_feat_erb" and "feat_erb" name formats
    erb_idx = None
    spec_idx = None
    for d in inp_details:
        if "feat_erb" in d["name"]:
            erb_idx = d["index"]
            erb_detail = d
        if "feat_spec" in d["name"]:
            spec_idx = d["index"]
            spec_detail = d

    def maybe_quantize(array, detail):
        if detail["dtype"] not in (np.int16, np.int8):
            return array.astype(np.float32)
        scale, zero_point = detail["quantization"]
        if scale == 0:
            raise ValueError(f"Invalid quantization scale for input {detail['name']}")
        quantized = np.round(array / scale + zero_point)
        if detail["dtype"] == np.int16:
            quantized = np.clip(quantized, -32768, 32767)
            return quantized.astype(np.int16)
        quantized = np.clip(quantized, -128, 127)
        return quantized.astype(np.int8)

    def maybe_dequantize(array, detail):
        if detail["dtype"] not in (np.int16, np.int8):
            return array.astype(np.float32)
        scale, zero_point = detail["quantization"]
        return (array.astype(np.float32) - zero_point) * scale

    enhanced_frames = []
    total_steps = n_frames + stream_lookahead
    for t in range(total_steps):
        if t < n_frames:
            start = t * hop_size
            frame = audio[start:start + fft_size].astype(np.float32)
        else:
            frame = np.zeros([fft_size], dtype=np.float32)
        windowed = frame * window
        stft = np.fft.rfft(windowed).astype(np.complex64)
        spec_re = stft.real[np.newaxis, :]  # [1, F]
        spec_im = stft.imag[np.newaxis, :]
        spec_ri = np.stack([spec_re[0], spec_im[0]], axis=-1)  # [F, 2]

        # ── Feature extraction (float32) ──
        power = spec_re * spec_re + spec_im * spec_im
        erb_energy = power @ erb_band_matrix
        erb_feat = 10.0 * np.log10(erb_energy + 1e-10)
        new_erb_state = erb_feat * (1.0 - norm_alpha) + erb_state * norm_alpha
        feat_erb = (erb_feat - new_erb_state) / 40.0
        erb_state = new_erb_state

        sdf_re = spec_re[:, :nb_df]
        sdf_im = spec_im[:, :nb_df]
        x_abs = np.sqrt(sdf_re * sdf_re + sdf_im * sdf_im + 1e-14)
        new_un_state = x_abs * (1.0 - norm_alpha) + un_state * norm_alpha
        denom = np.sqrt(new_un_state)
        feat_spec_re = sdf_re / denom
        feat_spec_im = sdf_im / denom
        un_state = new_un_state

        feat_erb_nhwc = feat_erb.reshape(1, 1, nb_erb, 1).astype(np.float32)
        feat_spec_nhwc = np.stack(
            [feat_spec_re[0], feat_spec_im[0]], axis=-1
        ).reshape(1, 1, nb_df, 2).astype(np.float32)

        # ── NN inference (quantized TFLite) ──
        interp.set_tensor(erb_idx, maybe_quantize(feat_erb_nhwc, erb_detail))
        interp.set_tensor(spec_idx, maybe_quantize(feat_spec_nhwc, spec_detail))
        interp.invoke()

        mask = maybe_dequantize(
            interp.get_tensor(out_details[0]["index"]), out_details[0])
        df_coefs = maybe_dequantize(
            interp.get_tensor(out_details[1]["index"]), out_details[1])

        # Check output shapes and swap if needed
        if mask.shape[-1] != 1:  # df_coefs was output[0]
            mask, df_coefs = df_coefs, mask

        # ── Spec delay for lookahead (input_specf mode) ──
        if spec_lookahead_buf is not None:
            # Push current spec into buffer, pop oldest as delayed spec
            delayed_spec_ri = spec_lookahead_buf[0].copy()  # [F, 2]
            spec_lookahead_buf = np.concatenate(
                [spec_lookahead_buf[1:], spec_ri[np.newaxis, :, :]], axis=0)
            delayed_re = delayed_spec_ri[:, 0][np.newaxis, :]  # [1, F]
            delayed_im = delayed_spec_ri[:, 1][np.newaxis, :]
        else:
            delayed_re = spec_re
            delayed_im = spec_im

        # ── ERB mask application (float32) ──
        m = mask[0, 0, :, 0]  # [E]
        m_freq = m @ erb_inv_fb  # [F]

        # Mask applied to delayed spec (for high-freq output)
        masked_delayed_re = delayed_re[0] * m_freq  # [F]
        masked_delayed_im = delayed_im[0] * m_freq

        # For pad_specf: DF uses current (non-delayed) masked spec
        if pad_specf and spec_lookahead_buf is not None:
            masked_cur_re = spec_re[0] * m_freq
            masked_cur_im = spec_im[0] * m_freq
            df_input_re = masked_cur_re[:nb_df]
            df_input_im = masked_cur_im[:nb_df]
        else:
            df_input_re = masked_delayed_re[:nb_df]
            df_input_im = masked_delayed_im[:nb_df]

        # ── Deep filtering (float32) ──
        spec_df_cur = np.stack(
            [df_input_re, df_input_im], axis=-1
        )[np.newaxis, :]  # [1, F_df, 2]
        win = np.concatenate(
            [spec_buf[0], spec_df_cur[0:1]], axis=0
        )  # [O, F_df, 2]
        spec_buf[0] = win[1:]  # slide

        coefs = df_coefs[0, :, 0, :, :]  # [O, F_df, 2]
        s_re = win[..., 0]  # [O, F_df]
        s_im = win[..., 1]
        c_re = coefs[..., 0]
        c_im = coefs[..., 1]
        out_re = np.sum(s_re * c_re - s_im * c_im, axis=0)  # [F_df]
        out_im = np.sum(s_re * c_im + s_im * c_re, axis=0)

        # Build full enhanced spectrum: DF for low-freq, delayed masked for high-freq
        enh_re = np.concatenate([out_re, masked_delayed_re[nb_df:]])
        enh_im = np.concatenate([out_im, masked_delayed_im[nb_df:]])
        enh_spec = enh_re + 1j * enh_im
        out_frame = np.fft.irfft(enh_spec, n=fft_size).astype(np.float32)
        out_frame *= window
        enhanced_frames.append(out_frame)

    # Drop leading lookahead frames (they are from the delay buffer init)
    if stream_lookahead > 0:
        enhanced_frames = enhanced_frames[stream_lookahead:stream_lookahead + n_frames]
    else:
        enhanced_frames = enhanced_frames[:n_frames]

    # Overlap-add
    output_len = (n_frames - 1) * hop_size + fft_size
    output = np.zeros(output_len, dtype=np.float32)
    for t, frame in enumerate(enhanced_frames):
        start = t * hop_size
        output[start:start + fft_size] += frame

    return output[:n_samples]


def export_tflite(wrapper, path, quantization=None, F_bins=481):
    """Export the full wrapper to TFLite.

    Args:
        wrapper: The full stateful TFLite wrapper module.
        path: Output .tflite file path.
        quantization: One of None, "dynamic", "float16".
            - None:      Full float32, no compression.
            - "dynamic": Weights quantized to int8, activations stay float32 at
                         runtime. Good size/quality tradeoff.
            - "float16": Weights + activations in float16. Near-perfect quality
                         at ~50% size reduction.
        F_bins: Number of frequency bins (fft_size // 2 + 1).

    Notes:
        int8 and int16x8 are intentionally NOT supported here. They must be
        exported through export_nn_tflite() so all feature extraction stays
        outside the quantized neural network.
    """
    if quantization in ("int8", "int16x8"):
        raise ValueError(
            "int8/int16x8 export must use export_nn_tflite(); "
            "full-wrapper export is only for float32/dynamic/float16"
        )

    saved_model_dir = tempfile.mkdtemp()
    tf.saved_model.save(
        wrapper, saved_model_dir,
        signatures={
            "forward": wrapper.forward,
            "reset_state": wrapper.reset_state,
        },
    )

    converter = tf.lite.TFLiteConverter.from_saved_model(
        saved_model_dir, signature_keys=["forward"])
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]

    if quantization == "dynamic":
        # Weight-only quantization: weights int8, activations float32 at runtime.
        # No representative dataset needed.
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

    elif quantization == "float16":
        # Float16 quantization: weights stored as float16, dequantized at runtime.
        # No representative dataset needed.
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]

    tflite_model = converter.convert()
    with open(path, "wb") as f:
        f.write(tflite_model)
    size_mb = len(tflite_model) / 1024 / 1024

    shutil.rmtree(saved_model_dir, ignore_errors=True)
    return size_mb


def run_tflite_on_audio(tflite_path, noisy_wav_path, fft_size=960, hop_size=480,
                        lookahead=0):
    """Run TFLite stateful model on real audio.

    Returns: enhanced audio [samples] at 48kHz
    """
    audio, sr = sf.read(noisy_wav_path)
    assert sr == 48000, f"Expected 48kHz, got {sr}"
    if audio.ndim > 1:
        audio = audio[:, 0]  # mono

    F = fft_size // 2 + 1
    window = np.sqrt(np.hanning(fft_size + 1)[:fft_size]).astype(np.float32)

    # Frame the audio
    n_samples = len(audio)
    n_frames = (n_samples - fft_size) // hop_size + 1

    # STFT frame by frame with TFLite
    interp = tf.lite.Interpreter(model_path=tflite_path)
    interp.allocate_tensors()
    inp_details = interp.get_input_details()
    out_details = interp.get_output_details()

    # With float32 I/O, no manual quantize/dequantize needed.
    enhanced_frames = []
    total_steps = n_frames + lookahead
    for t in range(total_steps):
        if t < n_frames:
            start = t * hop_size
            frame = audio[start:start + fft_size].astype(np.float32)
        else:
            frame = np.zeros([fft_size], dtype=np.float32)
        windowed = frame * window
        spec = np.fft.rfft(windowed).astype(np.complex64)
        spec_ri = np.stack([spec.real, spec.imag], axis=-1)  # [F, 2]
        spec_ri = spec_ri[np.newaxis, np.newaxis, np.newaxis, :, :]  # [1,1,1,F,2]

        interp.set_tensor(inp_details[0]["index"], spec_ri)
        interp.invoke()
        out = interp.get_tensor(out_details[0]["index"])  # [1,1,1,F,2]

        out_spec = out[0, 0, 0, :, 0] + 1j * out[0, 0, 0, :, 1]
        out_frame = np.fft.irfft(out_spec, n=fft_size).astype(np.float32)
        out_frame *= window
        enhanced_frames.append(out_frame)

    if lookahead > 0:
        enhanced_frames = enhanced_frames[lookahead:lookahead + n_frames]
    else:
        enhanced_frames = enhanced_frames[:n_frames]

    # Overlap-add
    output_len = (n_frames - 1) * hop_size + fft_size
    output = np.zeros(output_len, dtype=np.float32)
    for t, frame in enumerate(enhanced_frames):
        start = t * hop_size
        output[start:start + fft_size] += frame

    return output[:n_samples]


def run_tf_on_audio(model, noisy_wav_path, fft_size=960, hop_size=480,
                    lookahead=None):
    """Run TF stateful model on real audio (for reference)."""
    audio, sr = sf.read(noisy_wav_path)
    assert sr == 48000
    if audio.ndim > 1:
        audio = audio[:, 0]

    F = fft_size // 2 + 1
    window = np.sqrt(np.hanning(fft_size + 1)[:fft_size]).astype(np.float32)

    n_samples = len(audio)
    n_frames = (n_samples - fft_size) // hop_size + 1

    model.reset_state()
    if lookahead is None:
        lookahead = getattr(model, "stream_lookahead", 0)

    enhanced_frames = []
    total_steps = n_frames + lookahead
    for t in range(total_steps):
        if t < n_frames:
            start = t * hop_size
            frame = audio[start:start + fft_size].astype(np.float32)
        else:
            frame = np.zeros([fft_size], dtype=np.float32)
        windowed = frame * window
        spec = np.fft.rfft(windowed).astype(np.complex64)
        spec_ri = np.stack([spec.real, spec.imag], axis=-1)
        spec_ri = spec_ri[np.newaxis, np.newaxis, np.newaxis, :, :]

        out = model.forward(tf.constant(spec_ri)).numpy()
        out_spec = out[0, 0, 0, :, 0] + 1j * out[0, 0, 0, :, 1]
        out_frame = np.fft.irfft(out_spec, n=fft_size).astype(np.float32)
        out_frame *= window
        enhanced_frames.append(out_frame)

    if lookahead > 0:
        enhanced_frames = enhanced_frames[lookahead:lookahead + n_frames]
    else:
        enhanced_frames = enhanced_frames[:n_frames]

    output_len = (n_frames - 1) * hop_size + fft_size
    output = np.zeros(output_len, dtype=np.float32)
    for t, frame in enumerate(enhanced_frames):
        start = t * hop_size
        output[start:start + fft_size] += frame

    return output[:n_samples]


def plot_spectrograms(audio_dict, sr=48000, title="Spectrogram Comparison",
                      save_path=None):
    """Plot spectrograms for multiple audio signals."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(audio_dict)
    fig, axes = plt.subplots(n, 1, figsize=(14, 3 * n))
    if n == 1:
        axes = [axes]

    for ax, (label, audio) in zip(axes, audio_dict.items()):
        S = np.abs(np.fft.rfft(
            np.lib.stride_tricks.sliding_window_view(audio, 960)[::480] *
            np.sqrt(np.hanning(961)[:960]),
            axis=-1,
        ))
        S_db = 20 * np.log10(S.T + 1e-10)
        ax.imshow(S_db, aspect="auto", origin="lower",
                  extent=[0, len(audio) / sr, 0, sr / 2 / 1000],
                  vmin=-80, vmax=0, cmap="magma")
        ax.set_ylabel("Freq (kHz)")
        ax.set_title(label)

    axes[-1].set_xlabel("Time (s)")
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved spectrogram: {save_path}")
    plt.close()


def snr_db(ref, est):
    """Signal-to-noise ratio in dB."""
    noise = ref - est
    ref_pow = np.mean(ref ** 2)
    noise_pow = np.mean(noise ** 2)
    if noise_pow < 1e-30:
        return 300.0
    return 10 * np.log10(ref_pow / noise_pow)


def main():
    print("=" * 60)
    print("Stateful DeepFilterNet2 TFLite Export + Audio Test")
    print("=" * 60)

    # Build model
    model, wrapper, nn_wrapper, pt_model, p = build_model()
    F_bins = p.fft_size // 2 + 1

    # ── Export TFLite models ──
    print("\n--- Exporting TFLite models ---")

    f32_path = os.path.join(BASE_DIR, "dfnet2_stateful_float32.tflite")
    size = export_tflite(wrapper, f32_path, quantization=None, F_bins=F_bins)
    print(f"  float32:  {size:.2f} MB -> {f32_path}")

    dyn_path = os.path.join(BASE_DIR, "dfnet2_stateful_dynamic.tflite")
    size = export_tflite(wrapper, dyn_path, quantization="dynamic", F_bins=F_bins)
    print(f"  dynamic:  {size:.2f} MB -> {dyn_path}")

    fp16_path = os.path.join(BASE_DIR, "dfnet2_stateful_float16.tflite")
    size = export_tflite(wrapper, fp16_path, quantization="float16", F_bins=F_bins)
    print(f"  float16:  {size:.2f} MB -> {fp16_path}")

    # int8/int16x8: use NN-only model (feature extraction + post-processing in caller)
    int8_path = os.path.join(BASE_DIR, "dfnet2_stateful_int8.tflite")
    size = export_nn_tflite(nn_wrapper, model, int8_path, "int8", F_bins=F_bins)
    print(f"  int8 (nn-only): {size:.2f} MB -> {int8_path}")

    int16_path = os.path.join(BASE_DIR, "dfnet2_stateful_int16x8.tflite")
    size = export_nn_tflite(nn_wrapper, model, int16_path, "int16x8", F_bins=F_bins)
    print(f"  int16x8 (nn-only): {size:.2f} MB -> {int16_path}")

    # ── Real audio test ──
    noisy_path = os.path.join(AUDIO_SRC, "noisy.wav")
    clean_path = os.path.join(AUDIO_SRC, "clean.wav")

    if not os.path.exists(noisy_path):
        print(f"\nNo audio files found at {noisy_path}, skipping audio test.")
        return

    print(f"\n--- Running on real audio ---")
    print(f"  Noisy: {noisy_path}")

    # Copy source audio
    shutil.copy2(noisy_path, os.path.join(AUDIO_DIR, "noisy.wav"))
    if os.path.exists(clean_path):
        shutil.copy2(clean_path, os.path.join(AUDIO_DIR, "clean.wav"))

    # TF reference
    print("  Running TF stateful model...")
    enhanced_tf = run_tf_on_audio(model, noisy_path, lookahead=p.conv_lookahead)
    sf.write(os.path.join(AUDIO_DIR, "enhanced_tf_stateful.wav"), enhanced_tf, 48000)

    # TFLite float32
    print("  Running TFLite float32...")
    enhanced_f32 = run_tflite_on_audio(f32_path, noisy_path, lookahead=p.conv_lookahead)
    sf.write(os.path.join(AUDIO_DIR, "enhanced_tflite_float32.wav"), enhanced_f32, 48000)

    # TFLite dynamic range
    print("  Running TFLite dynamic range...")
    enhanced_dyn = run_tflite_on_audio(dyn_path, noisy_path, lookahead=p.conv_lookahead)
    sf.write(os.path.join(AUDIO_DIR, "enhanced_tflite_dynamic.wav"), enhanced_dyn, 48000)

    # TFLite float16
    print("  Running TFLite float16...")
    enhanced_fp16 = run_tflite_on_audio(fp16_path, noisy_path, lookahead=p.conv_lookahead)
    sf.write(os.path.join(AUDIO_DIR, "enhanced_tflite_float16.wav"), enhanced_fp16, 48000)

    # TFLite int8 (nn-only)
    print("  Running TFLite int8 (nn-only)...")
    enhanced_i8 = run_nn_tflite_on_audio(int8_path, model, noisy_path)
    sf.write(os.path.join(AUDIO_DIR, "enhanced_tflite_int8.wav"), enhanced_i8, 48000)

    # TFLite int16x8 (nn-only)
    print("  Running TFLite int16x8 (nn-only)...")
    enhanced_i16 = run_nn_tflite_on_audio(int16_path, model, noisy_path)
    sf.write(os.path.join(AUDIO_DIR, "enhanced_tflite_int16x8.wav"), enhanced_i16, 48000)

    # ── SNR comparison ──
    min_len = min(len(enhanced_tf), len(enhanced_f32), len(enhanced_dyn),
                  len(enhanced_fp16), len(enhanced_i8), len(enhanced_i16))
    enhanced_tf = enhanced_tf[:min_len]
    enhanced_f32 = enhanced_f32[:min_len]
    enhanced_dyn = enhanced_dyn[:min_len]
    enhanced_fp16 = enhanced_fp16[:min_len]
    enhanced_i8 = enhanced_i8[:min_len]
    enhanced_i16 = enhanced_i16[:min_len]

    print(f"\n--- SNR vs TF stateful reference ---")
    print(f"  TFLite float32      : {snr_db(enhanced_tf, enhanced_f32):7.1f} dB")
    print(f"  TFLite dynamic range: {snr_db(enhanced_tf, enhanced_dyn):7.1f} dB")
    print(f"  TFLite float16      : {snr_db(enhanced_tf, enhanced_fp16):7.1f} dB")
    print(f"  TFLite int8         : {snr_db(enhanced_tf, enhanced_i8):7.1f} dB")
    print(f"  TFLite int16x8      : {snr_db(enhanced_tf, enhanced_i16):7.1f} dB")

    # ── Spectrograms ──
    print("\n--- Generating spectrograms ---")
    noisy_audio, _ = sf.read(noisy_path)
    if noisy_audio.ndim > 1:
        noisy_audio = noisy_audio[:, 0]
    noisy_audio = noisy_audio[:min_len]

    audio_dict = {
        "Noisy": noisy_audio,
        "Enhanced (TF ref)": enhanced_tf,
        "Enhanced (TFLite float16)": enhanced_fp16,
        "Enhanced (TFLite dynamic)": enhanced_dyn,
        "Enhanced (TFLite int8)": enhanced_i8,
        "Enhanced (TFLite int16x8)": enhanced_i16,
    }

    if os.path.exists(clean_path):
        clean_audio, _ = sf.read(clean_path)
        if clean_audio.ndim > 1:
            clean_audio = clean_audio[:, 0]
        audio_dict = {"Clean": clean_audio[:min_len], **audio_dict}

    plot_spectrograms(
        audio_dict,
        title="Stateful DeepFilterNet2 TFLite — Quantization Comparison",
        save_path=os.path.join(BASE_DIR, "spectrogram_comparison.png"),
    )

    print("\n" + "=" * 60)
    print("Export complete!")
    print(f"  Models:  {BASE_DIR}/dfnet2_stateful_*.tflite")
    print(f"  Audio:   {AUDIO_DIR}/")
    print(f"  Specs:   {BASE_DIR}/spectrogram_comparison.png")
    print("=" * 60)


if __name__ == "__main__":
    main()
