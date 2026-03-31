"""
Tests for stateful streaming DeepFilterNet2.

Tests:
1. Feature extraction: TF erb/erb_norm/unit_norm vs libdf (Rust) reference
2. Stateful model vs old explicit-state streaming model
3. Reset state correctness
4. TFLite export (float32, dynamic, float16, int16x8) with two signatures
5. Real audio + spectrograms
"""

import os
import sys
import unittest
import math
import numpy as np
import warnings

os.environ["CUDA_VISIBLE_DEVICES"] = ""

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
import tensorflow as tf

# ── Load pretrained config ──
PRETRAINED_DIR = "/tmp/dfnet2_pretrained/DeepFilterNet2"
PRETRAINED_CONFIG = os.path.join(PRETRAINED_DIR, "config.ini")
PRETRAINED_CKPT = os.path.join(PRETRAINED_DIR, "checkpoints", "model_96.ckpt.best")

from df.config import config
if not os.path.exists(PRETRAINED_CONFIG):
    # Unzip if not already done
    import zipfile
    zip_path = os.path.join(os.path.dirname(__file__), "..", "..", "models", "DeepFilterNet2.zip")
    os.makedirs(PRETRAINED_DIR, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall("/tmp/dfnet2_pretrained")

config.load(PRETRAINED_CONFIG, allow_defaults=True, allow_reload=True)

from df.deepfilternet2 import ModelParams, init_model
from df.utils import get_norm_alpha, as_real
from libdf import DF, erb, erb_norm, unit_norm, unit_norm_init

from df.tf_stateful_streaming import (
    DfNetStreamingTF,
    DfNetStatefulStreamingTF,
    compute_erb_fb,
    compute_erb_inv_fb,
    compute_norm_alpha,
    tf_erb,
    tf_erb_norm,
    tf_unit_norm,
    build_tflite_module,
    build_tflite_nn_module,
    MEAN_NORM_INIT,
    UNIT_NORM_INIT,
)
from df.export_stateful_tflite import generate_speech_rep_data, generate_nn_rep_data


def snr_db(ref, est):
    """Signal-to-noise ratio in dB."""
    noise = ref - est
    ref_pow = np.mean(ref ** 2)
    noise_pow = np.mean(noise ** 2)
    if noise_pow < 1e-30:
        return 300.0
    return 10 * np.log10(ref_pow / noise_pow)


def get_pt_model_and_sd():
    """Create PT model from pretrained config, load checkpoint weights."""
    p = ModelParams()
    model = init_model()
    # Load pretrained checkpoint
    ckpt = torch.load(PRETRAINED_CKPT, map_location="cpu")
    model_sd = ckpt.get("model", ckpt)
    missing, unexpected = model.load_state_dict(model_sd, strict=False)
    print(f"  PT model loaded: {len(missing)} missing, {len(unexpected)} unexpected keys")
    model.eval()
    sd = {k: v.cpu().numpy() for k, v in model.state_dict().items()}
    sd["erb_inv_fb"] = sd["mask.erb_inv_fb"]
    return model, sd, p


class TestErbFbComputation(unittest.TestCase):
    """Test that Python erb_fb matches Rust/libdf."""

    def test_erb_fb_matches_libdf(self):
        """Our compute_erb_fb matches the DF state erb widths."""
        df_state = DF(sr=48000, fft_size=960, hop_size=480,
                      nb_bands=32, min_nb_erb_freqs=2)
        rust_widths = list(df_state.erb_widths())
        py_widths = compute_erb_fb(sr=48000, fft_size=960, nb_bands=32, min_nb_freqs=2)
        self.assertEqual(rust_widths, py_widths)
        print(f"  ERB widths match (min=2): {py_widths}")

    def test_erb_fb_default_matches_libdf_default(self):
        """compute_erb_fb default (min=1) matches DF() default."""
        df_state = DF(sr=48000, fft_size=960, hop_size=480, nb_bands=32)
        rust_widths = list(df_state.erb_widths())
        py_widths = compute_erb_fb(sr=48000, fft_size=960, nb_bands=32)
        self.assertEqual(rust_widths, py_widths)
        print(f"  ERB widths match (default/min=1): {py_widths}")

    def test_norm_alpha_matches(self):
        """Our compute_norm_alpha matches libdf get_norm_alpha."""
        rust_alpha = get_norm_alpha(False)
        py_alpha = compute_norm_alpha(sr=48000, hop_size=480, tau=1.0)
        self.assertAlmostEqual(rust_alpha, py_alpha, places=6)
        print(f"  Norm alpha: {py_alpha} (matches Rust: {rust_alpha})")


class TestFeatureExtraction(unittest.TestCase):
    """Test TF feature extraction functions vs libdf reference."""

    @classmethod
    def setUpClass(cls):
        cls.p = ModelParams()
        cls.alpha = get_norm_alpha(False)
        cls.df_state = DF(sr=48000, fft_size=960, hop_size=480,
                          nb_bands=32, min_nb_erb_freqs=2)
        cls.erb_widths = cls.df_state.erb_widths()  # numpy array for libdf
        cls.erb_widths_list = list(cls.erb_widths)  # list for TF

    def test_erb_single_frame(self):
        """TF erb matches libdf erb on single frames."""
        np.random.seed(42)
        # Create complex spec: [C=1, T=5, F=481]
        T = 5
        F = 481
        spec_re = np.random.randn(1, T, F).astype(np.float32) * 0.1
        spec_im = np.random.randn(1, T, F).astype(np.float32) * 0.1
        spec_complex = (spec_re + 1j * spec_im).astype(np.complex64)

        # Rust reference
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            rust_erb = erb(spec_complex, self.erb_widths, db=True)  # [C, T, E]

        # TF per-frame
        for t in range(T):
            tf_re = tf.constant(spec_re[:, t, :])  # [1, F]
            tf_im = tf.constant(spec_im[:, t, :])
            tf_result = tf_erb(tf_re, tf_im, self.erb_widths_list, db=True).numpy()  # [1, E]
            s = snr_db(rust_erb[0, t:t+1, :], tf_result)
            print(f"  ERB frame {t} SNR: {s:.1f} dB")
            np.testing.assert_allclose(rust_erb[0, t, :], tf_result[0], atol=1e-5)
            self.assertGreater(s, 120)

    def test_erb_norm_streaming(self):
        """TF erb_norm matches libdf erb_norm frame by frame."""
        np.random.seed(42)
        T = 10
        F = 481
        E = 32
        spec_re = np.random.randn(1, T, F).astype(np.float32) * 0.1
        spec_im = np.random.randn(1, T, F).astype(np.float32) * 0.1
        spec_complex = (spec_re + 1j * spec_im).astype(np.complex64)

        # Rust reference: erb then erb_norm (batch over all T)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            rust_erb_raw = erb(spec_complex, self.erb_widths, db=True)  # [C=1, T, E]
            rust_normed = erb_norm(rust_erb_raw.copy(), self.alpha)  # [C=1, T, E]
            # Also copy for per-frame TF comparison
            rust_erb_raw_copy = erb(spec_complex.copy(), self.erb_widths, db=True)

        # TF streaming: frame by frame with explicit state
        erb_state = np.linspace(MEAN_NORM_INIT[0], MEAN_NORM_INIT[1], E).astype(np.float32)
        erb_state = erb_state[np.newaxis, :]  # [1, E]

        for t in range(T):
            # TF erb
            tf_re = tf.constant(spec_re[:, t, :])
            tf_im = tf.constant(spec_im[:, t, :])
            erb_feat = tf_erb(tf_re, tf_im, self.erb_widths_list, db=True).numpy()

            # TF erb_norm
            normed, erb_state = tf_erb_norm(
                tf.constant(erb_feat),
                tf.constant(erb_state),
                self.alpha
            )
            normed = normed.numpy()
            erb_state = erb_state.numpy()

            s = snr_db(rust_normed[0, t:t+1, :], normed)
            print(f"  ERB norm frame {t} SNR: {s:.1f} dB")
            np.testing.assert_allclose(rust_normed[0, t, :], normed[0], atol=1e-5)
            self.assertGreater(s, 120)

    def test_unit_norm_streaming(self):
        """TF unit_norm matches libdf unit_norm frame by frame."""
        np.random.seed(42)
        T = 10
        F_df = 96
        spec_re = np.random.randn(1, T, F_df).astype(np.float32) * 0.1
        spec_im = np.random.randn(1, T, F_df).astype(np.float32) * 0.1
        spec_complex = (spec_re + 1j * spec_im).astype(np.complex64)

        # Rust reference: unit_norm (batch over all T)
        rust_normed = unit_norm(spec_complex.copy(), self.alpha)  # [C=1, T, F_df] complex

        # TF streaming: frame by frame
        un_state = np.linspace(UNIT_NORM_INIT[0], UNIT_NORM_INIT[1], F_df).astype(np.float32)
        un_state = un_state[np.newaxis, :]  # [1, F_df]

        for t in range(T):
            re_t = tf.constant(spec_re[:, t, :])
            im_t = tf.constant(spec_im[:, t, :])
            normed_re, normed_im, un_state_tf = tf_unit_norm(
                re_t, im_t, tf.constant(un_state), self.alpha)
            normed_re = normed_re.numpy()
            normed_im = normed_im.numpy()
            un_state = un_state_tf.numpy()

            ref_re = rust_normed[0, t, :].real
            ref_im = rust_normed[0, t, :].imag

            s_re = snr_db(ref_re[np.newaxis, :], normed_re)
            s_im = snr_db(ref_im[np.newaxis, :], normed_im)
            print(f"  Unit norm frame {t} SNR: re={s_re:.1f} dB, im={s_im:.1f} dB")
            np.testing.assert_allclose(ref_re, normed_re[0], atol=1e-4)
            np.testing.assert_allclose(ref_im, normed_im[0], atol=1e-4)
            self.assertGreater(s_re, 100)
            self.assertGreater(s_im, 100)


class TestStatefulModel(unittest.TestCase):
    """Test stateful streaming model vs explicit-state streaming model."""

    @classmethod
    def setUpClass(cls):
        """Build both models and load weights."""
        cls.pt_model, cls.sd, cls.p = get_pt_model_and_sd()
        p = cls.p

        # Use erb_inv_fb from PT model to ensure exact match.
        # init_model() creates DF() without min_nb_erb_freqs (Rust default=1).
        erb_inv_fb = cls.sd["mask.erb_inv_fb"]
        erb_widths = compute_erb_fb(sr=48000, fft_size=p.fft_size,
                                     nb_bands=p.nb_erb)

        F_bins = p.fft_size // 2 + 1

        # Common kwargs for both models
        common_kwargs = dict(
            nb_erb=p.nb_erb, nb_df=p.nb_df,
            conv_ch=p.conv_ch, df_order=p.df_order, df_lookahead=0,
            emb_hidden_dim=p.emb_hidden_dim, emb_num_layers=p.emb_num_layers,
            df_hidden_dim=p.df_hidden_dim, df_num_layers=p.df_num_layers,
            gru_groups=p.gru_groups, lin_groups=p.lin_groups,
            group_shuffle=p.group_shuffle,
            enc_concat=p.enc_concat,
            df_pathway_kernel_size_t=p.df_pathway_kernel_size_t,
            gru_type=p.gru_type,
        )

        # Build stateful model
        cls.stateful = DfNetStatefulStreamingTF(
            erb_widths_np=erb_widths,
            erb_inv_fb_np=erb_inv_fb,
            fft_size=p.fft_size, sr=48000, hop_size=480, norm_tau=1.0,
            df_gru_skip=p.df_gru_skip,
            batch_size=1,
            **common_kwargs,
        )
        # Build
        dummy = tf.zeros([1, 1, 1, F_bins, 2])
        cls.stateful.forward(dummy)
        cls.stateful.load_from_pt(cls.sd)
        cls.stateful.reset_state()  # Clean state after build

        # Build explicit-state model for reference
        cls.explicit = DfNetStreamingTF(
            erb_inv_fb_np=erb_inv_fb,
            fft_size=p.fft_size,
            df_gru_skip=p.df_gru_skip,
            **common_kwargs,
        )
        state = cls.explicit.get_initial_state(batch_size=1)
        _ = cls.explicit(
            tf.zeros([1, 1, 1, F_bins, 2]),
            tf.zeros([1, 1, 1, p.nb_erb]),
            tf.zeros([1, 1, 1, p.nb_df, 2]),
            state,
        )
        cls.explicit.load_from_pt(cls.sd)

    def test_stateful_vs_explicit_with_libdf_features(self):
        """Stateful model (internal features) vs explicit model (external libdf features).

        Both should produce identical NN outputs when given the same spec input,
        because the feature extraction is mathematically equivalent.
        """
        p = self.p
        T = 15
        F_bins = p.fft_size // 2 + 1
        np.random.seed(42)

        # Random spec data
        spec_np = np.random.randn(1, 1, T, F_bins, 2).astype(np.float32) * 0.1

        # ── Run libdf feature extraction for explicit model ──
        # Convert to complex for libdf
        spec_complex = (spec_np[0, 0, :, :, 0] + 1j * spec_np[0, 0, :, :, 1]).astype(np.complex64)
        spec_complex = spec_complex[np.newaxis, :, :]  # [C=1, T, F]

        alpha = get_norm_alpha(False)
        erb_widths = compute_erb_fb(sr=48000, fft_size=p.fft_size,
                                     nb_bands=p.nb_erb)
        erb_widths_np = np.array(erb_widths, dtype=np.uint64)  # libdf expects numpy

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            erb_raw = erb(spec_complex.copy(), erb_widths_np, db=True)  # [1, T, E]
            feat_erb_np = erb_norm(erb_raw.copy(), alpha)  # [1, T, E]
        feat_erb_np = feat_erb_np[np.newaxis, :, :, :]  # [B=1, C=1, T, E]

        feat_spec_complex = unit_norm(spec_complex[:, :, :p.nb_df].copy(), alpha)
        feat_spec_re = feat_spec_complex.real.astype(np.float32)
        feat_spec_im = feat_spec_complex.imag.astype(np.float32)
        feat_spec_np = np.stack([feat_spec_re, feat_spec_im], axis=-1)  # [1, T, F_df, 2]
        feat_spec_np = feat_spec_np[:, np.newaxis, :, :, :]  # [B=1, C=1, T, F_df, 2]

        # ── Run explicit-state model frame by frame ──
        state = {k: v.numpy() for k, v in
                 self.explicit.get_initial_state(batch_size=1).items()}
        explicit_out_list = []
        spec_running_explicit = spec_np.copy()

        for t in range(T):
            spec_frame = spec_running_explicit[:, :, t:t+1, :, :]
            erb_frame = feat_erb_np[:, :, t:t+1, :]
            feat_frame = feat_spec_np[:, :, t:t+1, :, :]

            tf_state = {k: tf.constant(v) for k, v in state.items()}
            spec_out, m, lsnr, alpha_out, new_state = self.explicit(
                tf.constant(spec_frame),
                tf.constant(erb_frame),
                tf.constant(feat_frame),
                tf_state,
                training=False,
            )
            state = {k: v.numpy() for k, v in new_state.items()}
            spec_running_explicit[:, :, t:t+1, :, :] = spec_out.numpy()
            explicit_out_list.append(spec_out.numpy())

        explicit_output = np.concatenate(explicit_out_list, axis=2)

        # ── Run stateful model frame by frame ──
        self.stateful.reset_state()
        stateful_out_list = []
        spec_running_stateful = spec_np.copy()

        for t in range(T):
            spec_frame = spec_running_stateful[:, :, t:t+1, :, :]
            spec_out = self.stateful.forward(tf.constant(spec_frame))
            spec_running_stateful[:, :, t:t+1, :, :] = spec_out.numpy()
            stateful_out_list.append(spec_out.numpy())

        stateful_output = np.concatenate(stateful_out_list, axis=2)

        # Compare
        s = snr_db(explicit_output, stateful_output)
        print(f"\n  Stateful vs explicit-state streaming SNR: {s:.1f} dB")
        np.testing.assert_allclose(explicit_output, stateful_output, atol=1e-4, rtol=1e-4)
        self.assertGreater(s, 100, f"SNR too low: {s:.1f} dB")

    def test_stateful_vs_pytorch(self):
        """Stateful streaming TF vs PyTorch batch.

        Note: The pretrained model has conv_lookahead=2, df_lookahead=2.
        In batch mode, PT uses padding that exposes future frames.
        In streaming causal mode (TF), we don't have future frames.
        So exact match is NOT expected — we verify the output is reasonable
        (not garbage) and the shapes are correct.
        """
        p = self.p
        T = 15
        F_bins = p.fft_size // 2 + 1
        np.random.seed(42)

        spec_np = np.random.randn(1, 1, T, F_bins, 2).astype(np.float32) * 0.1

        # Compute features using libdf for PT reference
        spec_complex = (spec_np[0, 0, :, :, 0] + 1j * spec_np[0, 0, :, :, 1]).astype(np.complex64)
        spec_complex = spec_complex[np.newaxis, :, :]  # [C=1, T, F]

        alpha_val = get_norm_alpha(False)
        erb_widths = compute_erb_fb(sr=48000, fft_size=p.fft_size,
                                     nb_bands=p.nb_erb)
        erb_widths_np = np.array(erb_widths, dtype=np.uint64)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            erb_raw = erb(spec_complex.copy(), erb_widths_np, db=True)
            feat_erb_np = erb_norm(erb_raw.copy(), alpha_val)
        feat_erb_np = feat_erb_np[np.newaxis, :, :, :]

        feat_spec_complex = unit_norm(spec_complex[:, :, :p.nb_df].copy(), alpha_val)
        feat_spec_re = feat_spec_complex.real.astype(np.float32)
        feat_spec_im = feat_spec_complex.imag.astype(np.float32)
        feat_spec_np = np.stack([feat_spec_re, feat_spec_im], axis=-1)
        feat_spec_np = feat_spec_np[:, np.newaxis, :, :, :]

        # PyTorch batch
        with torch.no_grad():
            pt_spec, pt_m, pt_lsnr, pt_alpha = self.pt_model(
                torch.from_numpy(spec_np.copy()),
                torch.from_numpy(feat_erb_np.astype(np.float32)),
                torch.from_numpy(feat_spec_np.astype(np.float32)),
            )
        pt_spec = pt_spec.numpy()

        # Stateful streaming TF
        self.stateful.reset_state()
        stateful_out_list = []
        spec_running = spec_np.copy()

        for t in range(T):
            spec_frame = spec_running[:, :, t:t+1, :, :]
            spec_out = self.stateful.forward(tf.constant(spec_frame))
            spec_running[:, :, t:t+1, :, :] = spec_out.numpy()
            stateful_out_list.append(spec_out.numpy())

        stateful_output = np.concatenate(stateful_out_list, axis=2)

        s = snr_db(pt_spec, stateful_output)
        print(f"\n  Stateful TF vs PyTorch SNR: {s:.1f} dB")
        print(f"  (Lower SNR expected: PT uses conv_lookahead={p.conv_lookahead}, "
              f"df_lookahead={p.df_lookahead}; TF streaming is causal)")

        # Shapes must match
        self.assertEqual(pt_spec.shape, stateful_output.shape)

        # Verify output is reasonable (not NaN, not all zeros)
        self.assertFalse(np.any(np.isnan(stateful_output)), "TF output has NaN")
        self.assertGreater(np.max(np.abs(stateful_output)), 1e-6,
                          "TF output is all zeros")

    def test_reset_state(self):
        """After reset_state, model produces same output as fresh model."""
        p = self.p
        F_bins = p.fft_size // 2 + 1

        # Start from clean state
        self.stateful.reset_state()

        # Run first frame from clean state
        np.random.seed(100)
        test_frame = np.random.randn(1, 1, 1, F_bins, 2).astype(np.float32) * 0.1
        out1 = self.stateful.forward(tf.constant(test_frame)).numpy()

        # Run a few frames to dirty state
        np.random.seed(42)
        for _ in range(5):
            spec = np.random.randn(1, 1, 1, F_bins, 2).astype(np.float32) * 0.1
            self.stateful.forward(tf.constant(spec))

        # Reset and run same first frame again
        self.stateful.reset_state()
        np.random.seed(100)
        test_frame = np.random.randn(1, 1, 1, F_bins, 2).astype(np.float32) * 0.1
        out2 = self.stateful.forward(tf.constant(test_frame)).numpy()

        s = snr_db(out2, out1)
        print(f"\n  Reset state SNR: {s:.1f} dB")
        np.testing.assert_allclose(out1, out2, atol=1e-6)
        self.assertGreater(s, 140, f"Reset state SNR too low: {s:.1f} dB")


class TestStatefulTFLite(unittest.TestCase):
    """Test stateful TFLite export with two signatures: forward and reset_state."""

    @classmethod
    def setUpClass(cls):
        """Build stateful model, freeze NN weights, create TFLite-ready wrapper."""
        _, cls.sd, cls.p = get_pt_model_and_sd()
        p = cls.p
        F_bins = p.fft_size // 2 + 1

        # Use erb_inv_fb from PT model to ensure exact match.
        erb_inv_fb = cls.sd["mask.erb_inv_fb"]
        erb_widths = compute_erb_fb(sr=48000, fft_size=p.fft_size,
                                     nb_bands=p.nb_erb)

        # Build stateful model and load weights
        cls.stateful = DfNetStatefulStreamingTF(
            erb_widths_np=erb_widths, erb_inv_fb_np=erb_inv_fb,
            nb_erb=p.nb_erb, nb_df=p.nb_df,
            fft_size=p.fft_size, sr=48000, hop_size=480, norm_tau=1.0,
            conv_ch=p.conv_ch, df_order=p.df_order, df_lookahead=0,
            emb_hidden_dim=p.emb_hidden_dim, emb_num_layers=p.emb_num_layers,
            df_hidden_dim=p.df_hidden_dim, df_num_layers=p.df_num_layers,
            gru_groups=p.gru_groups, lin_groups=p.lin_groups,
            group_shuffle=p.group_shuffle,
            enc_concat=p.enc_concat,
            df_pathway_kernel_size_t=p.df_pathway_kernel_size_t,
            gru_type=p.gru_type,
            df_gru_skip=p.df_gru_skip,
            batch_size=1,
        )
        dummy = tf.zeros([1, 1, 1, F_bins, 2])
        cls.stateful.forward(dummy)
        cls.stateful.load_from_pt(cls.sd)
        cls.stateful.reset_state()

        # Build TFLite-ready wrapper (frozen NN + state variables)
        cls.wrapper = build_tflite_module(cls.stateful)

        cls.F_bins = F_bins
        cls.erb_widths = erb_widths

    def _export_stateful_tflite(self, path, quantization=None):
        """Export stateful wrapper to TFLite via SavedModel.

        For float32/dynamic/float16: uses full-pipeline wrapper (spec → enhanced).
        For int8/int16x8: uses NN-only wrapper (feat_erb, feat_spec → mask, coefs)
        to avoid quantizing wide-range feature extraction ops.
        """
        import tempfile

        if quantization in ("int8", "int16x8"):
            return self._export_nn_only_tflite(path, quantization)

        # Full-pipeline export for float32/dynamic/float16
        saved_model_dir = tempfile.mkdtemp()
        tf.saved_model.save(
            self.wrapper,
            saved_model_dir,
            signatures={
                "forward": self.wrapper.forward,
                "reset_state": self.wrapper.reset_state,
            },
        )

        converter = tf.lite.TFLiteConverter.from_saved_model(
            saved_model_dir, signature_keys=["forward"])
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
        ]

        if quantization == "dynamic":
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
        elif quantization == "float16":
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]

        tflite_model = converter.convert()
        with open(path, "wb") as f:
            f.write(tflite_model)
        size_mb = len(tflite_model) / 1024 / 1024

        import shutil
        shutil.rmtree(saved_model_dir, ignore_errors=True)
        return size_mb

    def _export_nn_only_tflite(self, path, quantization):
        """Export NN-only wrapper for int8/int16x8 quantization.

        The NN-only model takes pre-computed features and returns mask + coefs.
        Feature extraction and post-processing stay in float32 caller code.
        """
        import tempfile

        nn_wrapper = build_tflite_nn_module(self.stateful)

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
        else:
            converter.inference_input_type = tf.int16
            converter.inference_output_type = tf.int16
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8,
                tf.lite.OpsSet.TFLITE_BUILTINS,
            ]

        def rep_data():
            for feats in generate_nn_rep_data(
                    self.stateful, num_sentences=200, sr=48000,
                    fft_size=960, hop_size=480, seed=42):
                yield feats
        converter.representative_dataset = rep_data

        tflite_model = converter.convert()
        with open(path, "wb") as f:
            f.write(tflite_model)
        size_mb = len(tflite_model) / 1024 / 1024

        import shutil
        shutil.rmtree(saved_model_dir, ignore_errors=True)
        return size_mb

    def _run_stateful_tflite(self, tflite_path, spec_np):
        """Run stateful TFLite model frame by frame.

        Args:
            tflite_path: path to TFLite file
            spec_np: [B=1, C=1, T, F, 2] noisy spec

        Returns:
            enhanced: [B=1, C=1, T, F, 2]
        """
        T = spec_np.shape[2]

        interp = tf.lite.Interpreter(model_path=tflite_path)
        interp.allocate_tensors()

        input_details = interp.get_input_details()
        output_details = interp.get_output_details()

        in_idx = input_details[0]["index"]
        out_idx = output_details[0]["index"]

        # With float32 I/O types, no manual quantize/dequantize needed.
        spec_running = spec_np.copy()
        out_list = []

        for t in range(T):
            frame = spec_running[:, :, t:t+1, :, :].astype(np.float32)
            interp.set_tensor(in_idx, frame)
            interp.invoke()
            spec_out = interp.get_tensor(out_idx)

            spec_running[:, :, t:t+1, :, :] = spec_out
            out_list.append(spec_out.copy())

        return np.concatenate(out_list, axis=2)

    def test_tflite_float32_export(self):
        """Export stateful model to TFLite float32."""
        path = "/tmp/dfnet2_stateful_f32.tflite"
        size = self._export_stateful_tflite(path, quantization=None)
        print(f"\n  Stateful TFLite float32 size: {size:.2f} MB")
        self.assertTrue(os.path.exists(path))
        self.assertGreater(size, 5.0, "TFLite too small — NN weights likely missing")

    def test_tflite_float32_accuracy(self):
        """Stateful TFLite float32 matches TF model."""
        p = self.p
        T = 10
        F_bins = self.F_bins
        np.random.seed(42)

        spec_np = np.random.randn(1, 1, T, F_bins, 2).astype(np.float32) * 0.1

        # TF reference
        self.stateful.reset_state()
        tf_out_list = []
        spec_running = spec_np.copy()
        for t in range(T):
            frame = spec_running[:, :, t:t+1, :, :]
            out = self.stateful.forward(tf.constant(frame))
            spec_running[:, :, t:t+1, :, :] = out.numpy()
            tf_out_list.append(out.numpy())
        ref = np.concatenate(tf_out_list, axis=2)

        # TFLite — always re-export to ensure fresh weights
        path = "/tmp/dfnet2_stateful_f32.tflite"
        self._export_stateful_tflite(path, quantization=None)
        est = self._run_stateful_tflite(path, spec_np)

        s = snr_db(ref, est)
        print(f"\n  Stateful TFLite float32 vs TF SNR: {s:.1f} dB")
        self.assertGreater(s, 80, f"TFLite float32 SNR too low: {s:.1f} dB")

    def test_tflite_dynamic_export(self):
        """Export stateful model to TFLite with dynamic range quantization."""
        path = "/tmp/dfnet2_stateful_dynamic.tflite"
        size = self._export_stateful_tflite(path, quantization="dynamic")
        print(f"\n  Stateful TFLite dynamic size: {size:.2f} MB")
        self.assertTrue(os.path.exists(path))
        # Dynamic range should be smaller than float32 (~2.5 vs ~9 MB)
        self.assertLess(size, 5.0, "Dynamic range model too large")

    def test_tflite_float16_export(self):
        """Export stateful model to TFLite with float16 quantization."""
        path = "/tmp/dfnet2_stateful_float16.tflite"
        size = self._export_stateful_tflite(path, quantization="float16")
        print(f"\n  Stateful TFLite float16 size: {size:.2f} MB")
        self.assertTrue(os.path.exists(path))
        # Float16 should be about half of float32 (~4.6 vs ~9 MB)
        self.assertLess(size, 6.0, "Float16 model too large")

    def test_tflite_int8_export(self):
        """Export NN-only model to TFLite with int8 quantization.

        Uses build_tflite_nn_module: frozen graph contains only convs/GRUs.
        Feature extraction (a^2+b^2, sqrt, ERB, log10, norms) and
        post-processing (mask, deep filtering) stay in float32 caller.
        """
        path = "/tmp/dfnet2_stateful_int8.tflite"
        size = self._export_stateful_tflite(path, quantization="int8")
        print(f"\n  Stateful TFLite int8 size: {size:.2f} MB")
        self.assertTrue(os.path.exists(path))
        interp = tf.lite.Interpreter(model_path=path)
        interp.allocate_tensors()
        for detail in interp.get_input_details():
            self.assertEqual(detail["dtype"], np.int8)
        for detail in interp.get_output_details():
            self.assertEqual(detail["dtype"], np.int8)

    def test_tflite_int16x8_export(self):
        """Export stateful model to TFLite int16x8."""
        path = "/tmp/dfnet2_stateful_int16x8.tflite"
        size = self._export_stateful_tflite(path, quantization="int16x8")
        print(f"\n  Stateful TFLite int16x8 size: {size:.2f} MB")
        self.assertTrue(os.path.exists(path))
        interp = tf.lite.Interpreter(model_path=path)
        interp.allocate_tensors()
        for detail in interp.get_input_details():
            self.assertEqual(detail["dtype"], np.int16)
        for detail in interp.get_output_details():
            self.assertEqual(detail["dtype"], np.int16)


if __name__ == "__main__":
    unittest.main(verbosity=2)
