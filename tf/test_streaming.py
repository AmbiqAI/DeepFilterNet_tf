"""Tests for DfNetStreaming: shape validation + batch-vs-streaming equivalence."""

import unittest
import numpy as np
import tensorflow as tf

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from deepfilternet2_tf import DfNet, DEFAULT_PARAMS
from deepfilternet2_tf_streaming import (
    GroupedGRUStep,
    GroupedGRULayerStep,
    SqueezedGRUStep,
    EncoderStep,
    ErbDecoderStep,
    DfDecoderStep,
    DfNetStreaming,
    copy_weights_from_batch_model,
)


class TestGroupedGRUStep(unittest.TestCase):
    def test_single_step(self):
        gru = GroupedGRUStep(128, 256, num_layers=1, groups=1, add_outputs=True)
        x = tf.random.normal([1, 128])
        h = tf.zeros([1, 256])
        out, new_h = gru(x, h)
        self.assertEqual(out.shape, (1, 256))
        self.assertEqual(new_h.shape, (1, 256))

    def test_multi_layer(self):
        gru = GroupedGRUStep(256, 256, num_layers=3, groups=1, add_outputs=True)
        x = tf.random.normal([1, 256])
        h = tf.zeros([1, 3 * 256])
        out, new_h = gru(x, h)
        self.assertEqual(out.shape, (1, 256))
        self.assertEqual(new_h.shape, (1, 3 * 256))

    def test_state_size(self):
        gru = GroupedGRUStep(128, 256, num_layers=3, groups=1)
        self.assertEqual(gru.state_size, 3 * 256)


class TestEncoderStep(unittest.TestCase):
    def test_shape(self):
        p = DEFAULT_PARAMS
        enc = EncoderStep(p)
        feat_erb = tf.random.normal([1, 1, 1, p["nb_erb"]])
        feat_spec = tf.random.normal([1, 2, 1, p["nb_df"]])
        state = tf.zeros([1, 1 * p["emb_hidden_dim"]])
        kn_inp_t = p["conv_kernel_inp"][0]
        erb_conv0_buf = tf.zeros([1, kn_inp_t - 1, p["nb_erb"], 1])
        df_conv0_buf = tf.zeros([1, kn_inp_t - 1, p["nb_df"], 2])

        (e0, e1, e2, e3, emb, c0, lsnr, new_state,
         new_erb_buf, new_df_buf) = enc(
            feat_erb, feat_spec, state, erb_conv0_buf, df_conv0_buf)
        self.assertEqual(emb.shape, (1, p["emb_hidden_dim"]))
        self.assertEqual(lsnr.shape, (1, 1))
        self.assertEqual(new_state.shape, state.shape)
        self.assertEqual(new_erb_buf.shape, erb_conv0_buf.shape)
        self.assertEqual(new_df_buf.shape, df_conv0_buf.shape)


class TestErbDecoderStep(unittest.TestCase):
    def test_shape(self):
        p = DEFAULT_PARAMS
        ch, nb_erb = p["conv_ch"], p["nb_erb"]
        dec = ErbDecoderStep(p)
        emb = tf.random.normal([1, p["emb_hidden_dim"]])
        e3 = tf.random.normal([1, 1, nb_erb // 4, ch])
        e2 = tf.random.normal([1, 1, nb_erb // 4, ch])
        e1 = tf.random.normal([1, 1, nb_erb // 2, ch])
        e0 = tf.random.normal([1, 1, nb_erb, ch])
        state = tf.zeros([1, max(p["emb_num_layers"] - 1, 1) * p["emb_hidden_dim"]])

        m, new_state = dec(emb, e3, e2, e1, e0, state)
        self.assertEqual(m.shape, (1, 1, 1, nb_erb))
        self.assertEqual(new_state.shape, state.shape)


class TestDfDecoderStep(unittest.TestCase):
    def test_shape(self):
        p = DEFAULT_PARAMS
        ch, nb_df = p["conv_ch"], p["nb_df"]
        n_ch_out = p["df_order"] * 2
        dec = DfDecoderStep(p, out_channels=n_ch_out)
        emb = tf.random.normal([1, p["emb_hidden_dim"]])
        c0 = tf.random.normal([1, 1, nb_df, ch])
        state = tf.zeros([1, p["df_num_layers"] * p["df_hidden_dim"]])
        kt = p["df_pathway_kernel_size_t"]
        df_convp_buf = tf.zeros([1, kt - 1, nb_df, ch])

        coefs, alpha, new_state, new_buf = dec(emb, c0, state, df_convp_buf)
        self.assertEqual(coefs.shape[1], p["df_order"])
        self.assertEqual(coefs.shape[2], 1)
        self.assertEqual(coefs.shape[3], nb_df)
        self.assertEqual(coefs.shape[4], 2)
        self.assertEqual(new_state.shape, state.shape)
        self.assertEqual(new_buf.shape, df_convp_buf.shape)


class TestDfNetStreaming(unittest.TestCase):
    def test_single_frame(self):
        p = DEFAULT_PARAMS
        model = DfNetStreaming(p)

        feat_erb = tf.random.normal([1, 1, 1, p["nb_erb"]])
        feat_spec = tf.random.normal([1, 1, 1, p["nb_df"], 2])
        state = model.get_initial_state(batch_size=1)

        m, lsnr, df_coefs, alpha, new_state = model(feat_erb, feat_spec, state)
        self.assertEqual(m.shape, (1, 1, 1, p["nb_erb"]))
        self.assertEqual(df_coefs.shape[1], p["df_order"])
        self.assertEqual(new_state.shape, state.shape)

    def test_state_evolves(self):
        """Verify state changes between frames (not stuck at zero)."""
        p = DEFAULT_PARAMS
        model = DfNetStreaming(p)

        feat_erb = tf.random.normal([1, 1, 1, p["nb_erb"]])
        feat_spec = tf.random.normal([1, 1, 1, p["nb_df"], 2])
        state0 = model.get_initial_state(batch_size=1)

        _, _, _, _, state1 = model(feat_erb, feat_spec, state0)
        self.assertFalse(np.allclose(state0.numpy(), state1.numpy()),
                         "State should change after processing a frame")

    def test_multi_frame_loop(self):
        """Process multiple frames sequentially."""
        p = DEFAULT_PARAMS
        T = 5
        model = DfNetStreaming(p)
        state = model.get_initial_state(batch_size=1)

        np.random.seed(42)
        for t in range(T):
            feat_erb = tf.constant(np.random.randn(1, 1, 1, p["nb_erb"]).astype(np.float32))
            feat_spec = tf.constant(np.random.randn(1, 1, 1, p["nb_df"], 2).astype(np.float32))
            m, lsnr, df_coefs, alpha, state = model(feat_erb, feat_spec, state)

        self.assertEqual(m.shape, (1, 1, 1, p["nb_erb"]))
        self.assertEqual(state.shape[1], model.total_state_size)


class TestBatchStreamingEquivalence(unittest.TestCase):
    def test_weight_copy(self):
        """Verify copy_weights_from_batch_model runs without error."""
        p = DEFAULT_PARAMS
        F = p["fft_size"] // 2 + 1
        T = 3
        erb_inv = np.random.randn(p["nb_erb"], F).astype(np.float32)

        batch_model = DfNet(erb_inv, p, run_df=True)
        streaming_model = DfNetStreaming(p)

        # Build both models
        feat_erb = tf.random.normal([1, 1, T, p["nb_erb"]])
        feat_spec = tf.random.normal([1, 1, T, p["nb_df"], 2])
        _ = batch_model(feat_erb, feat_spec)

        feat_erb1 = tf.random.normal([1, 1, 1, p["nb_erb"]])
        feat_spec1 = tf.random.normal([1, 1, 1, p["nb_df"], 2])
        state = streaming_model.get_initial_state(1)
        _ = streaming_model(feat_erb1, feat_spec1, state)

        # This should not raise
        copy_weights_from_batch_model(batch_model, streaming_model)


if __name__ == "__main__":
    unittest.main()
