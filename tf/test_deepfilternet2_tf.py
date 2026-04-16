"""Unit tests for each DeepFilterNet2 TF module.

Validates shapes and basic forward pass. Run with:
    python -m pytest tf/test_deepfilternet2_tf.py -v
"""

import unittest
import numpy as np
import tensorflow as tf

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from deepfilternet2_tf import (
    Conv2dNormAct,
    ConvTranspose2dNormAct,
    GroupedLinear,
    GroupedLinearEinsum,
    SqueezedGRU,
    Mask,
    DfOutputReshapeMF,
    DfOp,
    Encoder,
    ErbDecoder,
    DfDecoder,
    DfNet,
    DEFAULT_PARAMS,
)


class TestConv2dNormAct(unittest.TestCase):
    def test_basic_shape(self):
        layer = Conv2dNormAct(1, 16, kernel_size=(3, 3), bias=False, separable=True)
        x = tf.random.normal([1, 10, 32, 1])
        y = layer(x, training=False)
        self.assertEqual(y.shape[0], 1)
        self.assertEqual(y.shape[1], 10)  # causal: time preserved
        self.assertEqual(y.shape[3], 16)

    def test_fstride(self):
        layer = Conv2dNormAct(16, 16, kernel_size=(1, 3), fstride=2, bias=False, separable=True)
        x = tf.random.normal([1, 10, 32, 16])
        y = layer(x, training=False)
        self.assertEqual(y.shape[1], 10)
        self.assertEqual(y.shape[2], 16)  # F/2

    def test_1x1_conv(self):
        layer = Conv2dNormAct(16, 16, kernel_size=1, bias=False, separable=True)
        x = tf.random.normal([1, 10, 32, 16])
        y = layer(x, training=False)
        self.assertEqual(y.shape, x.shape)


class TestConvTranspose2dNormAct(unittest.TestCase):
    def test_upsample(self):
        layer = ConvTranspose2dNormAct(16, 16, kernel_size=(1, 3), fstride=2,
                                        bias=False, separable=True)
        x = tf.random.normal([1, 10, 16, 16])
        y = layer(x, training=False)
        self.assertEqual(y.shape[1], 10)
        self.assertEqual(y.shape[2], 32)  # F * 2

    def test_no_upsample(self):
        layer = ConvTranspose2dNormAct(16, 16, kernel_size=(1, 3), fstride=1,
                                        bias=False, separable=True)
        x = tf.random.normal([1, 10, 16, 16])
        y = layer(x, training=False)
        self.assertEqual(y.shape[2], 16)


class TestGroupedLinear(unittest.TestCase):
    def test_grouped(self):
        layer = GroupedLinear(128, 64, groups=4, shuffle=True)
        x = tf.random.normal([2, 10, 128])
        y = layer(x)
        self.assertEqual(y.shape, (2, 10, 64))

    def test_single_group(self):
        layer = GroupedLinear(64, 32, groups=1)
        x = tf.random.normal([2, 10, 64])
        y = layer(x)
        self.assertEqual(y.shape, (2, 10, 32))


class TestGroupedLinearEinsum(unittest.TestCase):
    def test_basic(self):
        layer = GroupedLinearEinsum(128, 64, groups=4)
        x = tf.random.normal([2, 10, 128])
        y = layer(x)
        self.assertEqual(y.shape, (2, 10, 64))

    def test_single_group(self):
        layer = GroupedLinearEinsum(256, 256, groups=1)
        x = tf.random.normal([1, 5, 256])
        y = layer(x)
        self.assertEqual(y.shape, (1, 5, 256))


class TestBuiltinGRU(unittest.TestCase):
    def test_shape(self):
        gru = tf.keras.layers.GRU(128, return_sequences=True, unroll=True, reset_after=True)
        x = tf.random.normal([2, 10, 64])
        out = gru(x)
        self.assertEqual(out.shape, (2, 10, 128))

    def test_with_initial_state(self):
        gru = tf.keras.layers.GRU(128, return_sequences=True, unroll=True, reset_after=True)
        x = tf.random.normal([2, 10, 64])
        h0 = tf.zeros([2, 128])
        out = gru(x, initial_state=h0)
        self.assertEqual(out.shape, (2, 10, 128))

    def test_cell(self):
        cell = tf.keras.layers.GRUCell(128, reset_after=True)
        h_new, _ = cell(tf.random.normal([2, 64]), [tf.zeros([2, 128])])
        self.assertEqual(h_new.shape, (2, 128))


class TestSqueezedGRU(unittest.TestCase):
    def test_basic(self):
        gru = SqueezedGRU(256, 256, num_layers=1, linear_groups=8)
        x = tf.random.normal([1, 10, 256])
        out, h = gru(x)
        self.assertEqual(out.shape, (1, 10, 256))

    def test_with_output_size(self):
        gru = SqueezedGRU(256, 256, output_size=128, num_layers=1, linear_groups=8)
        x = tf.random.normal([1, 10, 256])
        out, h = gru(x)
        self.assertEqual(out.shape, (1, 10, 128))


class TestMask(unittest.TestCase):
    def test_shape(self):
        nb_erb, freq_bins = 32, 481
        erb_inv = np.random.randn(nb_erb, freq_bins).astype(np.float32)
        layer = Mask(erb_inv)
        spec = tf.random.normal([1, 1, 10, freq_bins, 2])
        m = tf.random.uniform([1, 1, 10, nb_erb])
        out = layer(spec, m)
        self.assertEqual(out.shape, spec.shape)


class TestDfOutputReshapeMF(unittest.TestCase):
    def test_shape(self):
        layer = DfOutputReshapeMF(df_order=5, df_bins=96)
        x = tf.random.normal([2, 10, 96, 10])  # [B, T, F, O*2]
        y = layer(x)
        self.assertEqual(y.shape[1], 5)   # O
        self.assertEqual(y.shape[2], 10)  # T
        self.assertEqual(y.shape[3], 96)  # F
        self.assertEqual(y.shape[4], 2)


class TestDfOp(unittest.TestCase):
    def test_shape(self):
        op = DfOp(num_freqs=96, frame_size=5, lookahead=0)
        spec = tf.random.normal([1, 1, 10, 481, 2])
        coefs = tf.random.normal([1, 5, 10, 96, 2])
        out = op(spec, coefs)
        self.assertEqual(out.shape, spec.shape)

    def test_high_freqs_unchanged(self):
        op = DfOp(num_freqs=96, frame_size=5, lookahead=0)
        spec = tf.random.normal([1, 1, 10, 481, 2])
        coefs = tf.random.normal([1, 5, 10, 96, 2])
        out = op(spec, coefs)
        np.testing.assert_array_equal(
            spec[:, :, :, 96:, :].numpy(),
            out[:, :, :, 96:, :].numpy()
        )


class TestEncoder(unittest.TestCase):
    def test_shape(self):
        p = DEFAULT_PARAMS
        enc = Encoder(p)
        T = 10
        feat_erb = tf.random.normal([1, 1, T, p["nb_erb"]])
        feat_spec = tf.random.normal([1, 2, T, p["nb_df"]])
        e0, e1, e2, e3, emb, c0, lsnr = enc(feat_erb, feat_spec)
        self.assertEqual(emb.shape, (1, T, p["emb_hidden_dim"]))
        self.assertEqual(lsnr.shape, (1, T, 1))
        self.assertEqual(e0.shape[1], T)
        self.assertEqual(c0.shape[1], T)


class TestErbDecoder(unittest.TestCase):
    def test_shape(self):
        p = DEFAULT_PARAMS
        ch, nb_erb, T = p["conv_ch"], p["nb_erb"], 10
        dec = ErbDecoder(p)
        emb = tf.random.normal([1, T, p["emb_hidden_dim"]])
        e3 = tf.random.normal([1, T, nb_erb // 4, ch])
        e2 = tf.random.normal([1, T, nb_erb // 4, ch])
        e1 = tf.random.normal([1, T, nb_erb // 2, ch])
        e0 = tf.random.normal([1, T, nb_erb, ch])
        m = dec(emb, e3, e2, e1, e0)
        self.assertEqual(m.shape[0], 1)
        self.assertEqual(m.shape[1], 1)  # C
        self.assertEqual(m.shape[2], T)


class TestDfDecoder(unittest.TestCase):
    def test_shape(self):
        p = DEFAULT_PARAMS
        ch, nb_df, T = p["conv_ch"], p["nb_df"], 10
        dec = DfDecoder(p, out_channels=p["df_order"] * 2)
        emb = tf.random.normal([1, T, p["emb_hidden_dim"]])
        c0 = tf.random.normal([1, T, nb_df, ch])
        coefs, alpha = dec(emb, c0)
        self.assertEqual(coefs.shape[1], p["df_order"])
        self.assertEqual(coefs.shape[2], T)
        self.assertEqual(coefs.shape[3], nb_df)
        self.assertEqual(coefs.shape[4], 2)
        self.assertEqual(alpha.shape, (1, T, 1))


class TestDfNet(unittest.TestCase):
    def test_full_forward(self):
        p = DEFAULT_PARAMS
        freq_bins = p["fft_size"] // 2 + 1
        T = 10
        erb_inv = np.random.randn(p["nb_erb"], freq_bins).astype(np.float32)
        model = DfNet(erb_inv, p, run_df=True)

        feat_erb = tf.random.normal([1, 1, T, p["nb_erb"]])
        feat_spec = tf.random.normal([1, 1, T, p["nb_df"], 2])

        m, lsnr, df_coefs, alpha = model(feat_erb, feat_spec)
        self.assertEqual(m.shape, (1, 1, T, p["nb_erb"]))
        self.assertEqual(lsnr.shape, (1, T, 1))
        self.assertEqual(df_coefs.shape, (1, p["df_order"], T, p["nb_df"], 2))

    def test_no_df(self):
        p = DEFAULT_PARAMS
        freq_bins = p["fft_size"] // 2 + 1
        T = 10
        erb_inv = np.random.randn(p["nb_erb"], freq_bins).astype(np.float32)
        model = DfNet(erb_inv, p, run_df=False)

        feat_erb = tf.random.normal([1, 1, T, p["nb_erb"]])
        feat_spec = tf.random.normal([1, 1, T, p["nb_df"], 2])

        m, lsnr, df_coefs, alpha = model(feat_erb, feat_spec)
        self.assertEqual(m.shape, (1, 1, T, p["nb_erb"]))


if __name__ == "__main__":
    unittest.main()
