"""DeepFilterNet2 TensorFlow/Keras reimplementation.

Converts the PyTorch DeepFilterNet2 architecture to TensorFlow.
Feature extraction (STFT/ISTFT) is NOT included — this covers only
the neural network from spectral features to enhanced spectrum.

Conventions:
  - All conv layers use NHWC (channels_last) format.
  - Complex values are represented as 2-channel real tensors (..., 2)
    where [... , 0] = real, [..., 1] = imaginary.
  - RNNs are unrolled into per-timestep dense ops for TFLite compat.
  - ConvTranspose2d is replaced with upsample + Conv2D.
"""

import math
import tensorflow as tf
import numpy as np

# ==============================================================================
# Default hyperparameters (matching PyTorch defaults)
# ==============================================================================

DEFAULT_PARAMS = {
    "sr": 48000,
    "fft_size": 960,
    "hop_size": 480,
    "nb_erb": 32,
    "nb_df": 96,
    "df_order": 5,
    "df_lookahead": 2,
    "conv_ch": 64,
    "conv_kernel": (1, 3),
    "conv_kernel_inp": (3, 3),
    "emb_hidden_dim": 256,
    "emb_num_layers": 3,
    "df_hidden_dim": 256,
    "df_num_layers": 2,
    "gru_groups": 8,
    "lin_groups": 8,
    "group_shuffle": False,
    "enc_concat": True,
    "df_pathway_kernel_size_t": 5,
    "lsnr_max": 35,
    "lsnr_min": -15,
    "conv_lookahead": 2,
    "df_gru_skip": "none",
}


def _gcd(a, b):
    return math.gcd(a, b)


def _get_activation(name):
    if name is None or name == "none":
        return None
    if name == "relu":
        return tf.keras.layers.ReLU()
    if name == "sigmoid":
        return tf.keras.layers.Activation("sigmoid")
    if name == "tanh":
        return tf.keras.layers.Activation("tanh")
    if name == "elu":
        return tf.keras.layers.ELU()
    return tf.keras.layers.Activation(name)


# ==============================================================================
# Utility layers
# ==============================================================================

class CausalPad2D(tf.keras.layers.Layer):
    """Asymmetric padding on time axis (causal) and optional freq padding."""

    def __init__(self, time_pad_before, time_pad_after, **kwargs):
        super().__init__(**kwargs)
        self.padding = [
            [0, 0],
            [time_pad_before, time_pad_after],
            [0, 0],
            [0, 0],
        ]

    def call(self, x):
        return tf.pad(x, self.padding)


class Conv2dNormAct(tf.keras.layers.Layer):
    """Causal Conv2D block matching PyTorch Conv2dNormAct.

    Input/Output: [B, T, F, C] (NHWC).
    Separable: grouped conv (split channels) + 1x1 pointwise.
    """

    def __init__(self, in_ch, out_ch, kernel_size, fstride=1, bias=False,
                 separable=False, activation="relu", use_norm=True, **kwargs):
        super().__init__(**kwargs)
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.kernel_size = kernel_size

        # Causal time padding
        time_pad = kernel_size[0] - 1
        self.use_pad = time_pad > 0
        if self.use_pad:
            self.pad = CausalPad2D(time_pad, 0)

        fpad = kernel_size[1] // 2
        self.fpad = fpad

        groups = _gcd(in_ch, out_ch) if separable else 1
        if groups == 1:
            separable = False
        if max(kernel_size) == 1:
            separable = False

        self.groups = groups
        self.separable = separable

        if groups > 1:
            assert in_ch % groups == 0 and out_ch % groups == 0
            self.group_convs = [
                tf.keras.layers.Conv2D(
                    out_ch // groups, kernel_size,
                    strides=(1, fstride), padding="valid", use_bias=bias,
                )
                for _ in range(groups)
            ]
        else:
            self.conv = tf.keras.layers.Conv2D(
                out_ch, kernel_size,
                strides=(1, fstride), padding="valid", use_bias=bias,
            )

        if separable:
            self.pw_conv = tf.keras.layers.Conv2D(out_ch, 1, use_bias=False)
        else:
            self.pw_conv = None

        self.use_norm = use_norm
        if use_norm:
            self.norm = tf.keras.layers.BatchNormalization(axis=-1, epsilon=1e-5, momentum=0.9)

        self.act = _get_activation(activation)

    def call(self, x, training=False):
        if self.use_pad:
            x = self.pad(x)
        if self.fpad > 0:
            x = tf.pad(x, [[0, 0], [0, 0], [self.fpad, self.fpad], [0, 0]])

        if self.groups > 1:
            splits = tf.split(x, self.groups, axis=-1)
            x = tf.concat([c(s) for c, s in zip(self.group_convs, splits)], axis=-1)
        else:
            x = self.conv(x)

        if self.pw_conv is not None:
            x = self.pw_conv(x)
        if self.use_norm:
            x = self.norm(x, training=training)
        if self.act is not None:
            x = self.act(x)
        return x


class ConvTranspose2dNormAct(tf.keras.layers.Layer):
    """Transposed conv replacement: zero-insert upsample on freq + Conv2D.

    Implements ConvTranspose2d semantics using only Conv2D:
      1. Zero-insert upsample along freq (stride > 1)
      2. Pad freq with (kernel-1-orig_padding) on each side
      3. Conv2D with flipped kernel (kernel flipped at weight load time)
      4. Output padding to match ConvTranspose2d output size

    Only upsamples on freq axis. Time stride is always 1.
    Input/Output: [B, T, F, C] (NHWC).
    """

    def __init__(self, in_ch, out_ch, kernel_size, fstride=1, bias=False,
                 separable=False, activation="relu", use_norm=True, **kwargs):
        super().__init__(**kwargs)
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.kernel_size = kernel_size
        self.fstride = fstride

        time_pad = kernel_size[0] - 1
        self.use_pad = time_pad > 0
        if self.use_pad:
            self.pad = CausalPad2D(time_pad, 0)

        fpad = kernel_size[1] // 2
        self.fpad = fpad

        groups = _gcd(in_ch, out_ch) if separable else 1
        if groups == 1:
            separable = False
        self.groups = groups
        self.separable = separable

        if groups > 1:
            assert in_ch % groups == 0 and out_ch % groups == 0
            self.group_convs = [
                tf.keras.layers.Conv2D(
                    out_ch // groups, kernel_size,
                    strides=(1, 1), padding="valid", use_bias=bias,
                )
                for _ in range(groups)
            ]
        else:
            self.conv = tf.keras.layers.Conv2D(
                out_ch, kernel_size,
                strides=(1, 1), padding="valid", use_bias=bias,
            )

        if separable:
            self.pw_conv = tf.keras.layers.Conv2D(out_ch, 1, use_bias=False)
        else:
            self.pw_conv = None

        self.use_norm = use_norm
        if use_norm:
            self.norm = tf.keras.layers.BatchNormalization(axis=-1, epsilon=1e-5, momentum=0.9)

        self.act = _get_activation(activation)

    def call(self, x, training=False):
        if self.fstride > 1:
            # Zero-insert upsample along freq axis (axis 2)
            # [B, T, F, C] -> [B, T, (F-1)*stride + 1, C]
            # Use tf.pad (static padding) instead of tf.zeros (dynamic shape)
            # to avoid int16x8 quantizer crash.
            f_static = x.shape[2]  # static freq dim (known at graph build)
            x = tf.expand_dims(x, 3)  # [B, T, F, 1, C]
            x = tf.pad(x, [[0, 0], [0, 0], [0, 0], [0, self.fstride - 1], [0, 0]])  # [B, T, F, stride, C]
            new_f = f_static * self.fstride
            x = tf.reshape(x, [-1, tf.shape(x)[1], new_f, x.shape[4]])
            # Trim trailing zeros: keep (F-1)*stride + 1 elements
            trim_f = (f_static - 1) * self.fstride + 1
            x = x[:, :, :trim_f, :]

        if self.use_pad:
            x = self.pad(x)
        if self.fpad > 0:
            # For ConvTranspose2d: pad left=fpad, right=fpad+output_padding
            # output_padding = fstride - 1 (to match PyTorch ConvTranspose2d output size)
            right_pad = self.fpad + (self.fstride - 1 if self.fstride > 1 else 0)
            x = tf.pad(x, [[0, 0], [0, 0], [self.fpad, right_pad], [0, 0]])

        if self.groups > 1:
            splits = tf.split(x, self.groups, axis=-1)
            x = tf.concat([c(s) for c, s in zip(self.group_convs, splits)], axis=-1)
        else:
            x = self.conv(x)

        if self.pw_conv is not None:
            x = self.pw_conv(x)
        if self.use_norm:
            x = self.norm(x, training=training)
        if self.act is not None:
            x = self.act(x)
        return x


# ==============================================================================
# Grouped linear layers
# ==============================================================================

class GroupedLinear(tf.keras.layers.Layer):
    """Grouped linear: split input, apply per-group Dense, concat + optional shuffle."""

    def __init__(self, input_size, hidden_size, groups=1, shuffle=True, **kwargs):
        super().__init__(**kwargs)
        assert input_size % groups == 0
        assert hidden_size % groups == 0
        self.groups = groups
        self.input_size_per_group = input_size // groups
        self.hidden_size_per_group = hidden_size // groups
        self.hidden_size = hidden_size
        if groups == 1:
            shuffle = False
        self.shuffle = shuffle
        self.linears = [
            tf.keras.layers.Dense(self.hidden_size_per_group)
            for _ in range(groups)
        ]

    def call(self, x):
        outputs = []
        for i, layer in enumerate(self.linears):
            start = i * self.input_size_per_group
            end = start + self.input_size_per_group
            outputs.append(layer(x[..., start:end]))
        output = tf.concat(outputs, axis=-1)
        if self.shuffle:
            orig_shape = tf.shape(output)
            output = tf.reshape(output,
                tf.concat([orig_shape[:-1], [self.hidden_size_per_group, self.groups]], 0))
            # Swap last two dims
            ndim = len(output.shape)
            perm = list(range(ndim - 2)) + [ndim - 1, ndim - 2]
            output = tf.transpose(output, perm)
            output = tf.reshape(output, orig_shape)
        return output


class GroupedLinearEinsum(tf.keras.layers.Layer):
    """Grouped linear matching PyTorch GroupedLinearEinsum, without tf.einsum."""

    def __init__(self, input_size, hidden_size, groups=1, **kwargs):
        super().__init__(**kwargs)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.groups = groups
        assert input_size % groups == 0
        self.ws = input_size // groups

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(self.groups, self.ws, self.hidden_size // self.groups),
            name="weight",
            initializer="glorot_uniform",
        )

    def call(self, x):
        shape = tf.shape(x)
        # Reshape to (..., G, I/G)
        x = tf.reshape(x, tf.concat([shape[:-1], [self.groups, self.ws]], 0))
        # Batched matmul: (..., G, I/G) @ (G, I/G, H/G) -> (..., G, H/G)
        # Expand x to (..., G, 1, I/G) for broadcasting matmul
        x = tf.expand_dims(x, -2)          # (..., G, 1, I/G)
        x = tf.matmul(x, self.w)           # (..., G, 1, H/G)
        x = tf.squeeze(x, axis=-2)         # (..., G, H/G)
        x = tf.reshape(x, tf.concat([shape[:-1], [self.hidden_size]], 0))
        return x


# ==============================================================================
# GRU — uses built-in Keras GRU with unroll=True for TFLite compatibility
# ==============================================================================


class SqueezedGRU(tf.keras.layers.Layer):
    """SqueezedGRU: linear_in -> GRU(s) -> optional skip -> optional linear_out."""

    def __init__(self, input_size, hidden_size, output_size=None,
                 num_layers=1, linear_groups=8, gru_skip_op=None,
                 linear_act="relu", **kwargs):
        super().__init__(**kwargs)
        act = tf.keras.layers.ReLU() if linear_act == "relu" else tf.keras.layers.Lambda(lambda x: x)
        self.linear_in_proj = GroupedLinearEinsum(input_size, hidden_size, linear_groups)
        self.linear_in_act = act

        self.grus = [
            tf.keras.layers.GRU(
                hidden_size, return_sequences=True, unroll=True,
                reset_after=True, name=f"gru_{i}")
            for i in range(num_layers)
        ]
        self.gru_skip = gru_skip_op
        if output_size is not None:
            self.linear_out_proj = GroupedLinearEinsum(hidden_size, output_size, linear_groups)
            self.linear_out_act = tf.keras.layers.ReLU() if linear_act == "relu" else tf.keras.layers.Lambda(lambda x: x)
        else:
            self.linear_out_proj = None

    def call(self, x, h=None):
        x_in = self.linear_in_act(self.linear_in_proj(x))
        out = x_in
        for gru in self.grus:
            out = gru(out)
        if self.gru_skip is not None:
            out = out + self.gru_skip(x_in)
        if self.linear_out_proj is not None:
            out = self.linear_out_act(self.linear_out_proj(out))
        return out, None


# ==============================================================================
# Mask application
# ==============================================================================

class Mask(tf.keras.layers.Layer):
    """Apply ERB mask to spectrum using inverse ERB filterbank."""

    def __init__(self, erb_inv_fb_np, **kwargs):
        super().__init__(**kwargs)
        self._erb_inv_fb_np = erb_inv_fb_np  # shape (E, F)

    def build(self, input_shape):
        self.erb_inv_fb = self.add_weight(
            shape=self._erb_inv_fb_np.shape, name="erb_inv_fb",
            initializer=tf.keras.initializers.Constant(self._erb_inv_fb_np),
            trainable=False,
        )

    def call(self, spec, mask):
        # spec: [B, 1, T, F, 2], mask: [B, 1, T, E], erb_inv_fb: [E, F]
        mask_f = tf.matmul(mask, self.erb_inv_fb)  # [B, 1, T, F]
        return spec * tf.expand_dims(mask_f, -1)


# ==============================================================================
# Deep Filtering operator (2-channel real complex)
# ==============================================================================

class DfOutputReshapeMF(tf.keras.layers.Layer):
    """Reshape DF output: [B, T, F, O*2] -> [B, O, T, F, 2]."""

    def __init__(self, df_order, df_bins, **kwargs):
        super().__init__(**kwargs)
        self.df_order = df_order
        self.df_bins = df_bins

    def call(self, coefs):
        shape = tf.shape(coefs)
        b, t, f = shape[0], shape[1], shape[2]
        coefs = tf.reshape(coefs, [b, t, f, self.df_order, 2])
        coefs = tf.transpose(coefs, [0, 3, 1, 2, 4])
        return coefs


class DfOp(tf.keras.layers.Layer):
    """Deep filtering: unfold spectrogram + complex multiply with coefficients.

    All in 2-channel real representation (no tf.complex).
    """

    def __init__(self, num_freqs, frame_size, lookahead=0, **kwargs):
        super().__init__(**kwargs)
        self.num_freqs = num_freqs
        self.frame_size = frame_size
        self.lookahead = lookahead

    def call(self, spec, coefs):
        """
        spec:  [B, 1, T, F, 2]
        coefs: [B, O, T, Fdf, 2]
        """
        spec_df = spec[:, :, :, :self.num_freqs, :]  # [B, 1, T, Fdf, 2]

        # Pad time for unfolding: (frame_size - 1) frames before
        padded = tf.pad(spec_df,
                        [[0, 0], [0, 0], [self.frame_size - 1, 0], [0, 0], [0, 0]])

        # Unfold along time axis
        T = spec.shape[2]
        indices = tf.range(T)
        window_idx = tf.expand_dims(indices, 1) + tf.range(self.frame_size)  # [T, O]
        # Gather: padded is [B, 1, T+pad, Fdf, 2]
        unfolded = tf.gather(padded, window_idx, axis=2)  # [B, 1, T, O, Fdf, 2]
        unfolded = tf.transpose(unfolded, [0, 1, 2, 4, 3, 5])  # [B, 1, T, Fdf, O, 2]

        # Rearrange coefs: [B, O, T, F, 2] -> [B, 1, T, F, O, 2]
        c = tf.expand_dims(coefs, 1)  # [B, 1, O, T, F, 2]
        c = tf.transpose(c, [0, 1, 3, 4, 2, 5])  # [B, 1, T, F, O, 2]

        # Complex multiply + sum over O
        s_re, s_im = unfolded[..., 0], unfolded[..., 1]  # [B, 1, T, F, O]
        c_re, c_im = c[..., 0], c[..., 1]

        out_re = tf.reduce_sum(s_re * c_re - s_im * c_im, axis=-1)  # [B, 1, T, F]
        out_im = tf.reduce_sum(s_re * c_im + s_im * c_re, axis=-1)
        spec_f = tf.stack([out_re, out_im], axis=-1)  # [B, 1, T, Fdf, 2]

        # Assign filtered DF bins back, keep high freqs unchanged
        spec_hi = spec[:, :, :, self.num_freqs:, :]
        return tf.concat([spec_f, spec_hi], axis=3)


# ==============================================================================
# Encoder
# ==============================================================================

class Encoder(tf.keras.layers.Layer):
    """DeepFilterNet2 encoder: ERB path + DF path -> embedding + lSNR."""

    def __init__(self, p=None, **kwargs):
        super().__init__(**kwargs)
        if p is None:
            p = DEFAULT_PARAMS
        ch = p["conv_ch"]
        nb_erb = p["nb_erb"]
        nb_df = p["nb_df"]
        kn = tuple(p["conv_kernel"])
        kn_inp = tuple(p["conv_kernel_inp"])

        self.erb_conv0 = Conv2dNormAct(1, ch, kn_inp, bias=False, separable=True)
        self.erb_conv1 = Conv2dNormAct(ch, ch, kn, fstride=2, bias=False, separable=True)
        self.erb_conv2 = Conv2dNormAct(ch, ch, kn, fstride=2, bias=False, separable=True)
        self.erb_conv3 = Conv2dNormAct(ch, ch, kn, fstride=1, bias=False, separable=True)

        self.df_conv0 = Conv2dNormAct(2, ch, kn_inp, bias=False, separable=True)
        self.df_conv1 = Conv2dNormAct(ch, ch, kn, fstride=2, bias=False, separable=True)

        self.emb_in_dim = ch * nb_erb // 4
        self.emb_out_dim = p["emb_hidden_dim"]

        # gru_type=squeeze: use GroupedLinearEinsum + ReLU for df_fc_emb
        self.df_fc_emb = GroupedLinearEinsum(
            ch * nb_df // 2, self.emb_in_dim, groups=p["lin_groups"]
        )
        self.df_fc_emb_act = tf.keras.layers.ReLU()

        self.enc_concat = p["enc_concat"]
        emb_in = self.emb_in_dim * 2 if self.enc_concat else self.emb_in_dim

        # gru_type=squeeze: use SqueezedGRU
        self.emb_gru = SqueezedGRU(
            emb_in, self.emb_out_dim, output_size=None,
            num_layers=1, linear_groups=p["lin_groups"],
        )

        self.lsnr_dense = tf.keras.layers.Dense(1)
        self.lsnr_scale = float(p["lsnr_max"] - p["lsnr_min"])
        self.lsnr_offset = float(p["lsnr_min"])

    def call(self, feat_erb, feat_spec, training=False):
        """
        feat_erb:  [B, 1, T, E] (NCHW)
        feat_spec: [B, 2, T, Fc] (NCHW)
        Returns: e0..e3 [B,T,F_i,ch] (NHWC), emb [B,T,H], c0 [B,T,Fc,ch], lsnr [B,T,1]
        """
        erb = tf.transpose(feat_erb, [0, 2, 3, 1])    # [B, T, E, 1]
        spec = tf.transpose(feat_spec, [0, 2, 3, 1])  # [B, T, Fc, 2]

        e0 = self.erb_conv0(erb, training=training)
        e1 = self.erb_conv1(e0, training=training)
        e2 = self.erb_conv2(e1, training=training)
        e3 = self.erb_conv3(e2, training=training)

        c0 = self.df_conv0(spec, training=training)
        c1 = self.df_conv1(c0, training=training)

        # Flatten spatial dims for FC: [B, T, F', ch] -> [B, T, F'*ch]
        c1_s = tf.shape(c1)
        cemb = tf.reshape(c1, [c1_s[0], c1_s[1], -1])
        cemb = self.df_fc_emb_act(self.df_fc_emb(cemb))

        e3_s = tf.shape(e3)
        emb = tf.reshape(e3, [e3_s[0], e3_s[1], -1])

        if self.enc_concat:
            emb = tf.concat([emb, cemb], axis=-1)
        else:
            emb = emb + cemb

        emb, _ = self.emb_gru(emb)

        lsnr = tf.sigmoid(self.lsnr_dense(emb)) * self.lsnr_scale + self.lsnr_offset
        return e0, e1, e2, e3, emb, c0, lsnr


# ==============================================================================
# ERB Mask Decoder
# ==============================================================================

class ErbDecoder(tf.keras.layers.Layer):
    """Decode ERB mask from embedding + encoder skip connections."""

    def __init__(self, p=None, **kwargs):
        super().__init__(**kwargs)
        if p is None:
            p = DEFAULT_PARAMS
        ch = p["conv_ch"]
        nb_erb = p["nb_erb"]
        kn = tuple(p["conv_kernel"])

        self.emb_out_dim = p["emb_hidden_dim"]
        self.ch = ch
        self.nb_erb = nb_erb

        # gru_type=squeeze: SqueezedGRU with output_size and gru_skip=Identity
        self.emb_gru = SqueezedGRU(
            self.emb_out_dim, self.emb_out_dim,
            output_size=ch * nb_erb // 4,
            num_layers=p["emb_num_layers"] - 1,
            linear_groups=p["lin_groups"],
            gru_skip_op=tf.keras.layers.Lambda(lambda x: x),  # Identity
        )

        # Decoder conv layers
        self.conv3p = Conv2dNormAct(ch, ch, kernel_size=1, bias=False, separable=True)
        self.convt3 = Conv2dNormAct(ch, ch, kernel_size=kn, bias=False, separable=True)
        self.conv2p = Conv2dNormAct(ch, ch, kernel_size=1, bias=False, separable=True)
        self.convt2 = ConvTranspose2dNormAct(ch, ch, kernel_size=kn, fstride=2, bias=False, separable=True)
        self.conv1p = Conv2dNormAct(ch, ch, kernel_size=1, bias=False, separable=True)
        self.convt1 = ConvTranspose2dNormAct(ch, ch, kernel_size=kn, fstride=2, bias=False, separable=True)
        self.conv0p = Conv2dNormAct(ch, ch, kernel_size=1, bias=False, separable=True)
        self.conv0_out = Conv2dNormAct(ch, 1, kernel_size=kn, bias=False,
                                        separable=True, activation="sigmoid")

    def call(self, emb, e3, e2, e1, e0, training=False):
        """
        emb: [B, T, H], e0..e3: [B, T, F_i, ch] (NHWC)
        Returns: [B, 1, T, E] (NCHW)
        """
        f8 = e3.shape[2]  # nb_erb // 4

        # SqueezedGRU already includes linear_out projection
        emb, _ = self.emb_gru(emb)

        emb_s = tf.shape(emb)
        emb = tf.reshape(emb, [emb_s[0], emb_s[1], f8, self.ch])

        e3 = self.convt3(self.conv3p(e3, training=training) + emb, training=training)
        e2 = self.convt2(self.conv2p(e2, training=training) + e3, training=training)
        e1 = self.convt1(self.conv1p(e1, training=training) + e2, training=training)
        m = self.conv0_out(self.conv0p(e0, training=training) + e1, training=training)

        # NHWC -> NCHW: [B, T, E, 1] -> [B, 1, T, E]
        m = tf.transpose(m, [0, 3, 1, 2])
        return m


# ==============================================================================
# DF Coefficient Decoder
# ==============================================================================

class DfDecoder(tf.keras.layers.Layer):
    """Decode deep filtering coefficients from embedding."""

    def __init__(self, p=None, out_channels=-1, **kwargs):
        super().__init__(**kwargs)
        if p is None:
            p = DEFAULT_PARAMS
        ch = p["conv_ch"]
        self.emb_dim = p["emb_hidden_dim"]
        self.df_n_hidden = p["df_hidden_dim"]
        self.df_order = p["df_order"]
        self.df_bins = p["nb_df"]
        self.df_out_ch = out_channels if out_channels > 0 else p["df_order"] * 2

        kt = p["df_pathway_kernel_size_t"]
        self.df_convp = Conv2dNormAct(
            ch, self.df_out_ch, kernel_size=(kt, 1), fstride=1, bias=False, separable=True)

        # gru_type=squeeze: SqueezedGRU with gru_skip=Identity
        self.df_gru = SqueezedGRU(
            p["emb_hidden_dim"], p["df_hidden_dim"],
            num_layers=p["df_num_layers"],
            linear_groups=p["lin_groups"],
            gru_skip_op=tf.keras.layers.Lambda(lambda x: x),  # Identity
        )

        skip = p["df_gru_skip"]
        if skip == "groupedlinear":
            self.df_skip = GroupedLinearEinsum(
                p["emb_hidden_dim"], p["df_hidden_dim"], groups=p["lin_groups"])
        elif skip == "identity":
            self.df_skip = lambda x: x
        else:
            self.df_skip = None

        out_dim = self.df_bins * self.df_out_ch
        self.df_out_proj = GroupedLinearEinsum(
            self.df_n_hidden, out_dim, groups=p["lin_groups"])

        self.df_fc_a = tf.keras.layers.Dense(1)
        self.out_transform = DfOutputReshapeMF(self.df_order, self.df_bins)

    def call(self, emb, c0, training=False):
        """
        emb: [B, T, H], c0: [B, T, Fc, ch] (NHWC)
        Returns: coefs [B, O, T, F, 2], alpha [B, T, 1]
        """
        c, _ = self.df_gru(emb)
        if self.df_skip is not None:
            c = c + self.df_skip(emb)

        c0 = self.df_convp(c0, training=training)  # [B, T, Fc, df_out_ch]

        alpha = tf.sigmoid(self.df_fc_a(c))  # [B, T, 1]
        c = tf.tanh(self.df_out_proj(c))  # [B, T, F*O*2]

        b, t = tf.shape(c)[0], tf.shape(c)[1]
        c = tf.reshape(c, [b, t, self.df_bins, self.df_out_ch])
        c = c + c0  # residual from convolution pathway

        c = self.out_transform(c)  # [B, O, T, F, 2]
        return c, alpha


# ==============================================================================
# Full Model
# ==============================================================================

class DfNet(tf.keras.Model):
    """DeepFilterNet2 — neural network only (no spectrum processing).

    Inputs:
        feat_erb:  [B, 1, T, E]     ERB features (normalized)
        feat_spec: [B, 1, T, Fc, 2] complex spec features (2-ch real, normalized)

    Outputs:
        m:         [B, 1, T, E]     ERB gain mask (0–1)
        lsnr:      [B, T, 1]        local SNR estimate (dB)
        df_coefs:  [B, O, T, Fc, 2] deep filtering coefficients (if run_df)
        alpha:     [B, T, 1]        DF blending factor (0–1)
    """

    def __init__(self, erb_inv_fb_np, p=None, run_df=True, **kwargs):
        super().__init__(**kwargs)
        if p is None:
            p = DEFAULT_PARAMS
        self.run_df = run_df
        self.nb_df = p["nb_df"]

        self.enc = Encoder(p)
        self.erb_dec = ErbDecoder(p)
        # erb_inv_fb stored for weight transfer compatibility
        self._erb_inv_fb_np = erb_inv_fb_np

        if run_df:
            n_ch_out = p["df_order"] * 2
            self.df_dec = DfDecoder(p, out_channels=n_ch_out)

    def call(self, feat_erb, feat_spec, training=False):
        # feat_spec: [B, 1, T, Fc, 2] -> [B, 2, T, Fc]
        fs = tf.squeeze(feat_spec, axis=1)          # [B, T, Fc, 2]
        fs = tf.transpose(fs, [0, 3, 1, 2])         # [B, 2, T, Fc]

        e0, e1, e2, e3, emb, c0, lsnr = self.enc(feat_erb, fs, training=training)
        m = self.erb_dec(emb, e3, e2, e1, e0, training=training)

        if self.run_df:
            df_coefs, df_alpha = self.df_dec(emb, c0, training=training)
        else:
            df_coefs = tf.constant(0.0)
            df_alpha = tf.constant(0.0)

        return m, lsnr, df_coefs, df_alpha
