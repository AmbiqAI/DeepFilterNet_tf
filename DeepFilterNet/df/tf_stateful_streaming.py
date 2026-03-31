"""Stateful TFLite-ready streaming DeepFilterNet2.

Design goals (from user requirements + .github/instructions):
- Single input:  spec [B, 1, 1, F, 2]  (noisy spectrum, one frame)
- Single output: enhanced_spec [B, 1, 1, F, 2]
- All streaming state stored as tf.Variable, updated via assign internally
- Two signatures: forward(spec) and reset_state()

Architecture split -- ALL wide-range ops OUTSIDE frozen NN graph:
  The TFLite wrapper runs everything involving raw spectrum values in
  float32, *outside* the frozen/quantized NN graph:

  Wrapper (float32, never quantized):
    1. Feature extraction:
       spec -> re^2+im^2 -> ERB matmul -> log10 -> erb_norm -> feat_erb
       spec -> sqrt(re^2+im^2) -> unit_norm -> feat_spec
    2. Call frozen NN: feat_erb, feat_spec -> mask, df_coefs
    3. ERB mask application: spec * expand(mask) -> spec_masked
    4. Deep filtering: complex_MAC(spec_masked, df_coefs) -> spec_out

  Frozen NN graph (quantizable):
    (feat_erb, feat_spec, NN_states) -> mask, df_coefs, new_NN_states
    Contains ONLY: Encoder convs/GRUs, ERB decoder, DF decoder.
    No raw spectrum values, no a^2+b^2, no sqrt, no mask multiply, no DF.

TFLite export strategy (two-phase freeze-then-wrap):
  1. forward_stateless() -- pure NN forward pass (no assigns, no feature
     extraction), captures Keras layer weights via closures.
  2. Freeze NN weights to graph constants with convert_variables_to_constants_v2.
  3. Wrap frozen graph in tf.Module: does feature extraction + state assigns.
  4. Save wrapper as SavedModel; MLIR converter preserves mutable state as
     resource variables and frozen NN weights as constants.

Feature extraction pipeline (matches Rust libDF):
  1. erb():      spec [C,T,F] complex -> ERB energies [C,T,E]
                 compute_band_corr(|x|^2 avg per band), then 10*log10(v+1e-10)
  2. erb_norm(): running mean normalization
                 state = x*(1-alpha) + state*alpha;  x = (x - state) / 40
  3. unit_norm():running unit normalization for complex spec
                 state = |x|*(1-alpha) + state*alpha;  x = x / sqrt(state)

Constants:
  MEAN_NORM_INIT = [-60, -90]  -> linspace over nb_erb bands
  UNIT_NORM_INIT = [0.001, 0.0001] -> linspace over nb_df bins
  alpha = exp(-hop_size/sr / tau), tau=1.0, sr=48000, hop=480 -> ~0.99005
"""

import math
import numpy as np
import tensorflow as tf

from df.tf_modules import (
    Conv2dNormActTF,
    ConvTranspose2dNormActTF,
    GroupedGRUTF,
    GroupedLinearTF,
    GroupedLinearEinsumTF,
    MaskTF,
    DfOutputReshapeMFTF,
    SqueezedGRUTF,
)


# ============================================================================
# Constants matching Rust libDF
# ============================================================================

MEAN_NORM_INIT = [-60.0, -90.0]
UNIT_NORM_INIT = [0.001, 0.0001]


def compute_norm_alpha(sr=48000, hop_size=480, tau=1.0):
    """Compute exponential decay alpha matching Rust/Python get_norm_alpha."""
    dt = hop_size / sr
    a_ = math.exp(-dt / tau)
    # Round up until < 1.0 (matches Python impl)
    precision = 3
    a = 1.0
    while a >= 1.0:
        a = round(a_, precision)
        precision += 1
    return a


def compute_erb_fb(sr=48000, fft_size=960, nb_bands=32, min_nb_freqs=1):
    """Compute ERB band widths, matching Rust erb_fb().

    Note: Rust DF() constructor defaults to min_nb_erb_freqs=1.
    Use min_nb_freqs=1 (default) to match the Rust default, or pass
    p.min_nb_freqs from config if the model was trained with a different value.

    Returns:
        list of int: number of frequency bins per ERB band.
        Sum == fft_size // 2 + 1.
    """
    def freq2erb(f):
        return 9.265 * math.log1p(f / (24.7 * 9.265))

    def erb2freq(e):
        return 24.7 * 9.265 * (math.exp(e / 9.265) - 1.0)

    nyq_freq = sr / 2
    freq_width = sr / fft_size
    erb_low = freq2erb(0.0)
    erb_high = freq2erb(nyq_freq)
    step = (erb_high - erb_low) / nb_bands

    erb = [0] * nb_bands
    prev_freq = 0
    freq_over = 0
    for i in range(1, nb_bands + 1):
        f = erb2freq(erb_low + i * step)
        fb = round(f / freq_width)
        nb_freqs = fb - prev_freq - freq_over
        if nb_freqs < min_nb_freqs:
            freq_over = min_nb_freqs - nb_freqs
            nb_freqs = min_nb_freqs
        else:
            freq_over = 0
        erb[i - 1] = nb_freqs
        prev_freq = fb

    erb[-1] += 1  # fft_size//2+1 instead of fft_size//2
    too_large = sum(erb) - (fft_size // 2 + 1)
    if too_large > 0:
        erb[-1] -= too_large
    assert sum(erb) == fft_size // 2 + 1, f"ERB sum {sum(erb)} != {fft_size//2+1}"
    return erb


class EncoderStreamingTF(tf.keras.layers.Layer):
    """Streaming encoder: processes T=1 frames with explicit causal state."""

    def __init__(self, nb_erb=32, nb_df=96, conv_ch=16, emb_hidden_dim=256,
                 lin_groups=1, gru_groups=1, group_shuffle=True,
                 lsnr_max=35, lsnr_min=-15, enc_concat=False,
                 gru_type="grouped", **kwargs):
        super().__init__(**kwargs)
        self.nb_erb = nb_erb
        self.nb_df = nb_df
        self.conv_ch = conv_ch
        self.enc_concat = enc_concat
        self.gru_type = gru_type

        self.erb_conv0 = Conv2dNormActTF(
            1, conv_ch, (3, 3), bias=False, separable=True,
            causal_pad_t=0, name="erb_conv0")
        self.erb_conv1 = Conv2dNormActTF(
            conv_ch, conv_ch, (1, 3), fstride=2, bias=False, separable=True,
            name="erb_conv1")
        self.erb_conv2 = Conv2dNormActTF(
            conv_ch, conv_ch, (1, 3), fstride=2, bias=False, separable=True,
            name="erb_conv2")
        self.erb_conv3 = Conv2dNormActTF(
            conv_ch, conv_ch, (1, 3), fstride=1, bias=False, separable=True,
            name="erb_conv3")

        self.df_conv0 = Conv2dNormActTF(
            2, conv_ch, (3, 3), bias=False, separable=True,
            causal_pad_t=0, name="df_conv0")
        self.df_conv1 = Conv2dNormActTF(
            conv_ch, conv_ch, (1, 3), fstride=2, bias=False, separable=True,
            name="df_conv1")

        emb_in_dim = conv_ch * nb_erb // 4
        if gru_type == "grouped":
            self.df_fc_emb = GroupedLinearTF(
                conv_ch * nb_df // 2, emb_in_dim,
                groups=lin_groups, shuffle=group_shuffle, name="df_fc_emb")
        else:
            self.df_fc_emb = GroupedLinearEinsumTF(
                conv_ch * nb_df // 2, emb_in_dim,
                groups=lin_groups, name="df_fc_emb")
            self.df_fc_emb_relu = tf.keras.layers.ReLU()

        actual_emb_in = emb_in_dim * 2 if enc_concat else emb_in_dim
        if gru_type == "grouped":
            self.emb_gru = GroupedGRUTF(
                actual_emb_in, emb_hidden_dim, num_layers=1,
                groups=gru_groups, shuffle=group_shuffle,
                add_outputs=True, name="emb_gru")
        else:
            self.emb_gru = SqueezedGRUTF(
                actual_emb_in, emb_hidden_dim,
                num_layers=1, linear_groups=lin_groups,
                has_gru_skip=False, has_linear_act=True, name="emb_gru")

        self.lsnr_dense = tf.keras.layers.Dense(1, name="lsnr_dense")
        self.lsnr_scale = lsnr_max - lsnr_min
        self.lsnr_offset = lsnr_min

    def call(self, feat_erb, feat_spec, erb_buf, df_buf, h_emb, training=False):
        erb_padded = tf.concat([erb_buf, feat_erb], axis=1)
        erb_buf_new = erb_padded[:, 1:, :, :]
        e0 = self.erb_conv0(erb_padded, training=training)

        e1 = self.erb_conv1(e0, training=training)
        e2 = self.erb_conv2(e1, training=training)
        e3 = self.erb_conv3(e2, training=training)

        df_padded = tf.concat([df_buf, feat_spec], axis=1)
        df_buf_new = df_padded[:, 1:, :, :]
        c0 = self.df_conv0(df_padded, training=training)
        c1 = self.df_conv1(c0, training=training)

        c1_shape = tf.shape(c1)
        cemb = tf.reshape(c1, [c1_shape[0], c1_shape[1], -1])
        cemb = self.df_fc_emb(cemb)
        if self.gru_type == "squeeze":
            cemb = self.df_fc_emb_relu(cemb)

        e3_shape = tf.shape(e3)
        emb = tf.reshape(e3, [e3_shape[0], e3_shape[1], -1])

        if self.enc_concat:
            emb = tf.concat([emb, cemb], axis=-1)
        else:
            emb = emb + cemb

        emb, h_emb_new = self.emb_gru(emb, h_emb)
        lsnr = tf.sigmoid(self.lsnr_dense(emb)) * self.lsnr_scale + self.lsnr_offset

        return e0, e1, e2, e3, emb, c0, lsnr, erb_buf_new, df_buf_new, h_emb_new

    def load_from_pt(self, pt_sd, prefix="enc"):
        pfx = f"{prefix}."
        self.erb_conv0.load_from_pt(pt_sd, f"{pfx}erb_conv0")
        self.erb_conv1.load_from_pt(pt_sd, f"{pfx}erb_conv1")
        self.erb_conv2.load_from_pt(pt_sd, f"{pfx}erb_conv2")
        self.erb_conv3.load_from_pt(pt_sd, f"{pfx}erb_conv3")
        self.df_conv0.load_from_pt(pt_sd, f"{pfx}df_conv0")
        self.df_conv1.load_from_pt(pt_sd, f"{pfx}df_conv1")
        if self.gru_type == "grouped":
            self.df_fc_emb.load_from_pt(pt_sd, f"{pfx}df_fc_emb")
        else:
            self.df_fc_emb.load_from_pt(pt_sd, f"{pfx}df_fc_emb.0")
        self.emb_gru.load_from_pt(pt_sd, f"{pfx}emb_gru")
        w = pt_sd[f"{pfx}lsnr_fc.0.weight"]
        b = pt_sd[f"{pfx}lsnr_fc.0.bias"]
        self.lsnr_dense.set_weights([w.T, b])


class ErbDecoderStreamingTF(tf.keras.layers.Layer):
    """Streaming ERB decoder with explicit GRU state."""

    def __init__(self, nb_erb=32, conv_ch=16, emb_hidden_dim=256,
                 emb_num_layers=2, lin_groups=1, gru_groups=1,
                 group_shuffle=True, gru_type="grouped", **kwargs):
        super().__init__(**kwargs)
        self.conv_ch = conv_ch
        self.gru_type = gru_type

        if gru_type == "grouped":
            self.emb_gru = GroupedGRUTF(
                conv_ch * nb_erb // 4, emb_hidden_dim,
                num_layers=emb_num_layers - 1, groups=gru_groups,
                shuffle=group_shuffle, add_outputs=True, name="erb_dec_gru")
            self.fc_emb = GroupedLinearTF(
                emb_hidden_dim, conv_ch * nb_erb // 4,
                groups=lin_groups, shuffle=group_shuffle, name="fc_emb")
            self.fc_emb_relu = tf.keras.layers.ReLU()
        else:
            self.emb_gru = SqueezedGRUTF(
                emb_hidden_dim, emb_hidden_dim,
                output_size=conv_ch * nb_erb // 4,
                num_layers=emb_num_layers - 1,
                linear_groups=lin_groups,
                has_gru_skip=True, has_linear_act=True, name="emb_gru")
            self.fc_emb = None
            self.fc_emb_relu = None

        self.conv3p = Conv2dNormActTF(conv_ch, conv_ch, 1, bias=False, separable=True, name="conv3p")
        self.convt3 = Conv2dNormActTF(conv_ch, conv_ch, (1, 3), bias=False, separable=True, name="convt3")
        self.conv2p = Conv2dNormActTF(conv_ch, conv_ch, 1, bias=False, separable=True, name="conv2p")
        self.convt2 = ConvTranspose2dNormActTF(conv_ch, conv_ch, (1, 3), fstride=2, bias=False, separable=True, name="convt2")
        self.conv1p = Conv2dNormActTF(conv_ch, conv_ch, 1, bias=False, separable=True, name="conv1p")
        self.convt1 = ConvTranspose2dNormActTF(conv_ch, conv_ch, (1, 3), fstride=2, bias=False, separable=True, name="convt1")
        self.conv0p = Conv2dNormActTF(conv_ch, conv_ch, 1, bias=False, separable=True, name="conv0p")
        self.conv0_out = Conv2dNormActTF(
            conv_ch, 1, (1, 3), bias=False, separable=True,
            activation="sigmoid", name="conv0_out")

    def call(self, emb, e3, e2, e1, e0, h_erb, training=False):
        B = tf.shape(e3)[0]
        f8 = e3.shape[2] if e3.shape[2] is not None else tf.shape(e3)[2]

        emb, h_erb_new = self.emb_gru(emb, h_erb)
        if self.fc_emb is not None:
            emb = self.fc_emb_relu(self.fc_emb(emb))
        emb = tf.reshape(emb, [B, 1, f8, self.conv_ch])

        e3 = self.convt3(self.conv3p(e3, training=training) + emb, training=training)
        e2 = self.convt2(self.conv2p(e2, training=training) + e3, training=training)
        e1 = self.convt1(self.conv1p(e1, training=training) + e2, training=training)
        m = self.conv0_out(self.conv0p(e0, training=training) + e1, training=training)
        return m, h_erb_new

    def load_from_pt(self, pt_sd, prefix="erb_dec"):
        pfx = f"{prefix}."
        self.emb_gru.load_from_pt(pt_sd, f"{pfx}emb_gru")
        if self.gru_type == "grouped":
            self.fc_emb.load_from_pt(pt_sd, f"{pfx}fc_emb.0")
        self.conv3p.load_from_pt(pt_sd, f"{pfx}conv3p")
        self.convt3.load_from_pt(pt_sd, f"{pfx}convt3")
        self.conv2p.load_from_pt(pt_sd, f"{pfx}conv2p")
        self.convt2.load_from_pt(pt_sd, f"{pfx}convt2")
        self.conv1p.load_from_pt(pt_sd, f"{pfx}conv1p")
        self.convt1.load_from_pt(pt_sd, f"{pfx}convt1")
        self.conv0p.load_from_pt(pt_sd, f"{pfx}conv0p")
        self.conv0_out.load_from_pt(pt_sd, f"{pfx}conv0_out")


class DfDecoderStreamingTF(tf.keras.layers.Layer):
    """Streaming DF decoder with explicit GRU state."""

    def __init__(self, nb_df=96, df_order=5, conv_ch=16,
                 emb_hidden_dim=256, df_hidden_dim=256, df_num_layers=3,
                 gru_groups=1, group_shuffle=True, lin_groups=1,
                 df_pathway_kernel_size_t=1, gru_type="grouped",
                 df_gru_skip="none", **kwargs):
        super().__init__(**kwargs)
        self.nb_df = nb_df
        self.df_out_ch = df_order * 2
        self.gru_type = gru_type
        self.df_pathway_kernel_size_t = df_pathway_kernel_size_t

        self.df_convp = Conv2dNormActTF(
            conv_ch, self.df_out_ch, (df_pathway_kernel_size_t, 1),
            separable=True, bias=False,
            causal_pad_t=0 if df_pathway_kernel_size_t > 1 else None,
            name="df_convp")

        if gru_type == "grouped":
            self.df_gru = GroupedGRUTF(
                emb_hidden_dim, df_hidden_dim, num_layers=df_num_layers,
                groups=gru_groups, shuffle=group_shuffle,
                add_outputs=True, name="df_gru")
        else:
            self.df_gru = SqueezedGRUTF(
                emb_hidden_dim, df_hidden_dim,
                num_layers=df_num_layers,
                linear_groups=lin_groups,
                has_gru_skip=True, has_linear_act=True, name="df_gru")

        df_gru_skip = df_gru_skip.lower()
        if df_gru_skip == "none":
            self.df_skip = None
        elif df_gru_skip == "identity":
            self.df_skip = "identity"
        elif df_gru_skip == "groupedlinear":
            self.df_skip = GroupedLinearEinsumTF(
                emb_hidden_dim, df_hidden_dim, groups=lin_groups, name="df_skip")
        else:
            self.df_skip = None

        self.df_out = GroupedLinearEinsumTF(
            df_hidden_dim, nb_df * self.df_out_ch,
            groups=lin_groups, name="df_out")
        self.df_out_tanh = tf.keras.layers.Activation("tanh")
        self.df_fc_a_dense = tf.keras.layers.Dense(1, name="df_fc_a")
        self.df_fc_a_sigmoid = tf.keras.layers.Activation("sigmoid")
        self.out_transform = DfOutputReshapeMFTF(df_order, nb_df)

    def call(self, emb, c0, h_df, df_convp_buf=None, training=False):
        B = tf.shape(emb)[0]

        c, h_df_new = self.df_gru(emb, h_df)

        if self.df_skip is not None:
            if self.df_skip == "identity":
                c = c + emb
            else:
                c = c + self.df_skip(emb)

        if self.df_pathway_kernel_size_t > 1 and df_convp_buf is not None:
            c0_padded = tf.concat([df_convp_buf, c0], axis=1)
            df_convp_buf_new = c0_padded[:, 1:, :, :]
            c0 = self.df_convp(c0_padded, training=training)
        else:
            c0 = self.df_convp(c0, training=training)
            df_convp_buf_new = None

        alpha = self.df_fc_a_sigmoid(self.df_fc_a_dense(c))
        c = self.df_out_tanh(self.df_out(c))
        c = tf.reshape(c, [B, 1, self.nb_df, self.df_out_ch])
        c = c + c0
        coefs = self.out_transform(c)
        return coefs, alpha, h_df_new, df_convp_buf_new

    def load_from_pt(self, pt_sd, prefix="df_dec"):
        pfx = f"{prefix}."
        self.df_convp.load_from_pt(pt_sd, f"{pfx}df_convp")
        self.df_gru.load_from_pt(pt_sd, f"{pfx}df_gru")
        if self.df_skip is not None and self.df_skip != "identity":
            self.df_skip.load_from_pt(pt_sd, f"{pfx}df_skip")
        self.df_out.load_from_pt(pt_sd, f"{pfx}df_out.0")
        w = pt_sd[f"{pfx}df_fc_a.0.weight"]
        b = pt_sd[f"{pfx}df_fc_a.0.bias"]
        self.df_fc_a_dense.set_weights([w.T, b])


class MultiFrameDFStreamingTF(tf.keras.layers.Layer):
    """Streaming deep filtering with spec ring buffer."""

    def __init__(self, num_freqs, frame_size, lookahead=0, **kwargs):
        super().__init__(**kwargs)
        self.num_freqs = num_freqs
        self.frame_size = frame_size
        self.lookahead = lookahead
        self.buf_size = frame_size - 1

    def call(self, spec, coefs, spec_buf):
        F_df = self.num_freqs

        spec_df_cur = spec[:, 0, 0, :F_df, :]
        spec_df_cur = tf.expand_dims(spec_df_cur, axis=1)
        window = tf.concat([spec_buf, spec_df_cur], axis=1)
        spec_buf_new = window[:, 1:, :, :]

        coefs_sq = coefs[:, :, 0, :, :]
        s_re = window[..., 0]
        s_im = window[..., 1]
        c_re = coefs_sq[..., 0]
        c_im = coefs_sq[..., 1]

        out_re = tf.reduce_sum(s_re * c_re - s_im * c_im, axis=1)
        out_im = tf.reduce_sum(s_re * c_im + s_im * c_re, axis=1)
        out = tf.stack([out_re, out_im], axis=-1)

        spec_out = tf.concat([
            tf.reshape(out, [tf.shape(spec)[0], 1, 1, F_df, 2]),
            spec[:, :, :, F_df:, :],
        ], axis=3)
        return spec_out, spec_buf_new


class DfNetStreamingTF(tf.keras.Model):
    """Full reference streaming DfNet: processes one frame at a time."""

    def __init__(self, erb_inv_fb_np, nb_erb=32, nb_df=96,
                 fft_size=960, conv_ch=16, df_order=5, df_lookahead=0,
                 emb_hidden_dim=256, emb_num_layers=2,
                 df_hidden_dim=256, df_num_layers=3,
                 gru_groups=1, lin_groups=1, group_shuffle=True,
                 enc_concat=False, df_pathway_kernel_size_t=1,
                 gru_type="grouped", df_gru_skip="none", **kwargs):
        super().__init__(**kwargs)
        self.nb_erb = nb_erb
        self.nb_df = nb_df
        self.fft_size = fft_size
        self.freq_bins = fft_size // 2 + 1
        self.df_order = df_order
        self.conv_ch = conv_ch
        self.emb_hidden_dim = emb_hidden_dim
        self.df_hidden_dim = df_hidden_dim
        self.df_num_layers = df_num_layers
        self.emb_num_layers = emb_num_layers
        self.gru_groups = gru_groups
        self.gru_type = gru_type
        self.df_pathway_kernel_size_t = df_pathway_kernel_size_t

        self.enc = EncoderStreamingTF(
            nb_erb=nb_erb, nb_df=nb_df, conv_ch=conv_ch,
            emb_hidden_dim=emb_hidden_dim, lin_groups=lin_groups,
            gru_groups=gru_groups, group_shuffle=group_shuffle,
            enc_concat=enc_concat, gru_type=gru_type, name="encoder")
        self.erb_dec = ErbDecoderStreamingTF(
            nb_erb=nb_erb, conv_ch=conv_ch,
            emb_hidden_dim=emb_hidden_dim,
            emb_num_layers=emb_num_layers,
            lin_groups=lin_groups, gru_groups=gru_groups,
            group_shuffle=group_shuffle, gru_type=gru_type,
            name="erb_decoder")
        self.mask = MaskTF(erb_inv_fb_np, name="mask")
        self.df_dec = DfDecoderStreamingTF(
            nb_df=nb_df, df_order=df_order, conv_ch=conv_ch,
            emb_hidden_dim=emb_hidden_dim,
            df_hidden_dim=df_hidden_dim,
            df_num_layers=df_num_layers,
            gru_groups=gru_groups,
            group_shuffle=group_shuffle,
            lin_groups=lin_groups,
            df_pathway_kernel_size_t=df_pathway_kernel_size_t,
            gru_type=gru_type,
            df_gru_skip=df_gru_skip,
            name="df_decoder")
        self.df_op = MultiFrameDFStreamingTF(
            num_freqs=nb_df, frame_size=df_order,
            lookahead=df_lookahead, name="df_op")

    def get_initial_state(self, batch_size=1):
        state = {
            "erb_buf": tf.zeros([batch_size, 2, self.nb_erb, 1]),
            "df_buf": tf.zeros([batch_size, 2, self.nb_df, 2]),
            "spec_buf": tf.zeros([batch_size, self.df_order - 1, self.nb_df, 2]),
        }

        if self.gru_type == "grouped":
            h_per_group = self.emb_hidden_dim // self.gru_groups
            df_h_per_group = self.df_hidden_dim // self.gru_groups
            state["h_enc"] = tf.zeros([1 * self.gru_groups, batch_size, h_per_group])
            state["h_erb"] = tf.zeros([(self.emb_num_layers - 1) * self.gru_groups, batch_size, h_per_group])
            state["h_df"] = tf.zeros([self.df_num_layers * self.gru_groups, batch_size, df_h_per_group])
        else:
            state["h_enc"] = tf.zeros([1, batch_size, self.emb_hidden_dim])
            state["h_erb"] = tf.zeros([self.emb_num_layers - 1, batch_size, self.emb_hidden_dim])
            state["h_df"] = tf.zeros([self.df_num_layers, batch_size, self.df_hidden_dim])

        if self.df_pathway_kernel_size_t > 1:
            state["df_convp_buf"] = tf.zeros([
                batch_size, self.df_pathway_kernel_size_t - 1,
                self.nb_df, self.conv_ch])

        return state

    def call(self, spec, feat_erb, feat_spec, state, training=False):
        erb_buf = state["erb_buf"]
        df_buf = state["df_buf"]
        h_enc = state["h_enc"]
        h_erb = state["h_erb"]
        h_df = state["h_df"]
        spec_buf = state["spec_buf"]
        df_convp_buf = state.get("df_convp_buf", None)

        feat_spec_nhwc = tf.squeeze(feat_spec, axis=1)
        feat_erb_nhwc = tf.transpose(feat_erb, [0, 2, 3, 1])

        e0, e1, e2, e3, emb, c0, lsnr, erb_buf_new, df_buf_new, h_enc_new = \
            self.enc(feat_erb_nhwc, feat_spec_nhwc, erb_buf, df_buf, h_enc,
                     training=training)

        m_nhwc, h_erb_new = self.erb_dec(emb, e3, e2, e1, e0, h_erb,
                                         training=training)

        spec_tf = tf.transpose(spec, [0, 2, 3, 1, 4])
        spec_masked_tf = self.mask(spec_tf, m_nhwc)
        spec_masked = tf.transpose(spec_masked_tf, [0, 3, 1, 2, 4])

        df_coefs, df_alpha, h_df_new, df_convp_buf_new = \
            self.df_dec(emb, c0, h_df, df_convp_buf, training=training)

        spec_out, spec_buf_new = self.df_op(spec_masked, df_coefs, spec_buf)
        m_pt = tf.transpose(m_nhwc, [0, 3, 1, 2])

        new_state = {
            "erb_buf": erb_buf_new,
            "df_buf": df_buf_new,
            "h_enc": h_enc_new,
            "h_erb": h_erb_new,
            "h_df": h_df_new,
            "spec_buf": spec_buf_new,
        }
        if df_convp_buf_new is not None:
            new_state["df_convp_buf"] = df_convp_buf_new
        return spec_out, m_pt, lsnr, df_alpha, new_state

    def load_from_pt(self, pt_sd):
        self.enc.load_from_pt(pt_sd, "enc")
        self.erb_dec.load_from_pt(pt_sd, "erb_dec")
        self.df_dec.load_from_pt(pt_sd, "df_dec")


def compute_erb_inv_fb(erb_widths, normalized=True):
    """Compute inverse ERB filterbank matrix [nb_erb, freq_bins].

    Matches modules.py erb_fb(widths, sr, normalized=True, inverse=True).
    """
    n_freqs = int(sum(erb_widths))
    nb_bands = len(erb_widths)
    fb = np.zeros((n_freqs, nb_bands), dtype=np.float32)
    b = 0
    for i, w in enumerate(erb_widths):
        w = int(w)
        fb[b:b + w, i] = 1.0
        b += w
    # inverse: transpose then normalize
    fb_inv = fb.T  # [nb_erb, n_freqs]
    # For inverse, not-normalized means divide by row sums
    # But the code says: if inverse and not normalized: fb /= fb.sum(dim=1, keepdim=True)
    # Default is normalized=True for inverse, so no division
    if not normalized:
        row_sums = fb_inv.sum(axis=1, keepdims=True)
        fb_inv = fb_inv / np.maximum(row_sums, 1e-12)
    return fb_inv  # [nb_erb, n_freqs]


# ============================================================================
# TF Feature Extraction Ops (matching Rust libDF)
# ============================================================================

def tf_compute_band_corr(spec_re, spec_im, erb_widths_np):
    """Compute ERB band correlation (mean power per band).

    Matches Rust compute_band_corr(out, x, x, erb_fb):
        For each band b with width w:
            out[b] = sum_{j in band}(re[j]^2 + im[j]^2) / w

    Args:
        spec_re: [B, F] real part of one frame
        spec_im: [B, F] imag part of one frame
        erb_widths_np: list/array of int, ERB band widths

    Returns:
        erb_energy: [B, E] where E = len(erb_widths)
    """
    # |spec|^2 per frequency bin
    power = spec_re * spec_re + spec_im * spec_im  # [B, F]

    # Build a mapping matrix [F, E] where each col has 1/w for bins in that band
    F = int(sum(erb_widths_np))
    E = len(erb_widths_np)
    band_matrix = np.zeros((F, E), dtype=np.float32)
    b = 0
    for i, w in enumerate(erb_widths_np):
        w = int(w)
        band_matrix[b:b + w, i] = 1.0 / w
        b += w

    band_matrix_tf = tf.constant(band_matrix)  # [F, E]
    erb_energy = tf.matmul(power, band_matrix_tf)  # [B, E]
    return erb_energy


def tf_erb(spec_re, spec_im, erb_widths_np, db=True):
    """Compute ERB features from complex spec frame.

    Matches Rust erb() with db=True:
        compute_band_corr then 10*log10(v + 1e-10)

    Args:
        spec_re: [B, F] real part
        spec_im: [B, F] imag part
        erb_widths_np: ERB band widths
        db: if True, convert to dB scale

    Returns:
        erb_feat: [B, E]
    """
    erb_energy = tf_compute_band_corr(spec_re, spec_im, erb_widths_np)
    if db:
        erb_feat = 10.0 * tf.math.log(erb_energy + 1e-10) / tf.math.log(10.0)
    else:
        erb_feat = erb_energy
    return erb_feat


def tf_erb_norm(erb_feat, state, alpha):
    """Running mean normalization for ERB features (one frame).

    Matches Rust band_mean_norm_erb:
        state = x * (1 - alpha) + state * alpha
        x = (x - state) / 40

    Args:
        erb_feat: [B, E] — one frame of ERB features
        state: [B, E] — running mean state
        alpha: float

    Returns:
        normed: [B, E]
        new_state: [B, E]
    """
    new_state = erb_feat * (1.0 - alpha) + state * alpha
    normed = (erb_feat - new_state) / 40.0
    return normed, new_state


def tf_unit_norm(spec_re, spec_im, state, alpha, eps=1e-14):
    """Running unit normalization for complex spec (one frame).

    Matches Rust band_unit_norm:
        state = |x| * (1 - alpha) + state * alpha
        x = x / sqrt(state)

    Note: Rust uses x.norm() = sqrt(re^2 + im^2) for Complex32.

    Args:
        spec_re: [B, F] real part (F = nb_df)
        spec_im: [B, F] imag part
        state: [B, F] running norm state
        alpha: float
        eps: float for numerical stability

    Returns:
        normed_re: [B, F]
        normed_im: [B, F]
        new_state: [B, F]
    """
    x_abs = tf.sqrt(spec_re * spec_re + spec_im * spec_im + eps)
    new_state = x_abs * (1.0 - alpha) + state * alpha
    denom = tf.sqrt(new_state)
    normed_re = spec_re / denom
    normed_im = spec_im / denom
    return normed_re, normed_im, new_state


# ============================================================================
# Stateful Streaming Model
# ============================================================================

class DfNetStatefulStreamingTF(tf.keras.Model):
    """Stateful streaming DeepFilterNet2 for TFLite export.

    Single input: spec [B, 1, 1, F, 2]
    Single output: enhanced_spec [B, 1, 1, F, 2]

    All state is stored as tf.Variable and updated via assign.
    Feature extraction (erb, erb_norm, unit_norm) happens in forward().

    For TFLite export, use build_tflite_module(model) which:
    1. Gets a concrete function for forward_stateless (pure NN — no feature
       extraction, no a²+b², no √(a²+b²), no norm state)
    2. Freezes NN weights to constants with convert_variables_to_constants_v2
    3. Wraps in tf.Module that does feature extraction in float32 OUTSIDE
       the frozen graph, then calls frozen NN with pre-computed features

    Architecture split for quantization:
      Feature extraction (a²+b², ERB, log10, √, norms) stays in float32
      in the wrapper. The frozen NN graph only sees normalized features
      with narrow dynamic range → much better int8/int16x8 quantization.
    """

    def __init__(self, erb_widths_np, erb_inv_fb_np,
                 nb_erb=32, nb_df=96, fft_size=960,
                 sr=48000, hop_size=480, norm_tau=1.0,
                 conv_ch=16, df_order=5, df_lookahead=0,
                 emb_hidden_dim=256, emb_num_layers=2,
                 df_hidden_dim=256, df_num_layers=3,
                 gru_groups=1, lin_groups=1, group_shuffle=True,
                 enc_concat=False, df_pathway_kernel_size_t=1,
                 gru_type="grouped", df_gru_skip="none",
                 conv_lookahead=0, pad_mode="none",
                 batch_size=1):
        super().__init__()

        self.nb_erb = nb_erb
        self.nb_df = nb_df
        self.fft_size = fft_size
        self.freq_bins = fft_size // 2 + 1
        self.df_order = df_order
        self.conv_ch = conv_ch
        self.emb_hidden_dim = emb_hidden_dim
        self.df_hidden_dim = df_hidden_dim
        self.df_num_layers = df_num_layers
        self.emb_num_layers = emb_num_layers
        self.gru_groups = gru_groups
        self.gru_type = gru_type
        self.df_pathway_kernel_size_t = df_pathway_kernel_size_t
        self.batch_size = batch_size
        self.erb_widths_np = list(erb_widths_np)
        self.conv_lookahead = conv_lookahead
        self.pad_mode = pad_mode
        self.pad_specf = pad_mode.endswith("specf")
        self.input_feat_lookahead = conv_lookahead if pad_mode.startswith("input") else 0
        self.input_spec_lookahead = df_lookahead if self.pad_specf else 0
        if self.input_feat_lookahead > 0 and self.input_spec_lookahead > 0:
            if self.input_feat_lookahead != self.input_spec_lookahead:
                raise ValueError(
                    "Stateful TF input padding currently requires matching "
                    "conv_lookahead and df_lookahead"
                )
        self.stream_lookahead = max(self.input_feat_lookahead, self.input_spec_lookahead)

        # Norm alpha
        self.norm_alpha = compute_norm_alpha(sr, hop_size, norm_tau)

        # NN sub-modules (reuse existing streaming layers)
        self.enc = EncoderStreamingTF(
            nb_erb=nb_erb, nb_df=nb_df, conv_ch=conv_ch,
            emb_hidden_dim=emb_hidden_dim, lin_groups=lin_groups,
            gru_groups=gru_groups, group_shuffle=group_shuffle,
            enc_concat=enc_concat, gru_type=gru_type, name="encoder")
        self.erb_dec = ErbDecoderStreamingTF(
            nb_erb=nb_erb, conv_ch=conv_ch,
            emb_hidden_dim=emb_hidden_dim,
            emb_num_layers=emb_num_layers,
            lin_groups=lin_groups, gru_groups=gru_groups,
            group_shuffle=group_shuffle, gru_type=gru_type,
            name="erb_decoder")
        self.mask = MaskTF(erb_inv_fb_np, name="mask")
        self.df_dec = DfDecoderStreamingTF(
            nb_df=nb_df, df_order=df_order, conv_ch=conv_ch,
            emb_hidden_dim=emb_hidden_dim,
            df_hidden_dim=df_hidden_dim,
            df_num_layers=df_num_layers,
            gru_groups=gru_groups,
            group_shuffle=group_shuffle,
            lin_groups=lin_groups,
            df_pathway_kernel_size_t=df_pathway_kernel_size_t,
            gru_type=gru_type,
            df_gru_skip=df_gru_skip,
            name="df_decoder")
        self.df_op = MultiFrameDFStreamingTF(
            num_freqs=nb_df, frame_size=df_order,
            lookahead=df_lookahead, name="df_op")

        # ── State variables (resource variables for TFLite) ──
        B = batch_size
        E = nb_erb
        F_df = nb_df

        # Encoder conv buffers
        self.erb_buf = tf.Variable(
            tf.zeros([B, 2, E, 1]), trainable=False, name="state/erb_buf")
        self.df_buf = tf.Variable(
            tf.zeros([B, 2, F_df, 2]), trainable=False, name="state/df_buf")

        # GRU hidden states — shape depends on gru_type
        if gru_type == "grouped":
            h_per_group = emb_hidden_dim // gru_groups
            df_h_per_group = df_hidden_dim // gru_groups
            self.h_enc = tf.Variable(
                tf.zeros([1 * gru_groups, B, h_per_group]),
                trainable=False, name="state/h_enc")
            self.h_erb = tf.Variable(
                tf.zeros([(emb_num_layers - 1) * gru_groups, B, h_per_group]),
                trainable=False, name="state/h_erb")
            self.h_df = tf.Variable(
                tf.zeros([df_num_layers * gru_groups, B, df_h_per_group]),
                trainable=False, name="state/h_df")
        else:
            # squeeze: standard GRU, [num_layers, B, H]
            self.h_enc = tf.Variable(
                tf.zeros([1, B, emb_hidden_dim]),
                trainable=False, name="state/h_enc")
            self.h_erb = tf.Variable(
                tf.zeros([emb_num_layers - 1, B, emb_hidden_dim]),
                trainable=False, name="state/h_erb")
            self.h_df = tf.Variable(
                tf.zeros([df_num_layers, B, df_hidden_dim]),
                trainable=False, name="state/h_df")

        # Spec ring buffer for deep filtering
        self.spec_buf = tf.Variable(
            tf.zeros([B, df_order - 1, F_df, 2]),
            trainable=False, name="state/spec_buf")

        if self.stream_lookahead > 0 and self.pad_mode.startswith("input"):
            self.spec_lookahead_buf = tf.Variable(
                tf.zeros([B, self.stream_lookahead, self.freq_bins, 2]),
                trainable=False, name="state/spec_lookahead_buf")
        else:
            self.spec_lookahead_buf = None

        # df_convp causal buffer (if kernel_t > 1)
        if df_pathway_kernel_size_t > 1:
            self.df_convp_buf = tf.Variable(
                tf.zeros([B, df_pathway_kernel_size_t - 1, nb_df, conv_ch]),
                trainable=False, name="state/df_convp_buf")
        else:
            self.df_convp_buf = None

        # ── Feature extraction state variables ──
        # erb_norm state: initialized from linspace(MEAN_NORM_INIT[0], MEAN_NORM_INIT[1], nb_erb)
        erb_norm_init = np.linspace(MEAN_NORM_INIT[0], MEAN_NORM_INIT[1], E).astype(np.float32)
        erb_norm_init_tiled = np.tile(erb_norm_init[np.newaxis, :], [B, 1])  # [B, E]
        self.erb_norm_state = tf.Variable(
            erb_norm_init_tiled, trainable=False, name="state/erb_norm_state")
        self._erb_norm_init_val = erb_norm_init_tiled

        # unit_norm state: initialized from linspace(UNIT_NORM_INIT[0], UNIT_NORM_INIT[1], nb_df)
        unit_norm_init = np.linspace(UNIT_NORM_INIT[0], UNIT_NORM_INIT[1], F_df).astype(np.float32)
        unit_norm_init_tiled = np.tile(unit_norm_init[np.newaxis, :], [B, 1])  # [B, F_df]
        self.unit_norm_state = tf.Variable(
            unit_norm_init_tiled, trainable=False, name="state/unit_norm_state")
        self._unit_norm_init_val = unit_norm_init_tiled

        # Pre-compute ERB band correlation matrix as constant
        F = self.freq_bins
        band_matrix = np.zeros((F, E), dtype=np.float32)
        b_idx = 0
        for i, w in enumerate(self.erb_widths_np):
            w = int(w)
            band_matrix[b_idx:b_idx + w, i] = 1.0 / w
            b_idx += w
        self.erb_band_matrix = tf.constant(band_matrix, name="erb_band_matrix")

    def _build(self):
        """Build all sub-layers by running a dummy forward pass."""
        B = self.batch_size
        F = self.freq_bins
        dummy_spec = tf.zeros([B, 1, 1, F, 2])
        self.forward(dummy_spec)

    @tf.function(input_signature=[
        tf.TensorSpec([1, 1, 1, 481, 2], tf.float32, name="spec")
    ])
    def forward(self, spec):
        """Process one frame: extract features, run NN, return enhanced spec.

        Args:
            spec: [B, 1, 1, F, 2] — noisy spectrum (one frame)

        Returns:
            enhanced_spec: [B, 1, 1, F, 2]
        """
        F_df = self.nb_df
        E = self.nb_erb

        # ── 1. Feature extraction from spec ──
        # Extract real/imag: spec is [B, 1, 1, F, 2]
        spec_re = spec[:, 0, 0, :, 0]  # [B, F]
        spec_im = spec[:, 0, 0, :, 1]  # [B, F]

        # ERB features: compute_band_corr then dB
        power = spec_re * spec_re + spec_im * spec_im  # [B, F]
        erb_energy = tf.matmul(power, self.erb_band_matrix)  # [B, E]
        erb_feat = 10.0 * tf.math.log(erb_energy + 1e-10) / tf.math.log(10.0)  # [B, E]

        # ERB normalization (stateful)
        erb_state = self.erb_norm_state
        new_erb_state = erb_feat * (1.0 - self.norm_alpha) + erb_state * self.norm_alpha
        feat_erb_normed = (erb_feat - new_erb_state) / 40.0  # [B, E]
        self.erb_norm_state.assign(new_erb_state)

        # Unit norm for complex spec (stateful) — only first nb_df bins
        spec_df_re = spec_re[:, :F_df]  # [B, F_df]
        spec_df_im = spec_im[:, :F_df]  # [B, F_df]
        x_abs = tf.sqrt(spec_df_re * spec_df_re + spec_df_im * spec_df_im + 1e-14)
        un_state = self.unit_norm_state
        new_un_state = x_abs * (1.0 - self.norm_alpha) + un_state * self.norm_alpha
        denom = tf.sqrt(new_un_state)
        feat_spec_re = spec_df_re / denom  # [B, F_df]
        feat_spec_im = spec_df_im / denom  # [B, F_df]
        self.unit_norm_state.assign(new_un_state)

        # ── 2. Reshape features for encoder ──
        # feat_erb: [B, E] -> [B, 1, E, 1] (NHWC for encoder)
        feat_erb_nhwc = tf.reshape(feat_erb_normed, [-1, 1, E, 1])
        # feat_spec: [B, F_df, 2] -> [B, 1, F_df, 2] (NHWC for encoder)
        feat_spec_nhwc = tf.stack([feat_spec_re, feat_spec_im], axis=-1)  # [B, F_df, 2]
        feat_spec_nhwc = tf.expand_dims(feat_spec_nhwc, axis=1)  # [B, 1, F_df, 2]

        # ── 3. Encoder (streaming with conv buffers and GRU state) ──
        erb_buf = self.erb_buf
        df_buf = self.df_buf
        h_enc = self.h_enc

        e0, e1, e2, e3, emb, c0, _lsnr, erb_buf_new, df_buf_new, h_enc_new = \
            self.enc(feat_erb_nhwc, feat_spec_nhwc, erb_buf, df_buf, h_enc,
                     training=False)

        self.erb_buf.assign(erb_buf_new)
        self.df_buf.assign(df_buf_new)
        self.h_enc.assign(h_enc_new)

        # ── 4. ERB Decoder (streaming with GRU state) ──
        h_erb = self.h_erb
        m_nhwc, h_erb_new = self.erb_dec(emb, e3, e2, e1, e0, h_erb,
                                          training=False)
        self.h_erb.assign(h_erb_new)

        # ── 5. Apply ERB mask to spec ──
        if self.spec_lookahead_buf is not None:
            spec_frame = spec[:, 0, 0, :, :]
            spec_history = tf.concat(
                [self.spec_lookahead_buf, tf.expand_dims(spec_frame, axis=1)], axis=1)
            delayed_spec = tf.reshape(
                spec_history[:, 0, :, :], [-1, 1, 1, self.freq_bins, 2])
            self.spec_lookahead_buf.assign(spec_history[:, 1:, :, :])
        else:
            delayed_spec = spec

        def apply_mask(spec_in):
            spec_tf = tf.transpose(spec_in, [0, 2, 3, 1, 4])
            spec_masked_tf = self.mask(spec_tf, m_nhwc)
            return tf.transpose(spec_masked_tf, [0, 3, 1, 2, 4])

        spec_masked = apply_mask(delayed_spec)
        if self.pad_specf and self.spec_lookahead_buf is not None:
            spec_df_masked = apply_mask(spec)
        else:
            spec_df_masked = spec_masked

        # ── 6. DF Decoder (streaming with GRU state) ──
        h_df = self.h_df
        df_convp_buf = self.df_convp_buf if self.df_convp_buf is not None else None
        df_coefs, _df_alpha, h_df_new, df_convp_buf_new = \
            self.df_dec(emb, c0, h_df, df_convp_buf, training=False)
        self.h_df.assign(h_df_new)
        if df_convp_buf_new is not None:
            self.df_convp_buf.assign(df_convp_buf_new)

        # ── 7. Deep filtering (streaming with spec ring buffer) ──
        spec_buf = self.spec_buf
        spec_out_low, spec_buf_new = self.df_op(spec_df_masked, df_coefs, spec_buf)
        self.spec_buf.assign(spec_buf_new)

        if self.pad_specf:
            spec_out = tf.concat([
                spec_out_low[:, :, :, :F_df, :],
                spec_masked[:, :, :, F_df:, :],
            ], axis=3)
        else:
            spec_out = spec_out_low

        return spec_out

    @tf.function(input_signature=[])
    def reset_state(self):
        """Reset all streaming state to initial values."""
        self.erb_buf.assign(tf.zeros_like(self.erb_buf))
        self.df_buf.assign(tf.zeros_like(self.df_buf))
        self.h_enc.assign(tf.zeros_like(self.h_enc))
        self.h_erb.assign(tf.zeros_like(self.h_erb))
        self.h_df.assign(tf.zeros_like(self.h_df))
        self.spec_buf.assign(tf.zeros_like(self.spec_buf))
        if self.spec_lookahead_buf is not None:
            self.spec_lookahead_buf.assign(tf.zeros_like(self.spec_lookahead_buf))
        if self.df_convp_buf is not None:
            self.df_convp_buf.assign(tf.zeros_like(self.df_convp_buf))

        # Feature extraction state: reset to init values
        self.erb_norm_state.assign(
            tf.constant(self._erb_norm_init_val, dtype=tf.float32))
        self.unit_norm_state.assign(
            tf.constant(self._unit_norm_init_val, dtype=tf.float32))

        return tf.constant(0, dtype=tf.int32)  # dummy output for TFLite

    def forward_stateless(self, feat_erb_nhwc, feat_spec_nhwc,
                          erb_buf, df_buf, h_enc, h_erb, h_df,
                          df_convp_buf=None):
        """Pure NN forward pass: predicts ERB mask + DF coefficients.

        The frozen graph is a pure feature-to-prediction function:
          Input:  normalized features + NN streaming states
          Output: ERB mask [B,1,E,1] + DF coefs [B,O,1,F_df,2] + updated states

        Everything else runs in float32 in the wrapper:
          - Feature extraction: a²+b², ERB matmul, log10, √(a²+b²), norms
          - Mask application: erb_mask → freq expansion → spec × mask
          - Deep filtering: complex MAC with spec ring buffer

        No raw spectrum values ever enter the frozen graph. This gives the
        quantizer the narrowest possible activation ranges.

        Args:
            feat_erb_nhwc: [B, 1, E, 1] — normalized ERB features (after erb_norm)
            feat_spec_nhwc: [B, 1, F_df, 2] — normalized complex spec (after unit_norm)
            erb_buf, df_buf, h_enc, h_erb, h_df: NN streaming states
            df_convp_buf: optional DF pathway conv buffer

        Returns:
            (m_nhwc, df_coefs,
             new_erb_buf, new_df_buf, new_h_enc, new_h_erb, new_h_df
             [, new_df_convp_buf])
        """
        # ── 1. Encoder ──
        e0, e1, e2, e3, emb, c0, _lsnr, new_erb_buf, new_df_buf, new_h_enc = \
            self.enc(feat_erb_nhwc, feat_spec_nhwc, erb_buf, df_buf, h_enc,
                     training=False)

        # ── 2. ERB Decoder → mask ──
        m_nhwc, new_h_erb = self.erb_dec(emb, e3, e2, e1, e0, h_erb,
                                          training=False)

        # ── 3. DF Decoder → coefficients ──
        df_coefs, _df_alpha, new_h_df, new_df_convp_buf = \
            self.df_dec(emb, c0, h_df, df_convp_buf, training=False)

        result = [m_nhwc, df_coefs,
                  new_erb_buf, new_df_buf, new_h_enc, new_h_erb, new_h_df]
        if new_df_convp_buf is not None:
            result.append(new_df_convp_buf)
        return tuple(result)

    def load_from_pt(self, pt_sd):
        """Load NN weights from PyTorch state dict."""
        self.enc.load_from_pt(pt_sd, "enc")
        self.erb_dec.load_from_pt(pt_sd, "erb_dec")
        self.df_dec.load_from_pt(pt_sd, "df_dec")


# ============================================================================
# TFLite Export: Two-phase freeze-then-wrap approach
# ============================================================================

def build_tflite_module(model):
    """Build a TFLite-ready tf.Module from the stateful streaming model.

    Strategy:
    1. Get a concrete function for forward_stateless (no assigns, pure NN).
    2. Freeze all NN weights to constants with convert_variables_to_constants_v2.
    3. Wrap in a thin tf.Module that manages state variables and calls the
       frozen function.

    Args:
        model: DfNetStatefulStreamingTF with loaded weights.

    Returns:
        tf.Module with forward(spec) and reset_state() signatures.
    """
    from tensorflow.python.framework.convert_to_constants import (
        convert_variables_to_constants_v2,
    )

    B = model.batch_size
    F = model.freq_bins
    E = model.nb_erb
    F_df = model.nb_df
    df_order = model.df_order
    conv_ch = model.conv_ch
    emb_hidden_dim = model.emb_hidden_dim
    df_hidden_dim = model.df_hidden_dim
    df_num_layers = model.df_num_layers
    emb_num_layers = model.emb_num_layers
    gru_groups = model.gru_groups
    gru_type = model.gru_type
    df_pathway_kernel_size_t = model.df_pathway_kernel_size_t
    has_convp_buf = df_pathway_kernel_size_t > 1
    pad_mode = model.pad_mode
    pad_specf = model.pad_specf
    stream_lookahead = model.stream_lookahead

    # Compute GRU state shapes based on gru_type
    if gru_type == "grouped":
        h_per_group = emb_hidden_dim // gru_groups
        df_h_per_group = df_hidden_dim // gru_groups
        h_enc_shape = [1 * gru_groups, B, h_per_group]
        h_erb_shape = [(emb_num_layers - 1) * gru_groups, B, h_per_group]
        h_df_shape = [df_num_layers * gru_groups, B, df_h_per_group]
    else:
        h_enc_shape = [1, B, emb_hidden_dim]
        h_erb_shape = [emb_num_layers - 1, B, emb_hidden_dim]
        h_df_shape = [df_num_layers, B, df_hidden_dim]

    norm_alpha = model.norm_alpha
    erb_band_matrix_np = model.erb_band_matrix.numpy()  # [F, E]
    erb_inv_fb_np = model.mask.erb_inv_fb.numpy()        # [E, F]

    # ── 1. Get concrete function for stateless forward (pure NN) ──
    # The frozen graph contains ONLY the neural network:
    #   encoder convs/GRUs → ERB decoder (mask) → DF decoder (coefs)
    # NO spec, NO spec_buf, NO mask application, NO deep filtering.
    # All raw-spectrum ops stay in float32 in the wrapper.
    input_specs = [
        tf.TensorSpec([B, 1, E, 1], tf.float32),          # feat_erb_nhwc
        tf.TensorSpec([B, 1, F_df, 2], tf.float32),       # feat_spec_nhwc
        tf.TensorSpec([B, 2, E, 1], tf.float32),           # erb_buf
        tf.TensorSpec([B, 2, F_df, 2], tf.float32),        # df_buf
        tf.TensorSpec(h_enc_shape, tf.float32),            # h_enc
        tf.TensorSpec(h_erb_shape, tf.float32),            # h_erb
        tf.TensorSpec(h_df_shape, tf.float32),             # h_df
    ]
    if has_convp_buf:
        input_specs.append(
            tf.TensorSpec([B, df_pathway_kernel_size_t - 1, F_df, conv_ch],
                          tf.float32))  # df_convp_buf

    stateless_cf = tf.function(model.forward_stateless).get_concrete_function(
        *input_specs)

    print(f"  Stateless CF captured {len(stateless_cf.variables)} variables")

    # ── 2. Freeze NN weights to constants ──
    frozen_cf = convert_variables_to_constants_v2(stateless_cf)
    print(f"  Frozen graph: {len(frozen_cf.graph.as_graph_def().node)} ops")

    # ── 3. Build wrapper ──
    # The wrapper runs ALL wide-range ops in float32, outside frozen graph:
    #   - Feature extraction: a²+b², ERB matmul, log10, √(a²+b²), norms
    #   - Mask application: ERB mask → freq expansion → spec × mask
    #   - Deep filtering: complex MAC with spec ring buffer
    # The frozen NN only predicts mask + DF coefs from normalized features.
    # This gives the quantizer the narrowest possible activation ranges.
    class _StatefulWrapper(tf.Module):
        def __init__(self):
            super().__init__()

            # ── NN streaming state variables ──
            self.erb_buf = tf.Variable(
                tf.zeros([B, 2, E, 1]), trainable=False, name="state/erb_buf")
            self.df_buf = tf.Variable(
                tf.zeros([B, 2, F_df, 2]), trainable=False, name="state/df_buf")
            self.h_enc = tf.Variable(
                tf.zeros(h_enc_shape), trainable=False, name="state/h_enc")
            self.h_erb = tf.Variable(
                tf.zeros(h_erb_shape), trainable=False, name="state/h_erb")
            self.h_df = tf.Variable(
                tf.zeros(h_df_shape), trainable=False, name="state/h_df")

            if has_convp_buf:
                self.df_convp_buf = tf.Variable(
                    tf.zeros([B, df_pathway_kernel_size_t - 1, F_df, conv_ch]),
                    trainable=False, name="state/df_convp_buf")

            # ── Post-processing state (outside frozen graph, always float32) ──
            self.spec_buf = tf.Variable(
                tf.zeros([B, df_order - 1, F_df, 2]),
                trainable=False, name="state/spec_buf")
            if stream_lookahead > 0 and pad_mode.startswith("input"):
                self.spec_lookahead_buf = tf.Variable(
                    tf.zeros([B, stream_lookahead, F, 2]),
                    trainable=False, name="state/spec_lookahead_buf")
            else:
                self.spec_lookahead_buf = None

            # ── Feature extraction state variables ──
            erb_norm_init = np.linspace(
                MEAN_NORM_INIT[0], MEAN_NORM_INIT[1], E).astype(np.float32)
            self.erb_norm_state = tf.Variable(
                np.tile(erb_norm_init[np.newaxis, :], [B, 1]),
                trainable=False, name="state/erb_norm_state")
            self._erb_norm_init = np.tile(
                erb_norm_init[np.newaxis, :], [B, 1])

            unit_norm_init = np.linspace(
                UNIT_NORM_INIT[0], UNIT_NORM_INIT[1], F_df).astype(np.float32)
            self.unit_norm_state = tf.Variable(
                np.tile(unit_norm_init[np.newaxis, :], [B, 1]),
                trainable=False, name="state/unit_norm_state")
            self._unit_norm_init = np.tile(
                unit_norm_init[np.newaxis, :], [B, 1])

            # ── Constants for feature extraction + post-processing ──
            self._erb_band_matrix = tf.constant(
                erb_band_matrix_np, name="feat/erb_band_matrix")
            self._erb_inv_fb = tf.constant(
                erb_inv_fb_np, name="feat/erb_inv_fb")
            self._norm_alpha = norm_alpha

        @tf.function(input_signature=[
            tf.TensorSpec([B, 1, 1, F, 2], tf.float32, name="spec")
        ])
        def forward(self, spec):
            # ════════════════════════════════════════════════════════════
            # Feature extraction — float32, OUTSIDE frozen graph
            # ════════════════════════════════════════════════════════════
            spec_re = spec[:, 0, 0, :, 0]  # [B, F]
            spec_im = spec[:, 0, 0, :, 1]  # [B, F]

            # ERB features: power → band avg → dB → running mean norm
            power = spec_re * spec_re + spec_im * spec_im  # [B, F]
            erb_energy = tf.matmul(power, self._erb_band_matrix)  # [B, E]
            erb_feat = 10.0 * tf.math.log(erb_energy + 1e-10) / tf.math.log(10.0)

            erb_st = self.erb_norm_state
            new_erb_st = erb_feat * (1.0 - self._norm_alpha) + erb_st * self._norm_alpha
            feat_erb = (erb_feat - new_erb_st) / 40.0  # [B, E]
            self.erb_norm_state.assign(new_erb_st)

            # Unit norm: magnitude → running norm → normalized complex spec
            spec_df_re = spec_re[:, :F_df]  # [B, F_df]
            spec_df_im = spec_im[:, :F_df]
            x_abs = tf.sqrt(spec_df_re * spec_df_re + spec_df_im * spec_df_im + 1e-14)
            un_st = self.unit_norm_state
            new_un_st = x_abs * (1.0 - self._norm_alpha) + un_st * self._norm_alpha
            denom = tf.sqrt(new_un_st)
            feat_spec_re = spec_df_re / denom  # [B, F_df]
            feat_spec_im = spec_df_im / denom
            self.unit_norm_state.assign(new_un_st)

            # Reshape for encoder
            feat_erb_nhwc = tf.reshape(feat_erb, [-1, 1, E, 1])  # [B,1,E,1]
            feat_spec_nhwc = tf.expand_dims(
                tf.stack([feat_spec_re, feat_spec_im], axis=-1),
                axis=1)  # [B,1,F_df,2]

            # ════════════════════════════════════════════════════════════
            # Frozen NN — only predicts mask + DF coefs (quantizable)
            # ════════════════════════════════════════════════════════════
            args = [
                feat_erb_nhwc, feat_spec_nhwc,
                self.erb_buf, self.df_buf, self.h_enc,
                self.h_erb, self.h_df,
            ]
            if has_convp_buf:
                args.append(self.df_convp_buf)
            result = frozen_cf(*args)

            m_nhwc = result[0]       # [B, 1, E, 1] — ERB mask (sigmoid)
            df_coefs = result[1]     # [B, O, 1, F_df, 2] — DF coefficients
            self.erb_buf.assign(result[2])
            self.df_buf.assign(result[3])
            self.h_enc.assign(result[4])
            self.h_erb.assign(result[5])
            self.h_df.assign(result[6])
            if has_convp_buf:
                self.df_convp_buf.assign(result[7])

            # ════════════════════════════════════════════════════════════
            # ERB mask application — float32, OUTSIDE frozen graph
            # ════════════════════════════════════════════════════════════
            if self.spec_lookahead_buf is not None:
                spec_frame = spec[:, 0, 0, :, :]
                spec_history = tf.concat(
                    [self.spec_lookahead_buf, tf.expand_dims(spec_frame, axis=1)],
                    axis=1)
                delayed_spec = tf.reshape(spec_history[:, 0, :, :], [-1, 1, 1, F, 2])
                self.spec_lookahead_buf.assign(spec_history[:, 1:, :, :])
            else:
                delayed_spec = spec

            m = tf.squeeze(m_nhwc, axis=-1)           # [B, 1, E]
            m_freq = tf.matmul(m, self._erb_inv_fb)   # [B, 1, F]
            m_freq = m_freq[:, :, :, tf.newaxis, tf.newaxis]  # [B,1,F,1,1]

            def apply_mask(spec_in):
                spec_nhwc = tf.transpose(spec_in, [0, 2, 3, 1, 4])
                return tf.transpose(spec_nhwc * m_freq, [0, 3, 1, 2, 4])

            spec_masked = apply_mask(delayed_spec)
            if pad_specf and self.spec_lookahead_buf is not None:
                spec_df_masked = apply_mask(spec)
            else:
                spec_df_masked = spec_masked

            # ════════════════════════════════════════════════════════════
            # Deep filtering — float32, OUTSIDE frozen graph
            # ════════════════════════════════════════════════════════════
            spec_df_cur = spec_df_masked[:, 0, 0, :F_df, :]      # [B, F_df, 2]
            spec_df_cur = tf.expand_dims(spec_df_cur, axis=1)  # [B, 1, F_df, 2]
            window = tf.concat(
                [self.spec_buf, spec_df_cur], axis=1)  # [B, O, F_df, 2]
            self.spec_buf.assign(window[:, 1:, :, :])  # slide ring buffer

            coefs_sq = df_coefs[:, :, 0, :, :]  # [B, O, F_df, 2]
            s_re = window[..., 0]       # [B, O, F_df]
            s_im = window[..., 1]
            c_re = coefs_sq[..., 0]
            c_im = coefs_sq[..., 1]
            # Complex multiply-accumulate over O taps
            out_re = tf.reduce_sum(s_re * c_re - s_im * c_im, axis=1)  # [B, F_df]
            out_im = tf.reduce_sum(s_re * c_im + s_im * c_re, axis=1)
            out = tf.stack([out_re, out_im], axis=-1)  # [B, F_df, 2]

            # Replace low-freq bins with DF output, keep high-freq masked
            if pad_specf:
                spec_out = tf.concat([
                    tf.reshape(out, [-1, 1, 1, F_df, 2]),
                    spec_masked[:, :, :, F_df:, :],
                ], axis=3)  # [B, 1, 1, F, 2]
            else:
                spec_out = tf.concat([
                    tf.reshape(out, [-1, 1, 1, F_df, 2]),
                    spec_df_masked[:, :, :, F_df:, :],
                ], axis=3)

            return spec_out

        @tf.function(input_signature=[])
        def reset_state(self):
            self.erb_buf.assign(tf.zeros_like(self.erb_buf))
            self.df_buf.assign(tf.zeros_like(self.df_buf))
            self.h_enc.assign(tf.zeros_like(self.h_enc))
            self.h_erb.assign(tf.zeros_like(self.h_erb))
            self.h_df.assign(tf.zeros_like(self.h_df))
            self.spec_buf.assign(tf.zeros_like(self.spec_buf))
            if self.spec_lookahead_buf is not None:
                self.spec_lookahead_buf.assign(tf.zeros_like(self.spec_lookahead_buf))
            if has_convp_buf:
                self.df_convp_buf.assign(tf.zeros_like(self.df_convp_buf))
            self.erb_norm_state.assign(
                tf.constant(self._erb_norm_init, dtype=tf.float32))
            self.unit_norm_state.assign(
                tf.constant(self._unit_norm_init, dtype=tf.float32))
            return tf.constant(0, dtype=tf.int32)

    wrapper = _StatefulWrapper()
    return wrapper


def build_tflite_nn_module(model):
    """Build a NN-only TFLite module for quantized export.

    Unlike build_tflite_module() which creates a full-pipeline wrapper
    (spec → enhanced_spec), this creates a pure NN module:
      Input:  feat_erb [B,1,E,1], feat_spec [B,1,F_df,2]  (pre-computed)
      Output: mask [B,1,E,1], df_coefs [B,O,1,F_df,2]

    The caller must handle:
      - Feature extraction: a²+b², ERB, log10, √, norms → feat_erb, feat_spec
      - Mask application: spec × expand(mask)
      - Deep filtering: complex MAC with ring buffer

    This is critical for int8/int16x8 quantization because:
      - The frozen NN graph contains ONLY convs/GRUs/linear layers
      - No raw spectrum values (0..66000), no sqrt, no log10
      - The quantizer calibrates on narrow-range normalized features (~±3)
      - The representative dataset yields (feat_erb, feat_spec) not raw spec
    """
    from tensorflow.python.framework.convert_to_constants import (
        convert_variables_to_constants_v2,
    )

    B = model.batch_size
    F = model.freq_bins
    E = model.nb_erb
    F_df = model.nb_df
    df_order = model.df_order
    conv_ch = model.conv_ch
    emb_hidden_dim = model.emb_hidden_dim
    df_hidden_dim = model.df_hidden_dim
    df_num_layers = model.df_num_layers
    emb_num_layers = model.emb_num_layers
    gru_groups = model.gru_groups
    gru_type = model.gru_type
    df_pathway_kernel_size_t = model.df_pathway_kernel_size_t
    has_convp_buf = df_pathway_kernel_size_t > 1

    if gru_type == "grouped":
        h_per_group = emb_hidden_dim // gru_groups
        df_h_per_group = df_hidden_dim // gru_groups
        h_enc_shape = [1 * gru_groups, B, h_per_group]
        h_erb_shape = [(emb_num_layers - 1) * gru_groups, B, h_per_group]
        h_df_shape = [df_num_layers * gru_groups, B, df_h_per_group]
    else:
        h_enc_shape = [1, B, emb_hidden_dim]
        h_erb_shape = [emb_num_layers - 1, B, emb_hidden_dim]
        h_df_shape = [df_num_layers, B, df_hidden_dim]

    # ── 1. Freeze the NN graph ──
    input_specs = [
        tf.TensorSpec([B, 1, E, 1], tf.float32),
        tf.TensorSpec([B, 1, F_df, 2], tf.float32),
        tf.TensorSpec([B, 2, E, 1], tf.float32),
        tf.TensorSpec([B, 2, F_df, 2], tf.float32),
        tf.TensorSpec(h_enc_shape, tf.float32),
        tf.TensorSpec(h_erb_shape, tf.float32),
        tf.TensorSpec(h_df_shape, tf.float32),
    ]
    if has_convp_buf:
        input_specs.append(
            tf.TensorSpec([B, df_pathway_kernel_size_t - 1, F_df, conv_ch],
                          tf.float32))

    stateless_cf = tf.function(model.forward_stateless).get_concrete_function(
        *input_specs)
    print(f"  NN-only CF captured {len(stateless_cf.variables)} variables")

    frozen_cf = convert_variables_to_constants_v2(stateless_cf)
    print(f"  NN-only frozen graph: {len(frozen_cf.graph.as_graph_def().node)} ops")

    # ── 2. Build thin NN-only wrapper ──
    class _NNOnlyWrapper(tf.Module):
        def __init__(self):
            super().__init__()
            self.erb_buf = tf.Variable(
                tf.zeros([B, 2, E, 1]), trainable=False, name="state/erb_buf")
            self.df_buf = tf.Variable(
                tf.zeros([B, 2, F_df, 2]), trainable=False, name="state/df_buf")
            self.h_enc = tf.Variable(
                tf.zeros(h_enc_shape), trainable=False, name="state/h_enc")
            self.h_erb = tf.Variable(
                tf.zeros(h_erb_shape), trainable=False, name="state/h_erb")
            self.h_df = tf.Variable(
                tf.zeros(h_df_shape), trainable=False, name="state/h_df")
            if has_convp_buf:
                self.df_convp_buf = tf.Variable(
                    tf.zeros([B, df_pathway_kernel_size_t - 1, F_df, conv_ch]),
                    trainable=False, name="state/df_convp_buf")

        @tf.function(input_signature=[
            tf.TensorSpec([B, 1, E, 1], tf.float32, name="feat_erb"),
            tf.TensorSpec([B, 1, F_df, 2], tf.float32, name="feat_spec"),
        ])
        def forward(self, feat_erb, feat_spec):
            args = [
                feat_erb, feat_spec,
                self.erb_buf, self.df_buf, self.h_enc,
                self.h_erb, self.h_df,
            ]
            if has_convp_buf:
                args.append(self.df_convp_buf)
            result = frozen_cf(*args)

            m_nhwc = result[0]
            df_coefs = result[1]
            self.erb_buf.assign(result[2])
            self.df_buf.assign(result[3])
            self.h_enc.assign(result[4])
            self.h_erb.assign(result[5])
            self.h_df.assign(result[6])
            if has_convp_buf:
                self.df_convp_buf.assign(result[7])
            return m_nhwc, df_coefs

        @tf.function(input_signature=[])
        def reset_state(self):
            self.erb_buf.assign(tf.zeros_like(self.erb_buf))
            self.df_buf.assign(tf.zeros_like(self.df_buf))
            self.h_enc.assign(tf.zeros_like(self.h_enc))
            self.h_erb.assign(tf.zeros_like(self.h_erb))
            self.h_df.assign(tf.zeros_like(self.h_df))
            if has_convp_buf:
                self.df_convp_buf.assign(tf.zeros_like(self.df_convp_buf))
            return tf.constant(0, dtype=tf.int32)

    wrapper = _NNOnlyWrapper()
    return wrapper
