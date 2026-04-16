"""Stateful streaming DeepFilterNet2 — processes one frame at a time.

All GRU hidden states and the DF spectrogram ring buffer are explicit
inputs/outputs, so this model can be used frame-by-frame in TFLite.

Convention: T=1 for all tensors. States flow in and out as flat tensors.
"""

import tensorflow as tf
import numpy as np

from deepfilternet2_tf import (
    DEFAULT_PARAMS,
    Conv2dNormAct,
    ConvTranspose2dNormAct,
    GroupedLinearEinsum,
    DfOutputReshapeMF,
    _gcd,
    _get_activation,
)


# ==============================================================================
# Conv buffer helper for streaming
# ==============================================================================

def _conv_step(conv_layer, x_frame, buf):
    """Single-frame conv with explicit state buffer for causal context.

    For convs with temporal kernel > 1, we need previous input frames.
    Prepends the buffer to the current frame, runs the conv (which internally
    zero-pads via CausalPad2D), and returns only the last output frame.

    Args:
        conv_layer: Conv2dNormAct with kernel_size[0] > 1
        x_frame: [B, 1, F, C] current input frame
        buf: [B, time_pad, F, C] previous input frames
    Returns:
        out_frame: [B, 1, F', C'] single output frame
        new_buf: [B, time_pad, F, C] updated buffer
    """
    x = tf.concat([buf, x_frame], axis=1)  # [B, time_pad+1, F, C]
    new_buf = x[:, 1:, :, :]  # drop oldest, keep last time_pad frames
    out = conv_layer(x, training=False)
    return out[:, -1:, :, :], new_buf


# ==============================================================================
# Stateful GRU helpers (single-step, explicit state in/out)
# ==============================================================================

class GroupedGRULayerStep(tf.keras.layers.Layer):
    """Single-step grouped GRU layer with explicit state."""

    def __init__(self, input_size, hidden_size, groups=1, **kwargs):
        super().__init__(**kwargs)
        assert input_size % groups == 0
        assert hidden_size % groups == 0
        self.groups = groups
        self.in_per_group = input_size // groups
        self.h_per_group = hidden_size // groups
        self.hidden_size = hidden_size
        self.cells = [
            tf.keras.layers.GRUCell(self.h_per_group, reset_after=True)
            for _ in range(groups)
        ]

    def call(self, x_t, h):
        """
        x_t: (B, I)  — single timestep input
        h:   (B, H)  — full hidden state (all groups concatenated)
        Returns: output (B, H), new_h (B, H)
        """
        outputs = []
        new_states = []
        for i, cell in enumerate(self.cells):
            s_in = i * self.in_per_group
            e_in = s_in + self.in_per_group
            s_h = i * self.h_per_group
            e_h = s_h + self.h_per_group
            h_new, _ = cell(x_t[:, s_in:e_in], [h[:, s_h:e_h]])
            outputs.append(h_new)
            new_states.append(h_new)
        return tf.concat(outputs, axis=-1), tf.concat(new_states, axis=-1)


class GroupedGRUStep(tf.keras.layers.Layer):
    """Multi-layer grouped GRU, single-step, explicit state in/out."""

    def __init__(self, input_size, hidden_size, num_layers=1, groups=1,
                 shuffle=True, add_outputs=False, **kwargs):
        super().__init__(**kwargs)
        self.groups = groups
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.h_per_group = hidden_size // groups
        if groups == 1:
            shuffle = False
        self.shuffle = shuffle
        self.add_outputs = add_outputs

        self.gru_layers = [GroupedGRULayerStep(input_size, hidden_size, groups)]
        for _ in range(1, num_layers):
            self.gru_layers.append(GroupedGRULayerStep(hidden_size, hidden_size, groups))

    @property
    def state_size(self):
        """Total hidden state size across all layers: num_layers * hidden_size."""
        return self.num_layers * self.hidden_size

    def call(self, x_t, states):
        """
        x_t:    (B, I)  — single timestep
        states: (B, num_layers * H) — concatenated hidden states
        Returns: output (B, H), new_states (B, num_layers * H)
        """
        b = tf.shape(x_t)[0]
        output = tf.zeros([b, self.hidden_size])
        new_layer_states = []

        for i, gru in enumerate(self.gru_layers):
            h_i = states[:, i * self.hidden_size:(i + 1) * self.hidden_size]
            x_t, h_new = gru(x_t, h_i)
            new_layer_states.append(h_new)
            if self.shuffle and i < self.num_layers - 1:
                x_t = tf.reshape(x_t, [b, self.h_per_group, self.groups])
                x_t = tf.transpose(x_t, [0, 2, 1])
                x_t = tf.reshape(x_t, [b, self.hidden_size])
            if self.add_outputs:
                output = output + x_t
            else:
                output = x_t

        return output, tf.concat(new_layer_states, axis=-1)


class SqueezedGRUStep(tf.keras.layers.Layer):
    """Single-step SqueezedGRU with explicit state."""

    def __init__(self, input_size, hidden_size, output_size=None,
                 num_layers=1, linear_groups=8, gru_skip_op=None,
                 linear_act="relu", **kwargs):
        super().__init__(**kwargs)
        act = tf.keras.layers.ReLU() if linear_act == "relu" else tf.keras.layers.Lambda(lambda x: x)
        self.linear_in_proj = GroupedLinearEinsum(input_size, hidden_size, linear_groups)
        self.linear_in_act = act
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Single-step GRU cells
        self.cells = [
            tf.keras.layers.GRUCell(hidden_size, reset_after=True)
            for _ in range(num_layers)
        ]
        self.gru_skip = gru_skip_op
        if output_size is not None:
            self.linear_out_proj = GroupedLinearEinsum(hidden_size, output_size, linear_groups)
            self.linear_out_act = tf.keras.layers.ReLU() if linear_act == "relu" else tf.keras.layers.Lambda(lambda x: x)
        else:
            self.linear_out_proj = None

    @property
    def state_size(self):
        return self.num_layers * self.hidden_size

    def call(self, x_t, states):
        """
        x_t: (B, I), states: (B, num_layers * H)
        Returns: output (B, O), new_states (B, num_layers * H)
        """
        x_in = self.linear_in_act(self.linear_in_proj(x_t))
        out = x_in
        new_states = []
        for i, cell in enumerate(self.cells):
            h_i = states[:, i * self.hidden_size:(i + 1) * self.hidden_size]
            out, _ = cell(out, [h_i])
            new_states.append(out)
        if self.gru_skip is not None:
            out = out + self.gru_skip(x_in)
        if self.linear_out_proj is not None:
            out = self.linear_out_act(self.linear_out_proj(out))
        return out, tf.concat(new_states, axis=-1)


# ==============================================================================
# One-step DF operator (with ring buffer state)
# ==============================================================================

class DfOpOneStep(tf.keras.layers.Layer):
    """Single-step deep filtering with explicit spec ring buffer.

    The ring buffer holds the last df_order frames of the DF-bin spectrum.
    """

    def __init__(self, num_freqs, frame_size, lookahead=0, **kwargs):
        super().__init__(**kwargs)
        self.num_freqs = num_freqs
        self.frame_size = frame_size
        self.lookahead = lookahead

    def call(self, spec_frame, coefs_frame, spec_buf):
        """
        spec_frame: [B, 1, 1, F, 2]    — current full-band spectrum frame
        coefs_frame: [B, O, 1, Fdf, 2]  — DF coefs for this frame
        spec_buf:   [B, 1, O, Fdf, 2]   — ring buffer of last O frames (DF bins only)

        Returns:
            enhanced_frame: [B, 1, 1, F, 2]
            new_spec_buf:   [B, 1, O, Fdf, 2]
        """
        # Update ring buffer: roll left, insert new frame at end
        new_entry = spec_frame[:, :, :, :self.num_freqs, :]  # [B, 1, 1, Fdf, 2]
        new_spec_buf = tf.concat([spec_buf[:, :, 1:, :, :], new_entry], axis=2)

        # Complex multiply: buf * coefs, summed over O
        # buf: [B, 1, O, Fdf, 2], coefs: [B, O, 1, Fdf, 2] -> need same layout
        # Reshape coefs: [B, O, 1, F, 2] -> [B, 1, O, F, 2]
        c = tf.transpose(coefs_frame, [0, 2, 1, 3, 4])  # [B, 1, O, Fdf, 2]

        s_re = new_spec_buf[..., 0]  # [B, 1, O, Fdf]
        s_im = new_spec_buf[..., 1]
        c_re = c[..., 0]
        c_im = c[..., 1]

        out_re = tf.reduce_sum(s_re * c_re - s_im * c_im, axis=2)  # [B, 1, Fdf]
        out_im = tf.reduce_sum(s_re * c_im + s_im * c_re, axis=2)
        spec_f = tf.stack([out_re, out_im], axis=-1)  # [B, 1, Fdf, 2]
        spec_f = tf.expand_dims(spec_f, 2)  # [B, 1, 1, Fdf, 2]

        # Assign DF bins, keep high freqs from original
        spec_hi = spec_frame[:, :, :, self.num_freqs:, :]
        enhanced = tf.concat([spec_f, spec_hi], axis=3)

        return enhanced, new_spec_buf


# ==============================================================================
# Stateful single-frame Encoder
# ==============================================================================

class EncoderStep(tf.keras.layers.Layer):
    """Single-frame encoder with explicit GRU state."""

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

        # gru_type=squeeze: GroupedLinearEinsum + ReLU
        self.df_fc_emb = GroupedLinearEinsum(
            ch * nb_df // 2, self.emb_in_dim, groups=p["lin_groups"]
        )
        self.df_fc_emb_act = tf.keras.layers.ReLU()

        self.enc_concat = p["enc_concat"]
        emb_in = self.emb_in_dim * 2 if self.enc_concat else self.emb_in_dim

        # gru_type=squeeze: SqueezedGRUStep
        self.emb_gru = SqueezedGRUStep(
            emb_in, self.emb_out_dim, output_size=None,
            num_layers=1, linear_groups=p["lin_groups"],
        )

        self.lsnr_dense = tf.keras.layers.Dense(1)
        self.lsnr_scale = float(p["lsnr_max"] - p["lsnr_min"])
        self.lsnr_offset = float(p["lsnr_min"])

    def call(self, feat_erb, feat_spec, enc_state, erb_conv0_buf, df_conv0_buf):
        """
        feat_erb:  [B, 1, 1, E]
        feat_spec: [B, 2, 1, Fc]
        enc_state: [B, state_size]
        erb_conv0_buf: [B, 2, E, 1] — previous 2 input frames for erb_conv0
        df_conv0_buf:  [B, 2, Fc, 2] — previous 2 input frames for df_conv0
        Returns: e0..e3, emb, c0, lsnr, new_enc_state, new_erb_conv0_buf, new_df_conv0_buf
        """
        erb = tf.transpose(feat_erb, [0, 2, 3, 1])    # [B, 1, E, 1]
        spec = tf.transpose(feat_spec, [0, 2, 3, 1])  # [B, 1, Fc, 2]

        e0, erb_conv0_buf = _conv_step(self.erb_conv0, erb, erb_conv0_buf)
        e1 = self.erb_conv1(e0, training=False)
        e2 = self.erb_conv2(e1, training=False)
        e3 = self.erb_conv3(e2, training=False)

        c0, df_conv0_buf = _conv_step(self.df_conv0, spec, df_conv0_buf)
        c1 = self.df_conv1(c0, training=False)

        c1_s = tf.shape(c1)
        cemb = tf.reshape(c1, [c1_s[0], -1])       # [B, Fc/2 * ch]
        cemb = self.df_fc_emb_act(self.df_fc_emb(cemb))  # [B, emb_in]

        e3_s = tf.shape(e3)
        emb = tf.reshape(e3, [e3_s[0], -1])          # [B, F/4 * ch]

        if self.enc_concat:
            emb = tf.concat([emb, cemb], axis=-1)
        else:
            emb = emb + cemb

        emb, new_enc_state = self.emb_gru(emb, enc_state)  # (B, H), (B, state)

        lsnr = tf.sigmoid(self.lsnr_dense(emb)) * self.lsnr_scale + self.lsnr_offset
        return e0, e1, e2, e3, emb, c0, lsnr, new_enc_state, erb_conv0_buf, df_conv0_buf


# ==============================================================================
# Stateful single-frame ERB Decoder
# ==============================================================================

class ErbDecoderStep(tf.keras.layers.Layer):
    """Single-frame ERB decoder with explicit GRU state."""

    def __init__(self, p=None, **kwargs):
        super().__init__(**kwargs)
        if p is None:
            p = DEFAULT_PARAMS
        ch = p["conv_ch"]
        nb_erb = p["nb_erb"]
        kn = tuple(p["conv_kernel"])

        self.emb_out_dim = p["emb_hidden_dim"]
        self.ch = ch

        # gru_type=squeeze: SqueezedGRUStep with output_size and gru_skip=Identity
        self.emb_gru = SqueezedGRUStep(
            self.emb_out_dim, self.emb_out_dim,
            output_size=ch * nb_erb // 4,
            num_layers=p["emb_num_layers"] - 1,
            linear_groups=p["lin_groups"],
            gru_skip_op=tf.keras.layers.Lambda(lambda x: x),  # Identity
        )

        self.conv3p = Conv2dNormAct(ch, ch, kernel_size=1, bias=False, separable=True)
        self.convt3 = Conv2dNormAct(ch, ch, kernel_size=kn, bias=False, separable=True)
        self.conv2p = Conv2dNormAct(ch, ch, kernel_size=1, bias=False, separable=True)
        self.convt2 = ConvTranspose2dNormAct(ch, ch, kernel_size=kn, fstride=2, bias=False, separable=True)
        self.conv1p = Conv2dNormAct(ch, ch, kernel_size=1, bias=False, separable=True)
        self.convt1 = ConvTranspose2dNormAct(ch, ch, kernel_size=kn, fstride=2, bias=False, separable=True)
        self.conv0p = Conv2dNormAct(ch, ch, kernel_size=1, bias=False, separable=True)
        self.conv0_out = Conv2dNormAct(ch, 1, kernel_size=kn, bias=False,
                                        separable=True, activation="sigmoid")

    def call(self, emb, e3, e2, e1, e0, erb_dec_state):
        """
        emb: [B, H], e0..e3: [B, 1, F_i, ch]
        erb_dec_state: [B, state_size]
        Returns: m [B, 1, 1, E], new_state
        """
        f8 = e3.shape[2]

        # SqueezedGRUStep already includes output projection
        emb, new_state = self.emb_gru(emb, erb_dec_state)

        emb = tf.reshape(emb, [tf.shape(emb)[0], 1, f8, self.ch])

        e3 = self.convt3(self.conv3p(e3, training=False) + emb, training=False)
        e2 = self.convt2(self.conv2p(e2, training=False) + e3, training=False)
        e1 = self.convt1(self.conv1p(e1, training=False) + e2, training=False)
        m = self.conv0_out(self.conv0p(e0, training=False) + e1, training=False)

        # [B, 1, E, 1] -> [B, 1, 1, E]
        m = tf.transpose(m, [0, 3, 1, 2])
        return m, new_state


# ==============================================================================
# Stateful single-frame DF Decoder
# ==============================================================================

class DfDecoderStep(tf.keras.layers.Layer):
    """Single-frame DF coefficient decoder with explicit GRU state."""

    def __init__(self, p=None, out_channels=-1, **kwargs):
        super().__init__(**kwargs)
        if p is None:
            p = DEFAULT_PARAMS
        ch = p["conv_ch"]
        self.df_n_hidden = p["df_hidden_dim"]
        self.df_order = p["df_order"]
        self.df_bins = p["nb_df"]
        self.df_out_ch = out_channels if out_channels > 0 else p["df_order"] * 2

        kt = p["df_pathway_kernel_size_t"]
        self.df_convp = Conv2dNormAct(
            ch, self.df_out_ch, kernel_size=(kt, 1), fstride=1, bias=False, separable=True)

        # gru_type=squeeze: SqueezedGRUStep with gru_skip=Identity
        self.df_gru = SqueezedGRUStep(
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

    def call(self, emb, c0, df_dec_state, df_convp_buf):
        """
        emb: [B, H], c0: [B, 1, Fc, ch]
        df_dec_state: [B, state_size]
        df_convp_buf: [B, 4, Fc, ch] — previous 4 input frames for df_convp
        Returns: coefs [B, O, 1, F, 2], alpha [B, 1], new_state, new_df_convp_buf
        """
        c, new_state = self.df_gru(emb, df_dec_state)
        if self.df_skip is not None:
            c = c + self.df_skip(emb)

        c0, df_convp_buf = _conv_step(self.df_convp, c0, df_convp_buf)
        # Squeeze T=1: [B, 1, Fc, ch] -> [B, Fc*ch] ... no, keep for reshape
        c0 = tf.reshape(c0, [tf.shape(c0)[0], self.df_bins, self.df_out_ch])  # [B, F, O*2]

        alpha = tf.sigmoid(self.df_fc_a(c))  # [B, 1]
        c_out = tf.tanh(self.df_out_proj(c))  # [B, F*O*2]
        c_out = tf.reshape(c_out, [tf.shape(c_out)[0], self.df_bins, self.df_out_ch])  # [B, F, O*2]
        c_out = c_out + c0

        # Reshape: [B, F, O*2] -> [B, 1, F, O*2] -> out_transform expects [B, T, F, O*2]
        c_out = tf.expand_dims(c_out, 1)
        c_out = self.out_transform(c_out)  # [B, O, 1, F, 2]
        return c_out, alpha, new_state, df_convp_buf


# ==============================================================================
# Full Stateful Streaming Model
# ==============================================================================

class DfNetStreaming(tf.keras.Model):
    """Frame-by-frame stateful DeepFilterNet2 — neural network only.

    Features in, mask + DF coefficients out. No spectrum processing.
    All hidden states are explicit inputs/outputs. Process one frame at a time.

    State layout (concatenated into one flat tensor):
        [enc_gru | erb_dec_gru | df_dec_gru | erb_conv0_buf | df_conv0_buf | df_convp_buf]

    Individual sizes (with default params, conv_ch=64):
        enc_gru:       SqueezedGRU(1 layer)  = 1 * 256         =   256
        erb_dec_gru:   SqueezedGRU(2 layers) = 2 * 256         =   512
        df_dec_gru:    SqueezedGRU(2 layers) = 2 * 256         =   512
        erb_conv0_buf: 2 * nb_erb * 1        = 2 * 32 * 1      =    64
        df_conv0_buf:  2 * nb_df * 2         = 2 * 96 * 2      =   384
        df_convp_buf:  4 * nb_df * conv_ch   = 4 * 96 * 64     = 24576
        -----------------------------------------------------------
        Total:                                                 = 26304
    """

    def __init__(self, p=None, **kwargs):
        super().__init__(**kwargs)
        if p is None:
            p = DEFAULT_PARAMS
        self.p = p
        self.nb_df = p["nb_df"]
        self.nb_erb = p["nb_erb"]
        self.conv_ch = p["conv_ch"]
        self.df_order = p["df_order"]

        self.enc = EncoderStep(p)
        self.erb_dec = ErbDecoderStep(p)
        n_ch_out = p["df_order"] * 2
        self.df_dec = DfDecoderStep(p, out_channels=n_ch_out)

        # GRU state sizes
        self.enc_state_size = 1 * p["emb_hidden_dim"]
        self.erb_dec_state_size = (p["emb_num_layers"] - 1) * p["emb_hidden_dim"]
        self.df_dec_state_size = p["df_num_layers"] * p["df_hidden_dim"]

        # Conv buffer sizes (kernel_t - 1 frames of input)
        kn_inp_t = p["conv_kernel_inp"][0]  # 3
        kt = p["df_pathway_kernel_size_t"]  # 5
        self.erb_conv0_buf_size = (kn_inp_t - 1) * p["nb_erb"] * 1   # 64
        self.df_conv0_buf_size = (kn_inp_t - 1) * p["nb_df"] * 2     # 384
        self.df_convp_buf_size = (kt - 1) * p["nb_df"] * p["conv_ch"]  # 24576

        # Buffer shapes for reshape
        self.erb_conv0_buf_shape = (kn_inp_t - 1, p["nb_erb"], 1)
        self.df_conv0_buf_shape = (kn_inp_t - 1, p["nb_df"], 2)
        self.df_convp_buf_shape = (kt - 1, p["nb_df"], p["conv_ch"])

        self.total_state_size = (
            self.enc_state_size + self.erb_dec_state_size +
            self.df_dec_state_size +
            self.erb_conv0_buf_size + self.df_conv0_buf_size +
            self.df_convp_buf_size
        )

    def get_initial_state(self, batch_size=1):
        """Return zero-initialized state tensor."""
        return tf.zeros([batch_size, self.total_state_size])

    def _split_state(self, state):
        """Split flat state into GRU states + conv buffers."""
        idx = 0
        enc_state = state[:, idx:idx + self.enc_state_size]
        idx += self.enc_state_size
        erb_dec_state = state[:, idx:idx + self.erb_dec_state_size]
        idx += self.erb_dec_state_size
        df_dec_state = state[:, idx:idx + self.df_dec_state_size]
        idx += self.df_dec_state_size

        erb_conv0_buf = tf.reshape(
            state[:, idx:idx + self.erb_conv0_buf_size],
            [-1, *self.erb_conv0_buf_shape])
        idx += self.erb_conv0_buf_size
        df_conv0_buf = tf.reshape(
            state[:, idx:idx + self.df_conv0_buf_size],
            [-1, *self.df_conv0_buf_shape])
        idx += self.df_conv0_buf_size
        df_convp_buf = tf.reshape(
            state[:, idx:idx + self.df_convp_buf_size],
            [-1, *self.df_convp_buf_shape])

        return (enc_state, erb_dec_state, df_dec_state,
                erb_conv0_buf, df_conv0_buf, df_convp_buf)

    def _merge_state(self, enc_state, erb_dec_state, df_dec_state,
                     erb_conv0_buf, df_conv0_buf, df_convp_buf):
        """Merge GRU states + conv buffers back into flat state."""
        return tf.concat([
            enc_state, erb_dec_state, df_dec_state,
            tf.reshape(erb_conv0_buf, [tf.shape(enc_state)[0], -1]),
            tf.reshape(df_conv0_buf, [tf.shape(enc_state)[0], -1]),
            tf.reshape(df_convp_buf, [tf.shape(enc_state)[0], -1]),
        ], axis=-1)

    def call(self, feat_erb_frame, feat_spec_frame, state):
        """Process a single frame.

        Args:
            feat_erb_frame:  [B, 1, 1, E]       — ERB features for this frame
            feat_spec_frame: [B, 1, 1, Fc, 2]   — complex spec features for this frame
            state:           [B, total_state_size] — packed hidden state

        Returns:
            m:         [B, 1, 1, E]     ERB gain mask (0–1)
            lsnr:      [B, 1]           local SNR estimate (dB)
            df_coefs:  [B, O, 1, Fc, 2] deep filtering coefficients
            alpha:     [B, 1]           DF blending factor (0–1)
            new_state: [B, total_state_size]
        """
        (enc_state, erb_dec_state, df_dec_state,
         erb_conv0_buf, df_conv0_buf, df_convp_buf) = self._split_state(state)

        # feat_spec: [B, 1, 1, Fc, 2] -> [B, 2, 1, Fc]
        fs = tf.squeeze(feat_spec_frame, axis=1)  # [B, 1, Fc, 2]
        fs = tf.transpose(fs, [0, 3, 1, 2])       # [B, 2, 1, Fc]

        (e0, e1, e2, e3, emb, c0, lsnr, new_enc_state,
         erb_conv0_buf, df_conv0_buf) = self.enc(
            feat_erb_frame, fs, enc_state, erb_conv0_buf, df_conv0_buf)

        m, new_erb_dec_state = self.erb_dec(emb, e3, e2, e1, e0, erb_dec_state)

        df_coefs, alpha, new_df_dec_state, df_convp_buf = self.df_dec(
            emb, c0, df_dec_state, df_convp_buf)

        new_state = self._merge_state(
            new_enc_state, new_erb_dec_state, new_df_dec_state,
            erb_conv0_buf, df_conv0_buf, df_convp_buf)

        return m, lsnr, df_coefs, alpha, new_state


def copy_weights_from_batch_model(batch_model, streaming_model):
    """Copy weights from stateless DfNet to stateful DfNetStreaming.

    Both models use the same underlying layer classes (Conv2dNormAct, etc.),
    but GRUs differ (GroupedGRU vs GroupedGRUStep). The GRU cells share
    the same weight structure, so we copy cell-by-cell.
    """

    def _copy_conv_block(src, dst):
        """Copy Conv2dNormAct or ConvTranspose2dNormAct weights."""
        if hasattr(src, 'group_convs') and src.groups > 1:
            for s_conv, d_conv in zip(src.group_convs, dst.group_convs):
                d_conv.set_weights(s_conv.get_weights())
        else:
            if hasattr(src, 'conv') and hasattr(dst, 'conv'):
                dst.conv.set_weights(src.conv.get_weights())
        if src.pw_conv is not None and dst.pw_conv is not None:
            dst.pw_conv.set_weights(src.pw_conv.get_weights())
        if src.use_norm and dst.use_norm:
            dst.norm.set_weights(src.norm.get_weights())

    def _copy_squeezed_gru_to_step(src_sgru, dst_sgru):
        """Copy SqueezedGRU -> SqueezedGRUStep weights (cell-by-cell)."""
        # linear_in
        dst_sgru.linear_in_proj.set_weights(src_sgru.linear_in_proj.get_weights())
        # GRU cells: batch GRU.cell is a GRUCell, streaming uses GRUCell directly
        for s_gru, d_cell in zip(src_sgru.grus, dst_sgru.cells):
            d_cell.set_weights(s_gru.cell.get_weights())
        # linear_out (if present)
        if src_sgru.linear_out_proj is not None and dst_sgru.linear_out_proj is not None:
            dst_sgru.linear_out_proj.set_weights(src_sgru.linear_out_proj.get_weights())

    def _copy_dense(src, dst):
        dst.set_weights(src.get_weights())

    def _copy_grouped_linear_einsum(src, dst):
        dst.set_weights(src.get_weights())

    # --- Encoder ---
    s_enc, d_enc = batch_model.enc, streaming_model.enc
    _copy_conv_block(s_enc.erb_conv0, d_enc.erb_conv0)
    _copy_conv_block(s_enc.erb_conv1, d_enc.erb_conv1)
    _copy_conv_block(s_enc.erb_conv2, d_enc.erb_conv2)
    _copy_conv_block(s_enc.erb_conv3, d_enc.erb_conv3)
    _copy_conv_block(s_enc.df_conv0, d_enc.df_conv0)
    _copy_conv_block(s_enc.df_conv1, d_enc.df_conv1)
    _copy_grouped_linear_einsum(s_enc.df_fc_emb, d_enc.df_fc_emb)
    _copy_squeezed_gru_to_step(s_enc.emb_gru, d_enc.emb_gru)
    _copy_dense(s_enc.lsnr_dense, d_enc.lsnr_dense)

    # --- ERB Decoder ---
    s_dec, d_dec = batch_model.erb_dec, streaming_model.erb_dec
    _copy_squeezed_gru_to_step(s_dec.emb_gru, d_dec.emb_gru)
    _copy_conv_block(s_dec.conv3p, d_dec.conv3p)
    _copy_conv_block(s_dec.convt3, d_dec.convt3)
    _copy_conv_block(s_dec.conv2p, d_dec.conv2p)
    _copy_conv_block(s_dec.convt2, d_dec.convt2)
    _copy_conv_block(s_dec.conv1p, d_dec.conv1p)
    _copy_conv_block(s_dec.convt1, d_dec.convt1)
    _copy_conv_block(s_dec.conv0p, d_dec.conv0p)
    _copy_conv_block(s_dec.conv0_out, d_dec.conv0_out)

    # --- DF Decoder ---
    s_df, d_df = batch_model.df_dec, streaming_model.df_dec
    _copy_conv_block(s_df.df_convp, d_df.df_convp)
    _copy_squeezed_gru_to_step(s_df.df_gru, d_df.df_gru)
    if s_df.df_skip is not None and d_df.df_skip is not None:
        if hasattr(s_df.df_skip, 'get_weights'):
            d_df.df_skip.set_weights(s_df.df_skip.get_weights())
    _copy_grouped_linear_einsum(s_df.df_out_proj, d_df.df_out_proj)
    _copy_dense(s_df.df_fc_a, d_df.df_fc_a)
