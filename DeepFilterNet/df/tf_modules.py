"""
TensorFlow re-implementation of DeepFilterNet2 modules.

Phase 1 of PyTorch -> TF -> TFLite conversion.
Each module mirrors its PyTorch counterpart in modules.py / deepfilternet2.py.

Rules (from .github/instructions/pytorch-to-tflite.instructions.md):
- No in-place ops
- No complex tensor types (real/imag as [..., 2])
- No dynamic shapes — all dims fixed at export
- GRU: use GRUCell + Python for-loop, NOT tf.keras.layers.GRU
- BatchNorm: fused=False
- Conv weights: PT (O, I, kH, kW) -> TF (kH, kW, I, O)
- DepthwiseConv2D: PT (Ch, 1, kH, kW) -> TF (kH, kW, Ch, 1)
- Activations: PT (B, C, T, F) -> TF (B, T, F, C)
- GRU gates: PT [reset, update, new] -> TF [update, reset, new]
"""

import numpy as np
import tensorflow as tf


def shift_future_nhwc(x, lookahead):
    if lookahead <= 0:
        return x
    zeros = tf.zeros_like(x[:, :lookahead, ...])
    return tf.concat([x[:, lookahead:, ...], zeros], axis=1)


def shift_future_spec(x, lookahead):
    if lookahead <= 0:
        return x
    zeros = tf.zeros_like(x[:, :, :lookahead, ...])
    return tf.concat([x[:, :, lookahead:, ...], zeros], axis=2)


# ============================================================================
# Weight conversion utilities
# ============================================================================

def convert_conv2d_weight(pt_w):
    """PT (O, I, kH, kW) -> TF (kH, kW, I, O)"""
    return np.transpose(pt_w, (2, 3, 1, 0))


def convert_depthwise_conv2d_weight(pt_w):
    """PT (Ch, 1, kH, kW) -> TF (kH, kW, Ch, 1)"""
    return np.transpose(pt_w, (2, 3, 0, 1))


def convert_conv_transpose2d_weight(pt_w):
    """PT ConvTranspose2d (I, O, kH, kW) -> TF Conv2DTranspose (kH, kW, O, I)"""
    return np.transpose(pt_w, (2, 3, 1, 0))


def convert_linear_weight(pt_w):
    """PT Linear (O, I) -> TF Dense (I, O)"""
    return pt_w.T


def reorder_gru_gates(w, hidden_size):
    """Reorder GRU gate weights: PT [reset, update, new] -> TF [update, reset, new].

    Works for both weight matrices and bias vectors.
    """
    if w.ndim == 1:
        r, z, n = np.split(w, 3)
        return np.concatenate([z, r, n])
    elif w.ndim == 2:
        r, z, n = np.split(w, 3, axis=0)
        return np.concatenate([z, r, n], axis=0)
    else:
        raise ValueError(f"Unexpected weight ndim={w.ndim}")


def convert_gru_weights(pt_state_dict, prefix, hidden_size, layer_suffix=""):
    """Convert a single nn.GRU layer's weights from PT to TF GRUCell format.

    PT GRU stores:
        {prefix}.weight_ih_l0: [3*H, I]
        {prefix}.weight_hh_l0: [3*H, H]
        {prefix}.bias_ih_l0:   [3*H]
        {prefix}.bias_hh_l0:   [3*H]

    TF GRUCell expects (set_weights order):
        kernel:            [I, 3*H]
        recurrent_kernel:  [H, 3*H]
        bias:              [2, 3*H]

    Gate reorder: PT [r, z, n] -> TF [z, r, n]

    Args:
        layer_suffix: suffix for multi-layer GRU, e.g. "_l0", "_l1".
                      If empty, defaults to "_l0".
    """
    suffix = layer_suffix if layer_suffix else "_l0"
    w_ih = pt_state_dict[f"{prefix}.weight_ih{suffix}"]  # [3H, I]
    w_hh = pt_state_dict[f"{prefix}.weight_hh{suffix}"]  # [3H, H]
    b_ih = pt_state_dict[f"{prefix}.bias_ih{suffix}"]     # [3H]
    b_hh = pt_state_dict[f"{prefix}.bias_hh{suffix}"]     # [3H]

    # Reorder gates
    w_ih = reorder_gru_gates(w_ih, hidden_size)
    w_hh = reorder_gru_gates(w_hh, hidden_size)
    b_ih = reorder_gru_gates(b_ih, hidden_size)
    b_hh = reorder_gru_gates(b_hh, hidden_size)

    kernel = w_ih.T              # [I, 3H]
    recurrent_kernel = w_hh.T    # [H, 3H]
    bias = np.stack([b_ih, b_hh], axis=0)  # [2, 3H]

    return [kernel, recurrent_kernel, bias]


# ============================================================================
# TF Layers
# ============================================================================

class GroupedLinearTF(tf.keras.layers.Layer):
    """TF equivalent of GroupedLinear (with groups=1, it's just Dense + optional shuffle)."""

    def __init__(self, input_size, hidden_size, groups=1, shuffle=True, **kwargs):
        super().__init__(**kwargs)
        self.groups = groups
        self.input_size_per_group = input_size // groups
        self.hidden_size_per_group = hidden_size // groups
        self.shuffle = shuffle if groups > 1 else False
        self.linears = []
        for i in range(groups):
            self.linears.append(
                tf.keras.layers.Dense(self.hidden_size_per_group, name=f"linear_{i}")
            )

    def call(self, x):
        outputs = []
        for i, layer in enumerate(self.linears):
            start = i * self.input_size_per_group
            end = (i + 1) * self.input_size_per_group
            outputs.append(layer(x[..., start:end]))
        output = tf.concat(outputs, axis=-1)
        if self.shuffle:
            orig_shape = tf.shape(output)
            # Reshape to [..., hidden_size_per_group, groups], transpose, reshape back
            output = tf.reshape(output, [-1, self.hidden_size_per_group, self.groups])
            output = tf.transpose(output, [0, 2, 1])
            output = tf.reshape(output, orig_shape)
        return output

    def load_from_pt(self, pt_state_dict, prefix):
        """Load weights from PyTorch GroupedLinear."""
        pfx = f"{prefix}." if prefix else ""
        for i, layer in enumerate(self.linears):
            w = pt_state_dict[f"{pfx}layers.{i}.weight"]  # [O, I]
            b = pt_state_dict[f"{pfx}layers.{i}.bias"]    # [O]
            layer.set_weights([w.T, b])


class GroupedLinearEinsumTF(tf.keras.layers.Layer):
    """TF equivalent of GroupedLinearEinsum."""

    def __init__(self, input_size, hidden_size, groups=1, **kwargs):
        super().__init__(**kwargs)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.groups = groups
        self.ws = input_size // groups
        self.hs = hidden_size // groups

    def build(self, input_shape):
        self.w = self.add_weight(
            name="weight",
            shape=(self.groups, self.ws, self.hs),
            initializer="glorot_uniform",
        )

    def call(self, x):
        # x: [..., I]
        shape = tf.shape(x)
        # Reshape last dim: [..., G, I/G]
        x = tf.reshape(x, tf.concat([shape[:-1], [self.groups, self.ws]], axis=0))
        # Einsum: [..., G, I/G] x [G, I/G, H/G] -> [..., G, H/G]
        x = tf.einsum("...gi,gih->...gh", x, self.w)
        # Flatten last two dims: [..., H]
        x = tf.reshape(x, tf.concat([shape[:-1], [self.hidden_size]], axis=0))
        return x

    def load_from_pt(self, pt_state_dict, prefix):
        """Load weight from PyTorch GroupedLinearEinsum."""
        pfx = f"{prefix}." if prefix else ""
        w = pt_state_dict[f"{pfx}weight"]  # [G, I/G, H/G]
        self.w.assign(w)


class GRUCellStateful(tf.keras.layers.Layer):
    """Single GRU cell using tf.keras.layers.GRUCell for streaming.

    For TFLite: single cell call per time step, no WHILE loop.
    """

    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        # reset_after=True is default in TF, matches PT GRU behavior
        self.cell = tf.keras.layers.GRUCell(units, reset_after=True)

    def call(self, x_t, h):
        """Process single timestep.
        Args:
            x_t: [B, D] input for one timestep
            h: [B, H] hidden state
        Returns:
            output: [B, H]
            h_new: [B, H]
        """
        output, [h_new] = self.cell(x_t, [h])
        return output, h_new

    def load_from_pt(self, pt_state_dict, prefix, hidden_size, layer_suffix=""):
        """Load from PT nn.GRU single layer.

        Args:
            pt_state_dict: dict
            prefix: e.g. "enc.emb_gru.gru" for SqueezedGRU, or "grus.0.layers.0" for GroupedGRU
            hidden_size: GRU hidden size
            layer_suffix: e.g. "_l0" for multi-layer nn.GRU
        """
        tf_weights = convert_gru_weights(pt_state_dict, prefix, hidden_size,
                                          layer_suffix=layer_suffix)
        self.cell.set_weights(tf_weights)


class GroupedGRULayerTF(tf.keras.layers.Layer):
    """TF equivalent of GroupedGRULayer.

    With groups=1, this is a standard GRU.
    Uses GRUCell for TFLite compatibility.
    """

    def __init__(self, input_size, hidden_size, groups=1, **kwargs):
        super().__init__(**kwargs)
        self.groups = groups
        self.input_size = input_size // groups
        self.hidden_size = hidden_size // groups
        self.out_size = hidden_size
        self.cells = []
        for i in range(groups):
            self.cells.append(GRUCellStateful(self.hidden_size, name=f"gru_cell_{i}"))

    def call(self, x_seq, h0):
        """
        Args:
            x_seq: [B, T, I]
            h0: [G, B, H/G]
        Returns:
            output: [B, T, H]
            h_out: [G, B, H/G]
        """
        B = tf.shape(x_seq)[0]
        T = x_seq.shape[1]  # static for TFLite
        if T is None:
            T = tf.shape(x_seq)[1]

        # Process each timestep
        outputs_list = []
        h_states = [h0[i] for i in range(self.groups)]  # list of [B, H/G]

        for t in range(T):
            x_t = x_seq[:, t, :]  # [B, I]
            group_outputs = []
            new_h_states = []
            for i, cell in enumerate(self.cells):
                x_g = x_t[:, i * self.input_size:(i + 1) * self.input_size]
                out_g, h_new = cell(x_g, h_states[i])
                group_outputs.append(out_g)
                new_h_states.append(h_new)
            h_states = new_h_states
            out_t = tf.concat(group_outputs, axis=-1)  # [B, H]
            outputs_list.append(out_t)

        output = tf.stack(outputs_list, axis=1)  # [B, T, H]
        h_out = tf.stack(h_states, axis=0)        # [G, B, H/G]
        return output, h_out

    def load_from_pt(self, pt_state_dict, prefix):
        """Load weights from PyTorch GroupedGRULayer."""
        pfx = f"{prefix}." if prefix else ""
        for i, cell in enumerate(self.cells):
            cell_prefix = f"{pfx}layers.{i}"
            cell.load_from_pt(pt_state_dict, cell_prefix, self.hidden_size)


class GroupedGRUTF(tf.keras.layers.Layer):
    """TF equivalent of GroupedGRU.

    Multi-layer grouped GRU with optional shuffle and add_outputs.
    """

    def __init__(self, input_size, hidden_size, num_layers=1, groups=1,
                 shuffle=True, add_outputs=False, **kwargs):
        super().__init__(**kwargs)
        self.num_layers = num_layers
        self.groups = groups
        self.hidden_size_per_group = hidden_size // groups
        self.hidden_size = hidden_size
        self.shuffle = shuffle if groups > 1 else False
        self.add_outputs = add_outputs

        self.grus = []
        self.grus.append(GroupedGRULayerTF(input_size, hidden_size, groups, name="gru_layer_0"))
        for i in range(1, num_layers):
            self.grus.append(GroupedGRULayerTF(hidden_size, hidden_size, groups, name=f"gru_layer_{i}"))

    def call(self, x, state=None):
        """
        Args:
            x: [B, T, I]
            state: [num_layers * G, B, H/G]  (or None for zeros)
        Returns:
            output: [B, T, H]
            state_out: [num_layers * G, B, H/G]
        """
        B = tf.shape(x)[0]
        T = x.shape[1]
        if state is None:
            state = tf.zeros([self.num_layers * self.groups, B, self.hidden_size_per_group])

        h_per_layer = self.groups
        out_states = []
        output = tf.zeros([B, T, self.hidden_size])

        for i, gru in enumerate(self.grus):
            h_i = state[i * h_per_layer:(i + 1) * h_per_layer]
            x, s = gru(x, h_i)
            out_states.append(s)

            if self.shuffle and i < self.num_layers - 1:
                # Shuffle: [B, T, H] -> [B, T, H/G, G] -> [B, T, G, H/G] -> [B, T, H]
                x = tf.reshape(x, [B, T, -1, self.groups])
                x = tf.transpose(x, [0, 1, 3, 2])
                x = tf.reshape(x, [B, T, self.hidden_size])

            if self.add_outputs:
                output = output + x
            else:
                output = x

        state_out = tf.concat(out_states, axis=0)  # [num_layers * G, B, H/G]
        return output, state_out

    def load_from_pt(self, pt_state_dict, prefix):
        """Load from PT GroupedGRU state dict."""
        pfx = f"{prefix}." if prefix else ""
        for i, gru_layer in enumerate(self.grus):
            gru_layer.load_from_pt(pt_state_dict, f"{pfx}grus.{i}")


class SqueezedGRUTF(tf.keras.layers.Layer):
    """TF equivalent of SqueezedGRU.

    Architecture: linear_in (GroupedLinearEinsum + optional ReLU) -> GRU -> optional skip -> optional linear_out.
    The GRU is a standard nn.GRU (not GroupedGRU), implemented via GRUCell for TFLite.
    """

    def __init__(self, input_size, hidden_size, output_size=None,
                 num_layers=1, linear_groups=8,
                 has_gru_skip=False, has_linear_act=True, **kwargs):
        super().__init__(**kwargs)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.has_gru_skip = has_gru_skip

        # linear_in: GroupedLinearEinsum + optional ReLU
        self.linear_in = GroupedLinearEinsumTF(
            input_size, hidden_size, groups=linear_groups, name="linear_in")
        self.linear_in_act = tf.keras.layers.ReLU() if has_linear_act else None

        # Standard GRU via cells (for TFLite: no WHILE loops)
        self.gru_cells = []
        for i in range(num_layers):
            self.gru_cells.append(
                GRUCellStateful(hidden_size, name=f"gru_cell_{i}"))

        # Optional skip connection (identity)
        # gru_skip in PT is nn.Identity() — just add input to output

        # Optional linear_out
        if output_size is not None:
            self.linear_out = GroupedLinearEinsumTF(
                hidden_size, output_size, groups=linear_groups, name="linear_out")
            self.linear_out_act = tf.keras.layers.ReLU() if has_linear_act else None
        else:
            self.linear_out = None
            self.linear_out_act = None

    def call(self, x, state=None):
        """
        Args:
            x: [B, T, I]
            state: [num_layers, B, H]

        Returns:
            output: [B, T, H] or [B, T, output_size] if linear_out
            state_out: [num_layers, B, H]
        """
        B = tf.shape(x)[0]
        T = x.shape[1]

        if state is None:
            state = tf.zeros([self.num_layers, B, self.hidden_size])

        # linear_in
        x = self.linear_in(x)
        if self.linear_in_act is not None:
            x = self.linear_in_act(x)

        lin_in_out = x  # save for skip

        # GRU layers
        h_states = [state[i] for i in range(self.num_layers)]
        new_h_states = []

        for layer_idx, cell in enumerate(self.gru_cells):
            outputs_t = []
            h = h_states[layer_idx]
            for t in range(T):
                x_t = x[:, t, :]
                out_t, h = cell(x_t, h)
                outputs_t.append(out_t)
            x = tf.stack(outputs_t, axis=1)  # [B, T, H]
            new_h_states.append(h)

        # Skip connection
        if self.has_gru_skip:
            x = x + lin_in_out

        # linear_out
        if self.linear_out is not None:
            x = self.linear_out(x)
            if self.linear_out_act is not None:
                x = self.linear_out_act(x)

        state_out = tf.stack(new_h_states, axis=0)
        return x, state_out

    def load_from_pt(self, pt_state_dict, prefix):
        """Load from PT SqueezedGRU state dict."""
        pfx = f"{prefix}." if prefix else ""

        # linear_in.0.weight -> GroupedLinearEinsum
        self.linear_in.load_from_pt(pt_state_dict, f"{pfx}linear_in.0")

        # gru weights: standard nn.GRU with num_layers
        for i, cell in enumerate(self.gru_cells):
            cell.load_from_pt(pt_state_dict, f"{pfx}gru", self.hidden_size,
                              layer_suffix=f"_l{i}")

        # linear_out if present
        if self.linear_out is not None:
            self.linear_out.load_from_pt(pt_state_dict, f"{pfx}linear_out.0")


class Conv2dNormActTF(tf.keras.layers.Layer):
    """TF equivalent of Conv2dNormAct.

    Handles causal padding, separable conv, batchnorm, and activation.
    Input/output in NHWC format (B, T, F, C).
    """

    def __init__(self, in_ch, out_ch, kernel_size, fstride=1, bias=True,
                 separable=True, activation="relu", use_bn=True,
                 causal_pad_t=None, **kwargs):
        super().__init__(**kwargs)
        import math
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.kernel_size = tuple(kernel_size)
        self.fstride = fstride
        self.in_ch = in_ch
        self.out_ch = out_ch

        # Causal time padding: pad (k_t - 1) on the left, 0 on the right
        # This is the lookahead=0 case
        self.t_pad = kernel_size[0] - 1 if causal_pad_t is None else causal_pad_t
        # Frequency padding
        self.f_pad = kernel_size[1] // 2

        groups = math.gcd(in_ch, out_ch) if separable else 1
        if groups == 1:
            separable = False
        if max(kernel_size) == 1:
            separable = False

        self.is_separable = separable
        self.groups = groups  # Track groups even for non-separable (grouped 1x1 case)
        self.layers_list = []

        if separable:
            # Use Conv2D with groups instead of DepthwiseConv2D
            # (DepthwiseConv2D doesn't support asymmetric strides in TF)
            self.dw_conv = tf.keras.layers.Conv2D(
                out_ch, kernel_size,
                strides=(1, fstride),
                padding="valid",
                use_bias=False,
                groups=groups,
                name="dw_conv",
            )
            # Pointwise conv
            self.pw_conv = tf.keras.layers.Conv2D(
                out_ch, (1, 1), use_bias=bias, padding="valid", name="pw_conv"
            )
        else:
            self.main_conv = tf.keras.layers.Conv2D(
                out_ch, kernel_size,
                strides=(1, fstride),
                padding="valid",
                use_bias=bias,
                groups=groups,
                name="main_conv",
            )

        self.use_bn = use_bn
        if use_bn:
            self.bn = tf.keras.layers.BatchNormalization(
                epsilon=1e-5, momentum=0.9, name="bn"
            )

        if activation == "relu":
            self.act = tf.keras.layers.ReLU()
        elif activation == "sigmoid":
            self.act = tf.keras.layers.Activation("sigmoid")
        elif activation is None:
            self.act = None
        else:
            self.act = tf.keras.layers.Activation(activation)

    def call(self, x, training=False):
        # x: [B, T, F, C] (NHWC)
        # Apply causal time padding (left only) and frequency padding (both sides)
        if self.t_pad > 0 or self.f_pad > 0:
            # tf.pad: [[batch], [T], [F], [C]]
            x = tf.pad(x, [[0, 0], [self.t_pad, 0], [self.f_pad, self.f_pad], [0, 0]])

        if self.is_separable:
            x = self.dw_conv(x)
            x = self.pw_conv(x)
        else:
            x = self.main_conv(x)

        if self.use_bn:
            x = self.bn(x, training=training)

        if self.act is not None:
            x = self.act(x)
        return x

    def load_from_pt(self, pt_state_dict, prefix):
        """Load weights from PyTorch Conv2dNormAct (nn.Sequential).

        The PT sequential layer indices vary:
          - With time pad: [0:pad, 1:conv, 2:bn/pw, 3:bn/act, 4:act]
          - Without time pad: [0:conv, 1:pw, 2:bn, 3:act]
        We identify layers by inspecting weight shapes.
        """
        pfx = f"{prefix}." if prefix else ""

        # Collect all 4D weights (conv) and 1D weights (bn gamma) by their index
        conv4d = {}  # idx -> weight array
        bn_idx = None
        for k, v in pt_state_dict.items():
            if not k.startswith(pfx):
                continue
            rest = k[len(pfx):]
            parts = rest.split(".")
            idx = int(parts[0])
            attr = ".".join(parts[1:])

            if attr == "weight" and v.ndim == 4:
                conv4d[idx] = v
            elif attr == "running_mean":
                bn_idx = idx

        if self.is_separable:
            # Two 4D weights: depthwise (shape [Ch,1,kH,kW]) and pointwise ([O,I,1,1])
            sorted_idxs = sorted(conv4d.keys())
            dw_idx = sorted_idxs[0]
            pw_idx = sorted_idxs[1]

            # Load depthwise: PT [Ch,1,kH,kW] grouped conv -> TF Conv2D groups
            w_dw = conv4d[dw_idx]
            # For Conv2D with groups: TF expects [kH, kW, in_ch/groups, out_ch]
            # PT depthwise: [out_ch, 1, kH, kW] with groups=out_ch
            # -> TF: [kH, kW, 1, out_ch]
            w_dw_tf = np.transpose(w_dw, (2, 3, 1, 0))
            self.dw_conv.set_weights([w_dw_tf])

            # Load pointwise: PT [O, I, 1, 1] -> TF [1, 1, I, O]
            w_pw = conv4d[pw_idx]
            w_pw_tf = convert_conv2d_weight(w_pw)
            pw_weights = [w_pw_tf]
            b_key = f"{pfx}{pw_idx}.bias"
            if b_key in pt_state_dict:
                pw_weights.append(pt_state_dict[b_key])
            self.pw_conv.set_weights(pw_weights)
        else:
            # Single 4D weight
            conv_idx = sorted(conv4d.keys())[0]
            w = conv4d[conv_idx]
            w_tf = convert_conv2d_weight(w)
            conv_weights = [w_tf]
            b_key = f"{pfx}{conv_idx}.bias"
            if b_key in pt_state_dict:
                conv_weights.append(pt_state_dict[b_key])
            self.main_conv.set_weights(conv_weights)

        # Load BatchNorm
        if self.use_bn and bn_idx is not None:
            gamma = pt_state_dict[f"{pfx}{bn_idx}.weight"]
            beta = pt_state_dict[f"{pfx}{bn_idx}.bias"]
            rm = pt_state_dict[f"{pfx}{bn_idx}.running_mean"]
            rv = pt_state_dict[f"{pfx}{bn_idx}.running_var"]
            self.bn.set_weights([gamma, beta, rm, rv])


class ConvTranspose2dNormActTF(tf.keras.layers.Layer):
    """TF equivalent of ConvTranspose2dNormAct, using Conv2D + zero-insertion upsampling.

    Replaces Conv2DTranspose with frequency upsampling (zero insertion) followed
    by a standard Conv2D.  This avoids Conv2DTranspose which can be problematic
    for TFLite quantisation and has no grouped-convolution support in TF.
    Input/output in NHWC format (B, T, F, C).
    """

    def __init__(self, in_ch, out_ch, kernel_size, fstride=1, bias=True,
                 separable=True, activation="relu", use_bn=True, **kwargs):
        super().__init__(**kwargs)
        import math
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.kernel_size = tuple(kernel_size)
        self.fstride = fstride
        self.in_ch = in_ch
        self.out_ch = out_ch

        # Causal time padding (same as Conv2dNormAct)
        self.t_pad = kernel_size[0] - 1
        self.f_pad = kernel_size[1] // 2

        groups = math.gcd(in_ch, out_ch) if separable else 1
        if groups == 1:
            separable = False
        self.is_separable = separable
        self.groups = groups

        # Conv2D instead of Conv2DTranspose.
        # The frequency stride is handled by zero-insertion upsampling before the conv.
        if separable:
            # Depthwise via grouped Conv2D (no block-diagonal hack needed)
            self.dw_conv = tf.keras.layers.Conv2D(
                in_ch, kernel_size, strides=(1, 1),
                padding="valid", use_bias=False,
                groups=groups, name="dw_conv",
            )
            self.pw_conv = tf.keras.layers.Conv2D(
                out_ch, (1, 1), use_bias=False,
                padding="valid", name="pw_conv",
            )
        else:
            self.conv = tf.keras.layers.Conv2D(
                out_ch, kernel_size, strides=(1, 1),
                padding="valid", use_bias=bias,
                name="conv",
            )
            self.pw_conv = None

        self.use_bn = use_bn
        if use_bn:
            self.bn = tf.keras.layers.BatchNormalization(
                epsilon=1e-5, momentum=0.9, name="bn"
            )

        if activation == "relu":
            self.act = tf.keras.layers.ReLU()
        elif activation == "sigmoid":
            self.act = tf.keras.layers.Activation("sigmoid")
        elif activation is None:
            self.act = None
        else:
            self.act = tf.keras.layers.Activation(activation)

    def call(self, x, training=False):
        # x: [B, T, F, C] (NHWC)
        kT, kF = self.kernel_size

        # Causal time padding
        if self.t_pad > 0:
            x = tf.pad(x, [[0, 0], [self.t_pad, 0], [0, 0], [0, 0]])

        # Upsample frequency axis by inserting (fstride-1) zeros between elements
        if self.fstride > 1:
            x = tf.expand_dims(x, 3)                          # [B, T, F, 1, C]
            x = tf.pad(x, [[0, 0], [0, 0], [0, 0],
                           [0, self.fstride - 1], [0, 0]])    # [B, T, F, fstride, C]
            s = tf.shape(x)
            x = tf.reshape(x, [s[0], s[1], s[2] * s[3], s[4]])  # [B, T, F*fstride, C]
            x = x[:, :, :-(self.fstride - 1), :]              # [B, T, (F-1)*fstride+1, C]

        # Pad for transposed-conv-equivalent: (kT-1) each side in time,
        # (kF-1) each side in frequency
        pad_t = kT - 1
        pad_f = kF - 1
        if pad_t > 0 or pad_f > 0:
            x = tf.pad(x, [[0, 0], [pad_t, pad_t], [pad_f, pad_f], [0, 0]])

        # Convolution (replaces Conv2DTranspose)
        if self.is_separable:
            x = self.dw_conv(x)
        else:
            x = self.conv(x)

        # Crop time (same as original Conv2DTranspose path)
        if kT > 1:
            x = x[:, kT - 1:, :, :]

        # Crop frequency (same as original Conv2DTranspose path)
        if self.f_pad > 0:
            x = x[:, :, self.f_pad:, :]

        if self.pw_conv is not None:
            x = self.pw_conv(x)

        if self.use_bn:
            x = self.bn(x, training=training)

        if self.act is not None:
            x = self.act(x)
        return x

    def load_from_pt(self, pt_state_dict, prefix):
        """Load weights from PyTorch ConvTranspose2dNormAct.

        Conv2D equivalent kernel = spatially-flipped, channel-transposed
        version of the original Conv2DTranspose kernel.
        """
        pfx = f"{prefix}." if prefix else ""

        conv4d = {}
        bn_idx = None
        for k, v in pt_state_dict.items():
            if not k.startswith(pfx):
                continue
            rest = k[len(pfx):]
            parts = rest.split(".")
            idx = int(parts[0])
            attr = ".".join(parts[1:])
            if attr == "weight" and v.ndim == 4:
                conv4d[idx] = v
            elif attr == "running_mean":
                bn_idx = idx

        sorted_idxs = sorted(conv4d.keys())

        if self.is_separable:
            tconv_idx = sorted_idxs[0]
            pw_idx = sorted_idxs[1]

            # PT depthwise ConvTranspose2d weight: [in_ch, 1, kH, kW] (groups=in_ch)
            # Conv2D with groups kernel: (kH, kW, 1, in_ch), spatially flipped
            w_pt = conv4d[tconv_idx]  # [in_ch, 1, kH, kW]
            w_tf = np.transpose(w_pt, (2, 3, 1, 0))       # (kH, kW, 1, in_ch)
            w_tf = np.flip(w_tf, axis=(0, 1)).copy()       # spatial flip, contiguous
            self.dw_conv.set_weights([w_tf])

            w = conv4d[pw_idx]
            w_tf = convert_conv2d_weight(w)
            self.pw_conv.set_weights([w_tf])
        else:
            tconv_idx = sorted_idxs[0]
            w = conv4d[tconv_idx]
            # PT ConvTranspose2d: (I, O, kH, kW)
            # Conv2D kernel: (kH, kW, I, O), spatially flipped
            w_tf = np.transpose(w, (2, 3, 0, 1))           # (kH, kW, I, O)
            w_tf = np.flip(w_tf, axis=(0, 1)).copy()        # spatial flip, contiguous
            tconv_weights = [w_tf]
            b_key = f"{pfx}{tconv_idx}.bias"
            if b_key in pt_state_dict:
                tconv_weights.append(pt_state_dict[b_key])
            self.conv.set_weights(tconv_weights)

        if self.use_bn and bn_idx is not None:
            gamma = pt_state_dict[f"{pfx}{bn_idx}.weight"]
            beta = pt_state_dict[f"{pfx}{bn_idx}.bias"]
            rm = pt_state_dict[f"{pfx}{bn_idx}.running_mean"]
            rv = pt_state_dict[f"{pfx}{bn_idx}.running_var"]
            self.bn.set_weights([gamma, beta, rm, rv])


class MaskTF(tf.keras.layers.Layer):
    """TF equivalent of Mask module.

    Applies ERB mask to spectrogram.
    """

    def __init__(self, erb_inv_fb_np, **kwargs):
        super().__init__(**kwargs)
        # erb_inv_fb: [nb_erb, F]
        self.erb_inv_fb = tf.constant(erb_inv_fb_np, dtype=tf.float32)

    def call(self, spec, mask):
        """
        Args:
            spec: [B, T, F, 1, 2] (NHWC equivalent of [B, 1, T, F, 2])
            mask: [B, T, E, 1] (NHWC equivalent of [B, 1, T, E])
        Returns:
            spec_masked: [B, T, F, 1, 2]
        """
        # mask: [B, T, E, 1] -> squeeze channel -> [B, T, E]
        m = tf.squeeze(mask, axis=-1)  # [B, T, E]
        # ERB to frequency: [B, T, E] x [E, F] -> [B, T, F]
        m_freq = tf.matmul(m, self.erb_inv_fb)  # [B, T, F]
        # Expand for channel and complex: [B, T, F, 1, 1]
        m_freq = m_freq[:, :, :, tf.newaxis, tf.newaxis]
        return spec * m_freq


class DfOutputReshapeMFTF(tf.keras.layers.Layer):
    """TF equivalent of DfOutputReshapeMF.

    Reshapes DF decoder output to multiframe coefficient format.
    PT: [B, T, F, O*2] -> [B, O, T, F, 2]
    TF (NHWC): [B, T, F, O*2] -> [B, O, T, F, 2]  (same logic, just operates on NHWC data)
    """

    def __init__(self, df_order, df_bins, **kwargs):
        super().__init__(**kwargs)
        self.df_order = df_order
        self.df_bins = df_bins

    def call(self, coefs):
        # coefs: [B, T, F, O*2]
        B = tf.shape(coefs)[0]
        T = coefs.shape[1] if coefs.shape[1] is not None else tf.shape(coefs)[1]
        F = self.df_bins
        O = self.df_order

        # [B, T, F, O*2] -> [B, T, F, O, 2]
        coefs = tf.reshape(coefs, [B, T, F, O, 2])
        # [B, T, F, O, 2] -> [B, O, T, F, 2]
        coefs = tf.transpose(coefs, [0, 3, 1, 2, 4])
        return coefs


class MultiFrameDFTF(tf.keras.layers.Layer):
    """TF equivalent of DF (MultiFrameModule subclass) from multiframe.py.

    Deep filtering: pads+unfolds spectrogram, applies filter coefficients.
    No complex ops — uses real/imag as last dim of size 2.
    """

    def __init__(self, num_freqs, frame_size, lookahead=0, **kwargs):
        super().__init__(**kwargs)
        self.num_freqs = num_freqs
        self.frame_size = frame_size
        self.lookahead = lookahead

    def call(self, spec, coefs):
        """
        Args:
            spec: [B, 1, T, F, 2] (kept in PT format for this op since it's 5D)
            coefs: [B, O, T, F_df, 2]
        Returns:
            spec_out: [B, 1, T, F, 2]
        """
        F_df = self.num_freqs
        O = self.frame_size

        # Extract DF frequency bins and convert to complex-like representation
        spec_df = spec[:, :, :, :F_df, :]  # [B, 1, T, F_df, 2]
        spec_df = tf.squeeze(spec_df, axis=1)  # [B, T, F_df, 2]

        # Pad time axis: (O-1-lookahead) on left, lookahead on right
        pad_left = O - 1 - self.lookahead
        pad_right = self.lookahead
        spec_padded = tf.pad(spec_df, [[0, 0], [pad_left, pad_right], [0, 0], [0, 0]])

        # Unfold: create windows of size O along time axis
        # spec_padded: [B, T+O-1, F_df, 2]
        T = tf.shape(spec)[2]
        # Manual unfold using gather
        indices = tf.range(O)[tf.newaxis, :] + tf.range(T)[:, tf.newaxis]  # [T, O]
        spec_unfolded = tf.gather(spec_padded, indices, axis=1)  # [B, T, O, F_df, 2]

        # Deep filter: complex multiply and sum over O
        # spec_unfolded: [B, T, O, F_df, 2] (re, im)
        # coefs: [B, O, T, F_df, 2] -> transpose to [B, T, O, F_df, 2]
        coefs_t = tf.transpose(coefs, [0, 2, 1, 3, 4])  # [B, T, O, F_df, 2]

        # Complex multiply: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
        s_re = spec_unfolded[..., 0]  # [B, T, O, F_df]
        s_im = spec_unfolded[..., 1]
        c_re = coefs_t[..., 0]
        c_im = coefs_t[..., 1]

        out_re = s_re * c_re - s_im * c_im
        out_im = s_re * c_im + s_im * c_re

        # Sum over filter order dimension
        out_re = tf.reduce_sum(out_re, axis=2)  # [B, T, F_df]
        out_im = tf.reduce_sum(out_im, axis=2)

        out = tf.stack([out_re, out_im], axis=-1)  # [B, T, F_df, 2]

        # Assign back: replace first F_df freq bins
        # spec: [B, 1, T, F, 2]
        spec_out = tf.concat([
            tf.expand_dims(out, axis=1),           # [B, 1, T, F_df, 2]
            spec[:, :, :, F_df:, :],               # [B, 1, T, F-F_df, 2]
        ], axis=3)
        return spec_out


# ============================================================================
# Full DfNet TF Model (Phase 1-3)
# ============================================================================

class EncoderTF(tf.keras.layers.Layer):
    """TF equivalent of Encoder from deepfilternet2.py."""

    def __init__(self, nb_erb=32, nb_df=96, conv_ch=16, emb_hidden_dim=256,
                 lin_groups=1, gru_groups=1, group_shuffle=True,
                 lsnr_max=35, lsnr_min=-15, enc_concat=False, **kwargs):
        super().__init__(**kwargs)
        self.nb_erb = nb_erb
        self.nb_df = nb_df
        self.conv_ch = conv_ch
        self.enc_concat = enc_concat

        self.erb_conv0 = Conv2dNormActTF(1, conv_ch, (3, 3), bias=False, separable=True, name="erb_conv0")
        self.erb_conv1 = Conv2dNormActTF(conv_ch, conv_ch, (1, 3), fstride=2, bias=False, separable=True, name="erb_conv1")
        self.erb_conv2 = Conv2dNormActTF(conv_ch, conv_ch, (1, 3), fstride=2, bias=False, separable=True, name="erb_conv2")
        self.erb_conv3 = Conv2dNormActTF(conv_ch, conv_ch, (1, 3), fstride=1, bias=False, separable=True, name="erb_conv3")

        self.df_conv0 = Conv2dNormActTF(2, conv_ch, (3, 3), bias=False, separable=True, name="df_conv0")
        self.df_conv1 = Conv2dNormActTF(conv_ch, conv_ch, (1, 3), fstride=2, bias=False, separable=True, name="df_conv1")

        emb_in_dim = conv_ch * nb_erb // 4
        self.df_fc_emb = GroupedLinearTF(conv_ch * nb_df // 2, emb_in_dim,
                                          groups=lin_groups, shuffle=group_shuffle, name="df_fc_emb")

        actual_emb_in = emb_in_dim * 2 if enc_concat else emb_in_dim
        self.emb_gru = GroupedGRUTF(actual_emb_in, emb_hidden_dim, num_layers=1,
                                     groups=gru_groups, shuffle=group_shuffle,
                                     add_outputs=True, name="emb_gru")
        self.lsnr_dense = tf.keras.layers.Dense(1, name="lsnr_dense")
        self.lsnr_scale = lsnr_max - lsnr_min
        self.lsnr_offset = lsnr_min

    def call(self, feat_erb, feat_spec, h_emb=None, training=False):
        """feat_erb: [B, T, E, 1], feat_spec: [B, T, F_df, 2]"""
        e0 = self.erb_conv0(feat_erb, training=training)
        e1 = self.erb_conv1(e0, training=training)
        e2 = self.erb_conv2(e1, training=training)
        e3 = self.erb_conv3(e2, training=training)

        c0 = self.df_conv0(feat_spec, training=training)
        c1 = self.df_conv1(c0, training=training)

        c1_shape = tf.shape(c1)
        cemb = tf.reshape(c1, [c1_shape[0], c1_shape[1], -1])
        cemb = self.df_fc_emb(cemb)

        e3_shape = tf.shape(e3)
        emb = tf.reshape(e3, [e3_shape[0], e3_shape[1], -1])

        if self.enc_concat:
            emb = tf.concat([emb, cemb], axis=-1)
        else:
            emb = emb + cemb

        emb, h_emb_out = self.emb_gru(emb, h_emb)
        lsnr = tf.sigmoid(self.lsnr_dense(emb)) * self.lsnr_scale + self.lsnr_offset
        return e0, e1, e2, e3, emb, c0, lsnr, h_emb_out

    def load_from_pt(self, pt_sd, prefix="enc"):
        pfx = f"{prefix}."
        self.erb_conv0.load_from_pt(pt_sd, f"{pfx}erb_conv0")
        self.erb_conv1.load_from_pt(pt_sd, f"{pfx}erb_conv1")
        self.erb_conv2.load_from_pt(pt_sd, f"{pfx}erb_conv2")
        self.erb_conv3.load_from_pt(pt_sd, f"{pfx}erb_conv3")
        self.df_conv0.load_from_pt(pt_sd, f"{pfx}df_conv0")
        self.df_conv1.load_from_pt(pt_sd, f"{pfx}df_conv1")
        self.df_fc_emb.load_from_pt(pt_sd, f"{pfx}df_fc_emb")
        self.emb_gru.load_from_pt(pt_sd, f"{pfx}emb_gru")
        w = pt_sd[f"{pfx}lsnr_fc.0.weight"]
        b = pt_sd[f"{pfx}lsnr_fc.0.bias"]
        self.lsnr_dense.set_weights([w.T, b])


class ErbDecoderTF(tf.keras.layers.Layer):
    """TF equivalent of ErbDecoder."""

    def __init__(self, nb_erb=32, conv_ch=16, emb_hidden_dim=256,
                 emb_num_layers=2, lin_groups=1, gru_groups=1,
                 group_shuffle=True, **kwargs):
        super().__init__(**kwargs)
        self.conv_ch = conv_ch

        self.emb_gru = GroupedGRUTF(
            conv_ch * nb_erb // 4, emb_hidden_dim,
            num_layers=emb_num_layers - 1, groups=gru_groups,
            shuffle=group_shuffle, add_outputs=True, name="erb_dec_gru")
        self.fc_emb = GroupedLinearTF(emb_hidden_dim, conv_ch * nb_erb // 4,
                                       groups=lin_groups, shuffle=group_shuffle, name="fc_emb")
        self.fc_emb_relu = tf.keras.layers.ReLU()

        self.conv3p = Conv2dNormActTF(conv_ch, conv_ch, 1, bias=False, separable=True, name="conv3p")
        self.convt3 = Conv2dNormActTF(conv_ch, conv_ch, (1, 3), bias=False, separable=True, name="convt3")
        self.conv2p = Conv2dNormActTF(conv_ch, conv_ch, 1, bias=False, separable=True, name="conv2p")
        self.convt2 = ConvTranspose2dNormActTF(conv_ch, conv_ch, (1, 3), fstride=2, bias=False, separable=True, name="convt2")
        self.conv1p = Conv2dNormActTF(conv_ch, conv_ch, 1, bias=False, separable=True, name="conv1p")
        self.convt1 = ConvTranspose2dNormActTF(conv_ch, conv_ch, (1, 3), fstride=2, bias=False, separable=True, name="convt1")
        self.conv0p = Conv2dNormActTF(conv_ch, conv_ch, 1, bias=False, separable=True, name="conv0p")
        self.conv0_out = Conv2dNormActTF(conv_ch, 1, (1, 3), bias=False, separable=True, activation="sigmoid", name="conv0_out")

    def call(self, emb, e3, e2, e1, e0, h_erb_dec=None, training=False):
        B = tf.shape(e3)[0]
        T = tf.shape(e3)[1]
        f8 = e3.shape[2] if e3.shape[2] is not None else tf.shape(e3)[2]

        emb, h_erb_dec_out = self.emb_gru(emb, h_erb_dec)
        emb = self.fc_emb_relu(self.fc_emb(emb))
        emb = tf.reshape(emb, [B, T, f8, self.conv_ch])

        e3 = self.convt3(self.conv3p(e3, training=training) + emb, training=training)
        e2 = self.convt2(self.conv2p(e2, training=training) + e3, training=training)
        e1 = self.convt1(self.conv1p(e1, training=training) + e2, training=training)
        m = self.conv0_out(self.conv0p(e0, training=training) + e1, training=training)
        return m, h_erb_dec_out

    def load_from_pt(self, pt_sd, prefix="erb_dec"):
        pfx = f"{prefix}."
        self.emb_gru.load_from_pt(pt_sd, f"{pfx}emb_gru")
        self.fc_emb.load_from_pt(pt_sd, f"{pfx}fc_emb.0")
        self.conv3p.load_from_pt(pt_sd, f"{pfx}conv3p")
        self.convt3.load_from_pt(pt_sd, f"{pfx}convt3")
        self.conv2p.load_from_pt(pt_sd, f"{pfx}conv2p")
        self.convt2.load_from_pt(pt_sd, f"{pfx}convt2")
        self.conv1p.load_from_pt(pt_sd, f"{pfx}conv1p")
        self.convt1.load_from_pt(pt_sd, f"{pfx}convt1")
        self.conv0p.load_from_pt(pt_sd, f"{pfx}conv0p")
        self.conv0_out.load_from_pt(pt_sd, f"{pfx}conv0_out")


class DfDecoderTF(tf.keras.layers.Layer):
    """TF equivalent of DfDecoder."""

    def __init__(self, nb_df=96, df_order=5, conv_ch=16,
                 emb_hidden_dim=256, df_hidden_dim=256, df_num_layers=3,
                 gru_groups=1, group_shuffle=True, lin_groups=1,
                 df_pathway_kernel_size_t=1, **kwargs):
        super().__init__(**kwargs)
        self.nb_df = nb_df
        self.df_out_ch = df_order * 2

        self.df_convp = Conv2dNormActTF(
            conv_ch, self.df_out_ch, (df_pathway_kernel_size_t, 1),
            separable=True, bias=False, name="df_convp")

        self.df_gru = GroupedGRUTF(
            emb_hidden_dim, df_hidden_dim, num_layers=df_num_layers,
            groups=gru_groups, shuffle=group_shuffle, add_outputs=True, name="df_gru")

        self.df_out = GroupedLinearEinsumTF(df_hidden_dim, nb_df * self.df_out_ch,
                                             groups=lin_groups, name="df_out")
        self.df_out_tanh = tf.keras.layers.Activation("tanh")
        self.df_fc_a_dense = tf.keras.layers.Dense(1, name="df_fc_a")
        self.df_fc_a_sigmoid = tf.keras.layers.Activation("sigmoid")
        self.out_transform = DfOutputReshapeMFTF(df_order, nb_df)

    def call(self, emb, c0, h_df=None, training=False):
        B = tf.shape(emb)[0]
        T = emb.shape[1] if emb.shape[1] is not None else tf.shape(emb)[1]

        c, h_df_out = self.df_gru(emb, h_df)
        c0 = self.df_convp(c0, training=training)
        alpha = self.df_fc_a_sigmoid(self.df_fc_a_dense(c))
        c = self.df_out_tanh(self.df_out(c))
        c = tf.reshape(c, [B, T, self.nb_df, self.df_out_ch])
        c = c + c0
        coefs = self.out_transform(c)
        return coefs, alpha, h_df_out

    def load_from_pt(self, pt_sd, prefix="df_dec"):
        pfx = f"{prefix}."
        self.df_convp.load_from_pt(pt_sd, f"{pfx}df_convp")
        self.df_gru.load_from_pt(pt_sd, f"{pfx}df_gru")
        self.df_out.load_from_pt(pt_sd, f"{pfx}df_out.0")
        w = pt_sd[f"{pfx}df_fc_a.0.weight"]
        b = pt_sd[f"{pfx}df_fc_a.0.bias"]
        self.df_fc_a_dense.set_weights([w.T, b])


class DfNetTF(tf.keras.Model):
    """Full TF re-implementation of DfNet. Same I/O format as PT."""

    def __init__(self, erb_inv_fb_np, nb_erb=32, nb_df=96,
                 fft_size=960, conv_ch=16, df_order=5, df_lookahead=0,
                 emb_hidden_dim=256, emb_num_layers=2,
                 df_hidden_dim=256, df_num_layers=3,
                 gru_groups=1, lin_groups=1, group_shuffle=True,
                 enc_concat=False, df_pathway_kernel_size_t=1,
                 conv_lookahead=0, pad_mode="none", **kwargs):
        super().__init__(**kwargs)
        self.nb_df = nb_df
        self.conv_lookahead = conv_lookahead
        self.df_lookahead = df_lookahead if pad_mode == "model" else 0
        self.pad_mode = pad_mode
        self.pad_specf = pad_mode.endswith("specf")
        self.input_spec_lookahead = df_lookahead if self.pad_specf else 0
        self.pad_out = pad_mode.startswith("output") and (
            conv_lookahead > 0 or df_lookahead > 0
        )
        if self.pad_out:
            assert conv_lookahead == df_lookahead
            self.output_lookahead = conv_lookahead
        else:
            self.output_lookahead = 0

        self.enc = EncoderTF(nb_erb=nb_erb, nb_df=nb_df, conv_ch=conv_ch,
                              emb_hidden_dim=emb_hidden_dim, lin_groups=lin_groups,
                              gru_groups=gru_groups, group_shuffle=group_shuffle,
                              enc_concat=enc_concat, name="encoder")
        self.erb_dec = ErbDecoderTF(nb_erb=nb_erb, conv_ch=conv_ch,
                                     emb_hidden_dim=emb_hidden_dim,
                                     emb_num_layers=emb_num_layers,
                                     lin_groups=lin_groups, gru_groups=gru_groups,
                                     group_shuffle=group_shuffle, name="erb_decoder")
        self.mask = MaskTF(erb_inv_fb_np, name="mask")
        self.df_dec = DfDecoderTF(nb_df=nb_df, df_order=df_order, conv_ch=conv_ch,
                                    emb_hidden_dim=emb_hidden_dim,
                                    df_hidden_dim=df_hidden_dim,
                                    df_num_layers=df_num_layers,
                                    gru_groups=gru_groups,
                                    group_shuffle=group_shuffle,
                                    lin_groups=lin_groups,
                                    df_pathway_kernel_size_t=df_pathway_kernel_size_t,
                                    name="df_decoder")
        self.df_op = MultiFrameDFTF(num_freqs=nb_df, frame_size=df_order,
                                     lookahead=self.df_lookahead, name="df_op")

    def call(self, spec, feat_erb, feat_spec, training=False):
        """Same interface as PT DfNet.forward.
        spec: [B,1,T,F,2], feat_erb: [B,1,T,E], feat_spec: [B,1,T,F_df,2]
        """
        # Convert to NHWC
        feat_spec_nhwc = tf.squeeze(feat_spec, axis=1)      # [B, T, F_df, 2]
        feat_erb_nhwc = tf.transpose(feat_erb, [0, 2, 3, 1])  # [B, T, E, 1]

        if self.conv_lookahead > 0 and self.pad_mode.startswith("input"):
            feat_erb_nhwc = shift_future_nhwc(feat_erb_nhwc, self.conv_lookahead)
            feat_spec_nhwc = shift_future_nhwc(feat_spec_nhwc, self.conv_lookahead)

        e0, e1, e2, e3, emb, c0, lsnr, _ = self.enc(
            feat_erb_nhwc, feat_spec_nhwc, training=training)
        m_nhwc, _ = self.erb_dec(emb, e3, e2, e1, e0, training=training)

        if self.output_lookahead > 0:
            m_nhwc = shift_future_nhwc(m_nhwc, self.output_lookahead)

        # Mask: need spec in [B, T, F, 1, 2] and mask in [B, T, E, 1]
        spec_tf = tf.transpose(spec, [0, 2, 3, 1, 4])
        spec_masked_tf = self.mask(spec_tf, m_nhwc)
        spec_masked = tf.transpose(spec_masked_tf, [0, 3, 1, 2, 4])

        df_coefs, df_alpha, _ = self.df_dec(emb, c0, training=training)

        if self.output_lookahead > 0:
            df_coefs = shift_future_nhwc(df_coefs, self.output_lookahead)

        if self.pad_specf:
            spec_f = shift_future_spec(spec_masked, self.input_spec_lookahead)
            spec_f_out = self.df_op(spec_f, df_coefs)
            spec_out = tf.concat([
                spec_f_out[:, :, :, :self.nb_df, :],
                spec_masked[:, :, :, self.nb_df:, :],
            ], axis=3)
        else:
            spec_out = self.df_op(spec_masked, df_coefs)

        m_pt = tf.transpose(m_nhwc, [0, 3, 1, 2])
        return spec_out, m_pt, lsnr, df_alpha

    def load_from_pt(self, pt_sd):
        self.enc.load_from_pt(pt_sd, "enc")
        self.erb_dec.load_from_pt(pt_sd, "erb_dec")
        self.df_dec.load_from_pt(pt_sd, "df_dec")
