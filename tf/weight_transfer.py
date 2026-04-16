"""Weight transfer from PyTorch DeepFilterNet2 to TensorFlow.

Usage:
    python tf/weight_transfer.py --checkpoint models/DeepFilterNet2/checkpoints/model_96.ckpt.best

Loads a PyTorch DeepFilterNet2 checkpoint, builds the TF model,
transfers all weights, and saves as TF SavedModel + optional TFLite.
"""

import argparse
import numpy as np

# Weight transformation functions
# ================================

def transpose_conv2d(weight):
    """PyTorch Conv2d: (out, in, kH, kW) -> Keras Conv2D: (kH, kW, in, out)"""
    return np.transpose(weight, (2, 3, 1, 0))


def transpose_linear(weight):
    """PyTorch Linear: (out, in) -> Keras Dense: (in, out)"""
    return weight.T


def _reorder_gates_rzn_to_zrh(w, axis):
    """Reorder gate chunks from PyTorch [r, z, n] to Keras [z, r, h]."""
    import numpy as np
    r, z, n = np.split(w, 3, axis=axis)
    return np.concatenate([z, r, n], axis=axis)


def transfer_gru_weights(weight_ih, weight_hh, bias_ih, bias_hh):
    """Transfer PyTorch nn.GRU weights to Keras built-in GRU.

    PyTorch gate order: r, z, n (reset, update, new)
    Keras gate order:   z, r, h (update, reset, new)

    Returns: kernel (I, 3H), recurrent_kernel (H, 3H), bias (2, 3H)
    """
    import numpy as np
    # Transpose: PyTorch (out, in) -> Keras (in, out)
    kernel = _reorder_gates_rzn_to_zrh(weight_ih.T, axis=1)
    recurrent_kernel = _reorder_gates_rzn_to_zrh(weight_hh.T, axis=1)
    # Keras GRU with reset_after=True uses bias shape (2, 3*H)
    bi = _reorder_gates_rzn_to_zrh(bias_ih, axis=0)
    bh = _reorder_gates_rzn_to_zrh(bias_hh, axis=0)
    bias = np.stack([bi, bh], axis=0)  # (2, 3*H)
    return kernel, recurrent_kernel, bias


def transfer_batchnorm(weight, bias, running_mean, running_var):
    """PyTorch BN: gamma, beta, mean, var -> Keras BN: same order."""
    return [weight, bias, running_mean, running_var]


# Main transfer function
# =======================

def transfer_weights(pt_state_dict, tf_model):
    """Transfer all weights from PyTorch state dict to TF DfNet model."""

    def pt(key):
        return pt_state_dict[key].cpu().numpy()

    def set_conv2d_norm_act(tf_layer, prefix, is_transpose=False):
        """Transfer weights for a Conv2dNormAct block.

        Handles both separable (group_convs + pw_conv) and non-separable (conv).
        Auto-detects structure from checkpoint keys.

        When is_transpose=True, the main conv kernel is flipped along kH and kW
        to convert PyTorch ConvTranspose2d weights for use in regular Conv2D.
        """
        # Find conv weight keys and determine structure
        conv_weights = {}
        prefix_dot = prefix + "."
        for k in sorted(pt_state_dict.keys()):
            if k.startswith(prefix_dot):
                rest = k[len(prefix_dot):]
                parts = rest.split(".")
                if len(parts) == 2:
                    idx = int(parts[0])
                    attr = parts[1]
                    if idx not in conv_weights:
                        conv_weights[idx] = {}
                    conv_weights[idx][attr] = k

        # Parse layer structure from checkpoint indices
        sorted_idxs = sorted(conv_weights.keys())

        # Find the main conv (first 4D weight)
        main_conv_idx = None
        pw_conv_idx = None
        bn_idx = None

        for idx in sorted_idxs:
            attrs = conv_weights[idx]
            if "weight" in attrs:
                w = pt(attrs["weight"])
                if w.ndim == 4 and main_conv_idx is None:
                    main_conv_idx = idx
                elif w.ndim == 4 and w.shape[2] == 1 and w.shape[3] == 1:
                    pw_conv_idx = idx
                elif w.ndim == 1 and "running_mean" in attrs:
                    bn_idx = idx

        if main_conv_idx is None:
            return

        # Transfer main conv weights
        w = pt(conv_weights[main_conv_idx]["weight"])

        # For ConvTranspose2d: flip kernel along spatial dims (kH, kW)
        if is_transpose:
            w = w[:, :, ::-1, ::-1].copy()
        # Check if the conv is grouped (depthwise): in_ch_per_group == 1
        pt_groups = w.shape[0] // (w.shape[0] // max(w.shape[1], 1)) if w.shape[1] == 1 and w.shape[0] > 1 else 1

        if hasattr(tf_layer, 'group_convs') and tf_layer.groups > 1:
            # Separable: split grouped conv weights into per-group
            n_groups = tf_layer.groups
            out_per = w.shape[0] // n_groups
            for g in range(n_groups):
                gw = w[g * out_per:(g + 1) * out_per]
                gw_tf = transpose_conv2d(gw)
                tf_layer.group_convs[g].set_weights([gw_tf])
        else:
            tf_layer.conv.set_weights([transpose_conv2d(w)])

        # Transfer pointwise conv
        if pw_conv_idx is not None and hasattr(tf_layer, 'pw_conv') and tf_layer.pw_conv is not None:
            pw_w = pt(conv_weights[pw_conv_idx]["weight"])
            tf_layer.pw_conv.set_weights([transpose_conv2d(pw_w)])
            # BN is after pw_conv
            next_idx = pw_conv_idx + 1
        else:
            next_idx = main_conv_idx + 1

        # Transfer BatchNorm
        if bn_idx is None:
            # Try next index after conv/pw
            for idx in sorted_idxs:
                if idx >= next_idx and "running_mean" in conv_weights[idx]:
                    bn_idx = idx
                    break

        if bn_idx is not None and hasattr(tf_layer, 'norm') and tf_layer.use_norm:
            attrs = conv_weights[bn_idx]
            tf_layer.norm.set_weights(transfer_batchnorm(
                pt(attrs["weight"]),
                pt(attrs["bias"]),
                pt(attrs["running_mean"]),
                pt(attrs["running_var"]),
            ))

    def set_squeezed_gru(tf_sgru, prefix):
        """Transfer weights for SqueezedGRU.

        PyTorch SqueezedGRU structure:
          linear_in: Sequential(GroupedLinearEinsum, ReLU)
          gru: nn.GRU(hidden, hidden, num_layers=N)
          gru_skip: optional
          linear_out: Sequential(GroupedLinearEinsum, ReLU) or Identity
        """
        # linear_in: GroupedLinearEinsum
        lin_in_key = f"{prefix}.linear_in.0.weight"
        if lin_in_key in pt_state_dict:
            tf_sgru.linear_in_proj.set_weights([pt(lin_in_key)])

        # GRU layers (each layer of nn.GRU -> one Keras GRU in TF)
        for i, gru in enumerate(tf_sgru.grus):
            w_ih = pt(f"{prefix}.gru.weight_ih_l{i}")
            w_hh = pt(f"{prefix}.gru.weight_hh_l{i}")
            b_ih = pt(f"{prefix}.gru.bias_ih_l{i}")
            b_hh = pt(f"{prefix}.gru.bias_hh_l{i}")
            kernel, rec_kernel, bias = transfer_gru_weights(w_ih, w_hh, b_ih, b_hh)
            gru.set_weights([kernel, rec_kernel, bias])

        # linear_out: GroupedLinearEinsum (if exists)
        lin_out_key = f"{prefix}.linear_out.0.weight"
        if lin_out_key in pt_state_dict and tf_sgru.linear_out_proj is not None:
            tf_sgru.linear_out_proj.set_weights([pt(lin_out_key)])

    def set_grouped_linear_einsum(tf_layer, prefix):
        """Transfer GroupedLinearEinsum weights."""
        w = pt(f"{prefix}.weight")
        tf_layer.set_weights([w])

    def set_dense(tf_layer, prefix):
        """Transfer nn.Linear -> Dense."""
        w = pt(f"{prefix}.weight")
        b = pt(f"{prefix}.bias")
        tf_layer.set_weights([transpose_linear(w), b])

    enc = tf_model.enc

    # --- Encoder ---
    set_conv2d_norm_act(enc.erb_conv0, "enc.erb_conv0")
    set_conv2d_norm_act(enc.erb_conv1, "enc.erb_conv1")
    set_conv2d_norm_act(enc.erb_conv2, "enc.erb_conv2")
    set_conv2d_norm_act(enc.erb_conv3, "enc.erb_conv3")
    set_conv2d_norm_act(enc.df_conv0, "enc.df_conv0")
    set_conv2d_norm_act(enc.df_conv1, "enc.df_conv1")

    # df_fc_emb: GroupedLinearEinsum (squeeze mode)
    set_grouped_linear_einsum(enc.df_fc_emb, "enc.df_fc_emb.0")

    # emb_gru: SqueezedGRU
    set_squeezed_gru(enc.emb_gru, "enc.emb_gru")

    # lsnr_fc: Sequential(Linear, Sigmoid) -> Dense
    set_dense(enc.lsnr_dense, "enc.lsnr_fc.0")

    # --- ERB Decoder ---
    dec = tf_model.erb_dec

    # emb_gru: SqueezedGRU with output_size (has linear_out)
    set_squeezed_gru(dec.emb_gru, "erb_dec.emb_gru")

    set_conv2d_norm_act(dec.conv3p, "erb_dec.conv3p")
    set_conv2d_norm_act(dec.convt3, "erb_dec.convt3")
    set_conv2d_norm_act(dec.conv2p, "erb_dec.conv2p")
    set_conv2d_norm_act(dec.convt2, "erb_dec.convt2", is_transpose=True)
    set_conv2d_norm_act(dec.conv1p, "erb_dec.conv1p")
    set_conv2d_norm_act(dec.convt1, "erb_dec.convt1", is_transpose=True)
    set_conv2d_norm_act(dec.conv0p, "erb_dec.conv0p")
    set_conv2d_norm_act(dec.conv0_out, "erb_dec.conv0_out")

    # --- DF Decoder ---
    dfd = tf_model.df_dec
    set_conv2d_norm_act(dfd.df_convp, "df_dec.df_convp")

    # df_gru: SqueezedGRU
    set_squeezed_gru(dfd.df_gru, "df_dec.df_gru")

    # df_out: Sequential(GroupedLinearEinsum, Tanh)
    set_grouped_linear_einsum(dfd.df_out_proj, "df_dec.df_out.0")

    # df_fc_a: Sequential(Linear, Sigmoid)
    set_dense(dfd.df_fc_a, "df_dec.df_fc_a.0")

    print("Weight transfer complete.")


def main():
    parser = argparse.ArgumentParser(description="Transfer PyTorch DeepFilterNet2 weights to TF")
    parser.add_argument("--checkpoint", required=True, help="Path to PyTorch checkpoint (.pth)")
    parser.add_argument("--output-dir", default="saved_model", help="Output SavedModel directory")
    parser.add_argument("--tflite", action="store_true", help="Also export TFLite float32")
    args = parser.parse_args()

    import torch
    import tensorflow as tf
    from deepfilternet2_tf import DfNet, DEFAULT_PARAMS

    print(f"Loading PyTorch checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    # Handle different checkpoint formats
    if "model" in ckpt:
        state_dict = ckpt["model"]
    elif "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    # Get erb_inv_fb from checkpoint
    erb_inv_fb = state_dict["mask.erb_inv_fb"].cpu().numpy()
    p = DEFAULT_PARAMS

    print("Building TF model...")
    model = DfNet(erb_inv_fb, p, run_df=True)

    # Build by running a dummy forward pass
    T = 10
    dummy_erb = tf.zeros([1, 1, T, p["nb_erb"]])
    dummy_fspec = tf.zeros([1, 1, T, p["nb_df"], 2])
    _ = model(dummy_erb, dummy_fspec, training=False)

    print("Transferring weights...")
    transfer_weights(state_dict, model)

    print(f"Saving TF model to: {args.output_dir}")
    model.export(args.output_dir)

    if args.tflite:
        print("Converting to TFLite...")
        converter = tf.lite.TFLiteConverter.from_saved_model(args.output_dir)
        tflite_model = converter.convert()
        tflite_path = args.output_dir.rstrip("/") + ".tflite"
        with open(tflite_path, "wb") as f:
            f.write(tflite_model)
        print(f"TFLite saved to: {tflite_path}")

    print("Done.")


if __name__ == "__main__":
    main()
