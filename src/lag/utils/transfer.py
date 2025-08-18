from functools import partial
from typing import Dict

# from lag.models.musicgen_lm import MusicgenLm


def transfer(src, dest, srcname, destname):
    dest[destname].copy_(src[srcname])


def transfer_weights(af_dict: Dict, lag_dict: Dict, n_layers: int,
                     has_xatt: bool) -> Dict:

    # af_dict = audiocraft_lm.state_dict()
    # lag_dict = lag_lm.state_dict()

    t = partial(transfer, af_dict, lag_dict)

    hop = 3 if has_xatt else 2

    # transfer conditioner
    for s in ("weight", "bias"):
        lag_dict[f"conditioner.output_proj.{s}"] = af_dict[
            f"condition_provider.conditioners."
            f"description.output_proj.{s}"].clone()
        # t(
        #     f"condition_provider.conditioners.description.output_proj.{s}",
        #     f"conditioner.output_proj.{s}",
        # )

    # transfer embeddings
    for i in range(4):
        t(f"emb.{i}.weight", f"decoder.token_emb.emb.{i}.weight")

    # transfer decoder layers
    for layeridx in range(n_layers):

        lamsi_prefix = "decoder.attn_layers.layers"
        af_prefix = f"transformer.layers.{layeridx}"

        # ---- LAMSI SUBLAYER 0: self attention ----
        lamsi_prefix_layer = f"{lamsi_prefix}.{layeridx * hop}"

        # att norm
        for s in ("weight", "bias"):
            t(f"{af_prefix}.norm1.{s}", f"{lamsi_prefix_layer}.0.0.{s}")

        # attention q, k, v, out
        af_qkv = af_dict[f"{af_prefix}.self_attn.in_proj_weight"]
        assert (af_qkv.shape[0] % 3) == 0
        dim = af_qkv.shape[0] // 3
        att_q = af_qkv[:dim]
        att_k = af_qkv[dim:dim * 2]
        att_v = af_qkv[dim * 2:]
        lag_dict[f"{lamsi_prefix_layer}.1.to_q.weight"].copy_(att_q)
        lag_dict[f"{lamsi_prefix_layer}.1.to_k.weight"].copy_(att_k)
        lag_dict[f"{lamsi_prefix_layer}.1.to_v.weight"].copy_(att_v)
        t(
            f"{af_prefix}.self_attn.out_proj.weight",
            f"{lamsi_prefix_layer}.1.to_out.weight",
        )

        # ---- LAMSI SUBLAYER 1: cross attention ----
        if has_xatt:
            lamsi_prefix_layer = f"{lamsi_prefix}.{layeridx * hop + 1}"

            # xatt norm
            for s in ("weight", "bias"):
                t(f"{af_prefix}.norm_cross.{s}",
                  f"{lamsi_prefix_layer}.0.0.{s}")

            # xatt q, k, v, out
            af_xqkv = af_dict[f"{af_prefix}.cross_attention.in_proj_weight"]
            assert (af_xqkv.shape[0] % 3) == 0
            dim = af_xqkv.shape[0] // 3
            xatt_q = af_xqkv[:dim]
            xatt_k = af_xqkv[dim:dim * 2]
            xatt_v = af_xqkv[dim * 2:]
            lag_dict[f"{lamsi_prefix_layer}.1.to_q.weight"].copy_(xatt_q)
            lag_dict[f"{lamsi_prefix_layer}.1.to_k.weight"].copy_(xatt_k)
            lag_dict[f"{lamsi_prefix_layer}.1.to_v.weight"].copy_(xatt_v)
            t(
                f"{af_prefix}.cross_attention.out_proj.weight",
                f"{lamsi_prefix_layer}.1.to_out.weight",
            )

        # ---- LAMSI SUBLAYER 2: feed forward ----
        lamsi_prefix_layer = f"{lamsi_prefix}.{layeridx * hop + (hop - 1)}"

        # ff norm
        for s in ("weight", "bias"):
            t(f"{af_prefix}.norm2.{s}", f"{lamsi_prefix_layer}.0.0.{s}")

        # ff layers
        t(f"{af_prefix}.linear1.weight",
          f"{lamsi_prefix_layer}.1.ff.0.0.weight")

        t(f"{af_prefix}.linear2.weight", f"{lamsi_prefix_layer}.1.ff.2.weight")

    # transfer final norm
    for s in ("weight", "bias"):
        t(f"out_norm.{s}", f"decoder.attn_layers.final_norm.{s}")

    # transfer output linears
    for i in range(4):
        t(f"linears.{i}.weight", f"decoder.to_logits.linears.{i}.weight")

    # lag_lm.load_state_dict(lag_dict)

    return lag_dict


if __name__ == "__main__":
    import torch
    from lag import config as cfg
    from lag.models.musicgen_lm import MusicgenLm
    from lag.hyperparameters import PretrainedMelodyLmParams

    lag_lm = MusicgenLm(PretrainedMelodyLmParams())

    lag_state_dict = lag_lm.state_dict()
    af_state_dict = torch.load(cfg.weights_dir() / "audiocraft-melody.pt")

    print(
        sum(p.numel()
            for k, p in af_state_dict.items()
            if "conditioner" not in k))
    print(
        sum(p.numel()
            for k, p in lag_state_dict.items()
            if "conditioner" not in k))
    print(list((k, p.numel()) for k, p in af_state_dict.items()))
    print(list((k, p.numel()) for k, p in lag_state_dict.items()))

    new_lag_dict = transfer_weights(af_state_dict, lag_state_dict, 48, False)

    torch.save(new_lag_dict, cfg.weights_dir() / "lm-melody-weights.pt")
