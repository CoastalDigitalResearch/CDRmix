#!/usr/bin/env python3
# configs/param_autosizer.py
"""
Param Autosizer for CDRmix Core (Backbone-Only)

Reads cdrmix-core YAML configs and estimates parameter counts for:
- Embeddings (tied readout assumed; RoPE has no params)
- Transformer blocks (MHSA + FFN or MoE-FFN)
- RWKV blocks (TimeMix + ChannelMix or MoE-ChannelMix)
- Norms

Outputs:
- Total parameters (including all MoE experts stored)
- Active params per token (MoE with top_k active experts)
- Per-component breakdown

-------------
ASSUMPTIONS
-------------
These are *approximations* to keep the script model-agnostic and stable across
minor implementation differences. Adjust the constants below to match your code.

Transformer block:
  - MHSA: Q,K,V,O projections => 4 * d * d
  - FFN (non-MoE): 2 * d * ffn_hidden
  - Norms: 2 * d per block

RWKV block:
  - TimeMix: TM_PROJ * d * d   (default 3 * d^2: receptance/key/value lanes)
  - ChannelMix (non-MoE): 2 * d * (channel_mult * d) = 2 * channel_mult * d^2
  - Norms: 2 * d per block

MoE:
  - Replaces FFN/ChannelMix with a bank of experts
  - Per-expert FFN: 2 * d * expert_ffn_hidden
  - Total params store all experts: experts * (2 * d * expert_hid)
  - Active per token uses only top_k experts: top_k * (2 * d * expert_hid)

Embedding:
  - input embeddings: vocab_size * d
  - readout tied => no extra readout matrix counted

Biases, gating vectors, and small scalars are ignored; they’re negligible at scale.

-------------
USAGE
-------------
$ python3 configs/param_autosizer.py
$ python3 configs/param_autosizer.py --dir configs
$ python3 configs/param_autosizer.py --files configs/cdrmix-core-1b.yaml

"""

import os
import glob
import math
import argparse
import yaml
from dataclasses import dataclass

# ---------- Tunable constants ----------
TM_PROJ = 3                 # RWKV TimeMix projections (~3*d^2)
NORMS_PER_BLOCK = 2         # two norms per block (pre/post)
COUNT_NORM_PARAMS = True    # include d params per norm
INCLUDE_EMB = True          # count input embeddings
TIED_READOUT = True         # if True, do not add extra readout matrix


@dataclass
class ModelSpec:
    name: str
    vocab_size: int
    d_model: int
    n_layers: int
    transformer_pct: float
    interleave_every: int | None
    transformer_heads: int | None
    ffn_hidden: int | None
    rwkv_channel_mult: float | None
    moe_enabled: bool
    moe_scope: str | None
    moe_experts: int | None
    moe_top_k: int | None
    moe_expert_hidden: int | None


def load_spec(path: str) -> ModelSpec:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    model = cfg.get("model", {})
    blocks = cfg.get("blocks", {})
    tfm = blocks.get("transformer", {}) if isinstance(blocks, dict) else {}
    rwkv = blocks.get("rwkv", {}) if isinstance(blocks, dict) else {}
    moe = cfg.get("moe", {})

    schedule = model.get("schedule", {})
    transformer_pct = float(schedule.get("transformer_pct", 0.25))
    interleave_every = schedule.get("interleave_every", None)

    return ModelSpec(
        name=model.get("name", os.path.basename(path)),
        vocab_size=int(model.get("vocab_size", 50272)),
        d_model=int(model.get("d_model", 2048)),
        n_layers=int(model.get("n_layers", 24)),
        transformer_pct=transformer_pct,
        interleave_every=interleave_every,
        transformer_heads=tfm.get("n_heads", None),
        ffn_hidden=tfm.get("ffn_hidden", None),
        rwkv_channel_mult=float(rwkv.get("channel_mult", 2.0)) if rwkv else 2.0,
        moe_enabled=bool(moe.get("enabled", False)),
        moe_scope=moe.get("scope", None),
        moe_experts=int(moe.get("experts", 0)) if moe.get("experts", None) is not None else None,
        moe_top_k=int(moe.get("top_k", 0)) if moe.get("top_k", None) is not None else None,
        moe_expert_hidden=int(moe.get("expert_ffn_hidden", 0)) if moe.get("expert_ffn_hidden", None) is not None else None,
    )


def round_layer_split(n_layers: int, transformer_pct: float) -> tuple[int, int]:
    n_t = int(round(n_layers * transformer_pct))
    n_r = n_layers - n_t
    return n_t, n_r


def params_transformer_block(d: int, ffn_hidden: int, use_moe: bool,
                             experts: int, top_k: int, expert_hid: int) -> tuple[int, int, dict]:
    """
    Returns (total_params, active_params, breakdown)
    """
    # MHSA params
    attn = 4 * d * d  # q,k,v,o
    # FFN params (either dense or MoE bank)
    if use_moe:
        ffn_total = experts * (2 * d * expert_hid)
        ffn_active = top_k * (2 * d * expert_hid)
    else:
        ffn_total = 2 * d * ffn_hidden
        ffn_active = ffn_total

    # Norms
    norms = 2 * d if COUNT_NORM_PARAMS else 0

    total = attn + ffn_total + norms
    active = attn + ffn_active + norms
    breakdown = {
        "attn": attn,
        "ffn_total_or_bank": ffn_total,
        "ffn_active": ffn_active,
        "norms": norms,
    }
    return total, active, breakdown


def params_rwkv_block(d: int, channel_mult: float, use_moe: bool,
                      experts: int, top_k: int, expert_hid: int) -> tuple[int, int, dict]:
    """
    Returns (total_params, active_params, breakdown)
    """
    # TimeMix projections
    timemix = TM_PROJ * d * d

    # ChannelMix (either dense or MoE bank)
    if use_moe:
        cm_total = experts * (2 * d * expert_hid)
        cm_active = top_k * (2 * d * expert_hid)
    else:
        hidden = int(channel_mult * d)
        cm_total = 2 * d * hidden  # in + out
        cm_active = cm_total

    norms = 2 * d if COUNT_NORM_PARAMS else 0

    total = timemix + cm_total + norms
    active = timemix + cm_active + norms
    breakdown = {
        "timemix": timemix,
        "channelmix_total_or_bank": cm_total,
        "channelmix_active": cm_active,
        "norms": norms,
    }
    return total, active, breakdown


def compute_params(spec: ModelSpec) -> dict:
    d = spec.d_model
    n_layers = spec.n_layers
    n_t, n_r = round_layer_split(n_layers, spec.transformer_pct)

    # Embeddings
    emb = spec.vocab_size * d if INCLUDE_EMB else 0
    readout = 0 if TIED_READOUT else spec.vocab_size * d

    # MoE usage flags
    use_moe_on_ffn = spec.moe_enabled and (spec.moe_scope in (None, "ffn_only", "all"))
    experts = spec.moe_experts or 0
    top_k = spec.moe_top_k or 0
    expert_hid = spec.moe_expert_hidden or 0

    # Fallback for dense sizes if not provided
    ffn_hidden = spec.ffn_hidden or (4 * d)
    channel_mult = spec.rwkv_channel_mult or 2.0

    # Per-block counts
    t_total, t_active, t_bd = params_transformer_block(
        d, ffn_hidden, use_moe_on_ffn, experts, top_k, expert_hid
    )
    r_total, r_active, r_bd = params_rwkv_block(
        d, channel_mult, use_moe_on_ffn, experts, top_k, expert_hid
    )

    # Sum across depth
    backbone_total = n_t * t_total + n_r * r_total
    backbone_active = n_t * t_active + n_r * r_active

    total_params = emb + readout + backbone_total
    active_params = emb + readout + backbone_active

    return {
        "name": spec.name,
        "d_model": d,
        "n_layers": n_layers,
        "layers_transformer": n_t,
        "layers_rwkv": n_r,
        "vocab_size": spec.vocab_size,
        "moe_enabled": spec.moe_enabled,
        "experts": experts,
        "top_k": top_k,
        "expert_hidden": expert_hid,
        "ffn_hidden_dense": ffn_hidden,
        "rwkv_channel_mult": channel_mult,
        "emb_params": emb,
        "readout_params": readout,
        "transformer_block_total": t_total,
        "transformer_block_active": t_active,
        "rwkv_block_total": r_total,
        "rwkv_block_active": r_active,
        "backbone_total": backbone_total,
        "backbone_active": backbone_active,
        "total_params": total_params,
        "active_params": active_params,
        "transformer_block_breakdown": t_bd,
        "rwkv_block_breakdown": r_bd,
    }


def human(n: int) -> str:
    # Pretty print large integers
    if n >= 10**12:
        return f"{n/10**12:.2f}T"
    if n >= 10**9:
        return f"{n/10**9:.2f}B"
    if n >= 10**6:
        return f"{n/10**6:.2f}M"
    if n >= 10**3:
        return f"{n/10**3:.2f}K"
    return str(n)


def print_report(path: str, res: dict):
    print("=" * 80)
    print(f"{res['name']}  ({os.path.basename(path)})")
    print("-" * 80)
    print(f"d_model: {res['d_model']}, n_layers: {res['n_layers']}, vocab: {res['vocab_size']}")
    print(f"Layers  → Transformer: {res['layers_transformer']}, RWKV: {res['layers_rwkv']}")
    if res["moe_enabled"]:
        print(f"MoE     → experts: {res['experts']}, top_k: {res['top_k']}, expert_hidden: {res['expert_hidden']}")
    else:
        print("MoE     → disabled")
    print()
    print("Per-block (approx.):")
    print(f"  Transformer block  → total: {human(res['transformer_block_total'])} | "
          f"active: {human(res['transformer_block_active'])} | "
          f"attn: {human(res['transformer_block_breakdown']['attn'])}, "
          f"ffn_bank/total: {human(res['transformer_block_breakdown']['ffn_total_or_bank'])}")
    print(f"  RWKV block         → total: {human(res['rwkv_block_total'])} | "
          f"active: {human(res['rwkv_block_active'])} | "
          f"timemix: {human(res['rwkv_block_breakdown']['timemix'])}, "
          f"cm_bank/total: {human(res['rwkv_block_breakdown']['channelmix_total_or_bank'])}")
    print()
    print("Backbone totals:")
    print(f"  Params (backbone total stored): {human(res['backbone_total'])}")
    print(f"  Params (backbone active/token): {human(res['backbone_active'])}")
    if INCLUDE_EMB or not TIED_READOUT:
        print()
        if INCLUDE_EMB:   print(f"  Embeddings: {human(res['emb_params'])}")
        if not TIED_READOUT: print(f"  Readout  : {human(res['readout_params'])}")
    print()
    print(f"TOTAL params (stored): {human(res['total_params'])}")
    print(f"ACTIVE params per token: {human(res['active_params'])}")
    print("=" * 80)
    print()


def main():
    ap = argparse.ArgumentParser(description="CDRmix Core Param Autosizer")
    ap.add_argument("--dir", default="configs", help="Directory to scan for YAMLs")
    ap.add_argument("--files", nargs="*", help="Specific YAML files to process")
    args = ap.parse_args()

    paths = []
    if args.files:
        for f in args.files:
            if os.path.isdir(f):
                paths.extend(glob.glob(os.path.join(f, "*.yaml")))
            else:
                paths.append(f)
    else:
        paths = glob.glob(os.path.join(args.dir, "*.yaml"))

    if not paths:
        print("No YAML files found. Use --dir or --files to specify configs.")
        return

    for p in sorted(paths):
        try:
            spec = load_spec(p)
            res = compute_params(spec)
            print_report(p, res)
        except Exception as e:
            print(f"[ERROR] {p}: {e}")

if __name__ == "__main__":
    main()
