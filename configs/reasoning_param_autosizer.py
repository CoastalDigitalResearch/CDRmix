#!/usr/bin/env python3
# configs/reasoning_param_autosizer.py
"""
Reasoning Param Autosizer for CDRmix (Hybrid Hebbian-RL + RWKX-V Backbone)

Extends the core autosizer with estimates for:
  • Hebbian plastic adapters (plastic-LoRA) applied to selected submodules
  • RL-MemAgent policy/value heads (from actions list)
  • Working-memory summarizer projections (d <-> summary_dim)

Outputs per-config:
  - Total params (stored)       : includes all MoE experts and all adapters
  - Active params per token     : MoE uses top_k experts; adapters and heads are active

--------------------------------------------------------------------------------
ASSUMPTIONS (tweak constants below to match your implementation)
--------------------------------------------------------------------------------
Backbone (same as core):
  Transformer block:
    - MHSA Q,K,V,O projections      : 4 * d * d
    - FFN dense (non-MoE)           : 2 * d * ffn_hidden
    - Norms per block               : 2 * d
  RWKV block:
    - TimeMix projections           : TM_PROJ * d * d      (default TM_PROJ = 3)
    - ChannelMix dense (non-MoE)    : 2 * channel_mult * d^2
    - Norms per block               : 2 * d
  MoE (applies to FFN/ChannelMix path when enabled):
    - Per-expert FFN                : 2 * d * expert_ffn_hidden
    - Stored bank                   : experts * (2 * d * expert_hidden)
    - Active per token              : top_k  * (2 * d * expert_hidden)

Reasoning additions:
  Hebbian plastic-LoRA (rank r):
    - For a weight W (out x in), LoRA params ≈ r*(in + out)
    - We apply LoRA to:
        * Transformer attention projections (q,k,v,o): 4 × (d x d)  → + (4 * 2*r*d)
        * Transformer FFN dense (if non-MoE): (d x h) & (h x d)     → + (2 * r * (d + h))
        * RWKV TimeMix: approximated as TM_PROJ × (d x d)           → + (TM_PROJ * 2*r*d)
        * RWKV ChannelMix dense (if non-MoE): (d x m) & (m x d)     → + (2 * r * (d + m)),
          where m = channel_mult * d
    - By default, we DO NOT apply LoRA per MoE expert (set PLASTIC_ON_MOE_EXPERTS=True to change).
  RL-MemAgent heads:
    - Policy logits head: d → A         (A = number of actions)
    - Value head:        d → 1
    - Optional hidden MLP for both heads: d→H→A and d→H→1 if RL_HEAD_HIDDEN > 0
  Working-memory summarizer:
    - Two projections: d → S and S → d  (S = rolling_summary_dim)

Embedding / Readout:
  - Count input embeddings (vocab_size * d)
  - Readout tied to embeddings by default (no extra matrix)

Biases, gates, small scalars are ignored.

USAGE
-----
$ python3 configs/reasoning_param_autosizer.py
$ python3 configs/reasoning_param_autosizer.py --dir configs
$ python3 configs/reasoning_param_autosizer.py --files configs/cdrmix-reason-1b.yaml
"""

import os, glob, argparse, yaml
from dataclasses import dataclass

# ---------------- Tunable constants ----------------
TM_PROJ = 3                 # number of dxd projections inside RWKV TimeMix
COUNT_NORM_PARAMS = True    # include d params per norm (2 norms / block)
INCLUDE_EMB = True          # include input embeddings
TIED_READOUT = True         # assume tied embeddings (no extra readout matrix)

# Hebbian plasticity application toggles
PLASTIC_ON_ATTN = True
PLASTIC_ON_DENSE_FFN = True     # applies only when FFN/ChannelMix is dense, not MoE
PLASTIC_ON_TIMEMIX = True
PLASTIC_ON_CHANNELMIX = True
PLASTIC_ON_MOE_EXPERTS = False  # usually too heavy; keep False unless you really want it

# RL heads
RL_HEAD_HIDDEN = 0              # 0 = simple linear heads; >0 uses d→H→A and d→H→1

# ---------------- Spec dataclasses ----------------
@dataclass
class ModelSpec:
    name: str
    vocab_size: int
    d_model: int
    n_layers: int
    transformer_pct: float
    interleave_every: int | None
    ffn_hidden: int | None
    rwkv_channel_mult: float | None
    moe_enabled: bool
    moe_scope: str | None
    moe_experts: int | None
    moe_top_k: int | None
    moe_expert_hidden: int | None

@dataclass
class ReasoningSpec:
    hebbian_rank: int
    plastic_adapters: str | None
    rolling_summary_dim: int
    actions: list[str]


# ---------------- Parsing ----------------
def load_specs(path: str) -> tuple[ModelSpec, ReasoningSpec]:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    model = cfg.get("model", {})
    blocks = cfg.get("blocks", {}) or {}
    tfm = blocks.get("transformer", {}) if isinstance(blocks, dict) else {}
    rwkv = blocks.get("rwkv", {}) if isinstance(blocks, dict) else {}
    moe = cfg.get("moe", {}) or {}
    schedule = model.get("schedule", {}) or {}

    transformer_pct = float(schedule.get("transformer_pct", 0.25))
    interleave_every = schedule.get("interleave_every", None)

    # Reasoning section
    reasoning = cfg.get("reasoning", {}) or {}
    hebb = reasoning.get("hebbian", {}) or {}
    rl = reasoning.get("rl_memagent", {}) or {}
    memory = cfg.get("memory", {}) or reasoning.get("memory", {}) or {}
    working = memory.get("working", {}) or {}

    actions = rl.get("actions", []) or []
    rolling_summary_dim = int(working.get("rolling_summary_dim", 0))

    ms = ModelSpec(
        name=model.get("name", os.path.basename(path)),
        vocab_size=int(model.get("vocab_size", 50272)),
        d_model=int(model.get("d_model", 2048)),
        n_layers=int(model.get("n_layers", 24)),
        transformer_pct=transformer_pct,
        interleave_every=interleave_every,
        ffn_hidden=tfm.get("ffn_hidden", None),
        rwkv_channel_mult=float(rwkv.get("channel_mult", 2.0)) if rwkv else 2.0,
        moe_enabled=bool(moe.get("enabled", False)),
        moe_scope=moe.get("scope", None),
        moe_experts=int(moe.get("experts", 0)) if moe.get("experts") is not None else None,
        moe_top_k=int(moe.get("top_k", 0)) if moe.get("top_k") is not None else None,
        moe_expert_hidden=int(moe.get("expert_ffn_hidden", 0)) if moe.get("expert_ffn_hidden") is not None else None,
    )
    rs = ReasoningSpec(
        hebbian_rank=int(hebb.get("rank", 0)),
        plastic_adapters=hebb.get("adapters", None),
        rolling_summary_dim=rolling_summary_dim,
        actions=actions,
    )
    return ms, rs


# ---------------- Helpers ----------------
def split_layers(n_layers: int, transformer_pct: float) -> tuple[int, int]:
    n_t = int(round(n_layers * transformer_pct))
    n_r = n_layers - n_t
    return n_t, n_r

def human(n: int) -> str:
    if n >= 10**12: return f"{n/1e12:.2f}T"
    if n >= 10**9:  return f"{n/1e9:.2f}B"
    if n >= 10**6:  return f"{n/1e6:.2f}M"
    if n >= 10**3:  return f"{n/1e3:.2f}K"
    return str(n)


# ---------------- Backbone params (same as core) ----------------
def params_transformer_block(d: int, ffn_hidden: int, use_moe: bool,
                             experts: int, top_k: int, expert_hid: int,
                             count_norms: bool) -> tuple[int, int]:
    attn = 4 * d * d
    if use_moe:
        ffn_total = experts * (2 * d * expert_hid)
        ffn_active = top_k * (2 * d * expert_hid)
    else:
        ffn_total = 2 * d * ffn_hidden
        ffn_active = ffn_total
    norms = 2 * d if count_norms else 0
    return attn + ffn_total + norms, attn + ffn_active + norms

def params_rwkv_block(d: int, channel_mult: float, use_moe: bool,
                      experts: int, top_k: int, expert_hid: int,
                      count_norms: bool) -> tuple[int, int]:
    timemix = TM_PROJ * d * d
    if use_moe:
        cm_total = experts * (2 * d * expert_hid)
        cm_active = top_k * (2 * d * expert_hid)
    else:
        hidden = int(channel_mult * d)
        cm_total = 2 * d * hidden
        cm_active = cm_total
    norms = 2 * d if count_norms else 0
    return timemix + cm_total + norms, timemix + cm_active + norms


# ---------------- Hebbian plastic-LoRA params ----------------
def lora_params_for_linear(in_dim: int, out_dim: int, rank: int) -> int:
    # LoRA adds A (out x r) and B (r x in) => r*(out + in)
    return rank * (out_dim + in_dim)

def hebbian_params_per_transformer_block(d: int, h: int, use_moe: bool, expert_hid: int, rank: int) -> int:
    if rank <= 0:
        return 0
    total = 0
    if PLASTIC_ON_ATTN:
        # q,k,v,o each dxd
        total += 4 * lora_params_for_linear(d, d, rank)  # = 4 * 2*r*d
    if PLASTIC_ON_DENSE_FFN and not use_moe:
        # dense FFN: d->h and h->d
        total += lora_params_for_linear(d, h, rank)
        total += lora_params_for_linear(h, d, rank)
    if PLASTIC_ON_MOE_EXPERTS and use_moe:
        # OPTIONAL (usually disabled): adapters per expert FFN (very heavy)
        total += (8 * (  # experts=8 is common; but read from use_moe/expert_hid
            lora_params_for_linear(d, expert_hid, rank) +
            lora_params_for_linear(expert_hid, d, rank)
        ))
    return total

def hebbian_params_per_rwkv_block(d: int, channel_mult: float, use_moe: bool, expert_hid: int, rank: int) -> int:
    if rank <= 0:
        return 0
    total = 0
    if PLASTIC_ON_TIMEMIX:
        # approximated as TM_PROJ × (d x d)
        total += TM_PROJ * lora_params_for_linear(d, d, rank)  # = TM_PROJ * 2*r*d
    if PLASTIC_ON_CHANNELMIX and not use_moe:
        m = int(channel_mult * d)
        total += lora_params_for_linear(d, m, rank)
        total += lora_params_for_linear(m, d, rank)
    if PLASTIC_ON_MOE_EXPERTS and use_moe:
        total += (8 * (  # per-expert option; off by default
            lora_params_for_linear(d, expert_hid, rank) +
            lora_params_for_linear(expert_hid, d, rank)
        ))
    return total


# ---------------- RL heads & memory summarizer ----------------
def rl_heads_params(d: int, num_actions: int, hidden: int) -> int:
    if num_actions <= 0:
        return 0
    if hidden and hidden > 0:
        # policy: d->H->A, value: d->H->1
        return (d * hidden + hidden * num_actions) + (d * hidden + hidden * 1)
    else:
        # linear heads: policy d->A, value d->1
        return d * num_actions + d * 1

def working_memory_params(d: int, summary_dim: int) -> int:
    if summary_dim <= 0:
        return 0
    # d->S and S->d
    return d * summary_dim + summary_dim * d


# ---------------- Aggregate compute ----------------
def compute_params(model: ModelSpec, reason: ReasoningSpec) -> dict:
    d = model.d_model
    n_layers = model.n_layers
    n_t, n_r = split_layers(n_layers, model.transformer_pct)

    emb = model.vocab_size * d if INCLUDE_EMB else 0
    readout = 0 if TIED_READOUT else model.vocab_size * d

    use_moe = model.moe_enabled  # scope assumed to cover FFN/ChannelMix
    experts = model.moe_experts or 0
    top_k = model.moe_top_k or 0
    expert_hid = model.moe_expert_hidden or 0
    ffn_hidden = model.ffn_hidden or (4 * d)
    channel_mult = model.rwkv_channel_mult or 2.0

    # Backbone per-block params
    t_total, t_active = params_transformer_block(d, ffn_hidden, use_moe, experts, top_k, expert_hid, COUNT_NORM_PARAMS)
    r_total, r_active = params_rwkv_block(d, channel_mult, use_moe, experts, top_k, expert_hid, COUNT_NORM_PARAMS)

    # Hebbian adapters per block
    rank = reason.hebbian_rank
    hebb_t = hebbian_params_per_transformer_block(d, ffn_hidden, use_moe, expert_hid, rank)
    hebb_r = hebbian_params_per_rwkv_block(d, channel_mult, use_moe, expert_hid, rank)

    # Sum backbone + adapters across depth
    backbone_total = n_t * (t_total + hebb_t) + n_r * (r_total + hebb_r)
    backbone_active = n_t * (t_active + hebb_t) + n_r * (r_active + hebb_r)

    # RL heads & working-memory summarizer
    num_actions = len(reason.actions)
    rl_params = rl_heads_params(d, num_actions, RL_HEAD_HIDDEN)
    wm_params = working_memory_params(d, reason.rolling_summary_dim)

    total_params = emb + readout + backbone_total + rl_params + wm_params
    active_params = emb + readout + backbone_active + rl_params + wm_params

    return {
        "name": model.name,
        "d_model": d,
        "n_layers": n_layers,
        "layers_transformer": n_t,
        "layers_rwkv": n_r,
        "vocab_size": model.vocab_size,
        "moe_enabled": model.moe_enabled,
        "experts": experts,
        "top_k": top_k,
        "expert_hidden": expert_hid,
        "ffn_hidden_dense": ffn_hidden,
        "rwkv_channel_mult": channel_mult,
        "hebbian_rank": rank,
        "actions": num_actions,
        "summary_dim": reason.rolling_summary_dim,
        "emb_params": emb,
        "readout_params": readout,
        "backbone_total": backbone_total,
        "backbone_active": backbone_active,
        "rl_heads_params": rl_params,
        "wm_params": wm_params,
        "total_params": total_params,
        "active_params": active_params,
    }


# ---------------- Reporting ----------------
def print_report(path: str, res: dict):
    print("=" * 88)
    print(f"{res['name']}  ({os.path.basename(path)})")
    print("-" * 88)
    print(f"d_model={res['d_model']}, n_layers={res['n_layers']}, vocab={res['vocab_size']}")
    print(f"Layers → Transformer: {res['layers_transformer']}, RWKV: {res['layers_rwkv']}")
    if res["moe_enabled"]:
        print(f"MoE    → experts: {res['experts']}, top_k: {res['top_k']}, expert_hidden: {res['expert_hidden']}")
    else:
        print("MoE    → disabled")
    print(f"Hebbian rank: {res['hebbian_rank']}, RL actions: {res['actions']}, WM summary_dim: {res['summary_dim']}")
    print()
    print("Totals (approx.):")
    print(f"  Backbone stored      : {human(res['backbone_total'])}")
    print(f"  Backbone active/token: {human(res['backbone_active'])}")
    if res["rl_heads_params"] > 0:
        print(f"  RL heads             : {human(res['rl_heads_params'])}")
    if res["wm_params"] > 0:
        print(f"  Working memory proj  : {human(res['wm_params'])}")
    if res["emb_params"] > 0:
        print(f"  Embeddings           : {human(res['emb_params'])}")
    if res["readout_params"] > 0:
        print(f"  Readout              : {human(res['readout_params'])}")
    print()
    print(f"TOTAL params (stored): {human(res['total_params'])}")
    print(f"ACTIVE params/token  : {human(res['active_params'])}")
    print("=" * 88)
    print()


# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser(description="CDRmix Reasoning Param Autosizer")
    ap.add_argument("--dir", default="configs", help="Directory to scan for YAMLs")
    ap.add_argument("--files", nargs="*", help="Specific YAMLs to process")
    args = ap.parse_args()

    if args.files:
        paths = []
        for f in args.files:
            if os.path.isdir(f):
                paths.extend(glob.glob(os.path.join(f, "*.yaml")))
            else:
                paths.append(f)
    else:
        paths = glob.glob(os.path.join(args.dir, "*.yaml"))

    if not paths:
        print("No YAML files found. Use --dir or --files.")
        return

    for p in sorted(paths):
        try:
            model, reason = load_specs(p)
            res = compute_params(model, reason)
            print_report(p, res)
        except Exception as e:
            print(f"[ERROR] {p}: {e}")

if __name__ == "__main__":
    main()
