# CDRmix Formal Specifications

This directory contains Lean 4 formal proofs for the CDRmix architecture properties.

## Structure

- `lean/CDRmix/` - Main proof modules
  - `Types.lean` - Core type definitions
  - `Schedule.lean` - 25%/75% layer mixing proofs
  - `MoE.lean` - Mixture of experts capacity guarantees
  - `Lipschitz.lean` - Stability bounds
  - `Complexity.lean` - FLOP analysis
  - `Blocks/` - Individual layer proofs
  - `Reasoning/` - Episodic reasoning proofs
  - `Model.lean` - End-to-end correctness

## Building

```bash
cd specs/lean
lake build
```

## Proof Status

- ✅ Basic structure and type safety
- 🚧 Schedule counting theorems (TODO: complete interleave proof)
- 🚧 MoE capacity bounds (TODO: ceil monotonicity)
- 🚧 Smoothness proofs (TODO: softmax/GELU)
- ✅ Adapter invariants
- 🚧 End-to-end composition

Replace `sorry` placeholders with actual proofs incrementally.
