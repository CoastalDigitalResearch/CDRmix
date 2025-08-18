#!/bin/zsh

# CDRmix Repository Builder - Complete Lean 4 Proof Scaffolding
# Builds entire repository structure with all proof files incorporated

set -euo pipefail

echo "🔨 Building CDRmix repository structure..."

# Create base directory structure
mkdir -p specs/lean/CDRmix/{Blocks,Reasoning}
mkdir -p configs data training eval ops

echo "📁 Directory structure created"

# Create Lakefile.lean (optional standalone build)
cat > specs/lean/Lakefile.lean << 'EOF'
import Lake
open Lake DSL

package «CDRmix» where
  -- add any config as needed

require mathlib from git
  "https://github.com/leanprover-community/mathlib4.git"

lean_lib CDRmix where
  srcDir := "."
EOF

# Create Types.lean - Core type definitions
cat > specs/lean/CDRmix/Types.lean << 'EOF'
import Mathlib.Data.Real.Basic
import Mathlib.Analysis.NormedSpace.Basic
import Mathlib.Topology.Algebra.Module

namespace CDRmix

abbrev R := ℝ
abbrev Vec (d : Nat) := Fin d → R

-- Standard ℝ-vector space structure comes from function space instance

-- Norm (sup or euclidean); pick euclidean for smoothness-friendly lemmas
-- mathlib provides instances for Pi types with ℓ2 or ℓ∞; we rely on these.

end CDRmix
EOF

# Create Schedule.lean - 25%/75% mixing proofs
cat > specs/lean/CDRmix/Schedule.lean << 'EOF'
namespace CDRmix

inductive BlockType | RWKV | Tx deriving DecidableEq, Repr

structure ScheduleSpec where
  L : Nat
  txRatio : Rat := (1/4)  -- 25%

/-- Deterministic schedule generator: top-of-x (Tx stacked at top). -/
def scheduleTopOfX (S : ScheduleSpec) : List BlockType :=
  let nT := Nat.ofNat <| Int.toNat ((S.txRatio * S.L).floor)
  (List.replicate (S.L - nT) BlockType.RWKV) ++ (List.replicate nT BlockType.Tx)

/-- Deterministic schedule generator: interleave every k (e.g., 4). -/
def scheduleInterleave (S : ScheduleSpec) (k : Nat := 4) : List BlockType :=
  (List.range S.L).map (fun i => if (i+1) % k = 0 then BlockType.Tx else BlockType.RWKV)

theorem scheduleTopOfX_len (S : ScheduleSpec) :
  (scheduleTopOfX S).length = S.L := by
  -- length of replicate concat
  simp [scheduleTopOfX]

theorem scheduleInterleave_len (S : ScheduleSpec) (k : Nat := 4) :
  (scheduleInterleave S k).length = S.L := by
  simp [scheduleInterleave]

/-- Interleave: count of Tx is ⌊L/k⌋. -/
theorem interleave_count_Tx (S : ScheduleSpec) (k : Nat := 4) :
  ((scheduleInterleave S k).count BlockType.Tx) = S.L / k := by
  -- TODO: prove by counting multiples of k in 1..L
  sorry

/-- Top-of-x: Tx count is ⌊txRatio·L⌋. -/
theorem topOfX_count_Tx (S : ScheduleSpec) :
  ((scheduleTopOfX S).count BlockType.Tx) =
    Int.toNat ((S.txRatio * S.L).floor) := by
  -- by construction
  simp [scheduleTopOfX]

end CDRmix
EOF

# Create MoE.lean - Mixture of Experts capacity proofs
cat > specs/lean/CDRmix/MoE.lean << 'EOF'
namespace CDRmix

structure MoEConfig where
  E : Nat           -- experts
  topK : Nat
  ϕ : ℝ            -- capacity factor, ≥ 1

/-- Capacity rule: with N tokens, per-expert capacity cap = ⌈ϕ * (N*topK / E)⌉. -/
def perExpertCapacity (cfg : MoEConfig) (N : Nat) : Nat :=
  Nat.ceil (cfg.ϕ * (N * cfg.topK : ℝ) / cfg.E)

-- Router is abstract; we require it returns exactly topK experts per token.
structure Routing (cfg : MoEConfig) (N : Nat) where
  assign : Fin N → Finset (Fin cfg.E)
  card_topK : ∀ t, (assign t).card = cfg.topK

/-- No-overflow property if each expert receives ≤ capacity. -/
def NoOverflow {cfg : MoEConfig} {N : Nat} (r : Routing cfg N) : Prop :=
  ∀ (e : Fin cfg.E),
    (Finset.card {t : Fin N | e ∈ r.assign t}). ≤ perExpertCapacity cfg N

/-- If tokens are balanced up to rounding under topK, capacity with ϕ≥1 suffices. -/
theorem capacity_sufficient
  (cfg : MoEConfig) (hϕ : cfg.ϕ ≥ 1) (N : Nat)
  (r : Routing cfg N)
  (Hbal : ∀ e, (Finset.card {t : Fin N | e ∈ r.assign t}) ≤
               Nat.ceil ((N * cfg.topK : ℝ) / cfg.E)) :
  NoOverflow r := by
  -- Compare right-hand sides; use monotonicity of ceil and hϕ ≥ 1
  -- ceil(a) ≤ ceil(ϕ·a) for ϕ≥1
  intro e; have := Hbal e
  -- TODO: turn inequality on ℝ via coercions; finish with ceil monotonicity
  sorry

end CDRmix
EOF

# Create Lipschitz.lean - Stability bounds
cat > specs/lean/CDRmix/Lipschitz.lean << 'EOF'
import Mathlib.Topology.MetricSpace.Lipschitz

namespace CDRmix

/-- Composition Lipschitz bound: `K(g ∘ f) ≤ K(g)*K(f)`; from mathlib. -/
-- You'll use mathlib lemmas: `LipschitzWith.comp`, etc.

/-- Suppose each block is `LipschitzWith K_i`. Then the composed network is
    `LipschitzWith (∏ K_i)`. -/
axiom lipschitz_of_blocks
  {d : Nat} (blocks : List (Vec d → Vec d))
  (Ks : List 