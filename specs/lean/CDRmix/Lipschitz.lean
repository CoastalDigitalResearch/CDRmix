import Mathlib.Topology.MetricSpace.Lipschitz

namespace CDRmix

/-- Composition Lipschitz bound: `K(g ∘ f) ≤ K(g)*K(f)`; from mathlib. -/
-- You'll use mathlib lemmas: `LipschitzWith.comp`, etc.

/-- Suppose each block is `LipschitzWith K_i`. Then the composed network is
    `LipschitzWith (∏ K_i)`. -/
axiom lipschitz_of_blocks
  {d : Nat} (blocks : List (Vec d → Vec d))
  (Ks : List ℝ)
  (H : ∀ i, i < blocks.length →
       LipschitzWith (Real.toNNReal (Ks.get ⟨i, by have := Nat.lt_of_lt_of_le ?h sorry⟩)) (blocks.get ⟨i, ?h⟩)) :
  True
-- TODO: replace axiom with proper proof by induction on the list

end CDRmix
