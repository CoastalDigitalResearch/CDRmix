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
