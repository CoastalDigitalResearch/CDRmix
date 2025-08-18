import Mathlib.Analysis.Calculus.ContDiff
import Mathlib.Analysis.Normed.Field.Basic

namespace CDRmix

/-- Smooth GELU (elementwise). -/
def gelu (x : ℝ) : ℝ := x * 0.5 * (1 + Real.erf (x / Real.sqrt 2))

-- TODO: prove smoothness: `ContDiff ℝ ⊤ gelu`
lemma gelu_smooth : ContDiff ℝ ⊤ gelu := by
  -- follows from smoothness of erf and polynomials
  sorry

/-- Softmax on finite index set with temperature τ>0 over a fixed dimension d. -/
def softmax {d : Nat} (τ : ℝ) (x : Fin d → ℝ) : Fin d → ℝ :=
  let ex : Fin d → ℝ := fun i => Real.exp (x i / τ)
  let Z  : ℝ := (Finset.univ : Finset (Fin d)).attach.fold (· + ·) 0 (fun i _ => ex i.1)  -- sum over Fin d
  fun i => ex i / Z

-- TODO: show softmax is smooth for τ>0 and Lipschitz on bounded sets
axiom softmax_smooth {d} {τ : ℝ} (hτ : τ > 0) :
  ContDiff ℝ ⊤ (softmax (d:=d) τ)

end CDRmix
