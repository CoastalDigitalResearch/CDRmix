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
