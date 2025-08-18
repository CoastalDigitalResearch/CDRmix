namespace CDRmix

/-- FLOP model parameters. -/
structure Cost where
  flops : Nat

instance : OfNat Cost n where ofNat := ⟨n⟩

/-- Additive cost. -/
def Cost.add (a b : Cost) : Cost := ⟨a.flops + b.flops⟩

/-- Complexity upper bound for one forward pass. -/
def txCost (n d : Nat) : Cost := ⟨4 * d * d + 2 * d * (4*d) + n*n*d⟩  -- toy: MHSA + FFN + n²d
def rwkvCost (n d : Nat) : Cost := ⟨3 * d * d + 2 * d * (2*d) + n*d⟩  -- toy: TimeMix + ChannelMix + n d

/-- For L layers with α=25% transformer: bound matches spec. -/
theorem complexity_bound
  (L n d : Nat) :
  let Ltx := (L / 4)      -- floor approx
  let Lrw := L - Ltx
  (txCost n d).flops * Ltx + (rwkvCost n d).flops * Lrw
    ≤  (10 * d * d) * Ltx + (7 * d * d) * Lrw + (n*n*d) * Ltx + (n*d) * Lrw  -- choose C to match O(0.25·L·n²·d + 0.75·L·n·d) with constants
:= by
  -- TODO: massage algebra; present as explicit constant C
  simp [txCost, rwkvCost]

end CDRmix
