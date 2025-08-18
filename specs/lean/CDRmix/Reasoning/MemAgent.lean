namespace CDRmix

/-- Abstract policy π(a|s) with KL constraint and advantage estimate Â. -/
structure PolicyUpdate where
  klLimit : ℝ
  -- … more fields …

/-- Classical monotonic improvement (TRPO/PPO-style) under KL and bounded advantage bias.
    We encode the assumption and expose the bound to be connected to our GRPO variant. -/
axiom monotone_improvement_bound
  (upd : PolicyUpdate)
  -- assumptions: valid Â, trust region small, etc.
  : True

end CDRmix
