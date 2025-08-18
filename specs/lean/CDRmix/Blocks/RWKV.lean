import CDRmix.Types
import CDRmix.Blocks.Linear
import CDRmix.Blocks.Nonlinear

namespace CDRmix

/-- RWKV TimeMix configuration -/
structure TimeMixConfig where
  d_model : Nat

/-- RWKV ChannelMix configuration -/
structure ChannelMixConfig where
  d_model : Nat
  d_ff : Nat

/-- TimeMix is linear in sequence length -/
-- TODO: Prove O(n*d) complexity and differentiability
axiom timemix_linear_complexity (cfg : TimeMixConfig) (n : Nat) : True

/-- ChannelMix is position-wise FFN with special gating -/
-- TODO: Prove differentiability and complexity bounds
axiom channelmix_differentiable (cfg : ChannelMixConfig) : True

end CDRmix
