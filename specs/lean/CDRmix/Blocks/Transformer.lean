import CDRmix.Types
import CDRmix.Blocks.Linear
import CDRmix.Blocks.Nonlinear

namespace CDRmix

/-- Multi-head self-attention block specification -/
structure MHSAConfig where
  d_model : Nat
  n_heads : Nat
  d_head : Nat := d_model / n_heads

/-- MHSA is composition of linear maps + softmax + residual -/
-- TODO: Formalize attention mechanism and prove differentiability
axiom mhsa_differentiable (cfg : MHSAConfig) : True

/-- FFN is two linear layers with GELU -/
-- TODO: Prove composition preserves smoothness
axiom ffn_differentiable (d_model d_ff : Nat) : True

end CDRmix
