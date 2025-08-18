import Lake
open Lake DSL

package «CDRmix» where
  -- add any config as needed

require mathlib from git
  "https://github.com/leanprover-community/mathlib4.git"

lean_lib CDRmix where
  srcDir := "."
