import CDRmix.Model

namespace CDRmix

-- Smoke tests to ensure basic definitions work
#check ScheduleSpec
#check BlockType.RWKV
#check BlockType.Tx
#check MoEConfig
#check EpisodeState
#check cdrmix_correctness

-- Basic evaluation tests
example : (scheduleTopOfX ⟨8, 1/4⟩).length = 8 := by simp [scheduleTopOfX_len]

example : let cfg : MoEConfig := ⟨8, 2, 1.25⟩; cfg.ϕ ≥ 1 := by norm_num

-- TODO: Add more smoke tests as proofs are completed

end CDRmix
