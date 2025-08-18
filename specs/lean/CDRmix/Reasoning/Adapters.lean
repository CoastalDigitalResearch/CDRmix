namespace CDRmix

abbrev R := ℝ
abbrev Vec (d : Nat) := Fin d → R

/-- Ephemeral adapter delta as an endomorphism on Vec d. -/
structure Adapter (d : Nat) where
  apply : Vec d → Vec d

/-- Episode state carries base params θ and a multiset of accepted deltas. -/
structure EpisodeState (d : Nat) where
  θ    : Vec d → Vec d
  acc  : List (Adapter d)    -- accepted
  prop : List (Adapter d)    -- currently proposed (uncommitted)

/-- Commit accepts all current proposals atomically; rollback clears them. -/
def commit (s : EpisodeState d) : EpisodeState d :=
  { s with acc := s.acc ++ s.prop, prop := [] }

def rollback (s : EpisodeState d) : EpisodeState d :=
  { s with prop := [] }

/-- Invariant: rolling commit is idempotent; rollback clears without changing `acc`. -/
theorem commit_idempotent (s : EpisodeState d) :
  commit (commit s) = commit s := by
  -- list append idempotence with empty second `prop`
  cases s <;> simp [commit]

theorem rollback_neutral_acc (s : EpisodeState d) :
  (rollback s).acc = s.acc := by
  cases s <;> simp [rollback]

end CDRmix
