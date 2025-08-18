namespace CDRmix

inductive BlockType | RWKV | Tx deriving DecidableEq, Repr

structure ScheduleSpec where
  L : Nat
  txRatio : Rat := (1/4)  -- 25%

/-- Deterministic schedule generator: top-of-x (Tx stacked at top). -/
def scheduleTopOfX (S : ScheduleSpec) : List BlockType :=
  let nT := Nat.ofNat <| Int.toNat ((S.txRatio * S.L).floor)
  (List.replicate (S.L - nT) BlockType.RWKV) ++ (List.replicate nT BlockType.Tx)

/-- Deterministic schedule generator: interleave every k (e.g., 4). -/
def scheduleInterleave (S : ScheduleSpec) (k : Nat := 4) : List BlockType :=
  (List.range S.L).map (fun i => if (i+1) % k = 0 then BlockType.Tx else BlockType.RWKV)

theorem scheduleTopOfX_len (S : ScheduleSpec) :
  (scheduleTopOfX S).length = S.L := by
  -- length of replicate concat
  simp [scheduleTopOfX]

theorem scheduleInterleave_len (S : ScheduleSpec) (k : Nat := 4) :
  (scheduleInterleave S k).length = S.L := by
  simp [scheduleInterleave]

/-- Interleave: count of Tx is ⌊L/k⌋. -/
theorem interleave_count_Tx (S : ScheduleSpec) (k : Nat := 4) :
  ((scheduleInterleave S k).count BlockType.Tx) = S.L / k := by
  -- TODO: prove by counting multiples of k in 1..L
  sorry

/-- Top-of-x: Tx count is ⌊txRatio·L⌋. -/
theorem topOfX_count_Tx (S : ScheduleSpec) :
  ((scheduleTopOfX S).count BlockType.Tx) =
    Int.toNat ((S.txRatio * S.L).floor) := by
  -- by construction
  simp [scheduleTopOfX]

end CDRmix
