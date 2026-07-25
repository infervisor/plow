import Plow.CLI.FastCheckD
open Plow.CLI.FastCheckD

def main : IO Unit := do
  let ne := 1391
  let n  := 587707
  let needMeet := Array.mkArray ne true
  let entries : Array EntryView := (Array.range ne).map fun i =>
    { name := s!"t{i}", offset := 0, size := 1,
      readers := (Array.range 135).map (fun k => (i * 37 + k * 4099) % n),
      writers := #[] }
  let rows : Array UInt64 := Array.mkArray (n * 16) 0x1234
  let t0 ← IO.monoMsNow
  let mut meets : Array UInt64 := Array.mkArray (ne * 16) 0x7FFFFFFFFFFFFFFF
  for _ in [0:130] do
    meets := meetKernel needMeet entries rows meets
  let t1 ← IO.monoMsNow
  IO.println s!"130 meetKernel rounds (~24.4M andRow16): {t1 - t0}ms (checksum {meets[0]!})"
