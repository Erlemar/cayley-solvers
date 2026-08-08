# Beam-search benchmark (cayleypy)

Sanity-check the exported value head by actually solving a few puzzles with cayleypy beam
search (`beam_mode='iterated'`, non-backtracking history) - the same harness as the other
community baseline notebooks, so numbers are directly comparable *within this harness*.

Two honest caveats:

- Our headline results (51/51, mean 87.5 on the stratified sample; full-1001 production
  runs) come from our own solver stack (multi-pass beam at width 16k->65k, a 4-rotation
  symmetry ensemble, and min-merging) - absolute path lengths from a single cayleypy beam
  at width 4096 will be longer. Use this cell to compare checkpoints against each other
  and against other models in the same harness, not to reproduce the headline numbers.
- Never pair this value head with a Q-shortlister distilled from a DIFFERENT value model
  (the shortlist ordering is calibrated to its teacher's V landscape; mixing costs ~5-10%
  path length in our measurements).
