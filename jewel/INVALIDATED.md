# Invalidated pre-official-validation artifacts

The original `ball_d7`, un-suffixed PDBs, `train_mixed_v1.npz`,
`train_mixed_v2_d8.npz`, `transformer_v1`, and `transformer_v2_d8` artifacts
were built before direct replay against the competition's `puzzle_info.json`.

That replay exposed a TWS parsing error: edge-orientation deltas are indexed
by moved source positions, while the first engine treated them as target
positions.  The official competition directions are also the inverse of the
TWS base directions.  `jewel.official.OfficialPuzzle` now proves the complete
12-action isomorphism and translates directions at the file boundary.

Do not use pre-`official` artifacts for competition solving.  Correct artifacts
carry the `_official` suffix and all current command defaults point to them.
