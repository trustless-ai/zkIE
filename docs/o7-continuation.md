# O7 continuation record — same-table LogUp table-side sharing

State: `0cc1f50` + this O7 change on `fix/soundness-repair` (test machine
`/data/zkie-soundness-fix`). O7 is NOT complete pending TL review: the
per-proof table-commit reuse is implemented; the deeper multiplicity/common-
alpha/beta aggregation requires a protocol/transcript change and is reported
below for approval.

## Inventory

Active bound path, per `Op::Lookup`:
- one `LookupMultiplicity` auxiliary commitment (per lookup);
- per-op alpha/beta from the transcript at that op;
- one witness-side and one table-side fractional proof;
- for static non-identity tables, a fresh Merkle tree/commitment of the
  padded table for the table opening at the lookup's own point.

Same-table lookups therefore duplicate the table-side tree/commitment and
opening work. Real A/B/C/full64 contain at most one static `Lookup` each
(B/full64: the activation table; C's gate table is dynamic), so O7 does not
change those runs. The legacy committed path has the same per-lookup table
columns/alpha/beta; its verifier is retired and it was not modified.

## Implemented (no transcript/reduction change)

- Bound prover pre-commits each STATIC, non-identity lookup table once per
  proof, keyed by the trusted table tensor id (`prepare_static_lookup_commits`).
  The commitment is checked against the trusted model `lookup_roots` before
  use. Different table ids are never merged even if their contents coincide.
- `prove_reverse_lookup` reuses the cached committed table and its Merkle
  prover data; if the table is uncached (plain path, or
  `ZKIE_O7_TABLE_CACHE=0`) it commits once for that lookup as before.
- Per-lookup multiplicity columns, alpha/beta, witness-side/table-side
  fractional proofs and table openings are unchanged. Proof bytes and
  transcript order are unchanged.
- `Store::o7_table_commits()` counts table Merkle trees actually committed.

## Measurements

`crates/zkie-ops/examples/o7_lookup.rs`, two Lookups into the same static
table (2^18 entries), release:

| mode | table commits | proof bytes / hash | wall | VmHWM |
|---|---|---|---|---|
| `ZKIE_O7_TABLE_CACHE=0` | 2 | 793,525 / `4905d4813e89947d` | 591.6 ms | 105,704 kB |
| shared (default) | 1 | 793,525 / `4905d4813e89947d` | 617.7 ms | 116,272 kB |

The table-tree count drops 2 -> 1 and the proof is byte-identical. However,
wall time and RSS do **not** improve in this shape: WHIR's `open` consumes the
prover data, so the cached Merkle tree is deep-cloned per opening, and the
cached table/tree is retained; the commit saving is offset. This is reported
as measured, not as a performance win.

Offline vs within-proof state: the table root is fixed offline in the trusted
model; the prover's Merkle tree is rebuilt once per proof (the commit cost is
inside the measured prove time, not hidden). Cross-inference persistence of
the committed table/prover data is not implemented.

## Required for the full gist O7 (approval requested)

The full sharing (same-table lookups) needs:
1. independent per-lookup witness/multiplicity commitments fixed first;
2. a common alpha/beta derived once per trusted table group (after all group
   commitments), with a bounded random combination binding each lookup's
   numerator/effective-length mask to its own xs/ys;
3. summed multiplicity per table entry and one aggregated table-side
   fractional proof/opening per group; per-lookup witness-side fractions
   checked against the group relation so counts cannot be shifted between
   lookups or zeroed out;
4. a borrowed (non-consuming) WHIR opening API (or a shared prover-data
   handle) so the cached table tree is not cloned per opening.

This changes the Lookup reduction/transcript structure; it is not included in
this commit and requires TL approval before implementation.

## Regression

- New `o7_same_table_lookups_share_table_commit`: two same-id Lookups produce
  one table commit, both per-lookup Lookup reverse steps remain (2), the
  proof verifies; the model uses the trusted table id and index shapes.
- `zkie-ops --lib`: 160/0 (159 + O7 test); the existing wrong-table,
  tampered-multiplicity and missing-lookup negatives stay green.
- Real segmented A/B/C: proof bytes identical to pre-O7 dumps
  (241,874 / 462,836 / 355,183 B); verify 15.6 ms (C) / 30.8 ms (B).
- Real full64: `CORE-CHAIN-OK`; 64 claims, one `W_t` opening; B 510,969 B;
  verify 153.6 ms.
- No transcript/authentication/security change; 90-bit.

## Remaining

O7 awaits TL review/decision on the deeper aggregation. The O queue is
otherwise addressed; S7 is not accepted; S8/S9 remain postponed.
