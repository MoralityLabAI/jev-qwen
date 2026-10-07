# Patches for Hermes-Skills (not applied)

These are proposals for your Hermes-Skills repo. I do not write to that checkout: its working tree
does not match HEAD (many staged deletions), so a commit there would delete files.

## 0001-flow-policy-reject-paths.patch

Against `research/scripts/run_intellect3_logic_flow_policy_sweep.py` at HEAD `07a177f2`. Its effect
is measured in SPEC-U5 part R (`docs/ubench/SPEC-U5.md`, `reports/ubench/u5.md`).

**Why.** In SPEC-U4, the flow policies `logic_trm_c_repair_if_c_fail` and
`logic_trm_dual_repair_if_any_sig_fail` never abstain. So every candidate their projection cannot
fix is committed broken: 27% of the U4 test items for the dual policy, e.g. every candidate with two
tents swapped (counts right, touching wrong) and every wrong-shape grid.

**What it adds.**
- **`campsite_verifier_pass(grid, expected)`.** The official Campsite rules: trees unchanged, tent
  counts, no touching tents, and a perfect tent-tree matching. It reads only what the puzzle shows,
  through `expected`, exactly as the projections already do. It agrees with hermes-lite's
  `verify_candidate` on 9,879 of 9,879 U4 train/val grids (candidates and their projections).
- **A `verifier_pass` field** on every candidate, and `verifier_pass` / `committed_failing` on every
  policy row.
- **Four policies:**
  - `<policy>_noop_reject` abstains when the grid to commit is the unchanged original (edit distance
    0) and that original fails the verifier. This needs no verifier call after the repair.
  - `<policy>_verify_reject` commits only a grid that passes the verifier, at the cost of one call
    after the repair.

  Neither can lower the exact-solve rate; both only withhold commits that would fail.

**To apply**, from the Hermes-Skills checkout once its working tree is back in sync with HEAD:

```bash
git apply "C:/Users/patri/OneDrive/Documents/GitHub/jev-qwen/patches/hermes-skills/0001-flow-policy-reject-paths.patch"
```
