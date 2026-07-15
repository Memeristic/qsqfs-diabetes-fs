# Label-leakage control protocol

The binary outcome is derived from ICD-coded diabetes diagnoses. Any feature that
**defines** that diagnosis, **asserts its management**, or is **administered because
of it** recovers the label rather than predicting it. Such features are excluded
before modelling under a three-tier policy, implemented in `src/leakage.py` and
enforced by `assert_no_leakage()` as a hard post-condition on the feature matrix.

| Tier | Excluded | Rationale |
|---|---|---|
| 1 | Glucose, HbA1c — resolved by `itemid` from `d_labitems` / `d_items` | The analytes on which diabetes is diagnostically defined (ADA: FPG ≥ 7.0 mmol/L, HbA1c ≥ 48 mmol/mol). Retaining them makes the task circular. |
| 2 | ICD-9 V58.67, ICD-10 Z79.4 (long-term insulin use) | Assert glycaemic management without naming diabetes — the label expressed in another vocabulary. |
| 3 | Antidiabetic agents; hypoglycaemia rescue agents (glucagon, 50% dextrose, oral glucose gel) | Rescue agents are given *because* a patient is under glycaemic management: consequences of the outcome, not predictors of it. |

## Why identifiers, not names

Laboratory and chart features are keyed by opaque numeric `itemid`s
(e.g. `labs_item_50931`). A filter applied to feature *names* cannot see them.
Exclusions are therefore resolved against the database's own dictionaries by
pattern-matching the dictionary `label`, then applied by identifier. The policy
holds on the full database, not only on a demo subset.

## Order of operations

Criterion-analyte rows are removed **before** top-*k* item selection. Filtering
afterwards would let an excluded item occupy one of the *k* slots and then be
dropped, silently shrinking the modality.

## Enforcement

`tests/test_leakage.py` fails the build if any excluded feature can reach the
feature matrix. Excluding a feature is a modelling decision, so it is tested like
one.

## Reporting

The exclusion is reported in `summary.json` under `dataset.leakage_policy` and
`dataset.n_criterion_itemids_excluded`, so any run is self-documenting.
