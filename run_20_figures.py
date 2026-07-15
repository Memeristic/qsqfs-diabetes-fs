"""
run_20_figures.py
=================
Command-line entry point for the twenty analysis figures (F01-F20).

Rebuilds the cohort under the exclusion policy, reads the comparative table from
a completed run, and writes the figure set plus a manifest.

Usage:
    python run_20_figures.py                     # data_root from config.yaml
    python run_20_figures.py /path/to/mimic-iv   # explicit MIMIC-IV folder
    python run_20_figures.py demo                # synthetic cohort

The output directory is `analysis_out` unless OUT_DIR is set.
"""

import json
import os
import sys

import yaml

from src.data_loader import MIMICDataLoader
from src.figures20 import build_figures
from src.leakage import assert_no_leakage, leaky_columns, resolve_leaky_itemids
from src.modality_builder import (build_combined_matrix, build_diagnoses, build_labs,
                                  build_medications, build_modality_map, build_vitals,
                                  extract_diabetes_label)

OUT = os.environ.get("OUT_DIR", "analysis_out")
os.makedirs(OUT, exist_ok=True)

summary_path = os.path.join(OUT, "summary.json")
if not os.path.exists(summary_path):
    sys.exit(f"error: {summary_path} not found -- run run_analysis.py first.")
SUMM = json.load(open(summary_path))

label_path = os.path.join(OUT, "feature_labels.json")
LAB = json.load(open(label_path)) if os.path.exists(label_path) else {}

print(">> loading data ...", flush=True)
cfg = yaml.safe_load(open("config.yaml"))
_arg = sys.argv[1] if len(sys.argv) > 1 else None

if _arg == "demo":
    cfg["demo_mode"] = True
else:
    cfg["demo_mode"] = False
    cfg["paths"] = dict(cfg.get("paths", {}))
    if _arg:
        cfg["paths"]["data_root"] = _arg
cfg["preprocessing"]["combine_how"] = "outer"

data = MIMICDataLoader(cfg).load_all()
label_df = extract_diabetes_label(data.get("patients"), data.get("diagnoses_icd"),
                                  data.get("d_icd_diagnoses"))

# The exclusion policy applies here exactly as it does in the analysis pipeline:
# criterion analytes are resolved from the dictionaries and withheld by itemid
# before the top-k ranking, so the figures describe the same cohort the results
# were computed on.
leaky_ids = resolve_leaky_itemids(data.get("d_labitems"), data.get("d_items"))

mod = {}
for key, fn, arg in [("labs", build_labs, data.get("labevents")),
                     ("vitals", build_vitals, data.get("chartevents")),
                     ("meds", build_medications, data.get("pharmacy")),
                     ("dx", build_diagnoses, data.get("diagnoses_icd"))]:
    df, _ = (fn(arg, label_df, leaky_itemids=leaky_ids)
             if key in ("labs", "vitals") else fn(arg, label_df))
    if df is not None:
        mod[key] = df

combined, names = build_combined_matrix(mod, how="outer")
names = [n for n in names if n not in set(leaky_columns(names, leaky_ids))]
assert_no_leakage(names, leaky_ids)

mmap = build_modality_map(mod, names)
X = combined[names].values.astype(float)
y = combined["label"].values.astype(int)
print(f"   {X.shape[0]} patients x {X.shape[1]} features, "
      f"prevalence {100 * y.mean():.1f}%", flush=True)

manifest = build_figures(
    X, y, names, mmap, cfg,
    comparative_table=SUMM.get("comparative", {}).get("table", []),
    out_dir=OUT,
    feature_labels=LAB,
    progress=lambda m: print("  saved", m, flush=True),
)

print(f"\n>> DONE: {len(manifest)} figures in {os.path.join(OUT, 'fig20')}")
