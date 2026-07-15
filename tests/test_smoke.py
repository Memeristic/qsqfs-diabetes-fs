"""Fast CI smoke tests (no torch/streamlit). Run: pytest -q"""
import numpy as np
import importlib.util

import pytest
import yaml
from src.data_loader import MIMICDataLoader
from src.modality_builder import (extract_diabetes_label, build_labs, build_vitals,
                                   build_medications, build_diagnoses,
                                   build_combined_matrix, build_modality_map)
from src.evaluation import nested_cv_evaluate
from src.qsfs import QSQFS


def _tiny_data():
    cfg = yaml.safe_load(open("config.yaml"))
    # Unit tests exercise the algorithm, not the data source: they must not
    # depend on a credentialed database being present on disk, nor on whatever
    # `demo_mode` happens to be set to in config.yaml. Force the synthetic
    # fixture explicitly so the suite is hermetic and CI is reproducible.
    cfg["demo_mode"] = True
    data = MIMICDataLoader(cfg).load_all()
    label = extract_diabetes_label(data["patients"], data["diagnoses_icd"],
                                   data["d_icd_diagnoses"])
    mod = {}
    for fn, arg, _ in [(build_labs, data["labevents"], 0), (build_vitals, data["chartevents"], 0),
                       (build_medications, data["pharmacy"], 0), (build_diagnoses, data["diagnoses_icd"], 0)]:
        df, _ = fn(arg, label)
        if df is not None:
            mod[df.columns[1].split("_")[0]] = df
    combined, names = build_combined_matrix(mod, "inner")
    return combined[names].values.astype(float), combined["label"].values.astype(int), names, mod


def test_qsqfs_runs_and_selects():
    X, y, _, _ = _tiny_data()
    m = QSQFS(n_colonies=8, max_iter_stage1=4, max_iter_stage2=4, cv_folds=3, verbose=False)
    m.fit(X, y)
    sel = m.get_selected_features()
    assert len(sel) >= 1
    assert m.get_best_fitness() > 0.5


def test_no_cache_changes_evals_not_accuracy():
    X, y, _, _ = _tiny_data()
    base = dict(n_colonies=8, max_iter_stage1=5, max_iter_stage2=0, cv_folds=3, verbose=False)
    a = QSQFS(**base, use_cache=True).fit(X, y)
    b = QSQFS(**base, use_cache=False).fit(X, y)
    assert b.n_evals >= a.n_evals            # disabling cache costs more evaluations
    assert abs(a.get_best_fitness() - b.get_best_fitness()) < 1e-6  # same accuracy


def test_nested_cv_is_bounded():
    X, y, names, mod = _tiny_data()
    cfg = yaml.safe_load(open("config.yaml"))
    cfg["stage1"].update(n_colonies=8, max_iter_stage1=4, cv_folds=3)
    cfg["stage2"].update(n_colonies=8, max_iter_stage2=4, cv_folds=3)
    mmap = build_modality_map(mod, names)
    res = nested_cv_evaluate(X, y, mmap, cfg, n_outer=3, classifier="knn")
    assert 0.0 <= res["auc_mean"] <= 1.0
    assert len(res["selection_frequency"]) == X.shape[1]


# --------------------------------------------------------------------------- #
# v2.4.0 additions: schema validation, generic CSV mapping, interactive plots, #
# and plain-language captions.                                                 #
# --------------------------------------------------------------------------- #
import pandas as pd


def test_validate_mimic_folder_rejects_bad_dir(tmp_path):
    """A non-MIMIC folder must be flagged (ok=False) with specific errors."""
    from src import schema
    rep = schema.validate_mimic_folder(str(tmp_path))
    assert rep["ok"] is False
    assert any("hosp" in e for e in rep["errors"])


def test_infer_and_build_generic_csv():
    """The generic column-mapping layer turns an arbitrary CSV into (X, y)."""
    from src import schema
    df = pd.DataFrame({
        "pid": range(40),
        "age": np.random.randint(20, 80, 40),
        "bmi": np.random.normal(27, 5, 40),
        "sex": np.random.choice(["M", "F"], 40),
        "outcome": np.random.choice([0, 1], 40),
    })
    guess = schema.infer_column_kinds(df)
    assert guess["label_col"] == "outcome"
    X, y, names, mmap = schema.build_matrix_from_mapping(
        df, guess["label_col"], guess["numeric"], guess["categorical"], guess["id_col"])
    assert X.shape[0] == 40
    assert set(np.unique(y)) <= {0, 1}
    assert "sex_M" in names and "sex_F" in names   # one-hot expanded
    assert mmap == {"uploaded": list(range(X.shape[1]))}


def test_generic_csv_rejects_non_binary_label():
    """A 3-class label must raise a clear ValueError (binary task only)."""
    from src import schema
    df = pd.DataFrame({"a": np.arange(30.0), "y": np.random.choice([0, 1, 2], 30)})
    import pytest
    with pytest.raises(ValueError):
        schema.build_matrix_from_mapping(df, "y", ["a"], [])


@pytest.mark.skipif(importlib.util.find_spec("plotly") is None,
                    reason="plotly is an optional dashboard dependency")
def test_interactive_plots_build():
    """Every Plotly helper returns a figure without error."""
    from src import plotting_interactive as pi
    y = np.array([0, 1, 0, 1, 1, 0]); p = np.array([.2, .8, .3, .7, .9, .1])
    assert pi.plot_roc_interactive(y, p) is not None
    assert pi.plot_confusion_interactive(y, (p >= .5).astype(int)) is not None
    assert pi.plot_convergence_interactive([.6, .7, .8], [5, 4, 3]) is not None
    assert pi.plot_selection_frequency_interactive(np.array([.9, .4, .8]), ["a", "b", "c"]) is not None
    comp = {"QSQ-FS": {"auc": np.array([.8, .82])}, "GA": {"auc": np.array([.7, .71])}}
    assert pi.plot_comparison_interactive(comp, "auc") is not None


def test_captions_are_plain_strings():
    """Caption helpers return non-empty plain strings for the UI."""
    from ui import captions as cap
    assert "coin-flip" in cap.auc_meaning(0.89)
    assert isinstance(cap.accuracy_meaning(0.8, 0.35), str) and cap.accuracy_meaning(0.8, 0.35)
    assert cap.f1_meaning(0.7) and cap.selection_frequency_meaning()
