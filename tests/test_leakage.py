"""Regression tests for the label-leakage exclusion policy.

The diabetes label is derived from ICD-coded diagnoses. Any feature that defines
the diagnosis (glucose, HbA1c), asserts its management (insulin-use codes), or is
administered because of it (hypoglycaemia rescue agents) recovers the label
rather than predicting it. These tests fail the build if such a feature can reach
the feature matrix.
"""
import pandas as pd
import pytest

from src.leakage import (
    assert_no_leakage,
    is_criterion_analyte,
    is_leaky_code,
    is_leaky_medication,
    leakage_report,
    leaky_columns,
    resolve_leaky_itemids,
)


def test_criterion_analytes_recognised():
    for label in ["Glucose", "Glucose, Whole Blood", "% Hemoglobin A1c",
                  "Glycated Hemoglobin", "Absolute A1c", "Glucose (serum)",
                  "Glucose finger stick (range 70-100)"]:
        assert is_criterion_analyte(label), label


def test_non_criterion_analytes_survive():
    for label in ["Creatinine", "Anion Gap", "Lactate", "Heart Rate", "Sodium"]:
        assert not is_criterion_analyte(label), label


def test_insulin_use_codes_are_leaky():
    # These name no diabetes, but assert its management -- the label restated.
    assert is_leaky_code("V5867")    # ICD-9  long-term use of insulin
    assert is_leaky_code("Z794")     # ICD-10 long term use of insulin
    assert is_leaky_code("Z79.4")    # dotted form
    assert is_leaky_code("E119")     # diabetes-defining
    assert is_leaky_code("25000")    # diabetes-defining


def test_ordinary_comorbidities_survive():
    for code in ["4019", "2724", "I10", "E785", "F329"]:
        assert not is_leaky_code(code), code


def test_rescue_medications_are_leaky():
    # Administered *because* the patient is under glycaemic management.
    for med in ["Glucagon", "Dextrose 50%", "D50W", "Glucose Gel",
                "Insulin Glargine", "MetFORMIN (Glucophage) 500 MG Tab",
                "Boost Glucose Control (Full)"]:
        assert is_leaky_medication(med), med


def test_ordinary_medications_survive():
    for med in ["Aspirin", "Furosemide", "Pantoprazole", "Metoprolol Tartrate"]:
        assert not is_leaky_medication(med), med


def test_itemids_resolved_from_dictionary():
    d_lab = pd.DataFrame({"itemid": [50931, 50852, 50912],
                          "label": ["Glucose", "% Hemoglobin A1c", "Creatinine"]})
    ids = resolve_leaky_itemids(d_lab, None)
    assert 50931 in ids and 50852 in ids
    assert 50912 not in ids


def test_leaky_columns_flags_itemid_encoded_features():
    cols = ["labs_item_50931", "labs_item_50912", "vitals_item_220621",
            "meds_Glucagon", "meds_Aspirin", "dx_V5867", "dx_4019"]
    bad = leaky_columns(cols, leaky_itemids={50931, 220621})
    assert set(bad) == {"labs_item_50931", "vitals_item_220621",
                        "meds_Glucagon", "dx_V5867"}


def test_assert_no_leakage_raises_on_violation():
    with pytest.raises(ValueError, match="Label leakage"):
        assert_no_leakage(["labs_item_50931", "meds_Aspirin"],
                          leaky_itemids={50931})


def test_assert_no_leakage_passes_on_clean_matrix():
    assert_no_leakage(["labs_item_50912", "meds_Aspirin", "dx_4019"],
                      leaky_itemids={50931})


def test_report_partitions_by_tier():
    rep = leakage_report(
        ["labs_item_50931", "dx_Z794", "meds_Glucagon", "meds_Aspirin"],
        leaky_itemids={50931})
    assert rep["tier1_criterion_analytes"] == ["labs_item_50931"]
    assert rep["tier2_treatment_codes"] == ["dx_Z794"]
    assert rep["tier3_medications"] == ["meds_Glucagon"]
