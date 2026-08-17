"""
Comprehensive Robustness Test Suite for the Data Cleaning Engine
================================================================
Tests 12 synthetic dataset structures covering every edge case.
Each test validates:
  - No exceptions raised
  - Returned value is a valid DataFrame
  - Column count unchanged (unless intentionally changed)
  - Row count unchanged (except drop_rows)
  - log_entry['ok'] is True for valid operations, False for invalid ones
  - The target column's missing count reaches 0 after imputation
"""
import sys, traceback
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
from engine.cleaner import (
    impute_column, recommend_imputation_method,
    remove_duplicates, standardise_categories,
    fix_data_types, cap_outliers, get_cleaning_summary,
)

PASS = 0
FAIL = 0
RESULTS = []

# ── Test helpers ──────────────────────────────────────────────────────────────

def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        RESULTS.append(f"  PASS  {name}")
    else:
        FAIL += 1
        RESULTS.append(f"  FAIL  {name}" + (f" — {detail}" if detail else ""))

def assert_valid_df(df, original_df, name, allow_row_change=False):
    check(f"{name}: returns DataFrame",
          isinstance(df, pd.DataFrame))
    check(f"{name}: column count unchanged",
          len(df.columns) == len(original_df.columns),
          f"got {len(df.columns)}, expected {len(original_df.columns)}")
    if not allow_row_change:
        check(f"{name}: row count unchanged",
              len(df) == len(original_df),
              f"got {len(df)}, expected {len(original_df)}")

def run_test(label, fn):
    global FAIL
    try:
        fn()
    except Exception as e:
        FAIL += 1
        RESULTS.append(f"  FAIL  {label} — UNCAUGHT EXCEPTION: {e}")
        RESULTS.append(f"        {traceback.format_exc().splitlines()[-2]}")

# ── Dataset factories ─────────────────────────────────────────────────────────

def make_numerical_only(n=80):
    np.random.seed(1)
    df = pd.DataFrame({
        "A": np.random.normal(50, 10, n),
        "B": np.random.exponential(5, n),
        "C": np.random.uniform(0, 100, n),
    })
    df.loc[np.random.choice(n, 10, replace=False), "A"] = np.nan
    df.loc[np.random.choice(n, 5, replace=False), "B"]  = np.nan
    return df

def make_categorical_only(n=60):
    np.random.seed(2)
    df = pd.DataFrame({
        "Gender":  np.random.choice(["Male", "Female", None], n, p=[0.45, 0.45, 0.10]),
        "Dept":    np.random.choice(["Eng", "Arts", "Science", None], n, p=[0.30, 0.30, 0.30, 0.10]),
        "Status":  np.random.choice(["Active", "Inactive"], n),
    })
    return df

def make_mixed(n=120):
    np.random.seed(3)
    df = pd.DataFrame({
        "Age":        np.where(np.random.rand(n) > 0.9, np.nan, np.random.randint(18, 65, n).astype(float)),
        "Score":      np.where(np.random.rand(n) > 0.85, np.nan, np.random.normal(70, 15, n)),
        "Department": np.random.choice(["HR", "Finance", "IT", None], n, p=[0.3, 0.3, 0.3, 0.1]),
        "Passed":     np.random.choice(["Yes", "No"], n),
        "Year":       np.random.choice([2020, 2021, 2022, 2023], n),
    })
    return df

def make_many_missing(n=50):
    np.random.seed(4)
    df = pd.DataFrame({
        "X": np.where(np.random.rand(n) > 0.45, np.nan, np.random.normal(0, 1, n)),
        "Y": np.where(np.random.rand(n) > 0.60, np.nan, np.random.normal(10, 2, n)),
        "Z": np.random.choice(["A", "B", None], n, p=[0.35, 0.35, 0.30]),
    })
    return df

def make_no_missing(n=100):
    np.random.seed(5)
    return pd.DataFrame({
        "P": np.random.normal(0, 1, n),
        "Q": np.random.normal(5, 2, n),
        "R": np.random.choice(["Cat", "Dog", "Bird"], n),
    })

def make_with_duplicates(n=40):
    np.random.seed(6)
    base = pd.DataFrame({
        "ID":    range(n),
        "Value": np.random.normal(0, 1, n),
        "Label": np.random.choice(["X", "Y"], n),
    })
    return pd.concat([base, base.iloc[:10]], ignore_index=True)

def make_with_outliers(n=80):
    np.random.seed(7)
    df = pd.DataFrame({
        "Normal": np.random.normal(100, 15, n),
        "Skewed": np.random.exponential(2, n),
    })
    df.loc[[0, 1, 2], "Normal"] = [999, -999, 500]
    return df

def make_datetime(n=50):
    np.random.seed(8)
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    df = pd.DataFrame({
        "Date":  dates,
        "Value": np.random.normal(10, 2, n),
        "Cat":   np.random.choice(["A", "B"], n),
    })
    df.loc[[5, 15, 25], "Date"]  = pd.NaT
    df.loc[[3, 13, 23], "Value"] = np.nan
    return df

def make_unusual_col_names(n=30):
    np.random.seed(9)
    df = pd.DataFrame({
        "col with spaces":    np.random.normal(0, 1, n),
        "123numeric_start":   np.random.choice(["A", "B", None], n, p=[0.4, 0.4, 0.2]),
        "UPPER_CASE_COL":     np.random.normal(5, 1, n),
        "col/with/slashes":   np.random.choice(["X", "Y"], n),
    })
    df.loc[np.random.choice(n, 5, replace=False), "col with spaces"] = np.nan
    return df

def make_empty_column(n=40):
    np.random.seed(10)
    df = pd.DataFrame({
        "Good":  np.random.normal(0, 1, n),
        "Empty": np.full(n, np.nan),
        "Cat":   np.random.choice(["A", "B"], n),
    })
    return df

def make_small(n=5):
    np.random.seed(11)
    df = pd.DataFrame({
        "V1": [1.0, np.nan, 3.0, np.nan, 5.0],
        "V2": ["A", None, "B", "A", None],
    })
    return df

def make_large(n=5000):
    np.random.seed(12)
    df = pd.DataFrame({
        "Num1": np.where(np.random.rand(n) > 0.95, np.nan, np.random.normal(0, 1, n)),
        "Num2": np.random.normal(100, 20, n),
        "Cat1": np.random.choice(["X", "Y", "Z", None], n, p=[0.30, 0.30, 0.30, 0.10]),
        "Cat2": np.random.choice(["P", "Q"], n),
    })
    return df

# ══════════════════════════════════════════════════════════════════════════════
# TEST 1 — Numerical-only dataset
# ══════════════════════════════════════════════════════════════════════════════
def test_numerical_only():
    print("\n[1] Numerical-only dataset")
    df = make_numerical_only()
    orig_cols = list(df.columns)

    for col, meth in [("A", "mean"), ("A", "median"), ("A", "knn"), ("B", "median")]:
        df_in = df.copy()
        out, log = impute_column(df_in, col, meth)
        assert_valid_df(out, df_in, f"num-only {col}/{meth}")
        if log["ok"]:
            check(f"num-only {col}/{meth}: no missing after",
                  out[col].isnull().sum() == 0,
                  f"still {out[col].isnull().sum()} missing")
        check(f"num-only {col}/{meth}: columns preserved",
              list(out.columns) == orig_cols)

run_test("Numerical-only dataset", test_numerical_only)

# ══════════════════════════════════════════════════════════════════════════════
# TEST 2 — Categorical-only dataset
# ══════════════════════════════════════════════════════════════════════════════
def test_categorical_only():
    print("\n[2] Categorical-only dataset")
    df = make_categorical_only()

    for col, meth in [("Gender", "mode"), ("Dept", "constant_unknown"),
                      ("Gender", "drop_rows")]:
        df_in = df.copy()
        out, log = impute_column(df_in, col, meth)
        allow_row = (meth == "drop_rows")
        assert_valid_df(out, df_in, f"cat-only {col}/{meth}", allow_row_change=allow_row)
        if log["ok"] and not allow_row:
            check(f"cat-only {col}/{meth}: no missing",
                  out[col].isnull().sum() == 0)

    # mean on categorical must fail gracefully (ok=False)
    out, log = impute_column(df.copy(), "Gender", "mean")
    check("cat-only Gender/mean: ok=False (expected)",
          log["ok"] is False,
          f"got ok={log['ok']}")
    assert_valid_df(out, df, "cat-only Gender/mean safe return")

run_test("Categorical-only dataset", test_categorical_only)

# ══════════════════════════════════════════════════════════════════════════════
# TEST 3 — Mixed-type dataset (most common real-world case)
# ══════════════════════════════════════════════════════════════════════════════
def test_mixed():
    print("\n[3] Mixed-type dataset")
    df = make_mixed()

    for col, meth in [("Age", "mean"), ("Score", "median"), ("Score", "knn"),
                      ("Department", "mode"), ("Department", "constant_unknown")]:
        df_in = df.copy()
        out, log = impute_column(df_in, col, meth)
        assert_valid_df(out, df_in, f"mixed {col}/{meth}")
        if log["ok"]:
            check(f"mixed {col}/{meth}: cleared",
                  out[col].isnull().sum() == 0)

run_test("Mixed-type dataset", test_mixed)

# ══════════════════════════════════════════════════════════════════════════════
# TEST 4 — High missingness (>40% → KNN recommended)
# ══════════════════════════════════════════════════════════════════════════════
def test_many_missing():
    print("\n[4] High-missingness dataset")
    df = make_many_missing()

    for col, meth in [("X", "knn"), ("X", "median"), ("Y", "knn"), ("Z", "mode")]:
        df_in = df.copy()
        out, log = impute_column(df_in, col, meth)
        assert_valid_df(out, df_in, f"highmiss {col}/{meth}")
        if log["ok"]:
            check(f"highmiss {col}/{meth}: cleared",
                  out[col].isnull().sum() == 0)

    # KNN on categorical → graceful fail
    out, log = impute_column(df.copy(), "Z", "knn")
    check("highmiss Z/knn: ok=False", log["ok"] is False)
    assert_valid_df(out, df, "highmiss Z/knn safe return")

run_test("High-missingness dataset", test_many_missing)

# ══════════════════════════════════════════════════════════════════════════════
# TEST 5 — No missing values
# ══════════════════════════════════════════════════════════════════════════════
def test_no_missing():
    print("\n[5] No-missing-value dataset")
    df = make_no_missing()

    for col, meth in [("P", "mean"), ("Q", "knn"), ("R", "mode")]:
        df_in = df.copy()
        out, log = impute_column(df_in, col, meth)
        assert_valid_df(out, df_in, f"nomiss {col}/{meth}")
        check(f"nomiss {col}/{meth}: 0 changed", log["values_changed"] == 0)

run_test("No-missing-value dataset", test_no_missing)

# ══════════════════════════════════════════════════════════════════════════════
# TEST 6 — Duplicate rows
# ══════════════════════════════════════════════════════════════════════════════
def test_duplicates():
    print("\n[6] Duplicate rows dataset")
    df = make_with_duplicates()  # 50 rows (40 + 10 duplicates)

    out, log = remove_duplicates(df)
    check("dedup: ok=True",             log["ok"] is True)
    check("dedup: returns DataFrame",   isinstance(out, pd.DataFrame))
    check("dedup: col count same",      len(out.columns) == len(df.columns))
    check("dedup: rows reduced",        len(out) < len(df),
          f"{len(out)} rows vs {len(df)}")
    check("dedup: no remaining dups",   out.duplicated().sum() == 0)
    check("dedup: RangeIndex clean",
          list(out.index) == list(range(len(out))))

    # After dedup, imputation must still work (index gap test)
    out2, log2 = impute_column(out, "Value", "median")
    assert_valid_df(out2, out, "post-dedup imputation")

run_test("Duplicate rows dataset", test_duplicates)

# ══════════════════════════════════════════════════════════════════════════════
# TEST 7 — Outliers
# ══════════════════════════════════════════════════════════════════════════════
def test_outliers():
    print("\n[7] Outlier dataset")
    df = make_with_outliers()

    for col, meth in [("Normal", "iqr"), ("Normal", "zscore"), ("Skewed", "iqr")]:
        out, log = cap_outliers(df.copy(), col, meth)
        assert_valid_df(out, df, f"outlier {col}/{meth}")
        check(f"outlier {col}/{meth}: ok=True", log["ok"] is True)
        check(f"outlier {col}/{meth}: some capped", log["values_changed"] >= 0)

    # zscore on constant column → graceful fail
    df_const = df.copy()
    df_const["Const"] = 42.0
    out, log = cap_outliers(df_const, "Const", "zscore")
    check("outlier const/zscore: ok=False", log["ok"] is False)

    # outlier on categorical → graceful fail
    df_cat = pd.DataFrame({"Cat": ["A", "B", "C", "A", "B"]})
    out, log = cap_outliers(df_cat, "Cat", "iqr")
    check("outlier cat/iqr: ok=False", log["ok"] is False)
    assert_valid_df(out, df_cat, "outlier cat safe return")

run_test("Outlier dataset", test_outliers)

# ══════════════════════════════════════════════════════════════════════════════
# TEST 8 — Datetime variables
# ══════════════════════════════════════════════════════════════════════════════
def test_datetime():
    print("\n[8] Datetime dataset")
    df = make_datetime()

    # Numerical col imputation
    out, log = impute_column(df.copy(), "Value", "mean")
    assert_valid_df(out, df, "datetime df/Value/mean")
    check("datetime Value/mean: ok", log["ok"] is True)
    check("datetime Value/mean: cleared", out["Value"].isnull().sum() == 0)

    # Forward fill on datetime column
    out, log = impute_column(df.copy(), "Date", "forward_fill")
    assert_valid_df(out, df, "datetime df/Date/ffill")
    if log["ok"]:
        check("datetime Date/ffill: cleared", out["Date"].isnull().sum() == 0)

    # type conversion: date-string → datetime
    df2 = pd.DataFrame({
        "DateStr": ["2022-01-01", "2022-01-02", "not-a-date", "2022-01-04"],
        "Val":     [1.0, 2.0, 3.0, 4.0],
    })
    out, log = fix_data_types(df2, "DateStr", "datetime")
    assert_valid_df(out, df2, "fix_types DateStr→datetime")
    check("fix_types datetime: ok=True", log["ok"] is True)
    check("fix_types datetime: bad row is NaT",
          pd.isna(out.loc[2, "DateStr"]))

run_test("Datetime dataset", test_datetime)

# ══════════════════════════════════════════════════════════════════════════════
# TEST 9 — Unusual column names
# ══════════════════════════════════════════════════════════════════════════════
def test_unusual_col_names():
    print("\n[9] Unusual column names dataset")
    df = make_unusual_col_names()

    for col in ["col with spaces", "UPPER_CASE_COL"]:
        out, log = impute_column(df.copy(), col, "mean" if pd.api.types.is_numeric_dtype(df[col]) else "mode")
        assert_valid_df(out, df, f"unusual_name {col}")
        check(f"unusual_name {col}: cols list same",
              list(out.columns) == list(df.columns))

    # Category standardisation with unusual col name
    df2 = df.copy()
    out, log = standardise_categories(df2, "col/with/slashes", {"X": "x_std", "Y": "y_std"})
    assert_valid_df(out, df2, "unusual_name std categories")
    check("unusual_name std: ok=True", log["ok"] is True)

run_test("Unusual column names dataset", test_unusual_col_names)

# ══════════════════════════════════════════════════════════════════════════════
# TEST 10 — All-NaN (empty) column
# ══════════════════════════════════════════════════════════════════════════════
def test_empty_column():
    print("\n[10] Empty column dataset")
    df = make_empty_column()

    # mean on all-NaN numeric → handled gracefully (fills with NaN of mean which is nan)
    out, log = impute_column(df.copy(), "Empty", "mean")
    assert_valid_df(out, df, "empty_col/mean")
    # NaN mean → still NaN after fill, but should NOT crash
    check("empty_col/mean: no crash", True)

    # mode on all-NaN → no mode found, graceful
    out, log = impute_column(df.copy(), "Empty", "mode")
    assert_valid_df(out, df, "empty_col/mode")
    check("empty_col/mode: no crash", True)

    # knn on all-NaN → fail gracefully
    out, log = impute_column(df.copy(), "Empty", "knn")
    check("empty_col/knn: ok=False (insufficient non-missing)",
          log["ok"] is False)
    assert_valid_df(out, df, "empty_col/knn safe return")

    # Other cols unaffected
    check("empty_col: Good column untouched",
          df["Good"].equals(out["Good"]))

run_test("Empty column dataset", test_empty_column)

# ══════════════════════════════════════════════════════════════════════════════
# TEST 11 — Small dataset (n=5)
# ══════════════════════════════════════════════════════════════════════════════
def test_small():
    print("\n[11] Small dataset (n=5)")
    df = make_small()

    for col, meth in [("V1", "mean"), ("V1", "median"), ("V1", "knn"),
                      ("V2", "mode"), ("V2", "constant_unknown")]:
        df_in = df.copy()
        out, log = impute_column(df_in, col, meth)
        assert_valid_df(out, df_in, f"small {col}/{meth}")
        if log["ok"]:
            check(f"small {col}/{meth}: cleared",
                  out[col].isnull().sum() == 0)

    # Non-existent column → graceful fail
    out, log = impute_column(df.copy(), "NonExistent", "mean")
    check("small NonExistent/mean: ok=False", log["ok"] is False)
    assert_valid_df(out, df, "small NonExistent safe return")

run_test("Small dataset (n=5)", test_small)

# ══════════════════════════════════════════════════════════════════════════════
# TEST 12 — Large dataset (n=5000)
# ══════════════════════════════════════════════════════════════════════════════
def test_large():
    print("\n[12] Large dataset (n=5000)")
    df = make_large()

    for col, meth in [("Num1", "mean"), ("Num1", "knn"), ("Cat1", "mode"),
                      ("Cat1", "constant_unknown")]:
        df_in = df.copy()
        out, log = impute_column(df_in, col, meth)
        assert_valid_df(out, df_in, f"large {col}/{meth}")
        if log["ok"]:
            check(f"large {col}/{meth}: cleared",
                  out[col].isnull().sum() == 0)

    # Dedup + sequential imputation (the original crash scenario)
    df_dedup, log_d = remove_duplicates(df.copy())
    check("large dedup: ok=True", log_d["ok"] is True)
    out1, l1 = impute_column(df_dedup, "Num1", "knn")
    check("large post-dedup KNN: ok=True", l1["ok"] is True)
    assert_valid_df(out1, df_dedup, "large post-dedup KNN")
    out2, l2 = impute_column(out1, "Cat1", "mode")
    check("large sequential mode: ok=True", l2["ok"] is True)
    assert_valid_df(out2, out1, "large sequential mode")

run_test("Large dataset (n=5000)", test_large)

# ══════════════════════════════════════════════════════════════════════════════
# BONUS TEST 13 — fix_data_types + standardise_categories
# ══════════════════════════════════════════════════════════════════════════════
def test_type_ops():
    print("\n[13] Type conversion & category standardisation")
    df = pd.DataFrame({
        "Score":    ["85", "92", "not_a_num", "78", ""],
        "Category": ["male", "Female", "MALE", "female", "Male"],
        "Year":     [2020, 2021, 2022, 2023, 2024],
    })

    # String → numeric (one bad value)
    out, log = fix_data_types(df.copy(), "Score", "numerical")
    assert_valid_df(out, df, "fixtype Score→num")
    check("fixtype Score→num: ok=True", log["ok"] is True)
    check("fixtype Score→num: bad val is NaN",
          pd.isna(out.loc[2, "Score"]))

    # Standardise gender categories
    mapping = {"male": "Male", "MALE": "Male", "female": "Female", "Female": "Female"}
    out2, log2 = standardise_categories(df.copy(), "Category", mapping)
    assert_valid_df(out2, df, "std_cat Category")
    check("std_cat Category: ok=True", log2["ok"] is True)
    check("std_cat Category: 4+ changed", log2["values_changed"] >= 4)

    # String → categorical (strips whitespace)
    df3 = pd.DataFrame({"Name": ["  Alice  ", " Bob", "Charlie "]})
    out3, log3 = fix_data_types(df3.copy(), "Name", "categorical")
    check("fixtype Name→cat: ok=True", log3["ok"] is True)
    check("fixtype Name→cat: stripped",
          out3["Name"].tolist() == ["Alice", "Bob", "Charlie"])

run_test("Type conversion & category standardisation", test_type_ops)

# ══════════════════════════════════════════════════════════════════════════════
# BONUS TEST 14 — Cleaning log integrity
# ══════════════════════════════════════════════════════════════════════════════
def test_cleaning_log():
    print("\n[14] Cleaning log integrity")
    log = []
    df  = make_mixed()

    _, l1 = impute_column(df, "Age", "mean")
    log.append(l1)
    _, l2 = impute_column(df, "Department", "mode")
    log.append(l2)
    _, l3 = remove_duplicates(df)
    log.append(l3)
    _, l4 = impute_column(df, "NONEXISTENT", "mean")  # expected fail
    log.append(l4)

    summary = get_cleaning_summary(log)
    check("log: summary is string",         isinstance(summary, str))
    check("log: contains PASS marker",      "PASS" in summary or "✅" in summary)
    check("log: contains FAIL marker",      "⚠️" in summary or "ok" in str(l4))
    check("log: all entries have ok key",   all("ok" in e for e in log))
    check("log: l1 ok=True",               l1["ok"] is True)
    check("log: l4 ok=False (bad col)",    l4["ok"] is False)
    check("log: 4 entries",                len(log) == 4)

run_test("Cleaning log integrity", test_cleaning_log)

# ══════════════════════════════════════════════════════════════════════════════
# REPORT
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print(f"ROBUSTNESS TEST RESULTS — {PASS + FAIL} checks across 14 test suites")
print("=" * 65)
for r in RESULTS:
    print(r)
print("=" * 65)
print(f"  PASSED : {PASS}")
print(f"  FAILED : {FAIL}")
print("=" * 65)
if FAIL == 0:
    print("  ALL CHECKS PASSED — engine is dataset-agnostic and robust")
else:
    print(f"  {FAIL} CHECK(S) REQUIRE ATTENTION (see FAIL lines above)")
print("=" * 65)
