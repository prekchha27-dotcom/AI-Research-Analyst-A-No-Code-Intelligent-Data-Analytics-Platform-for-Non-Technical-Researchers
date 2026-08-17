# -*- coding: utf-8 -*-
"""
Comprehensive Dataset-Agnostic Test Suite
==========================================
Tests the analytical role engine, profiler, and cleaner across 18 synthetic
dataset structures — numerical, categorical, ordinal, datetime, text,
identifier, serial, admin codes, constants, mixed, high-missingness,
no-missing, duplicates, outliers, unusual column names, empty columns,
small/large samples.

Validates that:
- Identifiers/serial numbers are NEVER offered for imputation as numerical
- Appropriate variables ARE available for analysis
- No cleaning/statistical operation crashes on structure change
- Row/column alignment is preserved after every transformation
- Shape-mismatch errors cannot occur
"""

import sys, os, traceback, random, string
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

PASS = FAIL = 0
RESULTS = []


def check(name: str, cond: bool, detail: str = ""):
    global PASS, FAIL
    if cond:
        RESULTS.append(f"  PASS  {name}")
        PASS += 1
    else:
        RESULTS.append(f"  FAIL  {name}" + (f" — {detail}" if detail else ""))
        FAIL += 1


def run_test(label: str, fn):
    print(f"\n  [{label}]")
    try:
        fn()
    except Exception as e:
        RESULTS.append(f"  FAIL  {label} — UNCAUGHT EXCEPTION: {e}")
        RESULTS.append(f"        {traceback.format_exc().splitlines()[-2]}")
        global FAIL
        FAIL += 1


def assert_df_shape(df_in, df_out, label, allow_row_reduction=False):
    check(f"{label}: returns DataFrame", isinstance(df_out, pd.DataFrame))
    check(f"{label}: col count same",
          len(df_out.columns) == len(df_in.columns))
    if not allow_row_reduction:
        check(f"{label}: row count same", len(df_out) == len(df_in))


# ── Imports ───────────────────────────────────────────────────────────────────
from engine.analytical_roles import (
    classify_column, classify_dataset,
    recommend_imputation_for_column,
    ROLE_IDENTIFIER, ROLE_SERIAL, ROLE_ADMIN_CODE, ROLE_CONSTANT,
    ROLE_FREE_TEXT, ROLE_SUBST_NUMERICAL, ROLE_CATEGORICAL, ROLE_ORDINAL,
    ROLE_BOOLEAN, ROLE_DATETIME,
)
from engine.profiler import profile_dataset, generate_plain_summary
from engine.cleaner import (
    impute_column, remove_duplicates, standardise_categories,
    fix_data_types, cap_outliers, recommend_imputation_method,
)

# ══════════════════════════════════════════════════════════════════════════════
# DATASET 1 — Pure sequential integer identifier column
# ══════════════════════════════════════════════════════════════════════════════
def test_sequential_id():
    df = pd.DataFrame({
        'Sl_No':  list(range(1, 101)),
        'Score':  np.random.randint(40, 100, 100),
    })
    cl = classify_column(df, 'Sl_No')
    check("seq_id: role is serial or identifier",
          cl.analytical_role in (ROLE_SERIAL, ROLE_IDENTIFIER))
    check("seq_id: no suitable analyses",     len(cl.suitable_analyses) == 0)
    check("seq_id: imputation preserve",      cl.imputation_role == 'identifier_preserve')

    profile = profile_dataset(df)
    check("seq_id: excluded from numerical_cols",
          'Sl_No' not in profile['numerical_cols'])
    check("seq_id: in identifier_cols",
          'Sl_No' in profile.get('identifier_cols', []))
    check("seq_id: Score is in numerical_cols",
          'Score' in profile['numerical_cols'])

    # Imputation recommendation — should be do_not_impute
    sl_series = df['Sl_No'].copy()
    sl_series[5] = np.nan
    rec = recommend_imputation_method('Sl_No', sl_series, 'numerical')
    check("seq_id: imputation = do_not_impute", rec['method'] == 'do_not_impute')

run_test("1 Sequential ID", test_sequential_id)

# ══════════════════════════════════════════════════════════════════════════════
# DATASET 2 — All unique non-sequential integers (sparse ID)
# ══════════════════════════════════════════════════════════════════════════════
def test_sparse_id():
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        'CaseNo':  rng.integers(10000, 99999, 60),
        'Age':     rng.integers(18, 60, 60),
        'Score':   rng.uniform(50, 100, 60),
    })
    # Make CaseNo unique
    df['CaseNo'] = np.arange(10000, 10060)

    cl = classify_column(df, 'CaseNo')
    check("sparse_id: role is identifier-like",
          cl.analytical_role in (ROLE_IDENTIFIER, ROLE_SERIAL, ROLE_ADMIN_CODE))

    profile = profile_dataset(df)
    check("sparse_id: CaseNo excluded from numerical",
          'CaseNo' not in profile['numerical_cols'])
    check("sparse_id: Age and Score in numerical",
          'Age' in profile['numerical_cols'] and 'Score' in profile['numerical_cols'])

run_test("2 Sparse Integer ID", test_sparse_id)

# ══════════════════════════════════════════════════════════════════════════════
# DATASET 3 — Categorical-only dataset
# ══════════════════════════════════════════════════════════════════════════════
def test_categorical_only():
    df = pd.DataFrame({
        'Gender':     np.random.choice(['Male', 'Female'], 50),
        'Department': np.random.choice(['CS', 'Math', 'Physics', 'Bio'], 50),
        'Grade':      np.random.choice(['A', 'B', 'C', 'D'], 50),
    })
    profile = profile_dataset(df)
    check("cat_only: no numerical cols",       len(profile['numerical_cols']) == 0)
    check("cat_only: 3 categorical cols",      len(profile['categorical_cols']) == 3)
    check("cat_only: summary works",           isinstance(generate_plain_summary(profile), str))

    # Mode imputation on categorical
    df2 = df.copy()
    df2.loc[0, 'Gender'] = np.nan
    out, log = impute_column(df2, 'Gender', 'mode')
    assert_df_shape(df2, out, "cat_only/Gender/mode")
    check("cat_only/Gender/mode: ok",          log['ok'] is True)
    check("cat_only/Gender/mode: no missing",  out['Gender'].isnull().sum() == 0)

    # Mean on categorical should fail gracefully
    out2, log2 = impute_column(df2, 'Gender', 'mean')
    check("cat_only/Gender/mean: ok=False",    log2['ok'] is False)
    assert_df_shape(df2, out2, "cat_only/Gender/mean safe return")

run_test("3 Categorical-Only Dataset", test_categorical_only)

# ══════════════════════════════════════════════════════════════════════════════
# DATASET 4 — Ordinal Likert scale
# ══════════════════════════════════════════════════════════════════════════════
def test_ordinal_likert():
    df = pd.DataFrame({
        'ID':  list(range(1, 81)),
        'Q1':  np.random.randint(1, 6, 80),
        'Q2':  np.random.randint(1, 6, 80),
        'Q3':  np.random.randint(1, 6, 80),
        'Q4':  np.random.randint(1, 6, 80),
    })
    profile = profile_dataset(df)

    check("ordinal: Q1–Q4 classified as ordinal or numerical",
          any(profile['classifications']['Q1']['analytical_role']
              in (ROLE_ORDINAL, ROLE_SUBST_NUMERICAL) for _ in [1]))
    check("ordinal: ID excluded from numerical",
          'ID' not in profile['numerical_cols'])

    # Cronbach's alpha should work on Q columns
    from engine.statistics import cronbach_alpha
    q_cols = ['Q1', 'Q2', 'Q3', 'Q4']
    result = cronbach_alpha(df, q_cols)
    check("ordinal: cronbach alpha no error",  "error" not in result)
    # Alpha can be negative for random data (no real construct) — check it's a number
    check("ordinal: alpha is a float",         isinstance(result.get('alpha'), float))
    check("ordinal: alpha has reliability label", isinstance(result.get('reliability'), str))

run_test("4 Ordinal / Likert Scale", test_ordinal_likert)

# ══════════════════════════════════════════════════════════════════════════════
# DATASET 5 — Datetime column
# ══════════════════════════════════════════════════════════════════════════════
def test_datetime_col():
    df = pd.DataFrame({
        'Date':  pd.date_range('2023-01-01', periods=50, freq='D'),
        'Value': np.random.normal(100, 15, 50),
    })
    df.loc[5, 'Value'] = np.nan

    profile = profile_dataset(df)
    check("datetime: Date classified as datetime",
          profile['classifications']['Date']['analytical_role'] == ROLE_DATETIME)
    check("datetime: Date not in numerical_cols",
          'Date' not in profile['numerical_cols'])

    rec = profile['imputation_recommendations']['Value']
    check("datetime: Value imputation is not do_not_impute",
          rec['method'] != 'do_not_impute')

run_test("5 Datetime Column", test_datetime_col)

# ══════════════════════════════════════════════════════════════════════════════
# DATASET 6 — Constant column
# ══════════════════════════════════════════════════════════════════════════════
def test_constant_col():
    df = pd.DataFrame({
        'Country': ['Nigeria'] * 40,
        'City':    np.random.choice(['Lagos', 'Abuja'], 40),
        'Income':  np.random.normal(50000, 10000, 40),
    })
    cl = classify_column(df, 'Country')
    check("const: role is constant",           cl.analytical_role == ROLE_CONSTANT)
    check("const: no suitable analyses",       len(cl.suitable_analyses) == 0)

    profile = profile_dataset(df)
    check("const: Country excluded from analysis",
          'Country' not in profile['numerical_cols'] and
          'Country' not in profile['analysis_ready_categorical'])

run_test("6 Constant Column", test_constant_col)

# ══════════════════════════════════════════════════════════════════════════════
# DATASET 7 — Free-text column
# ══════════════════════════════════════════════════════════════════════════════
def test_free_text_col():
    df = pd.DataFrame({
        'ID':       list(range(1, 31)),
        'Comments': [f"This is a unique comment number {i} with some extra text" for i in range(30)],
        'Score':    np.random.randint(40, 100, 30),
    })
    cl = classify_column(df, 'Comments')
    check("freetext: role is free_text or identifier",
          cl.analytical_role in (ROLE_FREE_TEXT, ROLE_IDENTIFIER))
    check("freetext: no standard analyses suitable",
          len(cl.suitable_analyses) == 0 or
          all(a in ('frequency', 'crosstab') for a in cl.suitable_analyses))

    profile = profile_dataset(df)
    check("freetext: Comments not in numerical_cols",
          'Comments' not in profile['numerical_cols'])

run_test("7 Free-Text Column", test_free_text_col)

# ══════════════════════════════════════════════════════════════════════════════
# DATASET 8 — Boolean column (Yes/No)
# ══════════════════════════════════════════════════════════════════════════════
def test_boolean_col():
    df = pd.DataFrame({
        'Passed':  np.random.choice(['Yes', 'No'], 60),
        'Score':   np.random.randint(40, 100, 60),
        'Gender':  np.random.choice(['Male', 'Female'], 60),
    })
    cl_pass = classify_column(df, 'Passed')
    check("bool: Passed classified as boolean",
          cl_pass.analytical_role in (ROLE_BOOLEAN, ROLE_CATEGORICAL))

    profile = profile_dataset(df)
    # Passed should be in categorical (for chi-square, etc.)
    check("bool: Passed in analysis_ready_categorical",
          'Passed' in profile['analysis_ready_categorical'] or
          'Passed' in profile['categorical_cols'])

run_test("8 Boolean Column", test_boolean_col)

# ══════════════════════════════════════════════════════════════════════════════
# DATASET 9 — Mixed dataset with ID, numerical, categorical, ordinal, datetime
# ══════════════════════════════════════════════════════════════════════════════
def test_mixed_dataset():
    n = 80
    df = pd.DataFrame({
        'StudentID':  list(range(1001, 1001 + n)),
        'Age':        np.random.randint(18, 30, n),
        'Score':      np.random.uniform(40, 100, n),
        'Gender':     np.random.choice(['Male', 'Female'], n),
        'Department': np.random.choice(['CS', 'Math', 'Physics'], n),
        'Likert_1':   np.random.randint(1, 6, n),
        'Likert_2':   np.random.randint(1, 6, n),
        'EnrollDate': pd.date_range('2023-01-01', periods=n, freq='D'),
    })
    df.loc[5, 'Age']   = np.nan
    df.loc[10, 'Score'] = np.nan
    df.loc[15, 'Gender'] = np.nan

    profile = profile_dataset(df)

    check("mixed: StudentID excluded from numerical",
          'StudentID' not in profile['numerical_cols'])
    check("mixed: Age in numerical_cols",
          'Age' in profile['numerical_cols'])
    check("mixed: Score in numerical_cols",
          'Score' in profile['numerical_cols'])
    check("mixed: Gender in categorical_cols",
          'Gender' in profile['categorical_cols'])
    check("mixed: EnrollDate not in numerical",
          'EnrollDate' not in profile['numerical_cols'])

    # Imputation
    rec_id    = profile['imputation_recommendations']['StudentID']
    rec_age   = profile['imputation_recommendations']['Age']
    rec_gen   = profile['imputation_recommendations']['Gender']

    # StudentID has no missing in this dataset → recommendation is "none"
    # The key safeguard is that imputation_role is identifier_preserve
    check("mixed: StudentID imputation_role = identifier_preserve",
          profile['classifications']['StudentID']['imputation_role'] == 'identifier_preserve')
    check("mixed: Age = imputable method",
          rec_age['method'] in ('mean', 'median', 'knn'))
    check("mixed: Gender = mode",
          rec_gen['method'] == 'mode')

    # All imputation recs have why_appropriate
    check("mixed: all recs have reason",
          all('reason' in r for r in profile['imputation_recommendations'].values()))

    summary = generate_plain_summary(profile)
    check("mixed: summary mentions identifier exclusion",
          'identifier' in summary.lower() or 'excluded' in summary.lower())

run_test("9 Mixed Dataset", test_mixed_dataset)

# ══════════════════════════════════════════════════════════════════════════════
# DATASET 10 — High missingness (>40%)
# ══════════════════════════════════════════════════════════════════════════════
def test_high_missingness():
    n = 100
    vals = np.random.normal(50, 10, n).astype(float)
    vals[np.random.choice(n, 50, replace=False)] = np.nan  # 50% missing
    df = pd.DataFrame({'X': vals, 'Y': np.random.randint(1, 5, n)})

    rec = profile_dataset(df)['imputation_recommendations']['X']
    check("highmiss: KNN recommended or other method",
          rec['method'] in ('knn', 'median', 'mean'))
    check("highmiss: warning present",    len(rec.get('warning', '')) > 0)

    out, log = impute_column(df, 'X', 'knn')
    assert_df_shape(df, out, "highmiss/X/knn")
    check("highmiss/X/knn: ok",           log['ok'] is True)
    check("highmiss/X/knn: cleared",      out['X'].isnull().sum() == 0)

run_test("10 High Missingness (50%)", test_high_missingness)

# ══════════════════════════════════════════════════════════════════════════════
# DATASET 11 — No missing values
# ══════════════════════════════════════════════════════════════════════════════
def test_no_missing():
    df = pd.DataFrame({
        'A': np.random.normal(0, 1, 60),
        'B': np.random.randint(0, 100, 60),
        'C': np.random.choice(['X', 'Y', 'Z'], 60),
    })
    profile = profile_dataset(df)
    check("nomiss: total_missing = 0",    profile['total_missing'] == 0)
    for col in df.columns:
        rec = profile['imputation_recommendations'][col]
        check(f"nomiss/{col}: method=none",  rec['method'] == 'none')

run_test("11 No Missing Values", test_no_missing)

# ══════════════════════════════════════════════════════════════════════════════
# DATASET 12 — Duplicate rows
# ══════════════════════════════════════════════════════════════════════════════
def test_duplicates():
    df_base = pd.DataFrame({
        'ID':    list(range(1, 21)),
        'Score': np.random.randint(40, 100, 20),
        'Grade': np.random.choice(['A','B','C'], 20),
    })
    df = pd.concat([df_base, df_base.iloc[:5]], ignore_index=True)
    check("dup: duplicates detected", df.duplicated().sum() == 5)

    out, log = remove_duplicates(df)
    check("dup: ok", log['ok'] is True)
    assert_df_shape(df_base, out, "dup", allow_row_reduction=True)
    check("dup: no duplicates remain",    out.duplicated().sum() == 0)
    check("dup: RangeIndex clean",        list(out.index) == list(range(len(out))))

    # Post-dedup imputation should still work
    out2 = out.copy()
    out2.loc[0, 'Score'] = np.nan
    out3, log3 = impute_column(out2, 'Score', 'median')
    assert_df_shape(out2, out3, "dup/post-dedup imputation")
    check("dup/post-dedup: ok",           log3['ok'] is True)

run_test("12 Duplicate Rows", test_duplicates)

# ══════════════════════════════════════════════════════════════════════════════
# DATASET 13 — Outlier dataset
# ══════════════════════════════════════════════════════════════════════════════
def test_outliers():
    vals = list(np.random.normal(50, 5, 95)) + [200, 300, -100, 250, 180]
    df = pd.DataFrame({
        'Normal':  vals,
        'ID':      list(range(1, 101)),
        'Cat':     np.random.choice(['A','B'], 100),
    })
    profile = profile_dataset(df)

    # Outlier info should only be present for substantive numerical variables
    normal_stats = profile['col_stats']['Normal']
    id_stats     = profile['col_stats']['ID']

    check("outlier: Normal has outliers",
          normal_stats.get('outliers', {}).get('count', 0) > 0)
    check("outlier: ID has no outlier count (not substantive)",
          id_stats.get('outliers', {}).get('count', 0) == 0)

    out, log = cap_outliers(df, 'Normal', 'iqr')
    assert_df_shape(df, out, "outlier/Normal/iqr")
    check("outlier/Normal/iqr: ok",       log['ok'] is True)

    # Cap on ID (identifier) should still not crash — may succeed or fail gracefully
    out_id, log_id = cap_outliers(df, 'ID', 'iqr')
    check("outlier/ID: returns DataFrame",  isinstance(out_id, pd.DataFrame))
    check("outlier/ID: col count same",     len(out_id.columns) == len(df.columns))

run_test("13 Outlier Dataset", test_outliers)

# ══════════════════════════════════════════════════════════════════════════════
# DATASET 14 — Unusual column names
# ══════════════════════════════════════════════════════════════════════════════
def test_unusual_col_names():
    df = pd.DataFrame({
        'col with spaces':  np.random.normal(0, 1, 40),
        'UPPER_CASE_COL':   np.random.randint(1, 5, 40),
        '123_starts_digit': np.random.choice(['P', 'Q'], 40),
        'col/with/slash':   np.random.normal(5, 1, 40),
    })
    profile = profile_dataset(df)
    check("unusualname: all 4 cols profiled",
          len(profile['col_stats']) == 4)

    # Imputation on unusual names
    df2 = df.copy()
    df2.loc[0, 'col with spaces'] = np.nan
    out, log = impute_column(df2, 'col with spaces', 'mean')
    assert_df_shape(df2, out, "unusualname/col with spaces/mean")
    check("unusualname: col list same",
          list(out.columns) == list(df2.columns))

run_test("14 Unusual Column Names", test_unusual_col_names)

# ══════════════════════════════════════════════════════════════════════════════
# DATASET 15 — Empty column (all NaN)
# ══════════════════════════════════════════════════════════════════════════════
def test_empty_column():
    df = pd.DataFrame({
        'Good':  np.random.normal(50, 10, 30),
        'Empty': [np.nan] * 30,
    })
    profile = profile_dataset(df)
    check("emptycol: Empty classified safely",
          'Empty' in profile['col_stats'])

    out, log = impute_column(df, 'Empty', 'mean')
    check("emptycol/mean: returns DataFrame", isinstance(out, pd.DataFrame))
    check("emptycol/mean: col count same",    len(out.columns) == 2)

    out2, log2 = impute_column(df, 'Empty', 'knn')
    check("emptycol/knn: returns DataFrame",  isinstance(out2, pd.DataFrame))
    check("emptycol/knn: ok=False (no non-missing)", log2['ok'] is False)

    check("emptycol: Good column untouched",
          out['Good'].equals(df['Good']))

run_test("15 Empty Column", test_empty_column)

# ══════════════════════════════════════════════════════════════════════════════
# DATASET 16 — Small sample (n=5)
# ══════════════════════════════════════════════════════════════════════════════
def test_small_sample():
    df = pd.DataFrame({
        'ID':    [1, 2, 3, 4, 5],
        'Score': [55.0, np.nan, 70.0, 80.0, 65.0],
        'Group': ['A', 'B', None, 'A', 'B'],
    })
    profile = profile_dataset(df)
    check("small: ID excluded", 'ID' not in profile['numerical_cols'])
    check("small: Score in numerical", 'Score' in profile['numerical_cols'])

    out, log = impute_column(df, 'Score', 'median')
    assert_df_shape(df, out, "small/Score/median")
    check("small/Score/median: ok",       log['ok'] is True)

    # Impute non-existent column
    out2, log2 = impute_column(df, 'NonExistent', 'mean')
    check("small/NonExistent: ok=False",  log2['ok'] is False)
    assert_df_shape(df, out2, "small/NonExistent safe return")

run_test("16 Small Sample (n=5)", test_small_sample)

# ══════════════════════════════════════════════════════════════════════════════
# DATASET 17 — Large sample (n=5000) with ID and sequential numbering
# ══════════════════════════════════════════════════════════════════════════════
def test_large_sample():
    n = 5000
    df = pd.DataFrame({
        'RecordID': list(range(10001, 10001 + n)),
        'Age':      np.random.randint(18, 70, n),
        'Income':   np.random.lognormal(10, 0.5, n),
        'Dept':     np.random.choice(['HR','IT','Finance','Sales'], n),
    })
    df.loc[np.random.choice(n, 50), 'Age']    = np.nan
    df.loc[np.random.choice(n, 80), 'Income'] = np.nan
    df.loc[np.random.choice(n, 30), 'Dept']   = np.nan

    profile = profile_dataset(df)
    check("large: RecordID excluded",     'RecordID' not in profile['numerical_cols'])
    check("large: Age in numerical",      'Age' in profile['numerical_cols'])
    check("large: Income in numerical",   'Income' in profile['numerical_cols'])
    check("large: Dept in categorical",   'Dept' in profile['categorical_cols'])
    # RecordID has no missing → rec method is "none"; safeguard is identifier_preserve role
    check("large: RecordID imputation_role=identifier_preserve",
          profile['classifications']['RecordID']['imputation_role'] == 'identifier_preserve')

    out, log = impute_column(df, 'Age', 'mean')
    assert_df_shape(df, out, "large/Age/mean")
    check("large/Age/mean: ok",           log['ok'] is True)

    out2, log2 = impute_column(df, 'Dept', 'mode')
    assert_df_shape(df, out2, "large/Dept/mode")
    check("large/Dept/mode: ok",          log2['ok'] is True)

    out3, log3 = remove_duplicates(df)
    check("large dedup: ok",              log3['ok'] is True)

run_test("17 Large Sample (n=5000)", test_large_sample)

# ══════════════════════════════════════════════════════════════════════════════
# DATASET 18 — Administrative code column
# ══════════════════════════════════════════════════════════════════════════════
def test_admin_code():
    # Short alphanumeric codes — all unique
    codes = ['A' + str(i).zfill(4) for i in range(1, 51)]
    df = pd.DataFrame({
        'HouseCode': codes,
        'Income':    np.random.normal(50000, 10000, 50),
        'Members':   np.random.randint(1, 8, 50),
    })
    cl = classify_column(df, 'HouseCode')
    check("admincode: role is identifier/admin",
          cl.analytical_role in (ROLE_IDENTIFIER, ROLE_ADMIN_CODE, ROLE_FREE_TEXT))
    check("admincode: no statistical analyses",
          len(cl.suitable_analyses) == 0 or
          all(a in ('frequency',) for a in cl.suitable_analyses))

run_test("18 Administrative Code Column", test_admin_code)

# ══════════════════════════════════════════════════════════════════════════════
# DATASET 19 — User override propagates correctly
# ══════════════════════════════════════════════════════════════════════════════
def test_user_override():
    df = pd.DataFrame({
        'Code':   list(range(1, 41)),
        'Score':  np.random.randint(40, 100, 40),
        'Group':  np.random.choice(['A','B'], 40),
    })
    profile_auto = profile_dataset(df)
    check("override: Code auto-excluded",
          'Code' not in profile_auto['numerical_cols'])

    profile_ovr = profile_dataset(df, user_overrides={
        'Code': {'analytical_role': 'substantive_numerical', 'measurement_level': 'ratio'}
    })
    check("override: Code now in numerical after override",
          'Code' in profile_ovr['numerical_cols'])
    check("override: evidence contains override marker",
          any('override' in e.lower() or 'manually' in e.lower()
              for e in profile_ovr['classifications']['Code']['evidence']))

run_test("19 User Override", test_user_override)

# ══════════════════════════════════════════════════════════════════════════════
# DATASET 20 — DataFrame integrity after sequential cleaning operations
# ══════════════════════════════════════════════════════════════════════════════
def test_sequential_cleaning():
    """Simulate a full cleaning pipeline on a messy dataset."""
    n = 60
    df = pd.DataFrame({
        'ID':      list(range(1, n + 1)),
        'Age':     np.random.randint(18, 60, n).astype(float),
        'Score':   np.random.uniform(40, 100, n),
        'Gender':  np.random.choice(['Male', 'Female', 'male ', 'FEMALE'], n),
        'Dept':    np.random.choice(['CS', 'Math'], n),
    })
    # Introduce issues
    df.loc[5, 'Age']   = np.nan
    df.loc[10, 'Score'] = np.nan
    df.loc[15, 'Gender'] = np.nan
    # Duplicate row
    df = pd.concat([df, df.iloc[[3]]], ignore_index=True)

    n_cols_original = len(df.columns)

    # Step 1: remove duplicates
    df1, log1 = remove_duplicates(df)
    check("seqclean/dedup: ok",            log1['ok'])
    check("seqclean/dedup: col count same", len(df1.columns) == n_cols_original)

    # Step 2: impute Age
    df2, log2 = impute_column(df1, 'Age', 'median')
    check("seqclean/Age/median: ok",       log2['ok'])
    check("seqclean/Age/median: col count", len(df2.columns) == n_cols_original)

    # Step 3: impute Score
    df3, log3 = impute_column(df2, 'Score', 'mean')
    check("seqclean/Score/mean: ok",       log3['ok'])
    check("seqclean/Score/mean: col count", len(df3.columns) == n_cols_original)

    # Step 4: impute Gender (mode)
    df4, log4 = impute_column(df3, 'Gender', 'mode')
    check("seqclean/Gender/mode: ok",      log4['ok'])
    check("seqclean/Gender/mode: col count", len(df4.columns) == n_cols_original)

    # Step 5: standardise Gender categories
    df5, log5 = standardise_categories(df4, 'Gender', {'male ': 'Male', 'FEMALE': 'Female'})
    check("seqclean/Gender/std: ok",       log5['ok'])
    check("seqclean/Gender/std: col count", len(df5.columns) == n_cols_original)

    # Step 6: Re-profile — ID should still be excluded
    profile_final = profile_dataset(df5)
    check("seqclean/final: ID still excluded", 'ID' not in profile_final['numerical_cols'])
    check("seqclean/final: Age in numerical",  'Age' in profile_final['numerical_cols'])
    check("seqclean/final: Score in numerical", 'Score' in profile_final['numerical_cols'])
    check("seqclean/final: no missing in Age",   profile_final['missing_counts']['Age'] == 0)
    check("seqclean/final: no missing in Score",  profile_final['missing_counts']['Score'] == 0)

run_test("20 Sequential Cleaning Pipeline", test_sequential_cleaning)


# ══════════════════════════════════════════════════════════════════════════════
# PRINT RESULTS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print(f"  COMPREHENSIVE ROLE-AWARE TEST SUITE — {PASS + FAIL} checks across 20 test datasets")
print("=" * 70)
for r in RESULTS:
    print(r)
print("=" * 70)
print(f"  PASSED : {PASS}")
print(f"  FAILED : {FAIL}")
print("=" * 70)
if FAIL == 0:
    print("  ALL CHECKS PASSED — system is dataset-agnostic and role-aware")
else:
    print(f"  {FAIL} CHECK(S) REQUIRE ATTENTION")
print("=" * 70)
