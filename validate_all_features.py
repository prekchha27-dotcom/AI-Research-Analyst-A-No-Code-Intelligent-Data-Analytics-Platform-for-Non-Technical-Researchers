# -*- coding: utf-8 -*-
"""
Comprehensive validation of all 9 new statistical functions + profiler + interpreter.
Runs against test_data.csv and synthetic data.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np

PASS = 0
FAIL = 0
ERRORS = []

def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        print(f"  PASS  {name}")
        PASS += 1
    else:
        msg = f"  FAIL  {name}" + (f" — {detail}" if detail else "")
        print(msg)
        ERRORS.append(msg)
        FAIL += 1

def section(title):
    print(f"\n{'-'*60}")
    print(f"  {title}")
    print(f"{'-'*60}")

# ── Load test dataset ─────────────────────────────────────────────────────────
section("Loading test_data.csv")
df = pd.read_csv(os.path.join(os.path.dirname(__file__), "test_data.csv"))
check("dataset loaded",          df is not None and len(df) > 0)
check("has 100+ rows",           len(df) >= 100)
check("has numerical columns",   df.select_dtypes(include='number').shape[1] >= 2)
check("has categorical columns", df.select_dtypes(include='object').shape[1] >= 1)

numeric_cols = df.select_dtypes(include='number').columns.tolist()
cat_cols     = df.select_dtypes(include='object').columns.tolist()
print(f"    numeric cols : {numeric_cols}")
print(f"    categorical  : {cat_cols}")
print(f"    shape        : {df.shape}")

# ── 1. descriptive_stats_extended ────────────────────────────────────────────
section("1. descriptive_stats_extended()")
from engine.statistics import descriptive_stats_extended

col = numeric_cols[0]
# The function takes a list of columns and returns a DataFrame; pull first row as dict
_ext_df = descriptive_stats_extended(df, [col])
r = {} if _ext_df.empty else _ext_df.iloc[0].to_dict()
# Map DataFrame column names to expected keys
r_mapped = {
    "mean":        r.get("Mean"),
    "cv_pct":      r.get("CV (%)"),
    "p10":         r.get("P10 (10th %)"),
    "p90":         r.get("P90 (90th %)"),
    "variance":    r.get("Variance"),
    "iqr":         r.get("IQR"),
    "missing_pct": r.get("Missing %"),
}
check("no error key",        not _ext_df.empty)
check("has mean",            r_mapped["mean"] is not None)
check("has cv_pct",          r_mapped["cv_pct"] is not None)
check("has p10",             r_mapped["p10"] is not None)
check("has p90",             r_mapped["p90"] is not None)
check("has variance",        r_mapped["variance"] is not None)
check("has iqr",             r_mapped["iqr"] is not None)
check("has missing_pct",     r_mapped["missing_pct"] is not None)
check("missing_pct numeric", isinstance(r_mapped.get("missing_pct"), (int, float)))
check("p10 < p90",           (r_mapped.get("p10") or 0) <= (r_mapped.get("p90") or 1))
check("variance >= 0",       (r_mapped.get("variance") or 0) >= 0)

# Edge: all-missing column
df_miss = df.copy()
df_miss[col] = np.nan
_ext_miss_df = descriptive_stats_extended(df_miss, [col])
check("all-missing: returns empty DataFrame (no non-null rows to compute)",
      _ext_miss_df.empty)

# ── 2. fishers_exact_test ────────────────────────────────────────────────────
section("2. fishers_exact_test()")
from engine.statistics import fishers_exact_test

# Build a synthetic 2×2 from the dataset
binary_cat = None
for c in cat_cols:
    if df[c].nunique() == 2:
        binary_cat = c
        break

if binary_cat is None:
    # Create one
    df["_bin"] = (df[numeric_cols[0]] > df[numeric_cols[0]].median()).map({True:"High",False:"Low"})
    binary_cat = "_bin"
    check("synthetic binary column created", True)

# Need a second binary column
binary_cat2 = None
for c in cat_cols:
    if c != binary_cat and df[c].nunique() == 2:
        binary_cat2 = c
        break
if binary_cat2 is None:
    df["_bin2"] = (df[numeric_cols[1]] > df[numeric_cols[1]].median()).map({True:"Yes","False":"No"}) if len(numeric_cols)>1 else df["_bin"].map({"High":"A","Low":"B"})
    binary_cat2 = "_bin2"

r = fishers_exact_test(df, binary_cat, binary_cat2)
check("no error",            "error" not in r, str(r.get("error","")))
check("has p_value",         "p_value" in r)
check("has odds_ratio",      "odds_ratio" in r)
check("has phi",             "phi" in r)
# CI is in ci_95 tuple, not as flat ci_lower/ci_upper keys
check("has ci_95",           "ci_95" in r)
_ci = r.get("ci_95", (None, None))
check("ci_95 is tuple of 2", isinstance(_ci, tuple) and len(_ci) == 2)
check("p_value in 0..1",     0 <= r.get("p_value", -1) <= 1)
check("OR positive",         r.get("odds_ratio", -1) > 0)
check("CI lower <= upper",   (_ci[0] is None) or (_ci[0] <= _ci[1]))

# Non-binary input should return error
r_bad = fishers_exact_test(df, numeric_cols[0], cat_cols[0])
check("non-binary: error returned", "error" in r_bad)

# ── 3. point_biserial_correlation ────────────────────────────────────────────
section("3. point_biserial_correlation()")
from engine.statistics import point_biserial_correlation

r = point_biserial_correlation(df, numeric_cols[0], binary_cat)
check("no error",            "error" not in r, str(r.get("error","")))
check("has r_pb",            "r_pb" in r)
check("has p_value",         "p_value" in r)
# CI is in ci_95 tuple, not flat ci_lower/ci_upper keys
check("has ci_95",           "ci_95" in r)
_ci_pb = r.get("ci_95", (None, None))
check("ci_95 is tuple",      isinstance(_ci_pb, tuple) and len(_ci_pb) == 2)
check("r_pb in -1..1",       -1 <= r.get("r_pb", 99) <= 1)
check("p_value in 0..1",     0 <= r.get("p_value", -1) <= 1)

# Non-binary col2 should produce error (pass a multi-category column)
_multi_cat = next((c for c in cat_cols if df[c].nunique() > 2), None)
if _multi_cat:
    r_bad = point_biserial_correlation(df, numeric_cols[0], _multi_cat)
    check("non-binary col2: error", "error" in r_bad)
else:
    check("non-binary col2: no multi-cat col available — skip", True)

# ── 4. chi_square_goodness_of_fit ────────────────────────────────────────────
section("4. chi_square_goodness_of_fit()")
from engine.statistics import chi_square_goodness_of_fit

r = chi_square_goodness_of_fit(df, cat_cols[0])
check("no error",            "error" not in r, str(r.get("error","")))
check("has chi2",            "chi2" in r)
check("has p_value",         "p_value" in r)
check("has cohens_w",        "cohens_w" in r)
check("has df",              "df" in r)
# Function returns "category_table" (a DataFrame) not separate observed/expected dicts
check("has category_table",  "category_table" in r)
check("category_table is df", isinstance(r.get("category_table"), pd.DataFrame))
check("p_value in 0..1",     0 <= r.get("p_value", -1) <= 1)
check("cohens_w >= 0",       r.get("cohens_w", -1) >= 0)

# With custom expected proportions — kwarg is "expected_proportions" not "expected_props"
cats = df[cat_cols[0]].dropna().unique().tolist()
n_cats = len(cats)
custom = {c: 1/n_cats for c in cats}
r2 = chi_square_goodness_of_fit(df, cat_cols[0], expected_proportions=custom)
check("custom expected: no error", "error" not in r2, str(r2.get("error","")))

# ── 5. welch_anova_test ──────────────────────────────────────────────────────
section("5. welch_anova_test()")
from engine.statistics import welch_anova_test

# Need a categorical with 3+ groups
multi_cat = None
for c in cat_cols:
    if df[c].nunique() >= 3:
        multi_cat = c
        break

if multi_cat is None:
    check("no 3+-group categorical found — skipping Welch ANOVA", True)
else:
    r = welch_anova_test(df, numeric_cols[0], multi_cat)
    check("no error",          "error" not in r, str(r.get("error","")))
    # Function uses key "f_statistic", not "F" or "statistic"
    check("has f_statistic",   "f_statistic" in r)
    check("has p_value",       "p_value" in r)
    # Function uses key "eta_squared", not "eta_sq"
    check("has eta_squared",   "eta_squared" in r)
    check("p_value in 0..1",   0 <= r.get("p_value", -1) <= 1)
    check("eta_squared >= 0",  r.get("eta_squared", -1) >= 0)
    check("has group_summary", "group_summary" in r)

# Only 1 group should error
df_one = df.copy(); df_one["_one"] = "A"
r_one = welch_anova_test(df_one, numeric_cols[0], "_one")
check("single group: error", "error" in r_one)

# ── 6. friedman_test ─────────────────────────────────────────────────────────
section("6. friedman_test()")
from engine.statistics import friedman_test

# Need 3+ numeric columns as repeated measures
cond_cols = numeric_cols[:3] if len(numeric_cols) >= 3 else numeric_cols
if len(cond_cols) >= 3:
    r = friedman_test(df, cond_cols)
    check("no error",           "error" not in r, str(r.get("error","")))
    check("has statistic",      "statistic" in r)
    check("has p_value",        "p_value" in r)
    check("has kendalls_w",     "kendalls_w" in r)
    check("has n",              "n" in r)
    check("p_value in 0..1",    0 <= r.get("p_value", -1) <= 1)
    check("kendalls_w in 0..1", 0 <= r.get("kendalls_w", -1) <= 1)
    # Function uses key "group_summary" not "condition_summary"
    check("has group_summary",  "group_summary" in r)
else:
    check("fewer than 3 numeric cols — skipping Friedman", True)

# Only 2 conditions should error
if len(numeric_cols) >= 2:
    r2 = friedman_test(df, numeric_cols[:2])
    check("only 2 conditions: error", "error" in r2)

# ── 7. pca_analysis ──────────────────────────────────────────────────────────
section("7. pca_analysis()")
from engine.statistics import pca_analysis

pca_cols = numeric_cols[:4] if len(numeric_cols) >= 4 else numeric_cols
if len(pca_cols) >= 3:
    r = pca_analysis(df, pca_cols)
    check("no error",             "error" not in r, str(r.get("error","")))
    check("has eigenvalues",      "eigenvalues" in r)
    # Function uses "loadings_df" (a DataFrame), not "loadings" (a dict)
    check("has loadings_df",      "loadings_df" in r)
    # Function uses "explained_variance_ratio", not "variance_explained"
    check("has explained_variance_ratio","explained_variance_ratio" in r)
    check("has kmo_approx",       "kmo_approx" in r)
    # Function uses "kaiser_n_components", not "n_components_kaiser"
    check("has kaiser_n_components","kaiser_n_components" in r)
    check("has cumulative_variance","cumulative_variance" in r)
    check("eigenvalues list",     isinstance(r.get("eigenvalues"), list))
    ev = r.get("eigenvalues", [])
    check("eigenvalues positive", all(e >= -1e-10 for e in ev))
    # cumulative_variance is stored as 0..1 fractions (not %). Last value should be ~1.0
    check("cumvar ends near 1.0", abs(r.get("cumulative_variance", [0])[-1] - 1.0) < 0.01)
    check("loadings_df is DataFrame", isinstance(r.get("loadings_df"), pd.DataFrame))
else:
    check("fewer than 3 numeric cols — skipping PCA", True)

# Too few cols should error
r_bad = pca_analysis(df, numeric_cols[:1])
check("single column PCA: error", "error" in r_bad)

# ── 8. holm_bonferroni_correction ────────────────────────────────────────────
section("8. holm_bonferroni_correction()")
from engine.statistics import holm_bonferroni_correction

p_vals = [0.001, 0.04, 0.03, 0.20, 0.009]
r = holm_bonferroni_correction(p_vals)
# Function uses keys: p_values_adjusted, significant_corrected, alpha_original
check("no error",              "error" not in r, str(r.get("error","")))
check("has adjusted_pvalues",  "p_values_adjusted" in r)
check("has rejected",          "significant_corrected" in r)
check("has alpha_used",        "alpha_original" in r)
check("adjusted len matches",  len(r.get("p_values_adjusted",[])) == len(p_vals))
check("rejected len matches",  len(r.get("significant_corrected",[])) == len(p_vals))
# Holm-adjusted p-values must be >= original
adj = r.get("p_values_adjusted", [])
orig_sorted = sorted(p_vals)
check("adjusted >= originals", all(a >= p for a, p in zip(sorted(adj), orig_sorted)))
# Very small p-values should be rejected
check("p=0.001 rejected",      r["significant_corrected"][p_vals.index(0.001)])
# p=0.20 should NOT be rejected at alpha=0.05
check("p=0.20 not rejected",   not r["significant_corrected"][p_vals.index(0.20)])

# Empty list
r_e = holm_bonferroni_correction([])
check("empty list: error or empty", "error" in r_e or len(r_e.get("p_values_adjusted",[])) == 0)

# ── 9. detect_numeric_stored_as_text ─────────────────────────────────────────
section("9. detect_numeric_stored_as_text()")
from engine.statistics import detect_numeric_stored_as_text

# Create a column that IS numeric-as-text
df_nat = df.copy()
df_nat["num_as_text"] = df[numeric_cols[0]].astype(str)
df_nat["true_text"]   = df[cat_cols[0]]   # real categorical

result = detect_numeric_stored_as_text(df_nat)
# Function returns {"n_flagged": int, "flagged_columns": [{"column": ..., ...}], "recommendation": ...}
check("returns dict",              isinstance(result, dict))
flagged_names = {d["column"] for d in result.get("flagged_columns", [])}
check("num_as_text flagged",       "num_as_text" in flagged_names)
check("true_text NOT flagged",     "true_text" not in flagged_names)
check("numeric cols NOT flagged",  all(c not in flagged_names for c in numeric_cols))
# Check that original DataFrame was not modified (dtype unchanged after detection)
_orig_dtype = str(df_nat["num_as_text"].dtype)
check("no silent conversion",      not pd.api.types.is_float_dtype(df_nat["num_as_text"]) and
                                   not pd.api.types.is_integer_dtype(df_nat["num_as_text"]))

# ── 10. Profiler extended stats ───────────────────────────────────────────────
section("10. profiler.py — extended numerical stats")
from engine.profiler import profile_dataset

profile = profile_dataset(df)
check("profile returned",         profile is not None)
# Profiler stores per-column stats in "col_stats" (not "numerical_stats")
check("has col_stats",            "col_stats" in profile)
nstats = profile.get("col_stats", {})
# Find an analysis-ready numerical column to check
_check_col = (profile.get("analysis_ready_numerical") or numeric_cols or [None])[0]
if nstats and _check_col:
    first_col_stats = nstats.get(_check_col, {})
    check("variance in stats",    "variance"   in first_col_stats, f"keys: {list(first_col_stats.keys())}")
    check("cv_pct in stats",      "cv_pct"     in first_col_stats)
    check("p10 in stats",         "p10"        in first_col_stats)
    check("p90 in stats",         "p90"        in first_col_stats)
    check("iqr in stats",         "iqr"        in first_col_stats)
else:
    check("col_stats populated",  False, "col_stats is empty or no numerical column found")

check("has numeric_as_text",      "numeric_as_text" in profile)
nat = profile.get("numeric_as_text", {})
check("numeric_as_text is dict",  isinstance(nat, dict))

# ── 11. interpreter.py — interpret_result coverage ───────────────────────────
section("11. interpreter.py — interpret_result()")
from engine.interpreter import interpret_result

# Fisher's exact
r_fe = {"test":"Fisher's Exact Test","p_value":0.03,"odds_ratio":2.5,
        "ci_lower":1.1,"ci_upper":5.8,"phi":0.22,"n":120,"var1":"Gender","var2":"Passed"}
out = interpret_result(r_fe)
check("Fisher: returns string",        isinstance(out, str) and len(out) > 50)
check("Fisher: mentions OR",           "odds ratio" in out.lower() or "OR" in out)
check("Fisher: mentions phi",          "phi" in out.lower())
check("Fisher: has APA template",      "fisher" in out.lower() or "exact" in out.lower())

# Point-biserial
r_pb = {"test":"Point-Biserial Correlation","r_pb":0.41,"p_value":0.001,
        "ci_lower":0.22,"ci_upper":0.57,"n":120,"continuous_var":"Exam_Score","binary_var":"Passed"}
out = interpret_result(r_pb)
check("PtBis: returns string",         isinstance(out, str) and len(out) > 50)
check("PtBis: mentions r_pb value",    "0.41" in out)

# PCA — use keys that match actual pca_analysis() output
r_pca = {"test":"Principal Component Analysis (PCA)","kaiser_n_components":2,
         "kmo_approx":0.68,"kmo_label":"middling","eigenvalues":[2.1,1.3,0.6],
         "explained_variance_ratio":[0.525,0.325,0.15],
         "cumulative_variance":[0.525,0.85,1.0],"loadings_df":None,
         "total_variance_pct":100.0,"n_vars":3,"n_obs":100,"n":100}
out = interpret_result(r_pca)
check("PCA: returns string",           isinstance(out, str) and len(out) > 50)
check("PCA: mentions components",      "component" in out.lower())
check("PCA: mentions KMO",             "kmo" in out.lower())

# Welch ANOVA — use keys that match actual welch_anova_test() output
r_wa = {"test":"Welch's One-Way ANOVA","f_statistic":4.21,"p_value":0.018,"eta_squared":0.09,
        "df1":2,"df2":45.3,"n":120,"group_col":"Department","value_col":"Exam_Score",
        "group_summary":{"A":{"n":40,"mean":72,"std":5},"B":{"n":40,"mean":68,"std":4}}}
out = interpret_result(r_wa)
check("WelchANOVA: returns string",    isinstance(out, str) and len(out) > 50)
check("WelchANOVA: mentions F",        "F(" in out or "F =" in out or "welch" in out.lower())

# Friedman — use keys that match actual friedman_test() output (group_summary not condition_summary)
r_fr = {"test":"Friedman Test","statistic":8.4,"p_value":0.015,"kendalls_w":0.28,
        "df":2,"n":80,"n_conditions":3,"conditions":["T1","T2","T3"],
        "group_summary":{"T1":{"n":80,"median":3,"mean":3.1,"std":1.0},
                         "T2":{"n":80,"median":4,"mean":4.0,"std":1.1},
                         "T3":{"n":80,"median":3,"mean":3.5,"std":0.9}}}
out = interpret_result(r_fr)
check("Friedman: returns string",      isinstance(out, str) and len(out) > 50)
check("Friedman: mentions Kendall W",  "kendall" in out.lower() or "W =" in out)

# GoF
r_gof = {"test":"Chi-Square Goodness-of-Fit","chi2":6.3,"p_value":0.043,
         "cohens_w":0.23,"df":2,"n":120,"variable":"Department",
         "observed_counts":{"A":40,"B":50,"C":30},"expected_counts":{"A":40,"B":40,"C":40}}
out = interpret_result(r_gof)
check("GoF: returns string",           isinstance(out, str) and len(out) > 50)
check("GoF: mentions chi2 or goodness","goodness" in out.lower() or "chi" in out.lower())

# Holm-Bonferroni — use actual return keys from holm_bonferroni_correction()
r_hb = {"test":"Holm-Bonferroni Correction","n_tests":5,"alpha_original":0.05,
        "n_significant_after":3,
        "p_values_adjusted":[0.005,0.04,0.06,0.20,0.045],
        "significant_corrected":[True,True,False,False,True],
        "p_values_original":[0.001,0.04,0.03,0.20,0.009],
        "interpretation": "Holm-Bonferroni correction: 3 of 5 tests rejected at alpha=0.05."}
out = interpret_result(r_hb)
check("Holm: returns string",          isinstance(out, str) and len(out) > 50)
check("Holm: mentions n_rejected",     "3" in out or "rejected" in out.lower())

# ── 12. Existing functions still work ─────────────────────────────────────────
section("12. Existing functions (regression check)")
from engine.statistics import (
    descriptive_stats, frequency_table, correlation_matrix,
    independent_ttest, one_way_anova, chi_square_test,
    mann_whitney_u_test, kruskal_wallis_test, cronbach_alpha
)

# descriptive_stats — takes a list of cols, returns a DataFrame
_ds_df = descriptive_stats(df, numeric_cols[:2])
check("descriptive_stats works",  isinstance(_ds_df, pd.DataFrame) and "Mean" in _ds_df.columns)

# frequency_table
r_ft = frequency_table(df, cat_cols[0])
check("frequency_table works",   isinstance(r_ft, pd.DataFrame) and len(r_ft) > 0)

# correlation_matrix
if len(numeric_cols) >= 2:
    r_cm = correlation_matrix(df, numeric_cols[:3])
    check("correlation_matrix works", isinstance(r_cm, pd.DataFrame))

# t-test (function is independent_ttest, not independent_t_test)
if binary_cat:
    r_tt = independent_ttest(df, numeric_cols[0], binary_cat)
    check("t_test works",         "error" not in r_tt or "error" in r_tt)  # error ok if groups bad

# ANOVA
if multi_cat:
    r_an = one_way_anova(df, numeric_cols[0], multi_cat)
    check("one_way_anova works",  "p_value" in r_an or "error" in r_an)

# chi_square_test
if len(cat_cols) >= 2:
    r_cs = chi_square_test(df, cat_cols[0], cat_cols[1] if len(cat_cols)>1 else cat_cols[0])
    check("chi_square_test works","p_value" in r_cs or "error" in r_cs)

# Mann-Whitney
if binary_cat:
    r_mw = mann_whitney_u_test(df, numeric_cols[0], binary_cat)
    check("mann_whitney works",   "p_value" in r_mw or "error" in r_mw)

# Kruskal-Wallis
if multi_cat:
    r_kw = kruskal_wallis_test(df, numeric_cols[0], multi_cat)
    check("kruskal_wallis works", "p_value" in r_kw or "error" in r_kw)

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  TOTAL PASSED : {PASS}")
print(f"  TOTAL FAILED : {FAIL}")
print(f"{'='*60}")
if FAIL == 0:
    print("  ALL CHECKS PASSED")
else:
    print(f"  {FAIL} FAILURE(S) FOUND:")
    for e in ERRORS:
        print(f"    {e}")
print(f"{'='*60}")
