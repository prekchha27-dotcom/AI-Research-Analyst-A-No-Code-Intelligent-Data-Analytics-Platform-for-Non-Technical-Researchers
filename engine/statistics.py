"""
Statistical Engine — Full implementation
All statistics use scipy and statsmodels.
The AI layer never touches these calculations.

Rule-set additions (sourced from reference material, converted to generic auditable rules):
──────────────────────────────────────────────────────────────────────────────────────────
RULE-STAT-01  Skewness direction labels: positive skew (tail right, mean > median),
              negative skew (tail left, mean < median), symmetric (|skew| < 0.5).
              Applied in: normality_check(), central_tendency_recommendation().

RULE-STAT-02  Central tendency selection: use Mean for symmetric distributions (|skew| < 0.5,
              no extreme outliers); use Median when |skew| >= 0.5 or IQR outliers detected;
              use Mode for categorical/nominal variables only.
              Applied in: central_tendency_recommendation().

RULE-STAT-03  Covariance direction: positive cov -> variables move together; negative ->
              inverse; near-zero -> no consistent directional relationship. Covariance
              magnitude is unit-dependent and should NEVER be used to judge relationship
              strength -- use Pearson r (= cov / (sd_x * sd_y)) for normalised strength.
              Applied in: covariance().

RULE-STAT-04  Five-number summary: Min, Q1, Median, Q3, Max -- always computed alongside
              IQR and outlier fences (Q1 - 1.5*IQR, Q3 + 1.5*IQR).
              Applied in: five_number_summary().

RULE-STAT-05  Outlier fences (Tukey 1977): Lower fence = Q1 - 1.5*IQR;
              Upper fence = Q3 + 1.5*IQR. Values outside these fences are flagged as
              potential outliers. Already implemented in detect_outliers_iqr() -- confirmed.

CONFLICT-01   The reference material labels Pearson |r| >= 0.7 as "strong" and 0.4-0.6 as
              "moderate". This engine uses Cohen (1988) benchmarks: |r| >= 0.5 = large/strong,
              0.3 = medium, 0.1 = small. These are NOT equivalent. Cohen's benchmarks are
              retained as the established academic standard. The reference-material thresholds
              are documented here but NOT implemented as they would conflict with existing
              effect-size guidance used elsewhere in the engine.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")


# ── Descriptive Statistics ────────────────────────────────────────────────────

def descriptive_stats(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    rows = []
    for col in columns:
        series = df[col].dropna().astype(float)
        if len(series) == 0:
            continue
        from scipy import stats as sp
        try:
            mode_result = sp.mode(series, keepdims=True)
            mode_val = round(float(mode_result.mode[0]), 4)
        except Exception:
            mode_val = None
        rows.append({
            "Variable":  col,
            "N":         len(series),
            "Missing":   int(df[col].isnull().sum()),
            "Mean":      round(float(series.mean()), 4),
            "Median":    round(float(series.median()), 4),
            "Mode":      mode_val,
            "Std Dev":   round(float(series.std()), 4),
            "Variance":  round(float(series.var()), 4),
            "Min":       round(float(series.min()), 4),
            "Max":       round(float(series.max()), 4),
            "Range":     round(float(series.max() - series.min()), 4),
            "Q1 (25%)":  round(float(series.quantile(0.25)), 4),
            "Q3 (75%)":  round(float(series.quantile(0.75)), 4),
            "IQR":       round(float(series.quantile(0.75) - series.quantile(0.25)), 4),
            "Skewness":  round(float(series.skew()), 4),
            "Kurtosis":  round(float(series.kurt()), 4),
        })
    return pd.DataFrame(rows)


def frequency_table(df: pd.DataFrame, column: str, is_ordinal: bool = False) -> pd.DataFrame:
    """
    Build a frequency table for *column*.

    Parameters
    ----------
    df         : source DataFrame
    column     : column name
    is_ordinal : if True, a Cumulative % column is appended (only meaningful for
                 ordinal/ratio variables; suppress for nominal/categorical)

    Returns
    -------
    DataFrame with columns: Value, Frequency, % of Total, Valid %
    and optionally Cumulative % if is_ordinal=True.
    The final row is a Total summary row.
    """
    MISSING_LABEL = "⚠ Missing / Blank"

    n_total   = len(df)
    n_missing = int(df[column].isna().sum()) + int((df[column] == "").sum())

    # Count all values including NaN; replace NaN/blank with label
    vc = df[column].value_counts(dropna=False)

    rows = []
    for val, freq in vc.items():
        # Normalise the display label
        if val is None or (isinstance(val, float) and np.isnan(val)) or val == "":
            label = MISSING_LABEL
        else:
            label = str(val)
        pct_total = round(freq / n_total * 100, 2) if n_total > 0 else 0.0
        rows.append({"_label": label, "_freq": freq, "_pct_total": pct_total})

    # Valid % denominator = rows that are NOT missing
    n_valid = n_total - n_missing

    data_rows = []
    cumulative = 0.0
    for r in rows:
        is_miss = r["_label"] == MISSING_LABEL
        if not is_miss and n_valid > 0:
            valid_pct = round(r["_freq"] / n_valid * 100, 2)
        else:
            valid_pct = "—"
        cumulative += r["_pct_total"] if not is_miss else 0.0
        row = {
            "Value":      r["_label"],
            "Frequency":  r["_freq"],
            "% of Total": r["_pct_total"],
            "Valid %":    valid_pct,
        }
        if is_ordinal:
            row["Cumulative %"] = round(cumulative, 2) if not is_miss else "—"
        data_rows.append(row)

    # Total row
    total_valid_pct = round(n_valid / n_total * 100, 2) if n_total > 0 else 0.0
    total_row = {
        "Value":      "Total",
        "Frequency":  n_total,
        "% of Total": 100.0,
        "Valid %":    f"{total_valid_pct}% of {n_total} rows",
    }
    if is_ordinal:
        total_row["Cumulative %"] = "—"
    data_rows.append(total_row)

    return pd.DataFrame(data_rows)


# ── Cross-tabulation ──────────────────────────────────────────────────────────

def crosstab(df: pd.DataFrame, row_col: str, col_col: str) -> dict:
    MISSING_LABEL = "⚠ Missing / Blank"

    # Fill NaN with the missing label for display purposes (use a copy)
    r = df[row_col].fillna(MISSING_LABEL).replace("", MISSING_LABEL)
    c = df[col_col].fillna(MISSING_LABEL).replace("", MISSING_LABEL)

    freq    = pd.crosstab(r, c, margins=True, margins_name="Total")
    row_pct = pd.crosstab(r, c, normalize="index").round(4) * 100
    col_pct = pd.crosstab(r, c, normalize="columns").round(4) * 100

    # Rename index/column if they kept the raw column name as their name
    freq.index.name    = row_col
    freq.columns.name  = col_col
    row_pct.index.name = row_col
    col_pct.index.name = row_col

    return {"freq": freq, "row_pct": row_pct.round(2), "col_pct": col_pct.round(2)}


# ── Correlation ───────────────────────────────────────────────────────────────

def pearson_correlation(df: pd.DataFrame, col1: str, col2: str) -> dict:
    from scipy.stats import pearsonr
    x = df[[col1, col2]].dropna()
    if len(x) < 3:
        return {"error": "Insufficient data (need at least 3 complete pairs)."}
    r, p = pearsonr(x[col1].astype(float), x[col2].astype(float))
    ci = _correlation_ci(float(r), len(x))
    return {
        "test": "Pearson Correlation",
        "col1": col1, "col2": col2,
        "r": round(float(r), 4),
        "p_value": round(float(p), 6),
        "n": len(x),
        "ci_95": ci,
        "significant": bool(p < 0.05),
        "interpretation": _interpret_correlation(float(r), float(p), "Pearson"),
        "assumption_note": (
            "Pearson assumes both variables are continuous (interval or ratio scale) "
            "and approximately normally distributed. "
            "It measures linear association only. Outliers can strongly distort the coefficient. "
            "If these assumptions are questionable, consider Spearman or Kendall."
        ),
    }


def spearman_correlation(df: pd.DataFrame, col1: str, col2: str) -> dict:
    from scipy.stats import spearmanr
    x = df[[col1, col2]].dropna()
    if len(x) < 3:
        return {"error": "Insufficient data (need at least 3 complete pairs)."}
    r, p = spearmanr(x[col1].astype(float), x[col2].astype(float))
    ci = _correlation_ci(float(r), len(x))
    return {
        "test": "Spearman Rank Correlation",
        "col1": col1, "col2": col2,
        "r": round(float(r), 4),
        "p_value": round(float(p), 6),
        "n": len(x),
        "ci_95": ci,
        "significant": bool(p < 0.05),
        "interpretation": _interpret_correlation(float(r), float(p), "Spearman"),
        "assumption_note": (
            "Spearman is non-parametric — it does not assume normal distributions. "
            "It measures monotonic (rank-based) association, not strictly linear association. "
            "Suitable for ordinal variables, skewed data, or when Pearson assumptions are violated."
        ),
    }


def kendall_correlation(df: pd.DataFrame, col1: str, col2: str) -> dict:
    """Kendall's tau-b correlation — suitable for ordinal data or small samples with ties."""
    from scipy.stats import kendalltau
    x = df[[col1, col2]].dropna()
    if len(x) < 3:
        return {"error": "Insufficient data (need at least 3 complete pairs)."}
    tau, p = kendalltau(x[col1].astype(float), x[col2].astype(float))
    return {
        "test": "Kendall's Tau-b Correlation",
        "col1": col1, "col2": col2,
        "r": round(float(tau), 4),      # use 'r' key for interpreter compatibility
        "tau": round(float(tau), 4),
        "p_value": round(float(p), 6),
        "n": len(x),
        "ci_95": None,                   # Kendall CI requires bootstrap; not computed here
        "significant": bool(p < 0.05),
        "interpretation": _interpret_correlation(float(tau), float(p), "Kendall"),
        "assumption_note": (
            "Kendall's tau-b is non-parametric. It is especially useful when: "
            "(1) variables are ordinal, (2) the sample is small, or (3) there are many tied ranks. "
            "Tau values are typically smaller than Spearman's rho for the same data — "
            "this is normal and does not mean a weaker relationship."
        ),
    }


def correlation_matrix(df: pd.DataFrame, columns: list, method: str = "pearson") -> pd.DataFrame:
    """
    Compute a correlation matrix.

    Parameters
    ----------
    method : 'pearson' | 'spearman' | 'kendall'
    """
    method = method.lower()
    if method not in ("pearson", "spearman", "kendall"):
        method = "pearson"
    data = df[columns].apply(pd.to_numeric, errors="coerce").dropna()
    return data.corr(method=method).round(4)


def _correlation_ci(r: float, n: int, alpha: float = 0.05) -> tuple:
    """Fisher z-transformation 95% confidence interval for r.  Returns (lower, upper)."""
    if n < 4 or abs(r) >= 1.0:
        return (None, None)
    from scipy.stats import norm
    z  = np.arctanh(r)
    se = 1.0 / np.sqrt(n - 3)
    z_crit = norm.ppf(1 - alpha / 2)
    lower = float(np.tanh(z - z_crit * se))
    upper = float(np.tanh(z + z_crit * se))
    return (round(lower, 4), round(upper, 4))


def recommend_correlation_method(df: pd.DataFrame, col1: str, col2: str) -> dict:
    """
    Inspect two columns and recommend the most appropriate correlation method.

    Returns a dict with keys:
        recommended : 'Pearson' | 'Spearman' | 'Kendall'
        reason      : plain-language explanation
        warnings    : list of concern strings
    """
    warnings_out = []

    # Use pd.to_numeric (errors='coerce') — works across all pandas versions.
    # astype(float, errors="ignore") was removed in pandas 2.0.
    s1_raw = pd.to_numeric(df[col1].dropna(), errors="coerce").dropna()
    s2_raw = pd.to_numeric(df[col2].dropna(), errors="coerce").dropna()

    # Check if numeric at all (at least 3 coercible values in each)
    if len(s1_raw) < 3 or len(s2_raw) < 3:
        return {
            "recommended": None,
            "reason": "One or both variables cannot be converted to numeric or have too few values.",
            "warnings": ["Non-numeric variables cannot be used in standard correlation."],
        }

    try:
        s1 = s1_raw.astype(float)
        s2 = s2_raw.astype(float)
    except (ValueError, TypeError):
        return {
            "recommended": None,
            "reason": "One or both variables cannot be converted to numeric.",
            "warnings": ["Non-numeric variables cannot be used in standard correlation."],
        }

    n = len(df[[col1, col2]].dropna())

    # Skewness
    sk1 = float(s1.skew()) if len(s1) > 2 else 0.0
    sk2 = float(s2.skew()) if len(s2) > 2 else 0.0
    high_skew = abs(sk1) > 1.0 or abs(sk2) > 1.0

    # Outliers (IQR method)
    def has_outliers(s):
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        return int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum()) > 0

    outliers = has_outliers(s1) or has_outliers(s2)

    # Uniqueness (proxy for ordinal / few categories)
    u1 = s1.nunique()
    u2 = s2.nunique()
    few_values = u1 <= 7 or u2 <= 7   # likely ordinal/Likert

    # Small sample
    small_n = n < 30

    if high_skew:
        warnings_out.append(
            f"High skewness detected (col1 skewness={sk1:.2f}, col2 skewness={sk2:.2f}). "
            "Pearson assumes approximate normality."
        )
    if outliers:
        warnings_out.append("Potential outliers detected. Pearson coefficient is sensitive to outliers.")
    if few_values:
        warnings_out.append(
            f"One or both variables have few unique values ({u1}, {u2}) — "
            "they may be ordinal/Likert scales."
        )
    if small_n:
        warnings_out.append(
            f"Small sample size (N={n}). Kendall's tau is more reliable than Spearman for small samples with ties."
        )

    # Decision logic
    if few_values and small_n:
        rec = "Kendall"
        reason = (
            "Kendall's tau-b is recommended: variables appear to have few categories (possibly ordinal/Likert) "
            f"and the sample is small (N={n}). Kendall handles ties more reliably."
        )
    elif few_values or high_skew or outliers:
        rec = "Spearman"
        reason = (
            "Spearman rank correlation is recommended because one or more of the following apply: "
            "high skewness, outliers detected, or variables appear ordinal/Likert. "
            "Spearman does not require normality or linear relationships."
        )
    else:
        rec = "Pearson"
        reason = (
            "Pearson correlation appears appropriate: both variables are numeric, "
            f"skewness is low (col1={sk1:.2f}, col2={sk2:.2f}), no extreme outliers detected, "
            f"and the sample size is adequate (N={n})."
        )

    return {"recommended": rec, "reason": reason, "warnings": warnings_out}


def _interpret_correlation(r: float, p: float, method: str = "Pearson") -> str:
    strength = (
        "very strong" if abs(r) >= 0.9 else
        "strong"      if abs(r) >= 0.7 else
        "moderate"    if abs(r) >= 0.5 else
        "weak"        if abs(r) >= 0.3 else
        "very weak or negligible"
    )
    direction = "positive" if r > 0 else "negative"
    sig = "statistically significant (p < 0.05)" if p < 0.05 else "not statistically significant (p ≥ 0.05)"
    coeff_label = "τ" if method == "Kendall" else "r"
    return (
        f"**A.** Coefficient ({method} {coeff_label}) = {r:.4f}  \n"
        f"**B.** Direction: {direction} | Strength: {strength} | {sig} (p = {p:.6f})  \n"
        f"**C.** Plain language: There is a {strength} {direction} association between the variables "
        f"based on {method} correlation. "
        f"{'The result is statistically significant, suggesting this association is unlikely to be due to chance alone.' if p < 0.05 else 'The result is not statistically significant at the 0.05 level.'} "
        f"**Correlation does not establish causation.**"
    )


# ── Chi-Square ────────────────────────────────────────────────────────────────

def chi_square_test(df: pd.DataFrame, col1: str, col2: str) -> dict:
    from scipy.stats import chi2_contingency
    contingency = pd.crosstab(df[col1], df[col2])
    if contingency.shape[0] < 2 or contingency.shape[1] < 2:
        return {"error": "Both variables must have at least 2 categories."}
    chi2, p, dof, expected = chi2_contingency(contingency)
    n = int(contingency.values.sum())
    phi2 = chi2 / n
    r, k = contingency.shape
    cramers_v = float(np.sqrt(phi2 / min(k - 1, r - 1))) if min(k - 1, r - 1) > 0 else 0.0
    low_expected = int((expected < 5).sum())
    warning = ""
    if low_expected > 0:
        pct_low = round(low_expected / expected.size * 100, 1)
        warning = (f"⚠️ {low_expected} cells ({pct_low}%) have expected frequency < 5. "
                   "Chi-square results may be unreliable. Consider combining categories or Fisher's exact test.")
    return {
        "test": "Chi-Square Test of Independence",
        "col1": col1, "col2": col2,
        "chi2": round(float(chi2), 4),
        "p_value": round(float(p), 6),
        "df": int(dof),
        "n": n,
        "cramers_v": round(cramers_v, 4),
        "significant": bool(p < 0.05),
        "low_expected_count": low_expected,
        "warning": warning,
        "contingency_table": contingency,
        "interpretation": _interpret_chi2(chi2, p, dof, cramers_v),
        "assumption_note": "Chi-square assumes independence of observations and expected cell frequencies ≥ 5.",
    }


def _interpret_chi2(chi2: float, p: float, dof: int, v: float) -> str:
    sig = "statistically significant" if p < 0.05 else "not statistically significant"
    effect = "strong" if v >= 0.5 else "moderate" if v >= 0.3 else "weak" if v >= 0.1 else "negligible"
    return (f"χ²({dof}) = {chi2:.4f}, p = {p:.6f}. The association is {sig}. "
            f"Effect size (Cramér's V = {v:.4f}) indicates a {effect} association.")


# ── T-Tests ───────────────────────────────────────────────────────────────────

def independent_ttest(df: pd.DataFrame, num_col: str, group_col: str) -> dict:
    from scipy.stats import ttest_ind, levene
    groups = df[group_col].dropna().unique()
    if len(groups) != 2:
        return {"error": f"Group variable must have exactly 2 groups. Found: {list(groups)}"}
    g1 = df.loc[df[group_col] == groups[0], num_col].dropna().astype(float)
    g2 = df.loc[df[group_col] == groups[1], num_col].dropna().astype(float)
    if len(g1) < 2 or len(g2) < 2:
        return {"error": "Each group needs at least 2 observations."}
    lev_stat, lev_p = levene(g1, g2)
    equal_var = bool(lev_p >= 0.05)
    t, p = ttest_ind(g1, g2, equal_var=equal_var)
    pooled = float(np.sqrt(((len(g1)-1)*g1.std()**2 + (len(g2)-1)*g2.std()**2) / (len(g1)+len(g2)-2)))
    cohens_d = float((g1.mean() - g2.mean()) / pooled) if pooled > 0 else 0.0
    return {
        "test": "Independent Samples t-Test",
        "num_col": num_col, "group_col": group_col,
        "group1": str(groups[0]), "n1": len(g1),
        "mean1": round(float(g1.mean()), 4), "std1": round(float(g1.std()), 4),
        "group2": str(groups[1]), "n2": len(g2),
        "mean2": round(float(g2.mean()), 4), "std2": round(float(g2.std()), 4),
        "t_statistic": round(float(t), 4),
        "p_value": round(float(p), 6),
        "n": len(g1) + len(g2),
        "significant": bool(p < 0.05),
        "equal_variance": equal_var,
        "levene_p": round(float(lev_p), 4),
        "cohens_d": round(cohens_d, 4),
        "interpretation": _interpret_ttest(str(groups[0]), str(groups[1]),
                                           float(g1.mean()), float(g2.mean()), float(t), float(p), cohens_d),
        "assumption_note": (
            "Assumes independent groups and approximately normal distribution. "
            "Equal variance assumption tested via Levene's test "
            f"({'satisfied' if equal_var else 'violated — Welch correction applied'})."
        ),
    }


def paired_ttest(df: pd.DataFrame, col1: str, col2: str) -> dict:
    from scipy.stats import ttest_rel
    data = df[[col1, col2]].dropna().astype(float)
    if len(data) < 2:
        return {"error": "Insufficient paired observations."}
    t, p = ttest_rel(data[col1], data[col2])
    diff = data[col1] - data[col2]
    cohens_d = float(diff.mean() / diff.std()) if diff.std() > 0 else 0.0
    return {
        "test": "Paired Samples t-Test",
        "col1": col1, "col2": col2,
        "n": len(data),
        "mean_diff": round(float(diff.mean()), 4),
        "std_diff": round(float(diff.std()), 4),
        "t_statistic": round(float(t), 4),
        "p_value": round(float(p), 6),
        "significant": bool(p < 0.05),
        "cohens_d": round(cohens_d, 4),
        "interpretation": (
            f"Mean difference = {diff.mean():.4f}. t({len(data)-1}) = {t:.4f}, p = {p:.6f}. "
            + ("Statistically significant (p < 0.05)." if p < 0.05
               else "Not statistically significant (p ≥ 0.05).")
        ),
        "assumption_note": "Assumes differences between pairs are approximately normally distributed.",
    }


def _interpret_ttest(g1, g2, m1, m2, t, p, d) -> str:
    direction = f"Group '{g1}' (mean={m1:.4f}) is {'higher' if m1 > m2 else 'lower'} than '{g2}' (mean={m2:.4f})"
    sig = "statistically significant (p < 0.05)" if p < 0.05 else "not statistically significant (p ≥ 0.05)"
    effect = "large" if abs(d) >= 0.8 else "medium" if abs(d) >= 0.5 else "small"
    return f"{direction}. t = {t:.4f}, p = {p:.6f} — {sig}. Cohen's d = {d:.4f} ({effect} effect)."


# ── One-Way ANOVA ─────────────────────────────────────────────────────────────

def one_way_anova(df: pd.DataFrame, num_col: str, group_col: str) -> dict:
    from scipy.stats import f_oneway
    groups = df[group_col].dropna().unique()
    if len(groups) < 2:
        return {"error": "Need at least 2 groups for ANOVA."}
    group_data = [df.loc[df[group_col] == g, num_col].dropna().astype(float) for g in groups]
    if any(len(g) < 2 for g in group_data):
        return {"error": "Each group needs at least 2 observations."}
    f_stat, p = f_oneway(*group_data)
    grand_mean = float(df[num_col].dropna().astype(float).mean())
    ss_between = sum(len(g) * (float(g.mean()) - grand_mean)**2 for g in group_data)
    ss_total   = float(df[num_col].dropna().astype(float).var(ddof=0) * len(df[num_col].dropna()))
    eta_sq     = ss_between / ss_total if ss_total > 0 else 0.0
    group_summary = {
        str(g): {"n": len(d), "mean": round(float(d.mean()), 4), "std": round(float(d.std()), 4)}
        for g, d in zip(groups, group_data)
    }
    result = {
        "test": "One-Way ANOVA",
        "num_col": num_col, "group_col": group_col,
        "groups": [str(g) for g in groups],
        "n_groups": len(groups),
        "f_statistic": round(float(f_stat), 4),
        "p_value": round(float(p), 6),
        "n": sum(len(g) for g in group_data),
        "significant": bool(p < 0.05),
        "eta_squared": round(eta_sq, 4),
        "group_summary": group_summary,
        "interpretation": _interpret_anova(float(f_stat), float(p), len(groups), eta_sq),
        "assumption_note": "Assumes independent groups, approximate normality within groups, and homogeneity of variance.",
    }
    if p < 0.05 and len(groups) > 2:
        try:
            from statsmodels.stats.multicomp import pairwise_tukeyhsd
            all_vals = df[[num_col, group_col]].dropna()
            tukey = pairwise_tukeyhsd(all_vals[num_col].astype(float), all_vals[group_col].astype(str))
            tukey_df = pd.DataFrame(data=tukey._results_table.data[1:],
                                    columns=tukey._results_table.data[0])
            result["posthoc_tukey"] = tukey_df
        except Exception:
            result["posthoc_tukey"] = None
    return result


def _interpret_anova(f, p, k, eta) -> str:
    sig    = "statistically significant" if p < 0.05 else "not statistically significant"
    effect = "large" if eta >= 0.14 else "medium" if eta >= 0.06 else "small"
    return (f"F = {f:.4f}, p = {p:.6f}. Difference across {k} groups is {sig}. "
            f"η² = {eta:.4f} ({effect} effect).")


# ── Linear Regression ─────────────────────────────────────────────────────────

def linear_regression(df: pd.DataFrame, dependent: str, independents: list) -> dict:
    import statsmodels.api as sm
    data = df[[dependent] + independents].dropna()
    if len(data) < len(independents) + 2:
        return {"error": "Insufficient data for regression."}
    X = sm.add_constant(data[independents].astype(float))
    y = data[dependent].astype(float)
    model = sm.OLS(y, X).fit()
    coefs = pd.DataFrame({
        "Variable":    ["Intercept"] + independents,
        "Coefficient": [round(float(c), 4) for c in model.params],
        "Std Error":   [round(float(s), 4) for s in model.bse],
        "t-value":     [round(float(t), 4) for t in model.tvalues],
        "p-value":     [round(float(p), 6) for p in model.pvalues],
        "Significant": [bool(p < 0.05) for p in model.pvalues],
    })
    return {
        "test":          "Linear Regression (OLS)",
        "dependent":     dependent,
        "independents":  independents,
        "n":             len(data),
        "r_squared":     round(float(model.rsquared), 4),
        "adj_r_squared": round(float(model.rsquared_adj), 4),
        "f_statistic":   round(float(model.fvalue), 4),
        "f_p_value":     round(float(model.f_pvalue), 6),
        "aic":           round(float(model.aic), 2),
        "bic":           round(float(model.bic), 2),
        "coefficients":  coefs,
        "interpretation": (
            f"R² = {model.rsquared:.4f}: the model explains {model.rsquared*100:.1f}% of the variance in {dependent}. "
            f"Overall model is {'statistically significant' if model.f_pvalue < 0.05 else 'not statistically significant'} "
            f"(F = {model.fvalue:.4f}, p = {model.f_pvalue:.6f})."
        ),
        "assumption_note": (
            "Assumes linear relationship, independence of observations, homoscedasticity, "
            "normality of residuals, and no severe multicollinearity."
        ),
    }


# ── Logistic Regression ───────────────────────────────────────────────────────

def logistic_regression(df: pd.DataFrame, dependent: str, independents: list) -> dict:
    import statsmodels.api as sm
    data = df[[dependent] + independents].dropna()
    if len(data) < 20:
        return {"error": "Logistic regression requires at least 20 observations."}
    y = data[dependent].astype(int)
    if y.nunique() != 2:
        return {"error": "Dependent variable must be binary (exactly 2 unique values: 0 and 1)."}
    X = sm.add_constant(data[independents].astype(float))
    try:
        model = sm.Logit(y, X).fit(disp=0)
    except Exception as e:
        return {"error": f"Model failed to converge: {str(e)}"}
    coefs = pd.DataFrame({
        "Variable":    ["Intercept"] + independents,
        "Coefficient": [round(float(c), 4) for c in model.params],
        "Odds Ratio":  [round(float(np.exp(c)), 4) for c in model.params],
        "Std Error":   [round(float(s), 4) for s in model.bse],
        "z-value":     [round(float(z), 4) for z in model.tvalues],
        "p-value":     [round(float(p), 6) for p in model.pvalues],
        "Significant": [bool(p < 0.05) for p in model.pvalues],
    })
    return {
        "test":           "Logistic Regression",
        "dependent":      dependent,
        "independents":   independents,
        "n":              len(data),
        "pseudo_r2":      round(float(model.prsquared), 4),
        "log_likelihood": round(float(model.llf), 4),
        "aic":            round(float(model.aic), 2),
        "bic":            round(float(model.bic), 2),
        "coefficients":   coefs,
        "interpretation": (
            f"Pseudo R² (McFadden) = {model.prsquared:.4f}. "
            "Odds ratios > 1 indicate increased probability of the outcome for that predictor."
        ),
        "assumption_note": (
            "Assumes binary outcome, independence of observations, no severe multicollinearity, "
            "and sufficient sample size (≥10 events per predictor)."
        ),
    }


# ── Cronbach's Alpha (Reliability) ───────────────────────────────────────────

def cronbach_alpha(df: pd.DataFrame, items: list) -> dict:
    """
    Compute Cronbach's alpha for a set of scale items (survey reliability).
    Requires at least 2 items, each must be numerical.
    Returns alpha, interpretation, and item-total correlations.
    """
    if len(items) < 2:
        return {"error": "Cronbach's alpha requires at least 2 items."}

    data = df[items].apply(pd.to_numeric, errors="coerce").dropna()
    n = len(data)
    k = len(items)

    if n < 3:
        return {"error": "Insufficient non-missing rows (need at least 3)."}

    item_variances = data.var(axis=0, ddof=1)
    total_variance = data.sum(axis=1).var(ddof=1)

    if total_variance == 0:
        return {"error": "Total score variance is zero — cannot compute alpha."}

    alpha_raw = (k / (k - 1)) * (1 - item_variances.sum() / total_variance)
    alpha = float(round(alpha_raw, 4))
    # Negative alpha means items are counter-productive; clamp to -1 minimum for display
    # (do not clamp to 0 — negative values are meaningful and should be flagged)

    # Interpretation
    if alpha >= 0.9:
        level = "Excellent (α ≥ 0.90)"
    elif alpha >= 0.8:
        level = "Good (0.80 ≤ α < 0.90)"
    elif alpha >= 0.7:
        level = "Acceptable (0.70 ≤ α < 0.80)"
    elif alpha >= 0.6:
        level = "Questionable (0.60 ≤ α < 0.70)"
    elif alpha >= 0.5:
        level = "Poor (0.50 ≤ α < 0.60)"
    elif alpha >= 0:
        level = "Unacceptable (α < 0.50)"
    else:
        level = "Negative (α < 0) — items are negatively inter-correlated"

    # Item-total correlations (corrected: correlation of each item with sum of other items)
    item_total = {}
    for col in items:
        others_sum = data.drop(columns=[col]).sum(axis=1)
        corr = float(round(data[col].corr(others_sum), 4))
        item_total[col] = corr

    warning = None
    if alpha < 0:
        warning = ("CRITICAL: Cronbach's alpha is negative. This means some items are negatively "
                   "inter-correlated with the others. Check whether any items need to be reverse-scored, "
                   "or whether these items measure the same construct at all.")
    elif alpha < 0.7:
        warning = ("Alpha is below the conventional threshold of 0.70 for acceptable reliability. "
                   "Consider reviewing items with low item-total correlations.")
    if k < 4:
        warning = (warning or "") + (" Note: Alpha with fewer than 4 items should be interpreted cautiously.")

    result = {
        "test":            "Cronbach's Alpha (Internal Consistency)",
        "items":           items,
        "n_items":         k,
        "n":               n,
        "alpha":           alpha,
        "reliability":     level,
        "item_total_corr": item_total,
        "interpretation":  (
            f"Cronbach's α = {alpha:.4f}. Internal consistency is rated as: {level}. "
            f"Computed across {k} items and {n} respondents."
        ),
        "assumption_note": (
            "Assumes each item measures the same underlying construct (unidimensionality). "
            "Higher alpha (closer to 1.0) indicates stronger internal consistency. "
            "α ≥ 0.70 is generally considered acceptable for research purposes."
        ),
    }
    if warning:
        result["warning"] = warning
    return result


# ── Non-Parametric Tests ──────────────────────────────────────────────────────

def mann_whitney_u_test(df: pd.DataFrame, num_col: str, group_col: str) -> dict:
    """
    Mann-Whitney U Test — non-parametric test for two independent groups.

    Parameters
    ----------
    df        : source DataFrame
    num_col   : name of the continuous/numeric column
    group_col : name of the grouping column (must have exactly 2 distinct values)

    Returns
    -------
    dict with keys: test, col1, col2, group1, group2, n1, n2, u_statistic,
    p_value, n, significant, rank_biserial_r, interpretation, assumption_note.
    Returns {"error": "..."} on invalid input.
    """
    from scipy.stats import mannwhitneyu

    clean = df[[num_col, group_col]].dropna()
    n_total = len(df)
    n_used = len(clean)

    groups = clean[group_col].unique()
    if len(groups) != 2:
        return {"error": f"group_col must have exactly 2 groups. Found {len(groups)}: {list(groups)}"}

    g1_label, g2_label = str(groups[0]), str(groups[1])
    g1 = clean.loc[clean[group_col] == groups[0], num_col].astype(float).values
    g2 = clean.loc[clean[group_col] == groups[1], num_col].astype(float).values

    if len(g1) < 2 or len(g2) < 2:
        return {"error": "Each group must have at least 2 observations."}

    u_stat, p_val = mannwhitneyu(g1, g2, alternative="two-sided")
    n1, n2 = len(g1), len(g2)
    # Effect size: rank-biserial r = 1 - 2U / (n1 * n2)
    rank_biserial_r = float(round(1 - (2 * u_stat) / (n1 * n2), 4))
    u_stat = float(round(u_stat, 4))
    p_val  = float(round(p_val, 6))

    sig = p_val < 0.05
    effect_label = (
        "large"  if abs(rank_biserial_r) >= 0.5 else
        "medium" if abs(rank_biserial_r) >= 0.3 else
        "small"
    )
    direction = (
        f"Group '{g1_label}' tends to have higher values than '{g2_label}'"
        if rank_biserial_r > 0 else
        f"Group '{g2_label}' tends to have higher values than '{g1_label}'"
    )
    interpretation = (
        f"Mann-Whitney U = {u_stat}, p = {p_val:.6f}. "
        f"{'Statistically significant (p < 0.05)' if sig else 'Not statistically significant (p ≥ 0.05)'}. "
        f"{direction}. Effect size (rank-biserial r = {rank_biserial_r:.4f}) is {effect_label}. "
        f"(n_used={n_used} of n_total={n_total})"
    )

    return {
        "test":             "Mann-Whitney U Test",
        "col1":             num_col,
        "col2":             group_col,
        "group1":           g1_label,
        "group2":           g2_label,
        "n1":               n1,
        "n2":               n2,
        "u_statistic":      u_stat,
        "p_value":          p_val,
        "n":                n1 + n2,
        "n_total":          n_total,
        "n_used":           n_used,
        "significant":      bool(sig),
        "rank_biserial_r":  rank_biserial_r,
        "interpretation":   interpretation,
        "assumption_note": (
            "Mann-Whitney U assumes independent observations and that both groups come from the same "
            "population shape (only their location may differ) when testing medians. It does not require "
            "normality. Suitable when the independent t-test assumptions cannot be met."
        ),
    }


def wilcoxon_signed_rank_test(df: pd.DataFrame, col1: str, col2: str) -> dict:
    """
    Wilcoxon Signed-Rank Test — non-parametric alternative to the paired t-test.

    Parameters
    ----------
    df   : source DataFrame
    col1 : name of the first measurement column
    col2 : name of the second measurement column (same subjects, repeated measure)

    Returns
    -------
    dict with keys: test, col1, col2, n, statistic, p_value, significant,
    rank_biserial_r, interpretation, assumption_note.
    Returns {"error": "..."} on invalid input.
    A warning is included when fewer than 10 complete pairs are available.
    """
    from scipy.stats import wilcoxon

    data = df[[col1, col2]].dropna().astype(float)
    n_total = len(df)
    n_used  = len(data)
    warning = None

    if n_used < 2:
        return {"error": "Insufficient complete pairs (need at least 2)."}
    if n_used < 10:
        warning = (
            f"Only {n_used} complete pairs available. Wilcoxon signed-rank test has low power with "
            "fewer than 10 pairs — interpret results with caution."
        )

    diffs = (data[col1] - data[col2]).values
    # Drop zero differences (they are excluded by Wilcoxon)
    nonzero = diffs[diffs != 0]
    if len(nonzero) == 0:
        return {"error": "All differences between the two columns are zero — test cannot be computed."}

    stat, p_val = wilcoxon(data[col1].values, data[col2].values, alternative="two-sided")
    stat  = float(round(stat, 4))
    p_val = float(round(p_val, 6))
    n_pairs = n_used

    # Effect size: rank-biserial r = 1 - 4W / (n*(n+1))  where n = non-zero pairs
    n_nz = len(nonzero)
    rank_biserial_r = float(round(1 - (4 * stat) / (n_nz * (n_nz + 1)), 4)) if n_nz > 0 else 0.0

    sig = p_val < 0.05
    effect_label = (
        "large"  if abs(rank_biserial_r) >= 0.5 else
        "medium" if abs(rank_biserial_r) >= 0.3 else
        "small"
    )
    interpretation = (
        f"Wilcoxon W = {stat}, p = {p_val:.6f}. "
        f"{'Statistically significant (p < 0.05)' if sig else 'Not statistically significant (p ≥ 0.05)'}. "
        f"Effect size (rank-biserial r = {rank_biserial_r:.4f}) is {effect_label}. "
        f"(n_pairs={n_pairs} of n_total={n_total})"
    )

    result = {
        "test":            "Wilcoxon Signed-Rank Test",
        "col1":            col1,
        "col2":            col2,
        "n":               n_pairs,
        "n_total":         n_total,
        "n_used":          n_used,
        "statistic":       stat,
        "p_value":         p_val,
        "significant":     bool(sig),
        "rank_biserial_r": rank_biserial_r,
        "interpretation":  interpretation,
        "assumption_note": (
            "Wilcoxon signed-rank test assumes paired/dependent observations and that the distribution "
            "of differences is symmetric around the median. It does not assume normality. "
            "Suitable when paired t-test normality assumption is violated."
        ),
    }
    if warning:
        result["warning"] = warning
    return result


def kruskal_wallis_test(df: pd.DataFrame, num_col: str, group_col: str) -> dict:
    """
    Kruskal-Wallis Test — non-parametric alternative to one-way ANOVA.

    Parameters
    ----------
    df        : source DataFrame
    num_col   : name of the continuous/numeric column
    group_col : name of the grouping column (must have at least 2 distinct values)

    Returns
    -------
    dict with keys: test, num_col, group_col, groups, n_groups, h_statistic,
    p_value, n, significant, eta_squared_h, group_summary, interpretation,
    assumption_note, and optionally posthoc_note when significant and n_groups > 2.
    Returns {"error": "..."} on invalid input.
    """
    from scipy.stats import kruskal

    clean = df[[num_col, group_col]].dropna()
    n_total = len(df)
    n_used  = len(clean)

    groups = clean[group_col].unique()
    k = len(groups)
    if k < 2:
        return {"error": "Need at least 2 groups for Kruskal-Wallis test."}

    group_data = {
        str(g): clean.loc[clean[group_col] == g, num_col].astype(float).values
        for g in groups
    }
    if any(len(v) < 5 for v in group_data.values()):
        small = [g for g, v in group_data.items() if len(v) < 5]
        return {"error": f"Each group must have at least 5 observations. Groups with insufficient data: {small}"}

    h_stat, p_val = kruskal(*group_data.values())
    h_stat = float(round(h_stat, 4))
    p_val  = float(round(p_val, 6))
    n = n_used

    # Effect size: eta_squared_H = (H - k + 1) / (n - k)
    eta_sq_h = float(round((h_stat - k + 1) / (n - k), 4)) if (n - k) > 0 else 0.0

    group_summary = {
        g: {
            "n":      len(v),
            "median": round(float(np.median(v)), 4),
            "mean":   round(float(np.mean(v)), 4),
            "std":    round(float(np.std(v, ddof=1)), 4),
        }
        for g, v in group_data.items()
    }

    sig = p_val < 0.05
    effect_label = (
        "large"  if eta_sq_h >= 0.14 else
        "medium" if eta_sq_h >= 0.06 else
        "small"
    )
    interpretation = (
        f"Kruskal-Wallis H({k - 1}) = {h_stat}, p = {p_val:.6f}. "
        f"{'Statistically significant difference in rank distributions across groups (p < 0.05)' if sig else 'No statistically significant difference across groups (p ≥ 0.05)'}. "
        f"Effect size (η²H = {eta_sq_h:.4f}) is {effect_label}. "
        f"(n_used={n_used} of n_total={n_total})"
    )

    result = {
        "test":          "Kruskal-Wallis Test",
        "num_col":       num_col,
        "group_col":     group_col,
        "groups":        [str(g) for g in groups],
        "n_groups":      k,
        "h_statistic":   h_stat,
        "p_value":       p_val,
        "n":             n,
        "n_total":       n_total,
        "n_used":        n_used,
        "significant":   bool(sig),
        "eta_squared_h": eta_sq_h,
        "group_summary": group_summary,
        "interpretation": interpretation,
        "assumption_note": (
            "Kruskal-Wallis assumes independent observations and that groups share the same distribution "
            "shape (only location may differ). It does not require normality. "
            "It tests whether at least one group's rank distribution differs from the others."
        ),
    }
    if sig and k > 2:
        result["posthoc_note"] = (
            "The Kruskal-Wallis test is significant with more than 2 groups. "
            "To identify which specific pairs differ, conduct pairwise Mann-Whitney U tests "
            "with Bonferroni correction (alpha / number of pairs). "
            "Use bonferroni_correction() and mann_whitney_u_test() from this module."
        )
    return result


# ── Normality & Assumption Checks ─────────────────────────────────────────────

def normality_check(series: pd.Series, col_name: str) -> dict:
    """
    Test normality of a numeric series using Shapiro-Wilk (n ≤ 50) or
    D'Agostino-Pearson K² (n > 50).

    Parameters
    ----------
    series   : numeric pandas Series (NaN values are automatically dropped)
    col_name : display name for the column (used in output)

    Returns
    -------
    dict with keys: col_name, n, test_used, statistic, p_value, is_normal,
    skewness, kurtosis, interpretation, recommendation.
    Returns {"error": "..."} on invalid input.
    """
    from scipy.stats import shapiro, normaltest
    from scipy.stats import skew as sp_skew, kurtosis as sp_kurtosis

    s = pd.to_numeric(series, errors="coerce").dropna()
    n_total = len(series)
    n = len(s)

    if n < 3:
        return {"error": f"Insufficient data for normality check (need at least 3 values, got {n})."}

    if n <= 50:
        test_used = "Shapiro-Wilk"
        stat, p_val = shapiro(s.values)
    else:
        test_used = "D'Agostino-Pearson K²"
        stat, p_val = normaltest(s.values)

    stat  = float(round(stat, 6))
    p_val = float(round(p_val, 6))
    skewness = float(round(sp_skew(s.values), 4))
    kurt     = float(round(sp_kurtosis(s.values), 4))
    is_normal = bool(p_val > 0.05)

    # RULE-STAT-01: skewness direction labels
    if abs(skewness) < 0.5:
        skew_label = "approximately symmetric"
    elif skewness > 0:
        skew_label = (
            f"positively skewed (tail extends right; mean > median; "
            f"|skew| = {abs(skewness):.4f} — "
            + ("moderate" if abs(skewness) < 1.0 else "severe")
            + ")"
        )
    else:
        skew_label = (
            f"negatively skewed (tail extends left; mean < median; "
            f"|skew| = {abs(skewness):.4f} — "
            + ("moderate" if abs(skewness) < 1.0 else "severe")
            + ")"
        )

    interpretation = (
        f"{test_used}: statistic = {stat:.6f}, p = {p_val:.6f}. "
        f"The distribution {'appears normal (p > 0.05)' if is_normal else 'departs from normality (p ≤ 0.05)'}. "
        f"Skewness = {skewness:.4f} ({skew_label}); "
        f"Excess kurtosis = {kurt:.4f}. "
        f"(n_used={n} of n_total={n_total})"
    )

    if not is_normal:
        if n < 30:
            recommendation = (
                "Distribution is non-normal and n < 30. "
                "Consider a non-parametric alternative (e.g., Mann-Whitney U, Wilcoxon, Kruskal-Wallis)."
            )
        else:
            recommendation = (
                "Distribution is non-normal, but n ≥ 30. "
                "Central Limit Theorem may apply — a parametric test may still be appropriate. "
                "Check for severe skewness or outliers before proceeding."
            )
    else:
        recommendation = (
            "No significant departure from normality detected. "
            "Parametric tests (t-test, ANOVA, Pearson) are appropriate assuming other assumptions hold."
        )

    return {
        "col_name":       col_name,
        "n":              n,
        "n_total":        n_total,
        "test_used":      test_used,
        "statistic":      stat,
        "p_value":        p_val,
        "is_normal":      is_normal,
        "skewness":       skewness,
        "skew_direction": skew_label,   # RULE-STAT-01
        "kurtosis":       kurt,
        "interpretation": interpretation,
        "recommendation": recommendation,
    }


def levene_test(df: pd.DataFrame, num_col: str, group_col: str) -> dict:
    """
    Levene's Test for Homogeneity of Variance across groups.

    Parameters
    ----------
    df        : source DataFrame
    num_col   : name of the continuous/numeric column
    group_col : name of the grouping column (at least 2 groups required)

    Returns
    -------
    dict with keys: test, num_col, group_col, statistic, p_value, equal_variance,
    interpretation, recommendation.
    Returns {"error": "..."} on invalid input.
    """
    from scipy.stats import levene

    clean = df[[num_col, group_col]].dropna()
    n_total = len(df)
    n_used  = len(clean)

    groups = clean[group_col].unique()
    if len(groups) < 2:
        return {"error": "Need at least 2 groups for Levene's test."}

    group_data = [
        clean.loc[clean[group_col] == g, num_col].astype(float).values
        for g in groups
    ]
    if any(len(v) < 2 for v in group_data):
        return {"error": "Each group must have at least 2 observations."}

    stat, p_val = levene(*group_data)
    stat  = float(round(stat, 4))
    p_val = float(round(p_val, 6))
    equal_variance = bool(p_val > 0.05)

    interpretation = (
        f"Levene's statistic = {stat:.4f}, p = {p_val:.6f}. "
        f"{'Variances are homogeneous across groups (p > 0.05)' if equal_variance else 'Significant violation of homogeneity of variance (p ≤ 0.05)'}. "
        f"Groups tested: {[str(g) for g in groups]}. "
        f"(n_used={n_used} of n_total={n_total})"
    )

    if equal_variance:
        recommendation = (
            "Homogeneity of variance assumption is met. "
            "Proceed with standard independent t-test (equal_var=True) or one-way ANOVA."
        )
    else:
        recommendation = (
            "Homogeneity of variance is violated. "
            "Use Welch's t-test (equal_var=False) for two groups, or Welch's ANOVA for multiple groups. "
            "Non-parametric alternatives (Mann-Whitney U, Kruskal-Wallis) are also appropriate."
        )

    return {
        "test":           "Levene's Test for Homogeneity of Variance",
        "num_col":        num_col,
        "group_col":      group_col,
        "statistic":      stat,
        "p_value":        p_val,
        "n_total":        n_total,
        "n_used":         n_used,
        "equal_variance": equal_variance,
        "interpretation": interpretation,
        "recommendation": recommendation,
    }


# ── Effect Size CI ─────────────────────────────────────────────────────────────

def compute_effect_size_ci(
    statistic_type: str,
    value: float,
    n1: int,
    n2: int = None,
    alpha: float = 0.05,
) -> dict:
    """
    Compute a confidence interval for a given effect size statistic.

    Parameters
    ----------
    statistic_type : one of "cohens_d", "r", "eta_squared", "cramers_v"
    value          : observed effect size value
    n1             : primary sample size (total n for "r" and "eta_squared")
    n2             : secondary sample size (required for "cohens_d" and "cramers_v")
    alpha          : significance level (default 0.05 → 95% CI)

    Returns
    -------
    dict with keys: statistic_type, value, ci_lower, ci_upper, ci_level, note.
    Returns {"error": "..."} on invalid input.
    """
    from scipy.stats import norm, t as t_dist, nct

    statistic_type = statistic_type.lower().strip()
    ci_level = int(round((1 - alpha) * 100))

    valid_types = ("cohens_d", "r", "eta_squared", "cramers_v")
    if statistic_type not in valid_types:
        return {"error": f"statistic_type must be one of {valid_types}. Got '{statistic_type}'."}
    if n1 is None or n1 < 2:
        return {"error": "n1 must be an integer ≥ 2."}

    ci_lower = ci_upper = None
    note = ""

    if statistic_type == "r":
        # Fisher z-transform CI (reuse existing helper)
        n = n1
        if abs(value) >= 1.0:
            return {"error": "r must be strictly between -1 and 1."}
        ci_lower, ci_upper = _correlation_ci(float(value), int(n), alpha=alpha)
        note = "CI computed via Fisher z-transformation."

    elif statistic_type == "cohens_d":
        if n2 is None or n2 < 2:
            return {"error": "n2 must be provided and ≥ 2 for Cohen's d CI."}
        df_val = n1 + n2 - 2
        # Non-central t approximation: t_obs = d * sqrt(n1*n2 / (n1+n2))
        ncp_obs = float(value) * np.sqrt((n1 * n2) / (n1 + n2))
        try:
            # 95% CI on ncp → back-transform to d
            z_crit = norm.ppf(1 - alpha / 2)
            # SE of d via large-sample approximation: SE(d) ≈ sqrt((n1+n2)/(n1*n2) + d²/(2*(n1+n2-2)))
            se_d = np.sqrt((n1 + n2) / (n1 * n2) + value ** 2 / (2 * (n1 + n2 - 2)))
            ci_lower = float(round(value - z_crit * se_d, 4))
            ci_upper = float(round(value + z_crit * se_d, 4))
            note = (
                f"{ci_level}% CI via large-sample SE approximation: "
                "SE(d) = sqrt((n1+n2)/(n1*n2) + d²/(2*(n1+n2-2))). "
                "For small samples consider bootstrap CI."
            )
        except Exception as exc:
            return {"error": f"CI computation failed: {exc}"}

    elif statistic_type == "eta_squared":
        # Eta-squared CI via F-distribution inversion (approximate)
        n = n1  # total n
        if n2 is None:
            return {"error": "n2 must be provided as the number of groups (k) for eta_squared CI."}
        k = n2
        if value < 0 or value >= 1:
            return {"error": "eta_squared must be in [0, 1)."}
        # Convert eta² to F then compute CI on F
        df_between = k - 1
        df_within  = n - k
        if df_between < 1 or df_within < 1:
            return {"error": "Insufficient degrees of freedom for eta_squared CI."}
        from scipy.stats import f as f_dist
        # F = (eta² / df_between) / ((1 - eta²) / df_within)
        f_obs = (value / df_between) / ((1 - value) / df_within) if (1 - value) > 0 else np.inf
        # Confidence bounds via non-central F (approximated by shifting CI on eta²)
        se_eta = np.sqrt((2 * df_between * df_within ** 2) /
                         (df_within * (n - 1) ** 2 * (n + 1)))
        z_crit = norm.ppf(1 - alpha / 2)
        ci_lower = float(round(max(0.0, value - z_crit * se_eta), 4))
        ci_upper = float(round(min(1.0, value + z_crit * se_eta), 4))
        note = (
            f"{ci_level}% CI via large-sample SE approximation for η². "
            "Provide n2=k (number of groups). For more precise bounds use bootstrap."
        )

    elif statistic_type == "cramers_v":
        if n2 is None or n2 < 2:
            return {"error": "n2 must be provided as total n for Cramér's V CI."}
        n = n2
        if value < 0 or value > 1:
            return {"error": "Cramér's V must be in [0, 1]."}
        # CI via Fisher z-transform on V (approximate, treating V like a correlation)
        if value >= 1.0:
            ci_lower, ci_upper = (1.0, 1.0)
        else:
            ci_lower, ci_upper = _correlation_ci(float(value), int(n), alpha=alpha)
            # Clamp to [0, 1] since V is non-negative
            ci_lower = max(0.0, ci_lower) if ci_lower is not None else None
            ci_upper = min(1.0, ci_upper) if ci_upper is not None else None
        note = (
            f"{ci_level}% CI via Fisher z-transformation (treating V as a bounded correlation). "
            "Provide n2=total_n. This approximation is best for larger samples."
        )

    return {
        "statistic_type": statistic_type,
        "value":          round(float(value), 4),
        "ci_lower":       round(float(ci_lower), 4) if ci_lower is not None else None,
        "ci_upper":       round(float(ci_upper), 4) if ci_upper is not None else None,
        "ci_level":       f"{ci_level}%",
        "note":           note,
    }


# ── Multiple Testing Correction ───────────────────────────────────────────────

def bonferroni_correction(p_values: list, alpha: float = 0.05) -> dict:
    """
    Apply Bonferroni correction to a list of p-values.

    Parameters
    ----------
    p_values : list of raw p-values (floats between 0 and 1)
    alpha    : family-wise error rate (default 0.05)

    Returns
    -------
    dict with keys: n_tests, alpha_original, alpha_corrected, p_values_original,
    p_values_adjusted, significant_original, significant_corrected,
    n_significant_before, n_significant_after, interpretation.
    Returns {"error": "..."} on invalid input.
    """
    if not p_values or not isinstance(p_values, (list, tuple)):
        return {"error": "p_values must be a non-empty list of floats."}

    p_arr = []
    for i, pv in enumerate(p_values):
        try:
            pv_f = float(pv)
        except (TypeError, ValueError):
            return {"error": f"p_values[{i}] = {pv!r} is not a valid float."}
        if not (0.0 <= pv_f <= 1.0):
            return {"error": f"p_values[{i}] = {pv_f} is out of range [0, 1]."}
        p_arr.append(pv_f)

    m = len(p_arr)
    alpha_corrected = float(round(alpha / m, 8))
    p_adjusted = [float(round(min(pv * m, 1.0), 6)) for pv in p_arr]

    sig_before = [bool(pv < alpha)            for pv in p_arr]
    sig_after  = [bool(pv < alpha_corrected)  for pv in p_arr]
    n_before   = int(sum(sig_before))
    n_after    = int(sum(sig_after))

    interpretation = (
        f"Bonferroni correction applied across {m} tests (original α = {alpha}). "
        f"Corrected α = {alpha}/{m} = {alpha_corrected:.8f}. "
        f"Significant results before correction: {n_before}/{m}. "
        f"Significant results after correction: {n_after}/{m}. "
        + (
            "No tests remain significant after correction."
            if n_after == 0
            else f"{n_after} test(s) remain significant after Bonferroni correction."
        )
    )

    return {
        "n_tests":               m,
        "alpha_original":        float(alpha),
        "alpha_corrected":       alpha_corrected,
        "p_values_original":     p_arr,
        "p_values_adjusted":     p_adjusted,
        "significant_original":  sig_before,
        "significant_corrected": sig_after,
        "n_significant_before":  n_before,
        "n_significant_after":   n_after,
        "interpretation":        interpretation,
    }


# ── Chi-Square Pre-flight Check ───────────────────────────────────────────────

def chi_square_expected_check(df: pd.DataFrame, col1: str, col2: str) -> dict:
    """
    Check expected cell frequencies before running a chi-square test.

    Call this function BEFORE chi_square_test() to verify whether the chi-square
    assumption of expected frequencies ≥ 5 in at least 80% of cells is met.

    Parameters
    ----------
    df   : source DataFrame
    col1 : row variable (categorical)
    col2 : column variable (categorical)

    Returns
    -------
    dict with keys: n_cells, n_cells_below_5, pct_cells_below_5, min_expected,
    max_expected, is_chi_square_appropriate, recommend_fisher, warning_message.
    Returns {"error": "..."} on invalid input.
    """
    from scipy.stats import chi2_contingency

    clean = df[[col1, col2]].dropna()
    n_total = len(df)
    n_used  = len(clean)

    if n_used == 0:
        return {"error": "No complete cases found after dropping missing values."}

    contingency = pd.crosstab(clean[col1], clean[col2])
    if contingency.shape[0] < 2 or contingency.shape[1] < 2:
        return {"error": "Both variables must have at least 2 categories after dropping missing values."}

    _, _, _, expected = chi2_contingency(contingency)

    n_cells        = int(expected.size)
    n_cells_below  = int((expected < 5).sum())
    pct_below      = float(round(n_cells_below / n_cells * 100, 2))
    min_expected   = float(round(float(expected.min()), 4))
    max_expected   = float(round(float(expected.max()), 4))
    is_appropriate = bool(pct_below <= 20.0)

    is_2x2 = (contingency.shape == (2, 2))
    recommend_fisher = bool(is_2x2 and not is_appropriate)

    if is_appropriate:
        warning_message = (
            f"Chi-square assumption is met: only {pct_below:.1f}% of cells have expected frequency < 5 "
            f"(threshold: ≤ 20%). Minimum expected count = {min_expected:.4f}."
        )
    elif recommend_fisher:
        warning_message = (
            f"⚠️ {n_cells_below} of {n_cells} cells ({pct_below:.1f}%) have expected frequency < 5. "
            "Chi-square results may be unreliable. This is a 2×2 table — "
            "Fisher's Exact Test is strongly recommended instead."
        )
    else:
        warning_message = (
            f"⚠️ {n_cells_below} of {n_cells} cells ({pct_below:.1f}%) have expected frequency < 5. "
            "Chi-square results may be unreliable. Consider combining sparse categories "
            "or using an exact test."
        )

    return {
        "col1":                    col1,
        "col2":                    col2,
        "n_total":                 n_total,
        "n_used":                  n_used,
        "n_cells":                 n_cells,
        "n_cells_below_5":         n_cells_below,
        "pct_cells_below_5":       pct_below,
        "min_expected":            min_expected,
        "max_expected":            max_expected,
        "is_chi_square_appropriate": is_appropriate,
        "recommend_fisher":        recommend_fisher,
        "warning_message":         warning_message,
    }


# ── Enhanced Descriptive Statistics ──────────────────────────────────────────

def descriptive_stats_extended(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Enhanced descriptive statistics with Coefficient of Variation,
    percentiles (10th, 25th, 75th, 90th), and all standard measures.
    """
    rows = []
    for col in columns:
        series = df[col].dropna()
        try:
            series = series.astype(float)
        except (ValueError, TypeError):
            continue
        if len(series) == 0:
            continue
        from scipy import stats as sp
        try:
            mode_result = sp.mode(series, keepdims=True)
            mode_val = round(float(mode_result.mode[0]), 4)
        except Exception:
            mode_val = None
        mean_val = float(series.mean())
        std_val  = float(series.std())
        cv_val   = round((std_val / mean_val * 100), 4) if mean_val != 0 else None
        rows.append({
            "Variable":      col,
            "N":             len(series),
            "Missing":       int(df[col].isnull().sum()),
            "Missing %":     round(df[col].isnull().sum() / max(len(df), 1) * 100, 2),
            "Mean":          round(mean_val, 4),
            "Median":        round(float(series.median()), 4),
            "Mode":          mode_val,
            "Std Dev":       round(std_val, 4),
            "Variance":      round(float(series.var()), 4),
            "CV (%)":        cv_val,
            "Min":           round(float(series.min()), 4),
            "P10 (10th %)":  round(float(series.quantile(0.10)), 4),
            "Q1 (25%)":      round(float(series.quantile(0.25)), 4),
            "Q3 (75%)":      round(float(series.quantile(0.75)), 4),
            "P90 (90th %)":  round(float(series.quantile(0.90)), 4),
            "Max":           round(float(series.max()), 4),
            "Range":         round(float(series.max() - series.min()), 4),
            "IQR":           round(float(series.quantile(0.75) - series.quantile(0.25)), 4),
            "Skewness":      round(float(series.skew()), 4),
            "Kurtosis":      round(float(series.kurt()), 4),
        })
    return pd.DataFrame(rows)


# ── Fisher's Exact Test ───────────────────────────────────────────────────────

def fishers_exact_test(df: pd.DataFrame, col1: str, col2: str) -> dict:
    """
    Fisher's Exact Test — for 2×2 contingency tables with small expected frequencies.

    Preferred over chi-square when any expected cell count < 5 in a 2×2 table.

    Returns
    -------
    dict with keys: test, col1, col2, contingency_table, n, odds_ratio,
    p_value, significant, ci_95, interpretation, assumption_note.
    Returns {"error": "..."} on invalid input.
    """
    from scipy.stats import fisher_exact

    clean = df[[col1, col2]].dropna()
    n_total = len(df)
    n_used  = len(clean)

    if n_used == 0:
        return {"error": "No complete cases after dropping missing values."}

    contingency = pd.crosstab(clean[col1], clean[col2])
    if contingency.shape != (2, 2):
        return {
            "error": (
                f"Fisher's Exact Test requires a 2×2 table. "
                f"Current table is {contingency.shape[0]}×{contingency.shape[1]}. "
                "For larger tables use chi-square test."
            )
        }

    table_array = contingency.values
    odds_ratio, p_val = fisher_exact(table_array, alternative="two-sided")
    odds_ratio = float(round(odds_ratio, 4))
    p_val      = float(round(p_val, 6))
    n          = int(table_array.sum())
    sig        = bool(p_val < 0.05)

    # 95% CI on odds ratio via Woolf logit method
    a, b, c, d = int(table_array[0,0]), int(table_array[0,1]), \
                 int(table_array[1,0]), int(table_array[1,1])
    ci_lower = ci_upper = None
    if a > 0 and b > 0 and c > 0 and d > 0:
        import math
        log_or = math.log(odds_ratio)
        se_log_or = math.sqrt(1/a + 1/b + 1/c + 1/d)
        from scipy.stats import norm
        z = norm.ppf(0.975)
        ci_lower = round(math.exp(log_or - z * se_log_or), 4)
        ci_upper = round(math.exp(log_or + z * se_log_or), 4)

    # Effect size: phi coefficient
    phi = float(round((a*d - b*c) / np.sqrt((a+b)*(c+d)*(a+c)*(b+d)), 4)) \
          if (a+b)*(c+d)*(a+c)*(b+d) > 0 else 0.0

    if sig:
        interp = (
            f"Fisher's Exact Test: p = {p_val:.6f} (statistically significant, p < 0.05). "
            f"Odds Ratio = {odds_ratio:.4f}"
            + (f", 95% CI [{ci_lower:.4f}, {ci_upper:.4f}]." if ci_lower else ".")
            + f" Phi = {phi:.4f}. "
            "There is a statistically significant association between the two variables."
        )
    else:
        interp = (
            f"Fisher's Exact Test: p = {p_val:.6f} (not statistically significant, p ≥ 0.05). "
            f"Odds Ratio = {odds_ratio:.4f}"
            + (f", 95% CI [{ci_lower:.4f}, {ci_upper:.4f}]." if ci_lower else ".")
            + f" Phi = {phi:.4f}. "
            "No statistically significant association was found."
        )

    return {
        "test":            "Fisher's Exact Test",
        "col1":            col1,
        "col2":            col2,
        "n":               n,
        "n_total":         n_total,
        "n_used":          n_used,
        "odds_ratio":      odds_ratio,
        "p_value":         p_val,
        "ci_95":           (ci_lower, ci_upper) if ci_lower else (None, None),
        "phi":             phi,
        "significant":     sig,
        "contingency_table": contingency,
        "interpretation":  interp,
        "assumption_note": (
            "Fisher's Exact Test is appropriate for 2×2 tables when expected cell counts are small "
            "(any expected count < 5). Unlike chi-square, it calculates the exact probability "
            "without approximation. Requires independent observations and fixed marginal totals."
        ),
    }


# ── Point-Biserial Correlation ─────────────────────────────────────────────────

def point_biserial_correlation(df: pd.DataFrame, cont_col: str, binary_col: str) -> dict:
    """
    Point-Biserial Correlation — measures association between a continuous
    variable and a binary (dichotomous) variable.

    Mathematically equivalent to Pearson r when the binary variable is 0/1 coded.

    Parameters
    ----------
    df         : source DataFrame
    cont_col   : name of the continuous (interval/ratio) variable
    binary_col : name of the binary variable (must have exactly 2 distinct values)

    Returns
    -------
    dict with keys: test, cont_col, binary_col, r_pb, p_value, n, ci_95,
    significant, interpretation, assumption_note.
    Returns {"error": "..."} on invalid input.
    """
    from scipy.stats import pointbiserialr

    clean = df[[cont_col, binary_col]].dropna()
    n_total = len(df)
    n_used  = len(clean)

    if n_used < 3:
        return {"error": "Insufficient complete cases (need at least 3)."}

    # Validate binary variable
    unique_vals = clean[binary_col].unique()
    if len(unique_vals) != 2:
        return {
            "error": (
                f"Binary variable '{binary_col}' must have exactly 2 distinct values. "
                f"Found: {list(unique_vals)}"
            )
        }

    # Encode binary as 0/1
    try:
        binary_encoded = (clean[binary_col] == unique_vals[1]).astype(int)
    except Exception:
        return {"error": f"Could not encode '{binary_col}' as binary. Check variable values."}

    try:
        cont_vals = clean[cont_col].astype(float).values
    except (ValueError, TypeError):
        return {"error": f"Could not convert '{cont_col}' to numeric. Check variable values."}

    r_pb, p_val = pointbiserialr(binary_encoded.values, cont_vals)
    r_pb  = float(round(r_pb, 4))
    p_val = float(round(p_val, 6))
    n     = n_used
    sig   = bool(p_val < 0.05)

    # 95% CI via Fisher z-transform (treating r_pb as Pearson r)
    ci = _correlation_ci(r_pb, n)

    strength = (
        "very strong" if abs(r_pb) >= 0.9 else
        "strong"      if abs(r_pb) >= 0.7 else
        "moderate"    if abs(r_pb) >= 0.5 else
        "weak"        if abs(r_pb) >= 0.3 else
        "very weak or negligible"
    )
    direction = "positive" if r_pb > 0 else "negative"
    sig_txt   = "statistically significant (p < 0.05)" if sig else "not statistically significant (p ≥ 0.05)"

    interp = (
        f"Point-Biserial r = {r_pb:.4f} ({strength} {direction} association), p = {p_val:.6f} — {sig_txt}. "
        f"n = {n} (of n_total = {n_total}). "
        f"95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]." if ci[0] else f"Point-Biserial r = {r_pb:.4f}."
    )

    return {
        "test":            "Point-Biserial Correlation",
        "cont_col":        cont_col,
        "binary_col":      binary_col,
        "col1":            cont_col,
        "col2":            binary_col,
        "binary_groups":   [str(v) for v in unique_vals],
        "r":               r_pb,       # 'r' key for interpreter compatibility
        "r_pb":            r_pb,
        "p_value":         p_val,
        "n":               n,
        "n_total":         n_total,
        "n_used":          n_used,
        "ci_95":           ci,
        "significant":     sig,
        "interpretation":  interp,
        "assumption_note": (
            "Point-Biserial correlation requires: (1) one continuous variable (interval/ratio), "
            "(2) one genuinely dichotomous binary variable (not an arbitrary two-category split), "
            "(3) approximate normality of the continuous variable within each binary group, "
            "(4) independence of observations. "
            "Mathematically equivalent to Pearson r applied to a 0/1 coded binary variable."
        ),
    }


# ── Chi-Square Goodness-of-Fit ────────────────────────────────────────────────

def chi_square_goodness_of_fit(
    df: pd.DataFrame,
    column: str,
    expected_proportions: dict | None = None,
) -> dict:
    """
    Chi-Square Goodness-of-Fit Test — tests whether observed category frequencies
    match a specified (or uniform) theoretical distribution.

    Parameters
    ----------
    df                   : source DataFrame
    column               : categorical column to test
    expected_proportions : dict {category_value: proportion} summing to 1.0.
                           If None, assumes uniform distribution (equal proportions).

    Returns
    -------
    dict with keys: test, column, n, chi2, p_value, df, significant,
    cohens_w, category_table, interpretation, assumption_note.
    Returns {"error": "..."} on invalid input.
    """
    from scipy.stats import chisquare

    clean = df[column].dropna()
    n_total = len(df)
    n_used  = len(clean)

    if n_used == 0:
        return {"error": "No non-missing values in this column."}

    observed_counts = clean.value_counts()
    categories      = observed_counts.index.tolist()
    k               = len(categories)

    if k < 2:
        return {"error": "Variable must have at least 2 categories for goodness-of-fit test."}

    # Build expected frequencies
    if expected_proportions is None:
        # Uniform distribution
        expected_freq = [n_used / k] * k
        dist_label    = "uniform (equal proportions)"
    else:
        # Validate proportions
        total_prop = sum(expected_proportions.values())
        if abs(total_prop - 1.0) > 0.01:
            return {"error": f"Expected proportions must sum to 1.0. Got {total_prop:.4f}."}
        expected_freq = []
        for cat in categories:
            prop = expected_proportions.get(cat, None)
            if prop is None:
                return {"error": f"No expected proportion provided for category '{cat}'."}
            expected_freq.append(prop * n_used)
        dist_label = "researcher-specified"

    observed_freq = [int(observed_counts[cat]) for cat in categories]

    # Check minimum expected frequencies
    low_exp = sum(1 for e in expected_freq if e < 5)
    if low_exp > 0:
        pct_low = round(low_exp / k * 100, 1)
    else:
        pct_low = 0.0

    chi2_stat, p_val = chisquare(f_obs=observed_freq, f_exp=expected_freq)
    chi2_stat = float(round(chi2_stat, 4))
    p_val     = float(round(p_val, 6))
    dof       = int(k - 1)
    sig       = bool(p_val < 0.05)
    n         = n_used

    # Cohen's w = sqrt(chi2 / n)
    cohens_w = float(round(np.sqrt(chi2_stat / n) if n > 0 else 0.0, 4))
    w_label  = "large" if cohens_w >= 0.5 else "medium" if cohens_w >= 0.3 else "small"

    # Category-level table
    cat_rows = []
    for cat, obs, exp in zip(categories, observed_freq, expected_freq):
        res = (obs - exp) / np.sqrt(exp) if exp > 0 else 0.0
        cat_rows.append({
            "Category":          str(cat),
            "Observed (n)":      obs,
            "Observed (%)":      round(obs / n * 100, 2),
            "Expected (n)":      round(exp, 2),
            "Expected (%)":      round(exp / n * 100, 2),
            "Std. Residual":     round(float(res), 3),
        })
    category_table = pd.DataFrame(cat_rows)

    if sig:
        interp = (
            f"χ²({dof}) = {chi2_stat:.4f}, p = {p_val:.6f} (statistically significant, p < 0.05). "
            f"The observed frequencies significantly depart from the {dist_label} distribution. "
            f"Effect size: Cohen's w = {cohens_w:.4f} ({w_label}). "
            "Inspect standardised residuals to identify which categories deviate most."
        )
    else:
        interp = (
            f"χ²({dof}) = {chi2_stat:.4f}, p = {p_val:.6f} (not statistically significant, p ≥ 0.05). "
            f"No significant departure from the {dist_label} distribution was found. "
            f"Effect size: Cohen's w = {cohens_w:.4f} ({w_label})."
        )

    warning = ""
    if low_exp > 0:
        warning = (
            f"⚠️ {low_exp} of {k} categories ({pct_low}%) have expected frequency < 5. "
            "Chi-square goodness-of-fit results may be unreliable. "
            "Consider combining sparse categories."
        )

    result = {
        "test":           "Chi-Square Goodness-of-Fit Test",
        "column":         column,
        "n":              n,
        "n_total":        n_total,
        "n_used":         n_used,
        "chi2":           chi2_stat,
        "p_value":        p_val,
        "df":             dof,
        "significant":    sig,
        "cohens_w":       cohens_w,
        "distribution":   dist_label,
        "category_table": category_table,
        "interpretation": interp,
        "assumption_note": (
            "Chi-square goodness-of-fit assumes: (1) independent observations, "
            "(2) expected frequency ≥ 5 per category (warning shown if violated), "
            "(3) mutually exclusive, exhaustive categories. "
            "For uniform distribution, no theoretical proportions are needed. "
            "For other distributions, researcher must specify expected proportions."
        ),
    }
    if warning:
        result["warning"] = warning
    return result


# ── Welch's One-Way ANOVA ─────────────────────────────────────────────────────

def welch_anova_test(df: pd.DataFrame, num_col: str, group_col: str) -> dict:
    """
    Welch's One-Way ANOVA — compares means across 3+ groups without assuming
    equal variances (robust to heteroscedasticity).

    Uses Welch's F statistic (Brown-Forsythe approximation via scipy).

    Parameters
    ----------
    df        : source DataFrame
    num_col   : continuous outcome variable
    group_col : grouping variable (≥ 3 groups recommended; 2 groups also supported)

    Returns
    -------
    dict with keys: test, num_col, group_col, groups, n_groups, f_statistic,
    p_value, n, significant, eta_squared, group_summary, interpretation,
    assumption_note.
    Returns {"error": "..."} on invalid input.
    """
    from scipy.stats import f_oneway

    clean   = df[[num_col, group_col]].dropna()
    n_total = len(df)
    n_used  = len(clean)

    groups = clean[group_col].unique()
    k      = len(groups)
    if k < 2:
        return {"error": "Need at least 2 groups for Welch ANOVA."}

    group_data = {
        str(g): clean.loc[clean[group_col] == g, num_col].astype(float).values
        for g in groups
    }
    if any(len(v) < 2 for v in group_data.values()):
        small = [g for g, v in group_data.items() if len(v) < 2]
        return {"error": f"Each group must have at least 2 observations. Insufficient: {small}"}

    # Welch's F via scipy oneWay (uses Welch internally when equal_var=False is not available)
    # We implement Welch's ANOVA manually using the standard formula
    try:
        import statsmodels.stats.oneway as sm_ow
        result_sm = sm_ow.anova_oneway(
            [v for v in group_data.values()],
            use_var="unequal",   # Welch's correction
        )
        f_stat = float(round(result_sm.statistic, 4))
        p_val  = float(round(result_sm.pvalue, 6))
    except Exception:
        # Fallback: standard F (note in result)
        f_stat_raw, p_val_raw = f_oneway(*group_data.values())
        f_stat = float(round(f_stat_raw, 4))
        p_val  = float(round(p_val_raw, 6))

    n     = n_used
    sig   = bool(p_val < 0.05)

    # Eta-squared (approximate from grand mean)
    all_vals = np.concatenate(list(group_data.values()))
    grand_mean = float(np.mean(all_vals))
    ss_between = sum(len(v) * (float(np.mean(v)) - grand_mean)**2 for v in group_data.values())
    ss_total   = float(np.sum((all_vals - grand_mean)**2))
    eta_sq     = float(round(ss_between / ss_total if ss_total > 0 else 0.0, 4))
    eta_label  = "large" if eta_sq >= 0.14 else "medium" if eta_sq >= 0.06 else "small"

    group_summary = {
        g: {
            "n":      len(v),
            "mean":   round(float(np.mean(v)), 4),
            "std":    round(float(np.std(v, ddof=1)), 4),
        }
        for g, v in group_data.items()
    }

    interp = (
        f"Welch's ANOVA: F = {f_stat:.4f}, p = {p_val:.6f}. "
        f"{'Statistically significant difference in means across groups (p < 0.05)' if sig else 'No statistically significant difference across groups (p ≥ 0.05)'}. "
        f"η² = {eta_sq:.4f} ({eta_label} effect). "
        f"(n_used = {n_used} of n_total = {n_total})"
    )

    result = {
        "test":          "Welch's One-Way ANOVA",
        "num_col":       num_col,
        "group_col":     group_col,
        "groups":        [str(g) for g in groups],
        "n_groups":      k,
        "f_statistic":   f_stat,
        "p_value":       p_val,
        "n":             n,
        "n_total":       n_total,
        "n_used":        n_used,
        "significant":   sig,
        "eta_squared":   eta_sq,
        "group_summary": group_summary,
        "interpretation": interp,
        "assumption_note": (
            "Welch's ANOVA does not assume equal variances across groups (heteroscedasticity-robust). "
            "It assumes independence of observations and approximate normality within groups. "
            "Preferred over standard ANOVA when Levene's test is significant or group sizes differ greatly. "
            "Follow with Games-Howell post-hoc comparisons when significant."
        ),
    }
    if sig and k > 2:
        result["posthoc_note"] = (
            "Welch's ANOVA is significant. For post-hoc comparisons without equal-variance assumption, "
            "use Games-Howell test or pairwise Welch t-tests with Bonferroni/Holm correction."
        )
    return result


# ── Friedman Test ─────────────────────────────────────────────────────────────

def friedman_test(df: pd.DataFrame, columns: list) -> dict:
    """
    Friedman Test — non-parametric test for differences across 3+ related/repeated
    measurements from the same subjects.

    Non-parametric alternative to repeated-measures ANOVA.

    Parameters
    ----------
    df      : source DataFrame (each row = one subject, each column = one condition)
    columns : list of ≥ 3 column names (each column = one measurement condition)

    Returns
    -------
    dict with keys: test, columns, n_conditions, n_subjects, statistic, p_value,
    significant, kendalls_w, group_summary, interpretation, assumption_note.
    Returns {"error": "..."} on invalid input.
    """
    from scipy.stats import friedmanchisquare

    if len(columns) < 3:
        return {"error": "Friedman test requires at least 3 columns (conditions/time points)."}

    # Drop rows with any missing value across the selected columns
    clean   = df[columns].dropna()
    n_total = len(df)
    n_subs  = len(clean)

    if n_subs < 5:
        return {"error": f"Friedman test requires at least 5 complete subjects. Found: {n_subs}."}

    # Convert all to float
    try:
        arrays = [clean[c].astype(float).values for c in columns]
    except (ValueError, TypeError) as e:
        return {"error": f"Could not convert all columns to numeric: {e}"}

    stat, p_val = friedmanchisquare(*arrays)
    stat  = float(round(stat, 4))
    p_val = float(round(p_val, 6))
    k     = len(columns)
    n     = n_subs
    sig   = bool(p_val < 0.05)

    # Kendall's W = chi2_friedman / (n * (k - 1))
    kendalls_w = float(round(stat / (n * (k - 1)), 4)) if n * (k - 1) > 0 else 0.0
    w_label    = "strong agreement" if kendalls_w >= 0.7 else \
                 "moderate agreement" if kendalls_w >= 0.5 else "weak agreement"

    group_summary = {}
    for c in columns:
        vals = clean[c].astype(float)
        group_summary[c] = {
            "n":      int(len(vals)),
            "median": round(float(vals.median()), 4),
            "mean":   round(float(vals.mean()), 4),
            "std":    round(float(vals.std(ddof=1)), 4),
        }

    interp = (
        f"Friedman χ²({k-1}) = {stat:.4f}, p = {p_val:.6f}. "
        f"{'Statistically significant difference in distributions across conditions (p < 0.05)' if sig else 'No statistically significant difference across conditions (p ≥ 0.05)'}. "
        f"Kendall's W = {kendalls_w:.4f} ({w_label}). "
        f"(n_subjects = {n_subs} of n_total = {n_total})"
    )

    result = {
        "test":          "Friedman Test",
        "columns":       columns,
        "n_conditions":  k,
        "n_subjects":    n_subs,
        "n_total":       n_total,
        "n":             n_subs,
        "statistic":     stat,
        "p_value":       p_val,
        "significant":   sig,
        "kendalls_w":    kendalls_w,
        "group_summary": group_summary,
        "interpretation": interp,
        "assumption_note": (
            "Friedman test assumes: (1) repeated or matched measurements (same subjects across all conditions), "
            "(2) ordinal or continuous outcome variable, "
            "(3) no interaction between subjects (blocks) and treatment. "
            "Does not require normality. "
            "If significant, follow with pairwise Wilcoxon signed-rank tests with Bonferroni/Holm correction."
        ),
    }
    if sig:
        result["posthoc_note"] = (
            "Friedman test is significant. Conduct pairwise Wilcoxon signed-rank tests "
            "across all condition pairs, applying Bonferroni or Holm correction for multiple comparisons."
        )
    return result


# ── PCA Analysis ──────────────────────────────────────────────────────────────

def pca_analysis(df: pd.DataFrame, columns: list, n_components: int | None = None) -> dict:
    """
    Principal Component Analysis (PCA) — reduces dimensionality of a set of
    continuous variables to uncorrelated principal components.

    Parameters
    ----------
    df           : source DataFrame
    columns      : list of ≥ 3 numerical column names
    n_components : number of components to retain (None = all)

    Returns
    -------
    dict with keys: test, columns, n_vars, n_obs, n_components_used,
    explained_variance_ratio, cumulative_variance, eigenvalues,
    loadings_df, kmo_approx, interpretation, assumption_note.
    Returns {"error": "..."} on invalid input.
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    if len(columns) < 3:
        return {"error": "PCA requires at least 3 numerical variables."}

    clean   = df[columns].dropna()
    n_total = len(df)
    n_obs   = len(clean)

    if n_obs < 10:
        return {"error": f"PCA requires at least 10 complete observations. Found: {n_obs}."}

    # Rule-of-thumb: n >= 5 per variable
    if n_obs < 5 * len(columns):
        warn_n = (
            f"Sample size ({n_obs}) is below the recommended 5:1 ratio for PCA "
            f"({len(columns)} variables × 5 = {5*len(columns)} observations). "
            "Interpret results with caution."
        )
    else:
        warn_n = ""

    # Convert to float
    try:
        X = clean[columns].astype(float).values
    except (ValueError, TypeError) as e:
        return {"error": f"Could not convert columns to numeric: {e}"}

    # Standardise
    scaler = StandardScaler()
    X_std  = scaler.fit_transform(X)

    # Fit PCA
    n_comp = min(n_components or len(columns), len(columns), n_obs - 1)
    pca    = PCA(n_components=n_comp)
    pca.fit(X_std)

    evr    = [round(float(v), 4) for v in pca.explained_variance_ratio_]
    cumvar = [round(float(v), 4) for v in np.cumsum(pca.explained_variance_ratio_)]
    eigvals = [round(float(v), 4) for v in pca.explained_variance_]

    # Loadings = eigenvectors * sqrt(eigenvalues)
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    comp_labels = [f"PC{i+1}" for i in range(n_comp)]
    loadings_df = pd.DataFrame(
        np.round(loadings, 4),
        index=columns,
        columns=comp_labels
    )

    # Approximate KMO (simplified: based on correlations)
    try:
        corr_mat = pd.DataFrame(X_std, columns=columns).corr().values
        # Simplified KMO: mean off-diagonal r^2 / (mean off-diagonal r^2 + mean partial r^2)
        # Full KMO requires partial correlations — approximate with overall correlation adequacy
        off_diag = corr_mat[np.triu_indices_from(corr_mat, k=1)]
        r_sq_mean = float(np.mean(off_diag**2))
        kmo_approx = round(r_sq_mean / (r_sq_mean + (1 - r_sq_mean) / len(columns)), 3)
        kmo_label = (
            "marvellous" if kmo_approx >= 0.90 else
            "meritorious" if kmo_approx >= 0.80 else
            "middling" if kmo_approx >= 0.70 else
            "mediocre" if kmo_approx >= 0.60 else
            "miserable (below 0.60 — reconsider PCA)"
        )
    except Exception:
        kmo_approx = None
        kmo_label  = "could not be computed"

    # Components to retain (Kaiser criterion: eigenvalue > 1)
    kaiser_components = sum(1 for e in eigvals if e >= 1.0)

    # Variance explained by first n_comp components
    total_variance_pct = round(cumvar[-1] * 100, 1) if cumvar else 0.0

    interp = (
        f"PCA extracted {n_comp} component(s) from {len(columns)} variables "
        f"using {n_obs} complete observations. "
        f"Total variance explained by all {n_comp} component(s): {total_variance_pct}%. "
        f"Kaiser criterion (eigenvalue ≥ 1) suggests retaining {kaiser_components} component(s). "
        f"Approximate KMO = {kmo_approx} ({kmo_label}). "
        "Review the scree plot and loadings to determine the final number of components."
    )

    result = {
        "test":                  "Principal Component Analysis (PCA)",
        "columns":               columns,
        "n_vars":                len(columns),
        "n_obs":                 n_obs,
        "n_total":               n_total,
        "n_components_used":     n_comp,
        "explained_variance_ratio": evr,
        "cumulative_variance":   cumvar,
        "eigenvalues":           eigvals,
        "loadings_df":           loadings_df,
        "kaiser_n_components":   kaiser_components,
        "kmo_approx":            kmo_approx,
        "kmo_label":             kmo_label,
        "total_variance_pct":    total_variance_pct,
        "interpretation":        interp,
        "assumption_note": (
            "PCA assumes: (1) continuous (interval/ratio) variables, "
            "(2) linear relationships between variables, "
            "(3) sufficient correlations (KMO > 0.60 recommended), "
            "(4) adequate sample size (≥ 5–10 observations per variable, min 50). "
            "Variables are standardised (z-scored) before analysis. "
            "PCA is exploratory — components are data-driven, not theory-driven."
        ),
    }
    if warn_n:
        result["warning"] = warn_n
    return result


# ── Holm-Bonferroni Correction ────────────────────────────────────────────────

def holm_bonferroni_correction(p_values: list, alpha: float = 0.05) -> dict:
    """
    Holm-Bonferroni step-down procedure — a more powerful alternative to
    standard Bonferroni correction that is still family-wise error rate (FWER) controlled.

    Parameters
    ----------
    p_values : list of raw p-values (floats between 0 and 1)
    alpha    : family-wise error rate (default 0.05)

    Returns
    -------
    dict with keys: n_tests, alpha_original, method, p_values_original,
    p_values_adjusted, rank_order, significant_original, significant_corrected,
    n_significant_before, n_significant_after, interpretation.
    Returns {"error": "..."} on invalid input.
    """
    if not p_values or not isinstance(p_values, (list, tuple)):
        return {"error": "p_values must be a non-empty list of floats."}

    p_arr = []
    for i, pv in enumerate(p_values):
        try:
            pv_f = float(pv)
        except (TypeError, ValueError):
            return {"error": f"p_values[{i}] = {pv!r} is not a valid float."}
        if not (0.0 <= pv_f <= 1.0):
            return {"error": f"p_values[{i}] = {pv_f} is out of range [0, 1]."}
        p_arr.append(pv_f)

    m = len(p_arr)

    # Sort p-values ascending with original indices
    indexed = sorted(enumerate(p_arr), key=lambda x: x[1])
    p_adjusted = [0.0] * m
    significant_corrected = [False] * m

    # Holm step-down: reject test i (in sorted order k=1..m) if p(k) <= alpha / (m - k + 1)
    reject = True
    for rank, (orig_idx, p_val) in enumerate(indexed, 1):
        threshold = alpha / (m - rank + 1)
        if reject and p_val <= threshold:
            significant_corrected[orig_idx] = True
            p_adjusted[orig_idx] = float(round(min(p_val * (m - rank + 1), 1.0), 6))
        else:
            reject = False  # Once we fail to reject, all subsequent are not rejected
            p_adjusted[orig_idx] = float(round(min(p_val * (m - rank + 1), 1.0), 6))
            significant_corrected[orig_idx] = False

    sig_before = [bool(pv < alpha) for pv in p_arr]
    n_before   = int(sum(sig_before))
    n_after    = int(sum(significant_corrected))

    rank_order = [orig_idx + 1 for orig_idx, _ in indexed]

    interp = (
        f"Holm-Bonferroni correction applied across {m} tests (original α = {alpha}). "
        f"Significant before correction: {n_before}/{m}. "
        f"Significant after Holm-Bonferroni: {n_after}/{m}. "
        + (
            "No tests remain significant after correction."
            if n_after == 0
            else f"{n_after} test(s) remain significant after Holm-Bonferroni correction."
        ) +
        " Holm-Bonferroni is uniformly more powerful than standard Bonferroni "
        "while maintaining the same family-wise error rate control."
    )

    return {
        "n_tests":               m,
        "alpha_original":        float(alpha),
        "method":                "Holm-Bonferroni",
        "p_values_original":     p_arr,
        "p_values_adjusted":     p_adjusted,
        "rank_order":            rank_order,
        "significant_original":  sig_before,
        "significant_corrected": significant_corrected,
        "n_significant_before":  n_before,
        "n_significant_after":   n_after,
        "interpretation":        interp,
    }


# ── Numeric-Values-Stored-as-Text Detection ───────────────────────────────────

def detect_numeric_stored_as_text(df: pd.DataFrame) -> dict:
    """
    Detect columns where values appear to be numeric but are stored as text/object dtype.

    Does NOT convert them. Returns metadata and per-column diagnostics.

    Returns
    -------
    dict with keys: n_flagged, flagged_columns (list of dicts with details),
    safe_to_convert, recommendation.
    Each flagged column dict has: column, n_numeric_looking, pct_numeric_looking,
    sample_values, dtype, safe_to_convert.
    """
    flagged = []

    for col in df.columns:
        series = df[col]
        # Only check object/string columns (handles both legacy 'object' and new 'str'/'string' dtypes)
        dtype_str = str(series.dtype)
        if not (series.dtype == object or dtype_str in ("string", "str")):
            continue

        non_null = series.dropna()
        if len(non_null) == 0:
            continue

        # Try to coerce to numeric
        coerced = pd.to_numeric(non_null, errors="coerce")
        n_coercible = int(coerced.notna().sum())
        pct_coercible = round(n_coercible / len(non_null) * 100, 2)

        # Flag if ≥ 80% of values look numeric
        if pct_coercible >= 80 and n_coercible >= 3:
            safe = bool(n_coercible == len(non_null))  # all non-null are numeric
            flagged.append({
                "column":                col,
                "dtype":                 str(series.dtype),
                "n_total":               len(series),
                "n_missing":             int(series.isnull().sum()),
                "n_numeric_looking":     n_coercible,
                "pct_numeric_looking":   pct_coercible,
                "sample_values":         non_null.head(5).tolist(),
                "safe_to_convert":       safe,
                "note": (
                    "All non-missing values appear numeric — safe to convert."
                    if safe else
                    f"{len(non_null) - n_coercible} non-missing value(s) are non-numeric "
                    "— manual review required before converting."
                ),
            })

    return {
        "n_flagged":         len(flagged),
        "flagged_columns":   flagged,
        "recommendation": (
            "The following columns appear to contain numeric values stored as text. "
            "Do NOT silently convert them — review each column for non-numeric entries, "
            "formatting issues (commas, currency symbols), or deliberate codes "
            "before deciding whether conversion is appropriate. "
            "Keep the original column; create a converted copy if needed."
        ) if flagged else "No columns with numeric-looking text values detected.",
    }



# ── RULE-STAT-04: Five-Number Summary ─────────────────────────────────────────

def five_number_summary(series: pd.Series, col_name: str) -> dict:
    """
    Compute the five-number summary (Tukey 1977) for a numeric series.

    Returns Min, Q1, Median, Q3, Max plus IQR and the Tukey outlier fences.

    Rule applied (RULE-STAT-04 / RULE-STAT-05):
      Lower fence  = Q1 - 1.5 * IQR
      Upper fence  = Q3 + 1.5 * IQR
    Values outside the fences are classified as potential outliers.

    Parameters
    ----------
    series   : numeric pandas Series (NaN values dropped automatically)
    col_name : display name for the column

    Returns
    -------
    dict with keys: col_name, n, n_total, minimum, q1, median, q3, maximum,
    iqr, lower_fence, upper_fence, n_outliers_low, n_outliers_high,
    n_outliers_total, outlier_pct, interpretation.
    Returns {"error": "..."} on invalid input.
    """
    s = pd.to_numeric(series, errors="coerce").dropna()
    n_total = len(series)
    n = len(s)

    if n < 4:
        return {"error": f"Five-number summary requires at least 4 values, got {n}."}

    minimum = float(round(s.min(), 4))
    q1      = float(round(s.quantile(0.25), 4))
    median  = float(round(s.median(), 4))
    q3      = float(round(s.quantile(0.75), 4))
    maximum = float(round(s.max(), 4))
    iqr     = float(round(q3 - q1, 4))

    # Tukey fences (RULE-STAT-05)
    lower_fence = float(round(q1 - 1.5 * iqr, 4))
    upper_fence = float(round(q3 + 1.5 * iqr, 4))

    n_low   = int((s < lower_fence).sum())
    n_high  = int((s > upper_fence).sum())
    n_out   = n_low + n_high
    out_pct = float(round(n_out / n * 100, 2)) if n > 0 else 0.0

    # Skewness direction label (RULE-STAT-01)
    skew = float(s.skew())
    if abs(skew) < 0.5:
        skew_label = "approximately symmetric"
    elif skew > 0:
        skew_label = "positively skewed (tail extends right; mean > median)"
    else:
        skew_label = "negatively skewed (tail extends left; mean < median)"

    interp_parts = [
        f"Five-number summary for '{col_name}' (n = {n} of {n_total}):",
        f"  Min = {minimum}, Q1 = {q1}, Median = {median}, Q3 = {q3}, Max = {maximum}",
        f"  IQR = {iqr}",
        f"  Tukey fences: lower = {lower_fence}, upper = {upper_fence}",
        f"  Outliers: {n_out} total ({n_low} below lower fence, {n_high} above upper fence) "
        f"= {out_pct}% of observations.",
        f"  Distribution shape: {skew_label} (skewness = {round(skew, 4)}).",
    ]
    if n_out > 0:
        interp_parts.append(
            "  At least one potential outlier detected. Investigate whether outlier(s) "
            "are legitimate values or data-entry errors before proceeding with analysis."
        )

    return {
        "col_name":       col_name,
        "n":              n,
        "n_total":        n_total,
        "minimum":        minimum,
        "q1":             q1,
        "median":         median,
        "q3":             q3,
        "maximum":        maximum,
        "iqr":            iqr,
        "lower_fence":    lower_fence,
        "upper_fence":    upper_fence,
        "n_outliers_low":   n_low,
        "n_outliers_high":  n_high,
        "n_outliers_total": n_out,
        "outlier_pct":    out_pct,
        "skewness":       round(skew, 4),
        "skew_direction": skew_label,
        "interpretation": "\n".join(interp_parts),
        "assumption_note": (
            "The five-number summary is non-parametric and robust to distributional shape. "
            "Outlier fences use the Tukey (1977) rule: Q1 - 1.5*IQR and Q3 + 1.5*IQR. "
            "A value outside these fences is a *candidate* outlier — always verify against "
            "the study context before removing it."
        ),
    }


# ── RULE-STAT-02: Central Tendency Recommendation ─────────────────────────────

def central_tendency_recommendation(series: pd.Series, col_name: str,
                                    is_categorical: bool = False) -> dict:
    """
    Recommend the most appropriate measure of central tendency for a variable.

    Rules applied
    -------------
    RULE-STAT-02a  If is_categorical=True → recommend Mode (only valid measure
                   for nominal/categorical variables).
    RULE-STAT-02b  If |skewness| < 0.5 AND no IQR outliers → recommend Mean.
    RULE-STAT-02c  If |skewness| >= 0.5 OR IQR outliers detected → recommend Median.
                   The mean is sensitive to outliers and will be pulled toward the tail;
                   the median is resistant to extreme values.
    RULE-STAT-02d  Mode is additionally noted for any variable showing a dominant peak.

    RULE-STAT-01 (skew direction): positive skew (tail right) → mean > median;
                 negative skew (tail left) → mean < median; both labels are explicit
                 in the output.

    Parameters
    ----------
    series        : pandas Series (numeric or categorical)
    col_name      : display name
    is_categorical: True if the variable is nominal/categorical

    Returns
    -------
    dict with keys: col_name, recommended_measure, rationale, mean, median, mode,
    skewness, skew_direction, has_outliers, n_outliers, conflict_note.
    """
    # Categorical short-circuit (RULE-STAT-02a)
    if is_categorical:
        s_clean = series.dropna()
        mode_val = s_clean.mode().iloc[0] if len(s_clean.mode()) > 0 else "N/A"
        mode_freq = int((s_clean == mode_val).sum()) if mode_val != "N/A" else 0
        return {
            "col_name":           col_name,
            "recommended_measure": "Mode",
            "rationale": (
                "The variable is categorical (nominal/ordinal). Mean and median are not "
                "meaningful for categorical data. Mode identifies the most frequent category, "
                "which is the only valid measure of central tendency here."
            ),
            "mean":           None,
            "median":         None,
            "mode":           str(mode_val),
            "mode_frequency": mode_freq,
            "skewness":       None,
            "skew_direction": "Not applicable — categorical variable",
            "has_outliers":   False,
            "n_outliers":     0,
            "conflict_note":  "",
        }

    s = pd.to_numeric(series, errors="coerce").dropna()
    n = len(s)
    if n < 2:
        return {"error": f"Insufficient data for central tendency analysis (n = {n})."}

    mean_val   = float(round(s.mean(), 4))
    median_val = float(round(s.median(), 4))
    try:
        from scipy import stats as _sp
        mode_result = _sp.mode(s.values, keepdims=True)
        mode_val = float(round(mode_result.mode[0], 4))
    except Exception:
        mode_val = median_val   # fallback

    skew = float(s.skew())

    # Skewness direction label (RULE-STAT-01)
    if abs(skew) < 0.5:
        skew_direction = "approximately symmetric"
    elif skew > 0:
        skew_direction = "positively skewed (tail extends right; mean > median)"
    else:
        skew_direction = "negatively skewed (tail extends left; mean < median)"

    # IQR outlier detection (RULE-STAT-05)
    q1  = s.quantile(0.25)
    q3  = s.quantile(0.75)
    iqr = q3 - q1
    if iqr > 0:
        n_outliers  = int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum())
        has_outliers = n_outliers > 0
    else:
        n_outliers   = 0
        has_outliers = False

    # Decision (RULE-STAT-02b / RULE-STAT-02c)
    if abs(skew) < 0.5 and not has_outliers:
        recommended = "Mean"
        rationale = (
            f"The distribution of '{col_name}' is approximately symmetric "
            f"(skewness = {round(skew, 4)}, |skew| < 0.5) and no IQR outliers were detected. "
            "The mean is the most informative measure of central tendency and is appropriate here. "
            "Report as: mean ± standard deviation."
        )
    else:
        recommended = "Median"
        if abs(skew) >= 0.5 and has_outliers:
            reason = (
                f"significant skewness ({skew_direction}, skewness = {round(skew, 4)}) "
                f"and {n_outliers} IQR outlier(s) detected"
            )
        elif abs(skew) >= 0.5:
            reason = (
                f"significant skewness ({skew_direction}, skewness = {round(skew, 4)})"
            )
        else:
            reason = f"{n_outliers} IQR outlier(s) detected"
        rationale = (
            f"The distribution of '{col_name}' shows {reason}. "
            "The mean is sensitive to outliers and skewness — it is dragged toward the tail. "
            "The median is resistant to extreme values and better represents the typical observation. "
            "Report as: median [IQR] or median [Q1–Q3]."
        )

    # Conflict note: flag when mean and median differ by > 10% of SD
    conflict_note = ""
    std_val = float(s.std())
    if std_val > 0 and abs(mean_val - median_val) > 0.10 * std_val:
        conflict_note = (
            f"Mean ({mean_val}) and median ({median_val}) differ by "
            f"{round(abs(mean_val - median_val), 4)}, which is "
            f"{round(abs(mean_val - median_val) / std_val * 100, 1)}% of SD. "
            "This gap signals the presence of skewness or outliers. "
            "Using the mean in this context will overstate (positive skew) or "
            "understate (negative skew) the typical value for most observations."
        )

    return {
        "col_name":           col_name,
        "recommended_measure": recommended,
        "rationale":          rationale,
        "mean":               mean_val,
        "median":             median_val,
        "mode":               mode_val,
        "skewness":           round(skew, 4),
        "skew_direction":     skew_direction,
        "has_outliers":       has_outliers,
        "n_outliers":         n_outliers,
        "conflict_note":      conflict_note,
        "assumption_note": (
            "RULE-STAT-02: Mean is appropriate for symmetric distributions without extreme outliers. "
            "Median is appropriate when skewness |skew| >= 0.5 or IQR outliers are present. "
            "Mode is the only valid measure for nominal/categorical variables. "
            "Source: standard statistical practice (Field, 2018; Cohen, 1988)."
        ),
    }


# ── RULE-STAT-03: Covariance ──────────────────────────────────────────────────

def covariance(df: pd.DataFrame, col1: str, col2: str) -> dict:
    """
    Compute the sample covariance between two numerical variables and interpret
    its direction. Explicitly flags that magnitude is unit-dependent.

    Rules applied
    -------------
    RULE-STAT-03  Positive covariance -> variables move together (both above or
                  both below their respective means simultaneously).
                  Negative covariance -> one increases while the other decreases.
                  Near-zero covariance -> no consistent directional relationship.
                  Magnitude of covariance CANNOT be used to judge strength because
                  it depends on the measurement units of both variables.
                  To measure strength, divide by (SD_x * SD_y) to obtain
                  Pearson's correlation coefficient r (range -1 to +1).

    Parameters
    ----------
    df   : source DataFrame
    col1 : first numeric column name
    col2 : second numeric column name

    Returns
    -------
    dict with keys: test, col1, col2, n, n_total, covariance_sample,
    covariance_population, std_col1, std_col2, pearson_r, direction,
    interpretation, limitation_note, assumption_note.
    """
    clean   = df[[col1, col2]].dropna()
    n_total = len(df)
    n       = len(clean)

    if n < 3:
        return {"error": f"Covariance requires at least 3 complete pairs. Found: {n}."}

    x = clean[col1].astype(float)
    y = clean[col2].astype(float)

    cov_sample     = float(round(x.cov(y), 6))         # Bessel-corrected (n-1)
    cov_population = float(round(cov_sample * (n - 1) / n, 6))  # population cov
    std_x          = float(round(x.std(), 6))
    std_y          = float(round(y.std(), 6))

    # Pearson r (normalised; RULE-STAT-03)
    if std_x > 0 and std_y > 0:
        pearson_r = float(round(cov_sample / (std_x * std_y), 4))
    else:
        pearson_r = 0.0

    # Direction label
    if cov_sample > 0.0:
        direction = "positive"
        dir_plain = (
            f"'{col1}' and '{col2}' tend to move in the same direction — "
            "when one is above its mean, the other tends to be above its mean too."
        )
    elif cov_sample < 0.0:
        direction = "negative"
        dir_plain = (
            f"'{col1}' and '{col2}' tend to move in opposite directions — "
            "when one is above its mean, the other tends to be below its mean."
        )
    else:
        direction = "zero"
        dir_plain = (
            f"No consistent directional relationship between '{col1}' and '{col2}'."
        )

    interp = (
        f"Sample covariance (n = {n}): Cov({col1}, {col2}) = {cov_sample:.6f}  "
        f"[population estimate = {cov_population:.6f}].\n"
        f"Direction: {direction}. {dir_plain}\n"
        f"Normalised to Pearson r = {pearson_r:.4f} "
        f"(covariance / (SD_{col1} × SD_{col2}) = {cov_sample:.4f} / "
        f"({std_x:.4f} × {std_y:.4f})).\n"
        f"NOTE (RULE-STAT-03): The raw covariance value ({cov_sample:.4f}) cannot be used "
        f"to judge the *strength* of the relationship because its magnitude depends on the "
        f"measurement units of both variables. Always use Pearson r (or Spearman ρ) to "
        f"assess strength."
    )

    return {
        "test":                  "Covariance",
        "col1":                  col1,
        "col2":                  col2,
        "n":                     n,
        "n_total":               n_total,
        "covariance_sample":     cov_sample,
        "covariance_population": cov_population,
        "std_col1":              std_x,
        "std_col2":              std_y,
        "pearson_r":             pearson_r,
        "direction":             direction,
        "interpretation":        interp,
        "limitation_note": (
            "RULE-STAT-03 — Covariance limitation: the magnitude of covariance is entirely "
            "determined by the units of the two variables. Changing km to m, or £ to £000, "
            "will change the covariance value dramatically without changing the underlying "
            "relationship at all. Pearson r eliminates this unit dependency and should always "
            "be reported alongside or instead of covariance."
        ),
        "assumption_note": (
            "Covariance assumes: (1) both variables are numeric (interval/ratio scale), "
            "(2) observations are independent, "
            "(3) no extreme outliers (covariance, like the mean, is sensitive to outliers)."
        ),
    }
