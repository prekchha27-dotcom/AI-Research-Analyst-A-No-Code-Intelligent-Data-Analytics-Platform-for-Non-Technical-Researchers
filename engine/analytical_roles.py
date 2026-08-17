"""
Analytical Role Detection Engine
=================================
Determines the *analytical/research role* of every column independently of
its *technical data type*.

Key distinction
---------------
  Technical data type  — what Python/pandas sees (int64, float64, object …)
  Analytical role      — what the variable means for research purposes
                         (substantive numeric, identifier, serial number,
                          categorical label, ordinal scale, free-text, etc.)

No column names are hard-coded.  Every decision is based on measurable
evidence: uniqueness ratio, sequential patterns, range-to-cardinality ratio,
distribution shape, value patterns, etc.

Returns
-------
classify_column()   → ColumnClassification (dataclass)
classify_dataset()  → dict[col_name → ColumnClassification]
"""

from __future__ import annotations

import re
import math
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class ColumnClassification:
    """Complete classification for one column."""

    col_name:          str
    data_type:         str          # pandas/numpy dtype label
    analytical_role:   str          # substantive_numerical | identifier | serial |
                                    # categorical | ordinal | datetime | boolean |
                                    # constant | free_text | administrative_code | unknown
    measurement_level: str          # nominal | ordinal | interval | ratio | datetime |
                                    # identifier | constant | text
    confidence:        float        # 0–1 how confident the engine is
    evidence:          List[str]    # human-readable evidence bullets
    warnings:          List[str]    # methodological warnings for the user
    suitable_analyses: List[str]    # list of analysis types this variable supports
    unsuitable_reason: str          # why certain analyses are inappropriate
    imputation_role:   str          # "imputable" | "identifier_preserve" |
                                    # "constant_no_impute" | "free_text_manual" |
                                    # "datetime_forward_fill"
    suggested_treatment: str        # plain-language suggestion
    needs_confirmation: bool        # ask researcher before acting?

    def to_dict(self) -> dict:
        return {
            "col_name":           self.col_name,
            "data_type":          self.data_type,
            "analytical_role":    self.analytical_role,
            "measurement_level":  self.measurement_level,
            "confidence":         round(self.confidence, 2),
            "evidence":           self.evidence,
            "warnings":           self.warnings,
            "suitable_analyses":  self.suitable_analyses,
            "unsuitable_reason":  self.unsuitable_reason,
            "imputation_role":    self.imputation_role,
            "suggested_treatment": self.suggested_treatment,
            "needs_confirmation": self.needs_confirmation,
        }


# ── Role constants ─────────────────────────────────────────────────────────────

ROLE_IDENTIFIER       = "identifier"
ROLE_SERIAL           = "serial_number"
ROLE_ADMIN_CODE       = "administrative_code"
ROLE_CONSTANT         = "constant"
ROLE_SUBST_NUMERICAL  = "substantive_numerical"
ROLE_CATEGORICAL      = "categorical"
ROLE_ORDINAL          = "ordinal"
ROLE_BOOLEAN          = "boolean"
ROLE_DATETIME         = "datetime"
ROLE_FREE_TEXT        = "free_text"
ROLE_UNKNOWN          = "unknown"

LEVEL_NOMINAL     = "nominal"
LEVEL_ORDINAL     = "ordinal"
LEVEL_INTERVAL    = "interval"
LEVEL_RATIO       = "ratio"
LEVEL_DATETIME    = "datetime"
LEVEL_IDENTIFIER  = "identifier"
LEVEL_CONSTANT    = "constant"
LEVEL_TEXT        = "text"

# Patterns in column names that strongly suggest an identifier/admin role
# (intentionally very general — not specific to any dataset)
_ID_NAME_PATTERNS = [
    r'\bid\b',           # bare "id"
    r'\bids\b',
    r'_id$',             # ends with _id
    r'^id_',             # starts with id_
    r'\bcode\b',
    r'\bkey\b',
    r'\bref\b',
    r'\bserial\b',
    r'\bsl\b',           # sl.no, sl no
    r'\bno\b',           # no., no
    r'\bnumber\b',
    r'\bnum\b',
    r'\bindex\b',
    r'\brecord\b',
    r'\bregist',         # registration
    r'\bcase\b',
    r'\bsubject\b',
    r'\bparticipant\b',
    r'\brespondent\b',
    r'\bhousehold\b',
    r'\bperson\b',
    r'\bpatient\b',
    r'\bclient\b',
    r'\bsample\b',
    r'\bentry\b',
    r'\brow\b',
    r'\bobs\b',          # observation
    r'\buid\b',
    r'\buuid\b',
    r'\bnhs\b',
    r'\bssn\b',
    r'\broll\b',
    r'\bcustomer\b',
    r'\bemployee\b',
    r'\bstudent\b',
    r'\bworker\b',
]

_ID_NAME_RE = re.compile("|".join(_ID_NAME_PATTERNS), re.IGNORECASE)


def _name_suggests_identifier(col: str) -> bool:
    """Return True if column name matches typical identifier patterns."""
    clean = re.sub(r'[^a-z0-9 _]', ' ', col.lower())
    return bool(_ID_NAME_RE.search(clean))


def _is_sequential(series: pd.Series) -> bool:
    """
    Return True if the numeric series looks like a sequential counter
    (e.g. 1,2,3,4,... or 101,102,103,...).
    Checks whether sorted differences are all approximately 1.
    """
    valid = series.dropna()
    if len(valid) < 3:
        return False
    try:
        vals   = np.sort(valid.astype(float).values)
        diffs  = np.diff(vals)
        median_diff = float(np.median(diffs))
        if median_diff <= 0:
            return False
        # Allow some small gaps/repetitions — within 5% of median diff
        tolerance = max(0.5, abs(median_diff) * 0.05)
        frac_unit = np.mean(np.abs(diffs - median_diff) <= tolerance)
        return frac_unit >= 0.90   # 90%+ of steps are uniform
    except Exception:
        return False


def _all_unique(series: pd.Series) -> bool:
    """True if every non-null value is unique."""
    valid = series.dropna()
    if len(valid) == 0:
        return False
    return int(valid.nunique()) == len(valid)


def _uniqueness_ratio(series: pd.Series) -> float:
    """Fraction of non-null values that are unique (0..1)."""
    valid = series.dropna()
    if len(valid) == 0:
        return 0.0
    return valid.nunique() / len(valid)


def _looks_like_code(series: pd.Series) -> bool:
    """
    Heuristic: object/string column where values look like codes
    (short, alphanumeric, consistent length, many unique values).
    """
    valid = series.dropna().astype(str)
    if len(valid) == 0:
        return False
    lengths = valid.str.len()
    mean_len = float(lengths.mean())
    std_len  = float(lengths.std()) if len(valid) > 1 else 0.0
    # Short tokens (≤12 chars), low length variance, high uniqueness
    return mean_len <= 12 and std_len <= 3 and _uniqueness_ratio(series) > 0.7


# Values that unambiguously represent boolean/binary semantics
_BOOLEAN_VALUE_SETS = [
    {'true', 'false'},
    {'yes', 'no'},
    {'y', 'n'},
    {'1', '0'},
    {1, 0},
    {1.0, 0.0},
    {'t', 'f'},
    {'positive', 'negative'},
    {'pass', 'fail'},
    {'present', 'absent'},
    {'active', 'inactive'},
]


def _is_boolean(series: pd.Series) -> bool:
    """
    True if column has exactly 1–2 distinct non-null values AND those values
    look like boolean/binary semantics (True/False, Yes/No, 0/1, etc.).
    A column with 2 arbitrary categories (e.g. Red/Blue) is NOT boolean.
    """
    valid = series.dropna()
    if valid.nunique() < 1 or valid.nunique() > 2:
        return False
    if pd.api.types.is_bool_dtype(series):
        return True
    # Check numeric 0/1
    if pd.api.types.is_numeric_dtype(series):
        vals = set(valid.unique())
        return vals.issubset({0, 1, 0.0, 1.0})
    # Check string values
    vals_lower = {str(v).strip().lower() for v in valid.unique()}
    for bool_set in _BOOLEAN_VALUE_SETS:
        if isinstance(next(iter(bool_set)), str):
            if vals_lower == bool_set or vals_lower.issubset(bool_set):
                return True
        else:
            if vals_lower == {str(v) for v in bool_set}:
                return True
    return False


def _is_constant(series: pd.Series) -> bool:
    """True if all non-null values are identical."""
    valid = series.dropna()
    return len(valid) > 0 and valid.nunique() == 1


def _infer_measurement_level(series: pd.Series, analytical_role: str,
                              dtype_label: str) -> str:
    """Infer measurement level from role + data characteristics."""
    if analytical_role in (ROLE_IDENTIFIER, ROLE_SERIAL, ROLE_ADMIN_CODE):
        return LEVEL_IDENTIFIER
    if analytical_role == ROLE_CONSTANT:
        return LEVEL_CONSTANT
    if analytical_role == ROLE_DATETIME:
        return LEVEL_DATETIME
    if analytical_role == ROLE_FREE_TEXT:
        return LEVEL_TEXT
    if analytical_role == ROLE_BOOLEAN:
        return LEVEL_NOMINAL
    if analytical_role == ROLE_CATEGORICAL:
        return LEVEL_NOMINAL
    if analytical_role == ROLE_ORDINAL:
        return LEVEL_ORDINAL
    if analytical_role == ROLE_SUBST_NUMERICAL:
        # Ratio vs interval: ratio has a true zero
        valid = series.dropna()
        if pd.api.types.is_numeric_dtype(series):
            if float(valid.min()) >= 0:
                return LEVEL_RATIO
            else:
                return LEVEL_INTERVAL
    return LEVEL_NOMINAL


def _suitable_analyses(analytical_role: str, measurement_level: str,
                        n_unique: int, n_rows: int) -> List[str]:
    """Return list of analyses appropriate for this variable."""
    if analytical_role in (ROLE_IDENTIFIER, ROLE_SERIAL, ROLE_ADMIN_CODE,
                           ROLE_CONSTANT, ROLE_FREE_TEXT):
        return []
    if analytical_role == ROLE_DATETIME:
        return ["time_series", "trend_analysis", "grouping_by_period"]
    if analytical_role == ROLE_BOOLEAN:
        return ["frequency", "chi_square", "logistic_regression", "crosstab"]
    if analytical_role == ROLE_CATEGORICAL:
        return ["frequency", "crosstab", "chi_square", "bar_chart", "pie_chart",
                "t_test_grouping", "anova_grouping"]
    if analytical_role == ROLE_ORDINAL:
        return ["frequency", "crosstab", "spearman_correlation", "chi_square",
                "median_comparison", "bar_chart"]
    if analytical_role == ROLE_SUBST_NUMERICAL:
        analyses = ["descriptive_stats", "histogram", "boxplot", "scatter_plot"]
        if n_unique > 5:
            analyses += ["pearson_correlation", "spearman_correlation",
                         "linear_regression", "t_test_outcome", "anova_outcome"]
        return analyses
    return []


def _unsuitable_reason(analytical_role: str) -> str:
    """Return explanation of why certain analyses are not appropriate."""
    if analytical_role == ROLE_IDENTIFIER:
        return ("This column appears to be a unique identifier or record number. "
                "Computing mean, median, correlation or regression on identifier values "
                "is statistically meaningless — identifiers do not represent a measured quantity.")
    if analytical_role == ROLE_SERIAL:
        return ("This column appears to be a sequential serial number. "
                "Statistical analysis on serial numbers is not meaningful because the values "
                "are assigned by position rather than representing a measured characteristic.")
    if analytical_role == ROLE_ADMIN_CODE:
        return ("This column appears to be an administrative code or reference number. "
                "While the values may be numeric, they represent labels, not measurements, "
                "so arithmetic operations (mean, sum, correlation) are not meaningful.")
    if analytical_role == ROLE_CONSTANT:
        return ("This column has only one unique value across all rows (constant). "
                "It provides no variation and cannot be used in statistical tests, "
                "correlations, or regression — all such tests require variation in the data.")
    if analytical_role == ROLE_FREE_TEXT:
        return ("This column contains free-form text. Quantitative statistical methods "
                "(mean, t-test, correlation, etc.) are not applicable to free text. "
                "Consider text analysis or content coding first.")
    if analytical_role == ROLE_DATETIME:
        return ("Datetime values should not be directly used in arithmetic statistical tests. "
                "Use them for grouping, trend analysis, or time-series decomposition instead.")
    return ""


def _imputation_role(analytical_role: str) -> str:
    if analytical_role in (ROLE_IDENTIFIER, ROLE_SERIAL, ROLE_ADMIN_CODE):
        return "identifier_preserve"
    if analytical_role == ROLE_CONSTANT:
        return "constant_no_impute"
    if analytical_role == ROLE_FREE_TEXT:
        return "free_text_manual"
    if analytical_role == ROLE_DATETIME:
        return "datetime_forward_fill"
    return "imputable"


def _suggested_treatment(analytical_role: str, n_missing: int,
                          pct_missing: float, measurement_level: str) -> str:
    if analytical_role in (ROLE_IDENTIFIER, ROLE_SERIAL):
        if n_missing == 0:
            return "No action needed. Exclude from statistical analysis."
        return ("Missing identifier values should NOT be imputed. "
                "Exclude from analysis or regenerate identifiers if required.")
    if analytical_role == ROLE_ADMIN_CODE:
        if n_missing == 0:
            return "No action needed. Treat as a categorical label, not a numeric quantity."
        return "Missing codes cannot be imputed. Mark as 'Unknown' or exclude."
    if analytical_role == ROLE_CONSTANT:
        return "Column has no variation — consider removing it from the analysis."
    if analytical_role == ROLE_FREE_TEXT:
        return "Free-text field. Use text coding or content analysis."
    if analytical_role == ROLE_DATETIME:
        if n_missing == 0:
            return "No action needed. Use for time-based analysis or grouping."
        return "Forward-fill is the standard approach for ordered datetime columns."
    if n_missing == 0:
        return "No missing values. Ready for analysis."
    if pct_missing > 40:
        return (f"{pct_missing:.1f}% missing — consider KNN or multiple imputation. "
                "High missingness may bias results regardless of method.")
    if measurement_level in (LEVEL_NOMINAL, LEVEL_ORDINAL) or analytical_role in (ROLE_CATEGORICAL, ROLE_BOOLEAN):
        return "Mode imputation (most frequent value) is recommended for categorical/ordinal data."
    if measurement_level in (LEVEL_RATIO, LEVEL_INTERVAL):
        return "Mean or median imputation (depending on distribution shape) is recommended."
    return "Review variable type and choose an appropriate imputation method."


# ── Main classifier ───────────────────────────────────────────────────────────

def classify_column(df: pd.DataFrame, col: str) -> ColumnClassification:
    """
    Classify a single column's analytical role using evidence-based heuristics.
    No column names are hard-coded — only patterns and data characteristics.
    """
    series     = df[col]
    valid      = series.dropna()
    n_rows     = len(series)
    n_valid    = len(valid)
    n_missing  = int(series.isnull().sum())
    n_unique   = int(valid.nunique())
    pct_miss   = round(n_missing / max(n_rows, 1) * 100, 2)
    uniq_ratio = _uniqueness_ratio(series)
    is_numeric = pd.api.types.is_numeric_dtype(series)
    is_dt      = pd.api.types.is_datetime64_any_dtype(series)

    dtype_label = str(series.dtype)
    evidence: List[str] = []
    warnings: List[str] = []
    confidence = 0.5
    role       = ROLE_UNKNOWN

    # ── 1. Constant column ──────────────────────────────────────────────────
    if n_valid > 0 and _is_constant(series):
        role = ROLE_CONSTANT
        confidence = 0.99
        evidence.append(f"Only 1 unique non-null value ({repr(valid.iloc[0])}) across all {n_rows} rows.")
        warnings.append("Constant columns cannot be used in any statistical test — remove from analysis.")

    # ── 2. Datetime ─────────────────────────────────────────────────────────
    elif is_dt:
        role = ROLE_DATETIME
        confidence = 0.99
        evidence.append("Column has datetime64 dtype — detected as a date/time variable.")

    # ── 3. Boolean ──────────────────────────────────────────────────────────
    elif _is_boolean(series) and n_unique <= 2:
        # Check for Yes/No, True/False, 0/1 patterns
        role = ROLE_BOOLEAN
        confidence = 0.90
        vals_str = set(str(v).strip().lower() for v in valid.unique())
        evidence.append(f"Only {n_unique} unique value(s): {list(valid.unique()[:5])} — treated as boolean/binary.")

    # ── 4. Numeric columns — most complex branch ─────────────────────────────
    elif is_numeric:

        # 4a. Sequential integer — very strong signal of serial/row number
        sequential = _is_sequential(series)
        name_id    = _name_suggests_identifier(col)
        all_uniq   = _all_unique(series)

        # Score accumulator for identifier probability
        id_score = 0.0
        if sequential:
            id_score += 0.55
            evidence.append("Values are sequential (1,2,3,… or similar step pattern) — "
                             "strongly suggests a row/serial number.")
        if name_id:
            id_score += 0.30
            evidence.append(f"Column name '{col}' matches a common identifier naming pattern.")
        if all_uniq and n_valid >= 5:
            id_score += 0.20
            evidence.append(f"All {n_valid} non-null values are unique — consistent with an identifier.")
        if uniq_ratio > 0.95 and n_valid >= 10:
            id_score += 0.15
            evidence.append(f"Uniqueness ratio = {uniq_ratio:.2%} — very high for a substantive variable.")

        # Check if values look like integer codes (small int, high cardinality)
        if pd.api.types.is_integer_dtype(series) and n_unique >= n_valid * 0.80 and n_valid >= 5:
            id_score += 0.10
            evidence.append("Integer column with very high cardinality — may be a coded identifier.")

        # Determine role
        if id_score >= 0.60:
            role = ROLE_SERIAL if sequential else ROLE_IDENTIFIER
            confidence = min(0.95, 0.50 + id_score * 0.45)
            warnings.append(
                "This numeric column appears to be an identifier or serial number — "
                "NOT a measurement. Do not use it in mean, correlation, regression, "
                "t-test, ANOVA or other substantive statistical analyses."
            )
            if confidence < 0.85:
                warnings.append(
                    "The system is not fully certain of this classification. "
                    "Please review and confirm whether this is an identifier."
                )
        else:
            # Substantive numerical variable
            role = ROLE_SUBST_NUMERICAL
            evidence.append(f"Numeric column with {n_unique} unique values and uniqueness ratio {uniq_ratio:.2%}.")

            # Check for low cardinality — might be ordinal
            if n_unique <= 10 and n_unique < n_valid * 0.10:
                role = ROLE_ORDINAL
                confidence = 0.70
                evidence.append(
                    f"Only {n_unique} distinct numeric values across {n_valid} observations — "
                    "likely an ordinal scale (e.g., 1–5 Likert scale)."
                )
            else:
                confidence = 0.80
                # Subtract from confidence for ambiguous edge cases
                if name_id and not sequential:
                    confidence -= 0.10
                    warnings.append(
                        f"Column name '{col}' partially matches identifier patterns, "
                        "but the data distribution does not confirm this. "
                        "Please verify whether this is a measurement or an identifier."
                    )

    # ── 5. Object/string columns ─────────────────────────────────────────────
    else:
        name_id = _name_suggests_identifier(col)

        # Try string-based identifier detection
        str_uniq_ratio = _uniqueness_ratio(series)
        looks_code     = _looks_like_code(series)
        very_high_uniq = str_uniq_ratio > 0.90 and n_valid >= 10

        if name_id and very_high_uniq:
            role = ROLE_IDENTIFIER
            confidence = 0.85
            evidence.append(f"Column name '{col}' matches identifier patterns.")
            evidence.append(f"String values are highly unique ({str_uniq_ratio:.2%}) — consistent with an ID.")
            warnings.append("Appears to be a text identifier. Exclude from statistical analysis.")
        elif looks_code and very_high_uniq:
            role = ROLE_ADMIN_CODE
            confidence = 0.75
            evidence.append("Short, consistent-length alphanumeric values with high uniqueness — "
                             "suggests administrative codes or reference numbers.")
            warnings.append("Administrative codes should be treated as labels, not measured quantities.")
        elif n_unique / max(n_valid, 1) > 0.60 and n_valid >= 5:
            role = ROLE_FREE_TEXT
            confidence = 0.70
            evidence.append(f"High proportion of unique string values ({str_uniq_ratio:.2%}) — "
                             "likely free-text responses or unstructured data.")
            warnings.append("Free-text columns are not suitable for standard quantitative analysis.")
        else:
            role = ROLE_CATEGORICAL
            confidence = 0.80
            evidence.append(f"{n_unique} unique categories out of {n_valid} non-null values "
                             f"({str_uniq_ratio:.2%} uniqueness ratio).")

    # ── Enrich evidence with missingness info ─────────────────────────────────
    if n_missing > 0:
        evidence.append(f"{n_missing} missing value(s) ({pct_miss}% of rows).")
        if pct_miss > 40:
            warnings.append(f"High missingness ({pct_miss}%). Any imputation method may introduce bias.")

    # ── Resolve measurement level, suitable analyses, imputation role ─────────
    measurement_level  = _infer_measurement_level(series, role, dtype_label)
    suitable           = _suitable_analyses(role, measurement_level, n_unique, n_rows)
    unsuitable         = _unsuitable_reason(role)
    imp_role           = _imputation_role(role)
    treatment          = _suggested_treatment(role, n_missing, pct_miss, measurement_level)
    needs_confirm      = confidence < 0.80 or (
        role in (ROLE_IDENTIFIER, ROLE_SERIAL, ROLE_ADMIN_CODE) and confidence < 0.90
    )

    return ColumnClassification(
        col_name          = col,
        data_type         = dtype_label,
        analytical_role   = role,
        measurement_level = measurement_level,
        confidence        = confidence,
        evidence          = evidence,
        warnings          = warnings,
        suitable_analyses = suitable,
        unsuitable_reason = unsuitable,
        imputation_role   = imp_role,
        suggested_treatment = treatment,
        needs_confirmation  = needs_confirm,
    )


def classify_dataset(df: pd.DataFrame) -> dict:
    """
    Classify every column in df.
    Returns {col_name: ColumnClassification}.
    """
    return {col: classify_column(df, col) for col in df.columns}


# ── Role-aware imputation method recommendation ───────────────────────────────

def recommend_imputation_for_column(
        col: str,
        series: pd.Series,
        classification: ColumnClassification) -> dict:
    """
    Recommend imputation (or non-imputation) strategy taking the
    analytical role into account.

    Returns a dict with:
        method           — recommended method (or 'do_not_impute')
        reason           — plain-language explanation
        why_appropriate  — statistical rationale
        warning          — methodological caution (may be empty)
        alternatives     — list of alternative methods
        ok_to_automate   — whether system can apply automatically (no confirmation needed)
    """
    n_missing = int(series.isnull().sum())
    n_total   = max(len(series), 1)
    pct       = round(n_missing / n_total * 100, 2)
    role      = classification.analytical_role
    level     = classification.measurement_level
    imp_role  = classification.imputation_role

    if n_missing == 0:
        return {
            "method":          "none",
            "reason":          "No missing values in this column.",
            "why_appropriate": "Nothing to impute.",
            "warning":         "",
            "alternatives":    [],
            "ok_to_automate":  True,
        }

    # ── 100% missing — imputation is statistically impossible ─────────────────
    if n_missing >= n_total:
        return {
            "method":          "cannot_impute_all_missing",
            "reason":          (
                f"ALL {n_total} values in this column are missing (100%). "
                "There is no observed data to base any imputation on. "
                "Statistical methods such as mean, median, mode, and KNN all require "
                "at least some observed values to produce an estimate."
            ),
            "why_appropriate": (
                "Imputation derives estimates from the distribution of observed values. "
                "When no values are observed, any imputed value would be entirely fabricated "
                "with no empirical basis — this is not imputation, it is invention."
            ),
            "warning": (
                "Do NOT impute this column. Investigate why all values are missing: "
                "it may indicate a data import/conversion error, a structural skip pattern, "
                "or a column that does not apply to this dataset."
            ),
            "alternatives":    ["drop_column", "investigate_source"],
            "ok_to_automate":  False,
        }

    # ── Identifiers / serial numbers — never impute ───────────────────────────
    if imp_role == "identifier_preserve":
        return {
            "method":  "do_not_impute",
            "reason":  (
                f"This column is classified as '{role}'. "
                "Missing values in identifier columns should NOT be filled with "
                "statistical estimates — doing so would create fictitious IDs."
            ),
            "why_appropriate": (
                "Identifiers are not measured quantities. Imputing them with mean, "
                "median or KNN would produce meaningless values that could corrupt "
                "record linkage, duplicate detection and other downstream operations."
            ),
            "warning": (
                "If these missing IDs are needed, regenerate them from the original "
                "source data or assign new sequential IDs only if methodologically appropriate."
            ),
            "alternatives":   ["exclude_column", "regenerate_from_source"],
            "ok_to_automate": False,
        }

    # ── Constant column — no imputation needed (but warn) ────────────────────
    if imp_role == "constant_no_impute":
        return {
            "method":          "constant_unknown",
            "reason":          "Column has only one unique value. Filling missing values with the same constant.",
            "why_appropriate": "Since all values are identical, filling with the mode preserves consistency.",
            "warning":         "This column has no variation and should be excluded from statistical analyses.",
            "alternatives":    ["drop_rows"],
            "ok_to_automate":  True,
        }

    # ── Datetime ─────────────────────────────────────────────────────────────
    if imp_role == "datetime_forward_fill":
        return {
            "method":  "forward_fill",
            "reason":  "Forward-fill propagates the nearest preceding date — standard for ordered time data.",
            "why_appropriate": (
                "Datetime columns represent ordered temporal observations. "
                "Statistical mean/median of dates is not meaningful. "
                "Forward-fill preserves chronological order."
            ),
            "warning":   (
                "Forward-fill is only valid if the data are ordered chronologically. "
                "Check whether the rows are sorted by time before applying."
            ),
            "alternatives":   ["drop_rows"],
            "ok_to_automate": False,
        }

    # ── Free text ─────────────────────────────────────────────────────────────
    if imp_role == "free_text_manual":
        return {
            "method":  "constant_unknown",
            "reason":  "Free-text columns cannot be imputed statistically. Filling with 'Unknown'.",
            "why_appropriate": (
                "Free-text fields contain unstructured content. "
                "Statistical imputation is not appropriate. "
                "Filling with 'Unknown' preserves row count without inventing data."
            ),
            "warning":   "Consider whether rows with missing free-text are meaningful for your analysis.",
            "alternatives":   ["drop_rows"],
            "ok_to_automate": False,
        }

    # ── Substantive / categorical — apply standard recommendation logic ────────
    valid = series.dropna()

    # Categorical / nominal / boolean
    if level in (LEVEL_NOMINAL, LEVEL_ORDINAL) or role in (ROLE_CATEGORICAL, ROLE_BOOLEAN, ROLE_ORDINAL):
        method = "mode"
        reason = (
            f"Variable type is {role} (measurement level: {level}). "
            "Mode (most frequent category) is the statistically appropriate imputation "
            "for nominal/ordinal data — mean and median are not applicable."
        )
        why    = (
            "Nominal and ordinal data do not have a meaningful arithmetic mean. "
            "The mode preserves the most common category, minimising distortion of the distribution."
        )
        warn   = ("Mode imputation reduces category variability. "
                  "With high missingness, check whether the imputed distribution "
                  "still reflects the true population.")
        alts   = ["constant_unknown", "drop_rows"]
        auto   = pct <= 20

    # Numerical / ratio / interval
    elif level in (LEVEL_RATIO, LEVEL_INTERVAL) or role == ROLE_SUBST_NUMERICAL:
        if len(valid) < 3:
            method = "median"
            reason = "Too few non-missing values. Median is used as a conservative default."
            why    = "With very few observations, skewness cannot be reliably estimated. Median is robust."
            warn   = "Very few non-missing values — any imputation will have high uncertainty."
            alts   = ["mean", "constant_unknown", "drop_rows"]
            auto   = False
        else:
            try:
                skew = abs(float(valid.skew()))
            except Exception:
                skew = 0.0

            if pct > 40:
                method = "knn"
                reason = (
                    f"{pct:.1f}% of values are missing. "
                    "KNN imputation uses patterns from similar rows to estimate missing values, "
                    "making it more appropriate than single-statistic methods at high missingness."
                )
                why    = (
                    "Single-statistic methods (mean/median) ignore the relationship between "
                    "variables. With >40% missingness, KNN produces more reliable estimates "
                    "by leveraging multivariate structure in the data."
                )
                warn   = (
                    f"High missingness ({pct:.1f}%) is a concern regardless of the method used. "
                    "Consider whether the missing data is random (MCAR) or systematic (MAR/MNAR) "
                    "before imputing — systematic missingness may bias results."
                )
                alts   = ["median", "mean", "drop_rows"]
                auto   = False

            elif skew > 1.0:
                method = "median"
                reason = (
                    f"Distribution is skewed (|skewness| = {skew:.2f}). "
                    "Median is more robust than mean for skewed distributions "
                    "because it is not pulled toward the tail."
                )
                why    = (
                    "When data is skewed, the mean is located in the tail and "
                    "does not represent the centre of the distribution. "
                    "Median imputation is more representative."
                )
                warn   = ""
                alts   = ["mean", "knn"]
                auto   = pct <= 15

            else:
                method = "mean"
                reason = (
                    f"Distribution is approximately symmetric (|skewness| = {skew:.2f}). "
                    f"Mean imputation is appropriate for {pct:.1f}% missingness "
                    "in a normally distributed variable."
                )
                why    = (
                    "For symmetric distributions, the mean is the best single-value estimate "
                    "of the population centre. Mean imputation preserves the variable's mean "
                    "while reducing its variance slightly."
                )
                warn   = ("Mean imputation reduces variance in the imputed column. "
                          "With >15% missingness, consider KNN for a more conservative estimate.")
                alts   = ["median", "knn"]
                auto   = pct <= 15

    else:
        method = "mode"
        reason = f"Variable type is uncertain ('{role}'). Mode imputation used as a conservative default."
        why    = "When the variable type cannot be confidently determined, mode imputation is the safest choice."
        warn   = "Please verify this column's analytical role and choose the most appropriate method."
        alts   = ["constant_unknown", "drop_rows"]
        auto   = False

    return {
        "method":          method,
        "reason":          reason,
        "why_appropriate": why,
        "warning":         warn,
        "alternatives":    alts,
        "ok_to_automate":  auto,
    }
