"""
Data Profiling Engine
=====================
Inspects an uploaded DataFrame and returns a structured profile.
NO data is modified here — original DataFrame is always preserved.

V2 additions:
- Integrates analytical_roles.py for per-column role classification
- Every column now carries: data_type, analytical_role, measurement_level,
  confidence, evidence, warnings, suitable_analyses, unsuitable_reason,
  imputation_role, suggested_treatment, needs_confirmation
- The 'numerical_cols' list now EXCLUDES identifiers/serials/constants —
  it contains only substantive numerical variables
- 'analysis_ready_numerical' and 'analysis_ready_categorical' are the
  correct lists to use when offering statistical analyses
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from datetime import datetime


def detect_variable_types(df: pd.DataFrame) -> dict:
    """
    Classify each column as numerical, categorical, datetime, or text.
    This is the *technical* type only — analytical role is determined
    separately by analytical_roles.py.
    """
    types = {}
    for col in df.columns:
        s = df[col].dropna()
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            types[col] = "datetime"
        elif pd.api.types.is_numeric_dtype(df[col]):
            # Low cardinality numerics may be ordinal/categorical
            if df[col].nunique() <= 10 and df[col].nunique() < max(len(df) * 0.05, 2):
                types[col] = "categorical_numeric"
            else:
                types[col] = "numerical"
        else:
            # Try parsing as datetime
            if s.dtype == object or str(s.dtype) in ("string", "str"):
                sample = s.head(20).astype(str)
                parsed = 0
                for v in sample:
                    try:
                        pd.to_datetime(v)
                        parsed += 1
                    except Exception:
                        pass
                if len(sample) > 0 and parsed >= len(sample) * 0.7:
                    types[col] = "datetime"
                    continue
            n_unique = df[col].nunique()
            n_total  = len(df[col].dropna())
            if n_total == 0:
                types[col] = "categorical"
            elif n_unique / max(n_total, 1) < 0.5 or n_unique <= 30:
                types[col] = "categorical"
            else:
                types[col] = "text"
    return types


def detect_outliers_iqr(series: pd.Series) -> dict:
    """Detect outliers using IQR method for a numerical series."""
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outliers = series[(series < lower) | (series > upper)]
    return {
        "count":        len(outliers),
        "lower_bound":  round(lower, 4),
        "upper_bound":  round(upper, 4),
        "outlier_values": outliers.tolist()[:20],
        "q1":           round(q1, 4),
        "q3":           round(q3, 4),
        "iqr":          round(iqr, 4),
    }


def profile_dataset(df: pd.DataFrame,
                    user_overrides: dict | None = None) -> dict:
    """
    Full dataset profile. Returns a dict with all profiling results.
    Does NOT modify df.

    user_overrides: {col_name: {"analytical_role": ..., "measurement_level": ...}}
    Applied on top of automatic classification, allowing the researcher to
    correct the system's classification.
    """
    from engine.analytical_roles import (
        classify_dataset, recommend_imputation_for_column,
        ROLE_IDENTIFIER, ROLE_SERIAL, ROLE_ADMIN_CODE, ROLE_CONSTANT,
        ROLE_FREE_TEXT, ROLE_SUBST_NUMERICAL, ROLE_CATEGORICAL,
        ROLE_ORDINAL, ROLE_BOOLEAN, ROLE_DATETIME,
    )

    profile = {}
    user_overrides = user_overrides or {}

    # --- Basic dimensions ---
    profile["n_rows"]     = len(df)
    profile["n_cols"]     = len(df.columns)
    profile["columns"]    = list(df.columns)
    profile["memory_kb"]  = round(df.memory_usage(deep=True).sum() / 1024, 2)

    # --- Technical variable types (legacy support) ---
    var_types = detect_variable_types(df)
    profile["variable_types"] = var_types

    # --- Analytical role classification ---
    classifications = classify_dataset(df)

    # Apply user overrides
    for col, overrides in user_overrides.items():
        if col in classifications:
            c = classifications[col]
            if "analytical_role" in overrides:
                c.analytical_role = overrides["analytical_role"]
            if "measurement_level" in overrides:
                c.measurement_level = overrides["measurement_level"]
            c.evidence.insert(0, "⚙️ Manually overridden by researcher.")
            c.needs_confirmation = False

    profile["classifications"] = {col: c.to_dict() for col, c in classifications.items()}

    # --- Derived column lists ---
    # Technical lists (legacy: include everything by dtype)
    numerical_cols_all  = [c for c, t in var_types.items() if t in ("numerical", "categorical_numeric")]
    categorical_cols_all = [c for c, t in var_types.items() if t in ("categorical", "categorical_numeric")]
    datetime_cols       = [c for c, t in var_types.items() if t == "datetime"]
    text_cols           = [c for c, t in var_types.items() if t == "text"]

    # Analysis-ready lists (EXCLUDE identifiers, serials, constants, free-text)
    _non_analytical = {ROLE_IDENTIFIER, ROLE_SERIAL, ROLE_ADMIN_CODE,
                       ROLE_CONSTANT, ROLE_FREE_TEXT}

    analysis_ready_numerical = [
        c for c, cl in classifications.items()
        if cl.analytical_role in (ROLE_SUBST_NUMERICAL,)
    ]
    analysis_ready_ordinal = [
        c for c, cl in classifications.items()
        if cl.analytical_role == ROLE_ORDINAL
    ]
    analysis_ready_categorical = [
        c for c, cl in classifications.items()
        if cl.analytical_role in (ROLE_CATEGORICAL, ROLE_BOOLEAN, ROLE_ORDINAL)
    ]
    identifier_cols = [
        c for c, cl in classifications.items()
        if cl.analytical_role in (ROLE_IDENTIFIER, ROLE_SERIAL, ROLE_ADMIN_CODE)
    ]
    excluded_cols = [
        c for c, cl in classifications.items()
        if cl.analytical_role in _non_analytical
    ]

    # Backward-compatible lists (used by existing UI code)
    # numerical_cols = analysis-ready numerical only
    profile["numerical_cols"]           = analysis_ready_numerical
    profile["numerical_cols_all"]       = numerical_cols_all   # includes identifiers
    profile["categorical_cols"]         = analysis_ready_categorical
    profile["datetime_cols"]            = datetime_cols
    profile["text_cols"]                = text_cols
    profile["identifier_cols"]          = identifier_cols
    profile["excluded_cols"]            = excluded_cols
    profile["analysis_ready_numerical"] = analysis_ready_numerical
    profile["analysis_ready_ordinal"]   = analysis_ready_ordinal
    profile["analysis_ready_categorical"] = analysis_ready_categorical

    # --- Missing values ---
    missing      = df.isnull().sum()
    missing_pct  = (missing / max(len(df), 1) * 100).round(2)
    profile["missing_counts"]      = missing.to_dict()
    profile["missing_pct"]         = missing_pct.to_dict()
    profile["total_missing"]       = int(missing.sum())
    profile["total_missing_pct"]   = round(
        missing.sum() / max(len(df) * max(len(df.columns), 1), 1) * 100, 2)

    # --- Duplicates ---
    profile["duplicate_rows"] = int(df.duplicated().sum())

    # --- Per-column stats (enriched with classification) ---
    col_stats = {}
    for col in df.columns:
        s     = df[col]
        ctype = var_types.get(col, "unknown")
        cl    = classifications[col]

        stat = {
            "type":        ctype,
            "role":        cl.analytical_role,
            "level":       cl.measurement_level,
            "missing":     int(s.isnull().sum()),
            "missing_pct": round(s.isnull().sum() / max(len(df), 1) * 100, 2),
            "unique":      int(s.nunique()),
            "unique_pct":  round(s.nunique() / max(len(df), 1) * 100, 2),
            "warnings":    cl.warnings,
            "evidence":    cl.evidence,
            "confidence":  cl.confidence,
        }

        if ctype == "numerical":
            sn = s.dropna()
            if len(sn) > 0:
                mean_v = float(sn.mean())
                std_v  = float(sn.std())
                cv_v   = round((std_v / mean_v * 100), 4) if mean_v != 0 else None
                stat.update({
                    "mean":          round(mean_v, 4),
                    "median":        round(float(sn.median()), 4),
                    "std":           round(std_v, 4),
                    "variance":      round(float(sn.var()), 4),
                    "cv_pct":        cv_v,
                    "min":           round(float(sn.min()), 4),
                    "p10":           round(float(sn.quantile(0.10)), 4),
                    "q1":            round(float(sn.quantile(0.25)), 4),
                    "q3":            round(float(sn.quantile(0.75)), 4),
                    "p90":           round(float(sn.quantile(0.90)), 4),
                    "max":           round(float(sn.max()), 4),
                    "range":         round(float(sn.max() - sn.min()), 4),
                    "iqr":           round(float(sn.quantile(0.75) - sn.quantile(0.25)), 4),
                    "skewness":      round(float(sn.skew()), 4),
                    "kurtosis":      round(float(sn.kurt()), 4),
                    # Only add outlier info for substantive numerical variables
                    "outliers": (detect_outliers_iqr(sn)
                                 if cl.analytical_role == ROLE_SUBST_NUMERICAL
                                 else {"count": 0, "note": "Not computed for non-analytical columns"}),
                })
        elif ctype in ("categorical", "categorical_numeric", "text"):
            vc = s.value_counts()
            stat["top_values"] = vc.head(10).to_dict()
            stat["mode"] = str(s.mode().iloc[0]) if len(s.mode()) > 0 else "N/A"

        col_stats[col] = stat
    profile["col_stats"] = col_stats

    # --- Role-aware imputation recommendations ---
    imputation_recs = {}
    for col in df.columns:
        cl  = classifications[col]
        rec = recommend_imputation_for_column(col, df[col], cl)
        imputation_recs[col] = rec
    profile["imputation_recommendations"] = imputation_recs

    # --- Inconsistent categories detection (only for analysis-ready categorical) ---
    inconsistencies = {}
    for col in analysis_ready_categorical:
        if col in df.columns:
            vals    = df[col].dropna().astype(str)
            stripped = vals.str.strip().str.lower()
            groups  = {}
            for orig, norm in zip(vals, stripped):
                if norm not in groups:
                    groups[norm] = set()
                groups[norm].add(orig)
            incons = {k: list(v) for k, v in groups.items() if len(v) > 1}
            if incons:
                inconsistencies[col] = incons
    profile["inconsistencies"] = inconsistencies

    # --- Numeric values stored as text detection ---
    numeric_as_text = {}
    for col in df.columns:
        s = df[col]
        if not (s.dtype == object or str(s.dtype) in ("string", "str")):
            continue
        non_null = s.dropna()
        if len(non_null) == 0:
            continue
        coerced = pd.to_numeric(non_null, errors="coerce")
        n_coercible = int(coerced.notna().sum())
        pct_coercible = round(n_coercible / len(non_null) * 100, 2)
        if pct_coercible >= 80 and n_coercible >= 3:
            numeric_as_text[col] = {
                "n_numeric_looking":   n_coercible,
                "pct_numeric_looking": pct_coercible,
                "safe_to_convert":     bool(n_coercible == len(non_null)),
                "sample_values":       non_null.head(5).tolist(),
            }
    profile["numeric_as_text"] = numeric_as_text

    # --- Profiling timestamp ---
    profile["profiled_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return profile


def generate_plain_summary(profile: dict) -> str:
    """Generate a plain-English summary of the dataset profile."""
    lines = []
    lines.append(
        f"Your dataset contains **{profile['n_rows']:,} rows** and "
        f"**{profile['n_cols']} variables** (columns)."
    )

    nc = len(profile['analysis_ready_numerical'])
    cc = len(profile['analysis_ready_categorical'])
    dc = len(profile['datetime_cols'])
    ic = len(profile.get('identifier_cols', []))

    if nc:
        lines.append(f"- **{nc} substantive numerical variable(s):** "
                     f"{', '.join(profile['analysis_ready_numerical'])}")
    if cc:
        lines.append(f"- **{cc} categorical/ordinal variable(s):** "
                     f"{', '.join(profile['analysis_ready_categorical'])}")
    if dc:
        lines.append(f"- **{dc} date/time variable(s):** "
                     f"{', '.join(profile['datetime_cols'])}")
    if ic:
        lines.append(f"- **{ic} identifier/administrative column(s) (excluded from analysis):** "
                     f"{', '.join(profile['identifier_cols'])}")

    tm  = profile['total_missing']
    tmp = profile['total_missing_pct']
    if tm == 0:
        lines.append("✅ **No missing values** detected across the entire dataset.")
    else:
        lines.append(f"⚠️ **{tm:,} missing values** detected ({tmp}% of all cells).")
        missing_cols = [c for c, v in profile['missing_counts'].items() if v > 0]
        lines.append(f"  Affected columns: {', '.join(missing_cols)}")

    dups = profile['duplicate_rows']
    if dups == 0:
        lines.append("✅ **No duplicate rows** detected.")
    else:
        lines.append(f"⚠️ **{dups} duplicate row(s)** detected.")

    incons = profile.get('inconsistencies', {})
    if incons:
        lines.append(
            f"⚠️ **Inconsistent category names** detected in: {', '.join(incons.keys())}")

    # Warn about columns needing confirmation
    confirm_needed = [
        col for col, cl in profile.get('classifications', {}).items()
        if cl.get('needs_confirmation', False)
    ]
    if confirm_needed:
        lines.append(
            f"\n⚠️ **{len(confirm_needed)} column(s) need your confirmation** of their analytical role: "
            f"{', '.join(confirm_needed)}. Go to **Variable Classification** to review."
        )

    return "\n".join(lines)
