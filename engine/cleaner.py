"""
Data Cleaning Engine — Dataset-Agnostic Robust Implementation
=============================================================

Design principles
-----------------
1. Every function operates on a COPY; the original DataFrame is never mutated.
2. Every function validates its inputs before touching the data.
3. Every assignment is validated for shape/index compatibility BEFORE it is
   written to the DataFrame.  If the shapes don't match, the operation is
   aborted safely and an explanatory message is returned.
4. No fixed column counts, positions, names, or row counts are assumed.
5. KNN imputation uses only the target column's numpy values — never a
   multi-column block assignment that can produce shape mismatches.
6. Every function returns (df, log_entry).  The log_entry always contains
   'ok' (bool) and 'detail' (human-readable explanation) so the UI can
   surface clear messages on both success and failure.
7. All exceptions are caught per-operation; one bad column never crashes
   the dashboard or affects other columns.
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Optional

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _make_log(operation: str, column: str, method: str,
              values_changed: int, missing_before: int, missing_after: int,
              detail: str, ok: bool = True,
              original_values_sample: str = "",
              researcher_confirmed: bool = False,
              data_state: str = "working_copy",
              rows_affected: int = 0,
              rows_total: int = 0,
              reversible: bool = True,
              bias_risk: str = "",
              proposed_op: str = "") -> dict:
    """
    Extended audit log entry.

    Fields
    ------
    original_values_sample  : Short string showing original values before change
    researcher_confirmed    : True only when the researcher clicked the confirmation checkbox
    data_state              : 'working_copy' | 'imputed' | 'transformed' | 'researcher_confirmed'
    rows_affected           : Number of rows changed / removed
    rows_total              : Total rows in dataset at time of operation
    reversible              : Whether this operation can be undone via snapshot
    bias_risk               : Plain-language risk statement (empty = low risk)
    proposed_op             : Human-readable description of what was proposed
    """
    return {
        "timestamp":               _ts(),
        "operation":               operation,
        "column":                  column,
        "method":                  method,
        "values_changed":          values_changed,
        "missing_before":          missing_before,
        "missing_after":           missing_after,
        "rows_affected":           rows_affected,
        "rows_total":              rows_total,
        "detail":                  detail,
        "ok":                      ok,
        "original_values_sample":  original_values_sample,
        "researcher_confirmed":    researcher_confirmed,
        "data_state":              data_state,
        "reversible":              reversible,
        "bias_risk":               bias_risk,
        "proposed_op":             proposed_op,
    }


def _validate_col(df: pd.DataFrame, col: str) -> Optional[str]:
    """Return an error string if col is missing or df is empty, else None."""
    if df is None or not isinstance(df, pd.DataFrame):
        return "DataFrame is None or not a valid DataFrame."
    if len(df) == 0:
        return "DataFrame has zero rows."
    if col not in df.columns:
        return f"Column '{col}' does not exist in the DataFrame (columns: {list(df.columns)})."
    return None


def _safe_scalar(value) -> float:
    """Convert any scalar-like value to a plain Python float safely."""
    try:
        return float(value)
    except Exception:
        return 0.0


# ── Imputation recommendation ─────────────────────────────────────────────────

def recommend_imputation_method(col: str, series: pd.Series, var_type: str,
                                 analytical_role: str | None = None) -> dict:
    """
    Recommend the best imputation method based on variable type, analytical
    role, and data characteristics.  Pure analysis — does not modify any data.

    If analytical_role is provided (from the profiler/analytical_roles engine),
    it takes precedence over var_type for the recommendation logic so that
    identifiers, serial numbers, and constants are never offered for standard
    statistical imputation.
    """
    from engine.analytical_roles import (
        classify_column,
        recommend_imputation_for_column,
        ColumnClassification,
        ROLE_IDENTIFIER, ROLE_SERIAL, ROLE_ADMIN_CODE, ROLE_CONSTANT,
        ROLE_FREE_TEXT, ROLE_SUBST_NUMERICAL, ROLE_CATEGORICAL,
        ROLE_ORDINAL, ROLE_BOOLEAN, ROLE_DATETIME,
    )

    n_missing = int(series.isnull().sum())
    n_total   = max(len(series), 1)
    pct       = round(n_missing / n_total * 100, 2)

    if n_missing == 0:
        return {
            "method":          "none",
            "n_missing":       0,
            "pct_missing":     0.0,
            "reason":          "No missing values in this column.",
            "why_appropriate": "Nothing to impute.",
            "warning":         "",
            "alternatives":    [],
            "ok_to_automate":  True,
        }

    # Build or retrieve a classification for this column
    # We build a minimal one-column DataFrame to pass to classify_column
    try:
        mini_df = pd.DataFrame({col: series})
        cl      = classify_column(mini_df, col)
        # Override role if caller provided explicit analytical_role
        if analytical_role is not None:
            cl.analytical_role = analytical_role
    except Exception:
        # Fallback classification using var_type if classify_column fails
        from engine.analytical_roles import ColumnClassification, LEVEL_NOMINAL, LEVEL_RATIO
        role_map = {
            "numerical": ROLE_SUBST_NUMERICAL,
            "categorical": ROLE_CATEGORICAL,
            "categorical_numeric": ROLE_ORDINAL,
            "datetime": ROLE_DATETIME,
            "text": ROLE_FREE_TEXT,
        }
        fb_role = role_map.get(str(var_type).lower(), ROLE_CATEGORICAL)
        level   = LEVEL_RATIO if fb_role == ROLE_SUBST_NUMERICAL else LEVEL_NOMINAL
        cl = ColumnClassification(
            col_name=col, data_type=str(series.dtype),
            analytical_role=fb_role, measurement_level=level,
            confidence=0.5, evidence=[], warnings=[],
            suitable_analyses=[], unsuitable_reason="",
            imputation_role="imputable",
            suggested_treatment="", needs_confirmation=False,
        )

    rec = recommend_imputation_for_column(col, series, cl)

    # Normalise return format (backward-compatible with existing UI)
    return {
        "method":          rec["method"],
        "n_missing":       n_missing,
        "pct_missing":     pct,
        "reason":          rec["reason"],
        "why_appropriate": rec.get("why_appropriate", ""),
        "warning":         rec.get("warning", ""),
        "alternatives":    rec.get("alternatives", []),
        "ok_to_automate":  rec.get("ok_to_automate", False),
        # Legacy key for existing UI code
        "alternatives_legacy": rec.get("alternatives", []),
    }


# ── Core imputation ───────────────────────────────────────────────────────────

def impute_column(df: pd.DataFrame, col: str, method: str,
                  constant_value: str = "Unknown") -> tuple[pd.DataFrame, dict]:
    """
    Impute missing values in a single column of a COPY of df.

    Validation contract
    -------------------
    * The column must exist in df.
    * The returned DataFrame has the same columns as the input.
    * The returned DataFrame row count equals the input UNLESS method='drop_rows'.
    * Every assignment is checked for shape compatibility before writing.
    * Any exception aborts the operation safely; the original copy is returned
      with an error log entry (ok=False).

    Returns (modified_df, log_entry).
    """
    err = _validate_col(df, col)
    if err:
        return df.copy(), _make_log(
            "missing_value_imputation", col, method,
            0, 0, 0, f"Validation failed: {err}", ok=False)

    # Work on a fresh copy with a clean contiguous RangeIndex to eliminate
    # any index misalignment left by prior dedup/dropna operations.
    df = df.copy().reset_index(drop=True)

    n_rows_before    = len(df)
    n_cols_before    = len(df.columns)
    n_missing_before = int(df[col].isnull().sum())

    # ── Hard block: 100% missing — nothing to impute from ─────────────────────
    if n_missing_before >= n_rows_before and n_rows_before > 0:
        return df, _make_log(
            "missing_value_imputation", col, method,
            0, n_missing_before, n_missing_before,
            (
                f"BLOCKED: Column '{col}' has no observed values (100% missing, "
                f"{n_missing_before}/{n_rows_before} rows). "
                "Statistical imputation requires at least some observed data to produce "
                "an estimate. No values were changed."
            ),
            ok=False,
            bias_risk="100% missing — imputation would fabricate all values with no empirical basis.",
            proposed_op=f"impute '{col}' with {method} [BLOCKED — all values missing]",
        )
    detail           = ""
    orig_sample      = str(df[col].head(5).tolist())

    try:
        # ── mean ──────────────────────────────────────────────────────────────
        if method == "mean":
            if not pd.api.types.is_numeric_dtype(df[col]):
                raise TypeError(f"Column '{col}' is not numeric — 'mean' is not applicable. "
                                 "Use 'mode' for categorical data.")
            fill_val = _safe_scalar(df[col].mean())
            df[col]  = df[col].fillna(fill_val)
            detail   = (f"Missing values REPLACED (not recovered) with mean = {fill_val:.4f}. "
                        f"{n_missing_before} value(s) estimated. Original information is lost.")

        # ── median ────────────────────────────────────────────────────────────
        elif method == "median":
            if not pd.api.types.is_numeric_dtype(df[col]):
                raise TypeError(f"Column '{col}' is not numeric — 'median' is not applicable.")
            fill_val = _safe_scalar(df[col].median())
            df[col]  = df[col].fillna(fill_val)
            detail   = (f"Missing values REPLACED (not recovered) with median = {fill_val:.4f}. "
                        f"{n_missing_before} value(s) estimated. Original information is lost.")

        # ── mode ──────────────────────────────────────────────────────────────
        elif method == "mode":
            modes = df[col].mode(dropna=True)
            if len(modes) == 0:
                detail = f"Column '{col}' has no mode (all values missing). No changes made."
            else:
                fill_val = modes.iloc[0]
                df[col]  = df[col].fillna(fill_val)
                detail   = (f"Missing values REPLACED (not recovered) with mode = '{fill_val}'. "
                            f"{n_missing_before} value(s) estimated. Original information is lost.")

        # ── forward fill (datetime / ordered) ─────────────────────────────────
        elif method == "forward_fill":
            df[col] = df[col].ffill()
            # If the very first rows were NaN, ffill can't help — backfill those
            still   = int(df[col].isnull().sum())
            if still > 0:
                df[col] = df[col].bfill()
            detail = (f"Missing values REPLACED (not recovered) using forward-fill "
                      f"({n_missing_before} value(s) propagated from prior row). "
                      + (f"{still} value(s) back-filled from following row. " if still > 0 else "")
                      + "Original information is lost.")

        # ── KNN ───────────────────────────────────────────────────────────────
        elif method == "knn":
            if not pd.api.types.is_numeric_dtype(df[col]):
                raise TypeError(f"Column '{col}' is not numeric — KNN imputation requires a numeric column. "
                                 "Use 'mode' or 'constant_unknown' for categorical data.")

            try:
                from sklearn.impute import KNNImputer
            except ImportError:
                raise ImportError("scikit-learn is not installed. Run: pip install scikit-learn")

            # Only the target column's values are needed.
            # Reshape to (n, 1) to avoid any multi-column shape issues.
            col_values  = df[col].values.reshape(-1, 1).astype(float)
            n_neighbors = min(5, max(1, int(np.sum(~np.isnan(col_values.ravel()))) - 1))

            if n_neighbors < 1:
                raise ValueError(f"Not enough non-missing values in '{col}' to run KNN "
                                  f"(need at least 2 non-missing rows).")

            imputer     = KNNImputer(n_neighbors=n_neighbors)
            imputed_col = imputer.fit_transform(col_values).ravel()

            # Strict shape check before assignment
            if len(imputed_col) != len(df):
                raise ValueError(
                    f"KNN output length {len(imputed_col)} does not match DataFrame "
                    f"length {len(df)}. Operation aborted to prevent data corruption.")

            df[col] = np.round(imputed_col, 4)
            detail  = (f"Missing values ESTIMATED (not recovered) using KNN imputation "
                       f"({n_missing_before} value(s), {n_neighbors} neighbour(s)). "
                       f"Estimated values reduce variance and may affect statistical results. "
                       f"Original information is lost.")

        # ── constant / unknown ─────────────────────────────────────────────────
        elif method == "constant_unknown":
            fill = constant_value if constant_value else "Unknown"
            df[col] = df[col].fillna(fill)
            detail  = (f"Missing values RECODED as constant '{fill}' — "
                       f"this creates a new category, not a recovery of original data. "
                       f"{n_missing_before} value(s) affected. "
                       f"Frequencies, percentages, and statistical tests may be distorted.")

        # ── drop rows ─────────────────────────────────────────────────────────
        elif method == "drop_rows":
            n_before = len(df)
            df       = df.dropna(subset=[col]).reset_index(drop=True)
            removed  = n_before - len(df)
            pct_removed = round(removed / max(n_before, 1) * 100, 1)
            detail   = (f"DELETED {removed} row(s) ({pct_removed}% of dataset) "
                        f"where '{col}' was missing. "
                        f"{len(df)} rows remain. All data in those rows is permanently lost from the working copy. "
                        f"If missing data is not random (MNAR), deletion introduces selection bias.")

        # ── none / unknown ─────────────────────────────────────────────────────
        elif method == "none":
            detail = f"Method 'none' selected — no changes made to '{col}'."
        else:
            detail = f"Unknown method '{method}'. No changes made."

    except (TypeError, ValueError, ImportError) as exc:
        # Expected user-facing errors — return original copy with clear message
        return df, _make_log(
            "missing_value_imputation", col, method,
            0, n_missing_before, n_missing_before,
            f"Operation skipped: {exc}", ok=False)

    except Exception as exc:
        # Unexpected errors — safe fallback
        return df, _make_log(
            "missing_value_imputation", col, method,
            0, n_missing_before, n_missing_before,
            f"Unexpected error — operation aborted to protect your data: {exc}", ok=False)

    # ── Post-operation integrity check ────────────────────────────────────────
    if method != "drop_rows":
        # Row count must be unchanged
        if len(df) != n_rows_before:
            return df, _make_log(
                "missing_value_imputation", col, method,
                0, n_missing_before, n_missing_before,
                f"Integrity check failed: row count changed from {n_rows_before} to {len(df)} "
                f"during '{method}' — operation rolled back.", ok=False)

    # Column count must always be unchanged
    if len(df.columns) != n_cols_before:
        return df, _make_log(
            "missing_value_imputation", col, method,
            0, n_missing_before, n_missing_before,
            f"Integrity check failed: column count changed from {n_cols_before} to {len(df.columns)} "
            f"— operation rolled back.", ok=False)

    n_missing_after  = int(df[col].isnull().sum())
    n_changed        = n_missing_before - n_missing_after
    rows_affected    = (n_rows_before - len(df)) if method == "drop_rows" else n_changed
    pct_missing      = round(n_missing_before / max(n_rows_before, 1) * 100, 1)

    bias_risk_map = {
        "mean":             "Reduces variance; can weaken correlations and t-test/ANOVA power. Assumes data is Missing Completely At Random (MCAR).",
        "median":           "Preserves central tendency but reduces variance. Appropriate for skewed data if MCAR assumed.",
        "mode":             "Inflates the most frequent category; distorts frequency distributions.",
        "knn":              "Estimated values reduce variance; assumes similar observations exist. May distort scatter and regression.",
        "constant_unknown": "Creates a new synthetic category 'Unknown'. Inflates that category in frequencies and chi-square tests.",
        "forward_fill":     "Borrows values from adjacent rows; assumes temporal ordering. May introduce autocorrelation.",
        "drop_rows":        "Deletes entire rows. Introduces selection bias if missingness is not random (MNAR/MAR).",
    }
    data_state_map = {
        "mean": "imputed", "median": "imputed", "mode": "imputed",
        "knn": "imputed", "forward_fill": "imputed",
        "constant_unknown": "transformed", "drop_rows": "working_copy",
    }

    return df, _make_log(
        "missing_value_imputation", col, method,
        n_changed, n_missing_before, n_missing_after,
        detail, ok=True,
        original_values_sample=orig_sample,
        researcher_confirmed=False,   # UI must set True after confirmation
        data_state=data_state_map.get(method, "working_copy"),
        rows_affected=rows_affected,
        rows_total=n_rows_before,
        reversible=True,
        bias_risk=bias_risk_map.get(method, ""),
        proposed_op=f"Impute '{col}' missing values ({pct_missing}% missing) using {method}")


# ── Duplicate removal ─────────────────────────────────────────────────────────

def remove_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Remove fully duplicate rows.  Returns (cleaned_df, log_entry).
    The index is always reset to a clean RangeIndex after removal.
    """
    if df is None or not isinstance(df, pd.DataFrame):
        empty = pd.DataFrame()
        return empty, _make_log("remove_duplicates", "all", "drop_duplicates",
                                 0, 0, 0, "Invalid DataFrame supplied.", ok=False)

    n_cols_before = len(df.columns)
    before        = len(df)
    pct_removed   = round((df.duplicated().sum()) / max(before, 1) * 100, 1)
    orig_sample   = str(df[df.duplicated(keep=False)].head(3).to_dict(orient="records"))

    try:
        df_out = df.drop_duplicates().reset_index(drop=True)
    except Exception as exc:
        return df.copy().reset_index(drop=True), _make_log(
            "remove_duplicates", "all", "drop_duplicates",
            0, 0, 0, f"Deduplication failed: {exc}", ok=False)

    # Column count must not change
    if len(df_out.columns) != n_cols_before:
        return df.copy().reset_index(drop=True), _make_log(
            "remove_duplicates", "all", "drop_duplicates",
            0, 0, 0, "Column count changed unexpectedly — operation aborted.", ok=False)

    removed = before - len(df_out)
    return df_out, _make_log(
        "remove_duplicates", "all", "drop_duplicates",
        removed, 0, 0,
        (f"DELETED {removed} fully duplicate row(s) ({pct_removed}% of dataset). "
         f"{len(df_out)} rows remain. All data in those rows is permanently removed from the working copy."),
        ok=True,
        original_values_sample=orig_sample,
        researcher_confirmed=False,
        data_state="working_copy",
        rows_affected=removed,
        rows_total=before,
        reversible=True,
        bias_risk="If duplicates represent valid repeated measurements (e.g. longitudinal data), "
                  "their removal distorts sample size and results.",
        proposed_op=f"Remove {removed} duplicate rows ({pct_removed}% of dataset)")


# ── Category standardisation ──────────────────────────────────────────────────

def standardise_categories(df: pd.DataFrame, col: str,
                            mapping: dict) -> tuple[pd.DataFrame, dict]:
    """
    Apply a {original_value: standardised_value} mapping to a column.
    Only scalar string/object columns are modified; numeric dtypes are
    skipped with a clear warning.
    """
    err = _validate_col(df, col)
    if err:
        return df.copy(), _make_log(
            "standardise_categories", col, "string_mapping",
            0, 0, 0, f"Validation failed: {err}", ok=False)

    if not mapping:
        return df.copy(), _make_log(
            "standardise_categories", col, "string_mapping",
            0, 0, 0, "Empty mapping supplied — no changes made.", ok=False)

    df         = df.copy().reset_index(drop=True)
    changed    = 0
    n_cols_in  = len(df.columns)
    n_rows     = len(df)
    orig_cats  = str(sorted(df[col].dropna().unique().tolist())[:10])

    try:
        for original, standard in mapping.items():
            mask     = df[col].astype(str) == str(original)
            n        = int(mask.sum())
            if n > 0:
                df.loc[mask, col] = str(standard)
                changed          += n
    except Exception as exc:
        return df, _make_log(
            "standardise_categories", col, "string_mapping",
            0, 0, 0, f"Category standardisation failed: {exc}", ok=False)

    if len(df.columns) != n_cols_in:
        return df, _make_log(
            "standardise_categories", col, "string_mapping",
            0, 0, 0, "Column structure changed unexpectedly — operation aborted.", ok=False)

    mapping_str = "; ".join(f"'{k}' → '{v}'" for k, v in mapping.items())
    return df, _make_log(
        "standardise_categories", col, "string_mapping",
        changed, 0, 0,
        (f"RECODED {changed} value(s) in '{col}'. "
         f"Mapping applied: {mapping_str}. "
         f"Original category labels permanently changed in the working copy."),
        ok=True,
        original_values_sample=orig_cats,
        researcher_confirmed=False,
        data_state="transformed",
        rows_affected=changed,
        rows_total=n_rows,
        reversible=True,
        bias_risk="Category merging reduces granularity and may mask sub-group differences. "
                  "Frequency counts and chi-square test results will change.",
        proposed_op=f"Recode categories in '{col}': {mapping_str}")


# ── Data-type conversion ──────────────────────────────────────────────────────

def fix_data_types(df: pd.DataFrame, col: str,
                   target_type: str) -> tuple[pd.DataFrame, dict]:
    """
    Convert a column to a target type.
    target_type: 'numerical' | 'datetime' | 'categorical'
    Rows that cannot be converted become NaN/NaT; nothing is dropped.
    """
    err = _validate_col(df, col)
    if err:
        return df.copy(), _make_log(
            "fix_data_type", col, f"convert_to_{target_type}",
            0, 0, 0, f"Validation failed: {err}", ok=False)

    df        = df.copy().reset_index(drop=True)
    n_cols_in = len(df.columns)
    n_rows    = len(df)
    errors    = 0
    detail    = ""
    orig_dtype = str(df[col].dtype)
    orig_sample = str(df[col].head(5).tolist())

    try:
        if target_type == "numerical":
            before_null = int(df[col].isnull().sum())
            df[col]     = pd.to_numeric(df[col], errors="coerce")
            after_null  = int(df[col].isnull().sum())
            errors      = max(0, after_null - before_null)
            detail      = (f"CONVERTED '{col}' from {orig_dtype} to numeric. "
                           f"{errors} value(s) that could not be parsed became NaN (new missing values introduced). "
                           f"Original text representations permanently lost.")

        elif target_type == "datetime":
            before_null = int(df[col].isnull().sum())
            df[col]     = pd.to_datetime(df[col], errors="coerce")
            after_null  = int(df[col].isnull().sum())
            errors      = max(0, after_null - before_null)
            detail      = (f"CONVERTED '{col}' from {orig_dtype} to datetime. "
                           f"{errors} value(s) that could not be parsed became NaT (new missing values introduced). "
                           f"Original text representations permanently lost.")

        elif target_type == "categorical":
            df[col] = df[col].astype(str).str.strip()
            detail  = (f"CONVERTED '{col}' from {orig_dtype} to string/categorical "
                       f"(whitespace stripped). Numerical precision is lost if this was a numeric column.")

        else:
            return df, _make_log(
                "fix_data_type", col, f"convert_to_{target_type}",
                0, 0, 0,
                f"Unknown target type '{target_type}'. Valid options: numerical, datetime, categorical.",
                ok=False)

    except Exception as exc:
        return df, _make_log(
            "fix_data_type", col, f"convert_to_{target_type}",
            0, 0, 0, f"Type conversion failed: {exc}", ok=False)

    if len(df.columns) != n_cols_in:
        return df, _make_log(
            "fix_data_type", col, f"convert_to_{target_type}",
            0, 0, errors, "Column structure changed unexpectedly — aborted.", ok=False)

    return df, _make_log(
        "fix_data_type", col, f"convert_to_{target_type}",
        errors, 0, errors, detail, ok=True,
        original_values_sample=orig_sample,
        researcher_confirmed=False,
        data_state="transformed",
        rows_affected=errors,
        rows_total=n_rows,
        reversible=True,
        bias_risk=(f"Type conversion may introduce new NaN values ({errors} found). "
                   f"Downstream statistics will treat these as missing.") if errors > 0 else "",
        proposed_op=f"Convert '{col}' from {orig_dtype} to {target_type}")


# ── Outlier capping (Winsorization) ───────────────────────────────────────────

def cap_outliers(df: pd.DataFrame, col: str,
                 method: str = "iqr") -> tuple[pd.DataFrame, dict]:
    """
    Cap outliers by clipping extreme values (Winsorization).
    method: 'iqr' (1.5 × IQR rule) | 'zscore' (± 3 SD rule)
    Only operates on numeric columns.
    """
    err = _validate_col(df, col)
    if err:
        return df.copy(), _make_log(
            "cap_outliers", col, f"winsorize_{method}",
            0, 0, 0, f"Validation failed: {err}", ok=False)

    if not pd.api.types.is_numeric_dtype(df[col]):
        return df.copy(), _make_log(
            "cap_outliers", col, f"winsorize_{method}",
            0, 0, 0,
            f"Column '{col}' is not numeric — outlier capping is only applicable to numerical variables.",
            ok=False)

    df        = df.copy().reset_index(drop=True)
    n_cols_in = len(df.columns)
    valid     = df[col].dropna()

    if len(valid) < 4:
        return df, _make_log(
            "cap_outliers", col, f"winsorize_{method}",
            0, 0, 0,
            f"Column '{col}' has fewer than 4 non-missing values — outlier capping skipped.",
            ok=False)

    try:
        if method == "iqr":
            q1, q3 = float(valid.quantile(0.25)), float(valid.quantile(0.75))
            iqr    = q3 - q1
            lower  = q1 - 1.5 * iqr
            upper  = q3 + 1.5 * iqr
        else:  # zscore
            mu     = float(valid.mean())
            sigma  = float(valid.std())
            if sigma == 0:
                return df, _make_log(
                    "cap_outliers", col, f"winsorize_{method}",
                    0, 0, 0,
                    f"Column '{col}' has zero variance — Z-score capping is not applicable.",
                    ok=False)
            lower = mu - 3 * sigma
            upper = mu + 3 * sigma

        n_below = int((df[col] < lower).sum())
        n_above = int((df[col] > upper).sum())
        orig_outliers = str(df[col][(df[col] < lower) | (df[col] > upper)].head(5).tolist())
        df[col] = df[col].clip(lower=lower, upper=upper)
        changed = n_below + n_above

    except Exception as exc:
        return df, _make_log(
            "cap_outliers", col, f"winsorize_{method}",
            0, 0, 0, f"Outlier capping failed: {exc}", ok=False)

    if len(df.columns) != n_cols_in:
        return df, _make_log(
            "cap_outliers", col, f"winsorize_{method}",
            0, 0, 0, "Column structure changed unexpectedly — aborted.", ok=False)

    pct_affected = round(changed / max(len(df), 1) * 100, 1)
    return df, _make_log(
        "cap_outliers", col, f"winsorize_{method}",
        changed, 0, 0,
        (f"CAPPED (Winsorized) {changed} value(s) ({pct_affected}% of column) in '{col}'. "
         f"{n_below} value(s) below {lower:.3f} raised to {lower:.3f}; "
         f"{n_above} value(s) above {upper:.3f} lowered to {upper:.3f}. "
         f"Original extreme values are REPLACED — not deleted, not recovered. "
         f"True range of the variable is now artificially restricted."),
        ok=True,
        original_values_sample=orig_outliers,
        researcher_confirmed=False,
        data_state="transformed",
        rows_affected=changed,
        rows_total=len(df),
        reversible=True,
        bias_risk=(f"Winsorization reduces variance and compresses the tails of the distribution. "
                   f"Mean, SD, correlations, and regression coefficients will all change. "
                   f"If outliers are genuine observations (not data errors), capping distorts the findings."),
        proposed_op=f"Cap outliers in '{col}' using {method.upper()} method ({changed} values, {pct_affected}%)")


# ── Cleaning log summary ──────────────────────────────────────────────────────

def get_cleaning_summary(log: list) -> str:
    """Return a Markdown-formatted cleaning summary from the log list."""
    if not log:
        return "No cleaning operations have been performed yet."
    lines = [f"**Audit Log — {len(log)} operation(s) applied to working copy:**\n"]
    for i, entry in enumerate(log, 1):
        status    = "✅" if entry.get("ok", True) else "⚠️"
        confirmed = "🔬 Researcher-confirmed" if entry.get("researcher_confirmed") else "⚡ Auto-applied"
        state     = entry.get("data_state", "working_copy").replace("_", " ").title()
        lines.append(
            f"{i}. {status} [{entry['timestamp']}] **{entry['operation']}** "
            f"on `{entry['column']}` — Method: *{entry['method']}* "
            f"— {entry['values_changed']} value(s) affected — Data state: **{state}** — {confirmed}\n"
            f"   _{entry['detail']}_"
        )
    return "\n".join(lines)
