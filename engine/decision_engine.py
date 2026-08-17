"""
decision_engine.py — 16-Step Statistical Decision Engine
=========================================================
Self-contained reasoning engine for a research analytics dashboard.

Given a pair of variables (described by their analytical role, measurement
level, missingness, and sample size) the engine works through 16 explicit
decision steps and returns a fully-annotated ``DecisionResult`` that tells
the researcher:

  * *which* method to use and *why* (referencing actual variable names / n)
  * all relevant assumption checks (with data if a DataFrame is supplied)
  * alternatives, effect-size notes, CI notes, multiple-testing corrections,
    interpretation guidance, and methodological limitations

Exports
-------
DecisionContext
    Input descriptor built from the dataset profile.

DecisionResult
    Output of the engine: method recommendation + all supporting rationale.

run_decision_engine(ctx, df=None, n_tests_in_session=1) -> DecisionResult
    Main entry point.

build_context_from_profile(var_a, var_b, profile, df) -> DecisionContext
    Helper that populates a DecisionContext from profiler.py output.

interpret_research_question(question, profile) -> dict
    Heuristic NLP helper that classifies a free-text research question.

Dependencies
------------
Standard library + optional scipy (guarded with try/except).
No UI code.  No column names are ever hard-coded.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

# Optional heavy-weight dependency -------------------------------------------
try:
    from scipy import stats as _scipy_stats
    _SCIPY_AVAILABLE = True
except ImportError:  # pragma: no cover
    _SCIPY_AVAILABLE = False

# Local knowledge base -------------------------------------------------------
from engine.stat_methods import METHOD_KB, get_candidate_methods

# Blocked-role set (mirrors stat_methods._BLOCKED_ROLES)
_BLOCKED_ROLES: frozenset[str] = frozenset({
    "identifier",
    "serial_number",
    "administrative_code",
    "constant",
    "free_text",
    "unknown",
})

# Effect-size mapping: method_key → recommended measure(s)
_EFFECT_SIZE_MAP: dict[str, str] = {
    "independent_ttest":        "Cohen's d (d ≈ 0.2 small, 0.5 medium, 0.8 large)",
    "welch_ttest":              "Cohen's d / Hedges' g",
    "paired_ttest":             "Cohen's d on difference scores",
    "mann_whitney_u":           "Rank-biserial correlation r (r ≈ 0.1 small, 0.3 medium, 0.5 large)",
    "wilcoxon_signed_rank":     "Matched-pairs rank-biserial r",
    "one_way_anova":            "η² eta-squared (η² ≈ 0.01 small, 0.06 medium, 0.14 large)",
    "welch_anova":              "η² / ω² (omega-squared)",
    "kruskal_wallis":           "Epsilon-squared η²_H",
    "friedman":                 "Kendall's W (concordance coefficient)",
    "pearson_correlation":      "Pearson r / r² (r ≈ 0.1 small, 0.3 medium, 0.5 large)",
    "spearman_correlation":     "Spearman ρ (same benchmarks as Pearson r)",
    "kendall_tau":              "Kendall's τ (τ ≈ 0.1 small, 0.2 medium, 0.3 large)",
    "point_biserial":           "r_pb (point-biserial r; equivalent to Pearson r)",
    "chi_square_independence":  "Cramér's V (V ≈ 0.10 small, 0.30 medium, 0.50 large)",
    "chi_square_goodness_of_fit": "Cohen's w",
    "fishers_exact":            "Odds Ratio with 95% CI",
    "cramers_v":                "Cramér's V (0 = no association, 1 = perfect association)",
    "odds_ratio":               "Odds Ratio (OR = 1: no effect; OR > 1: increased odds)",
    "simple_linear_regression": "R² / Cohen's f² (f² ≈ 0.02 small, 0.15 medium, 0.35 large)",
    "multiple_linear_regression": "Adjusted R² / semi-partial r²",
    "logistic_regression":      "Nagelkerke R² / AUC-ROC",
    "ordinal_logistic":         "Nagelkerke R² / McFadden's R²",
    "poisson_regression":       "Incidence Rate Ratio (IRR)",
}

# CI availability hint -------------------------------------------------------
def _ci_note(method_key: str) -> str:
    if method_key in METHOD_KB:
        m = METHOD_KB[method_key]
        if m.ci_available:
            return "95% confidence interval is routinely available for this method."
        else:
            return (
                "Confidence interval is not directly available for this method; "
                "bootstrap resampling is recommended to estimate a CI."
            )
    return "CI availability unknown for this method."


# ============================================================================
# Dataclasses
# ============================================================================

@dataclass
class DecisionContext:
    """All information needed by the decision engine for one variable pair.

    Attributes
    ----------
    var_a, var_b : str
        Column names of the two variables under analysis.
    role_a, role_b : str
        Analytical roles (e.g. ``"substantive_numerical"``, ``"categorical"``).
    level_a, level_b : str
        Measurement levels (e.g. ``"ratio"``, ``"nominal"``).
    missing_pct_a, missing_pct_b : float
        Fraction of missing values (0.0–1.0).
    unique_a, unique_b : int
        Number of distinct values observed (non-missing).
    n_valid_a, n_valid_b : int
        Count of non-missing observations in each variable.
    warnings_a, warnings_b : list[str]
        Data-quality warnings from the profiler for each variable.
    n_total : int
        Total rows in the dataset.
    n_valid_pairs : int
        Rows where *both* variables are non-missing.
    n_missing_pairs : int
        Rows where at least one variable is missing.
    missing_pairs_pct : float
        ``n_missing_pairs / n_total`` (0.0–1.0).
    n_groups : int
        Distinct values of the grouping/categorical variable (0 if not applicable).
    paired : bool
        Researcher-specified paired design flag.
    research_question : str
        Free-text research question (may be empty).
    """

    # Variable A (required fields first)
    var_a: str
    role_a: str
    level_a: str
    missing_pct_a: float
    unique_a: int
    n_valid_a: int

    # Variable B (required fields)
    var_b: str
    role_b: str
    level_b: str
    missing_pct_b: float
    unique_b: int
    n_valid_b: int

    # All fields with defaults come last
    warnings_a: list[str] = field(default_factory=list)
    warnings_b: list[str] = field(default_factory=list)

    # Dataset-level
    n_total: int = 0
    n_valid_pairs: int = 0
    n_missing_pairs: int = 0
    missing_pairs_pct: float = 0.0

    # Design
    n_groups: int = 0
    paired: bool = False
    research_question: str = ""


@dataclass
class DecisionResult:
    """Full output of the 16-step decision engine.

    Attributes
    ----------
    context : DecisionContext
        The input context that produced this result.
    step_log : list[str]
        Human-readable log of every decision step taken.
    blocked : bool
        True when analysis cannot proceed.
    block_reason : str
        Plain-language explanation of why analysis is blocked.
    primary_method : str
        Key into ``stat_methods.METHOD_KB`` for the recommended method,
        or ``""`` if blocked.
    primary_rationale : str
        Full plain-language explanation that references actual variable names
        and sample sizes.
    primary_assumptions : list[str]
        Assumptions that must hold for the primary method to be valid.
    alternatives : list[dict]
        Each entry has keys: ``method_key``, ``name``, ``why_alternative``,
        ``when_prefer``.
    data_quality_warnings : list[str]
        Collated data-quality warnings from both variables.
    assumption_check_results : list[dict]
        Each entry: ``{assumption, status, detail, severity, implication,
        suggestion}``.
    effect_size_note : str
        Which effect-size measure to report alongside the primary method.
    ci_note : str
        Whether/how a confidence interval is available.
    multiple_testing_note : str
        Bonferroni guidance based on ``n_tests_in_session``.
    interpretation_guidance : str
        Plain-language guide to interpreting results.
    limitations : list[str]
        Known limitations of the recommended method in this specific context.
    """

    context: DecisionContext
    step_log: list[str] = field(default_factory=list)

    blocked: bool = False
    block_reason: str = ""

    primary_method: str = ""
    primary_rationale: str = ""
    primary_assumptions: list[str] = field(default_factory=list)

    alternatives: list[dict] = field(default_factory=list)
    data_quality_warnings: list[str] = field(default_factory=list)
    assumption_check_results: list[dict] = field(default_factory=list)

    effect_size_note: str = ""
    ci_note: str = ""
    multiple_testing_note: str = ""
    interpretation_guidance: str = ""
    limitations: list[str] = field(default_factory=list)


# ============================================================================
# Internal helpers
# ============================================================================

def _pct_label(frac: float) -> str:
    """Format a 0–1 fraction as a percentage string."""
    return f"{frac * 100:.1f}%"


def _assumption_entry(
    assumption: str,
    status: str,
    detail: str,
    severity: str,
    implication: str,
    suggestion: str,
) -> dict:
    """Build a standardised assumption-check dict."""
    return {
        "assumption": assumption,
        "status": status,            # "ok" | "warn" | "fail"
        "detail": detail,
        "severity": severity,        # "low" | "medium" | "high"
        "implication": implication,
        "suggestion": suggestion,
    }


def _check_assumptions_with_data(
    ctx: DecisionContext,
    df,
    primary_method: str,
) -> list[dict]:
    """Run data-driven assumption checks when a DataFrame is available.

    Checks performed (when variables are numeric):
      * Skewness: |skew| > 1.0 → warn; > 2.0 → fail
      * IQR outlier count: any outliers → note severity by proportion
      * Shapiro-Wilk normality (only when n < 50 and scipy available)

    Returns a list of assumption-check dicts.
    """
    results: list[dict] = []

    num_roles = {"substantive_numerical"}
    num_levels = {"interval", "ratio"}

    a_is_numeric = ctx.role_a in num_roles or ctx.level_a in num_levels
    b_is_numeric = ctx.role_b in num_roles or ctx.level_b in num_levels

    for var_name, is_numeric, role, level in (
        (ctx.var_a, a_is_numeric, ctx.role_a, ctx.level_a),
        (ctx.var_b, b_is_numeric, ctx.role_b, ctx.level_b),
    ):
        if not is_numeric:
            continue
        if var_name not in df.columns:
            continue

        series = df[var_name].dropna()
        n = len(series)
        if n < 2:
            continue

        # ── Skewness ────────────────────────────────────────────────────────
        try:
            skew = float(series.skew())
            abs_skew = abs(skew)
            if abs_skew <= 1.0:
                results.append(_assumption_entry(
                    assumption=f"Normality / symmetry of '{var_name}'",
                    status="ok",
                    detail=f"Skewness = {skew:.3f} (|skew| ≤ 1.0)",
                    severity="low",
                    implication="Distribution appears approximately symmetric.",
                    suggestion="Parametric methods are likely appropriate.",
                ))
            elif abs_skew <= 2.0:
                results.append(_assumption_entry(
                    assumption=f"Normality / symmetry of '{var_name}'",
                    status="warn",
                    detail=f"Skewness = {skew:.3f} (1.0 < |skew| ≤ 2.0)",
                    severity="medium",
                    implication="Moderate skew may affect parametric test validity, especially in small samples.",
                    suggestion=(
                        "Consider a non-parametric alternative or apply a log/sqrt "
                        "transformation if the skew is theoretically justifiable."
                    ),
                ))
            else:
                results.append(_assumption_entry(
                    assumption=f"Normality / symmetry of '{var_name}'",
                    status="fail",
                    detail=f"Skewness = {skew:.3f} (|skew| > 2.0 — severe skew)",
                    severity="high",
                    implication="Severe skew undermines parametric assumptions; p-values may be unreliable.",
                    suggestion=(
                        "Use a non-parametric method. "
                        "If a parametric approach is required, transform the variable first "
                        "and re-verify assumptions."
                    ),
                ))
        except Exception:
            pass

        # ── IQR Outliers ────────────────────────────────────────────────────
        try:
            q1 = float(series.quantile(0.25))
            q3 = float(series.quantile(0.75))
            iqr = q3 - q1
            if iqr == 0:
                results.append(_assumption_entry(
                    assumption=f"No extreme outliers in '{var_name}'",
                    status="warn",
                    detail="IQR = 0 — data has very low spread; outlier detection is not meaningful.",
                    severity="medium",
                    implication="Near-constant values may indicate a data-quality problem or mis-classified variable.",
                    suggestion="Review the variable's role classification and check for data-entry errors.",
                ))
            else:
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                n_out = int(((series < lower) | (series > upper)).sum())
                pct_out = n_out / n if n > 0 else 0.0

                if n_out == 0:
                    results.append(_assumption_entry(
                        assumption=f"No extreme outliers in '{var_name}'",
                        status="ok",
                        detail="No IQR outliers detected.",
                        severity="low",
                        implication="Outliers are unlikely to distort results.",
                        suggestion="Proceed with the recommended method.",
                    ))
                elif pct_out < 0.05:
                    results.append(_assumption_entry(
                        assumption=f"No extreme outliers in '{var_name}'",
                        status="warn",
                        detail=(
                            f"{n_out} IQR outlier(s) detected "
                            f"({_pct_label(pct_out)} of observations)."
                        ),
                        severity="medium",
                        implication="A small number of outliers may inflate variance and affect means.",
                        suggestion=(
                            "Investigate outliers. Report results with and without them "
                            "to assess sensitivity."
                        ),
                    ))
                else:
                    results.append(_assumption_entry(
                        assumption=f"No extreme outliers in '{var_name}'",
                        status="fail",
                        detail=(
                            f"{n_out} IQR outlier(s) detected "
                            f"({_pct_label(pct_out)} of observations ≥ 5%)."
                        ),
                        severity="high",
                        implication=(
                            "High outlier proportion seriously compromises parametric tests "
                            "and may indicate data-quality issues."
                        ),
                        suggestion=(
                            "Consider robust methods or non-parametric alternatives. "
                            "Investigate whether outliers are legitimate values or errors."
                        ),
                    ))
        except Exception:
            pass

        # ── Shapiro-Wilk (n < 50, scipy available) ──────────────────────────
        if n < 50 and _SCIPY_AVAILABLE:
            try:
                stat, p_val = _scipy_stats.shapiro(series)
                if p_val >= 0.05:
                    results.append(_assumption_entry(
                        assumption=f"Normality of '{var_name}' (Shapiro-Wilk)",
                        status="ok",
                        detail=(
                            f"W = {stat:.4f}, p = {p_val:.4f} "
                            "(cannot reject normality at α = 0.05)"
                        ),
                        severity="low",
                        implication="Normality assumption appears tenable.",
                        suggestion="Parametric methods are appropriate for this variable.",
                    ))
                else:
                    sev = "high" if p_val < 0.01 else "medium"
                    results.append(_assumption_entry(
                        assumption=f"Normality of '{var_name}' (Shapiro-Wilk)",
                        status="warn" if p_val >= 0.01 else "fail",
                        detail=(
                            f"W = {stat:.4f}, p = {p_val:.4f} "
                            f"(normality rejected at α = {'0.01' if p_val < 0.01 else '0.05'})"
                        ),
                        severity=sev,
                        implication=(
                            "The distribution departs significantly from normality. "
                            "Parametric tests may produce inaccurate p-values, "
                            "especially with small samples."
                        ),
                        suggestion=(
                            "Consider a non-parametric alternative "
                            f"(e.g., {METHOD_KB.get(primary_method, None) and METHOD_KB[primary_method].nonparametric_alternative or 'see alternatives'})."
                        ),
                    ))
            except Exception:
                pass
        elif n >= 50 and not _SCIPY_AVAILABLE:
            results.append(_assumption_entry(
                assumption=f"Normality of '{var_name}' (formal test)",
                status="warn",
                detail="scipy not available — Shapiro-Wilk test could not be run.",
                severity="low",
                implication="Normality assumption could not be formally evaluated.",
                suggestion="Install scipy to enable the Shapiro-Wilk test, or inspect a Q-Q plot manually.",
            ))

    return results


def _check_assumptions_without_data() -> list[dict]:
    """Return a placeholder assumption-check list when no DataFrame is provided."""
    return [
        _assumption_entry(
            assumption="Normality / distribution shape",
            status="warn",
            detail="Cannot check without data — no DataFrame was supplied.",
            severity="low",
            implication="Assumption status is unknown.",
            suggestion=(
                "Pass the DataFrame to run_decision_engine() to enable automated "
                "skewness, outlier, and Shapiro-Wilk checks."
            ),
        )
    ]


def _build_multiple_testing_note(n_tests: int) -> str:
    """Generate a Bonferroni-correction note for n_tests analyses."""
    if n_tests < 1:
        n_tests = 1
    # P(≥1 false positive) = 1 - (1 - 0.05)^N
    p_fp = 1.0 - (0.95 ** n_tests)
    bonferroni_alpha = 0.05 / n_tests
    return (
        f"You have run {n_tests} analysis/analyses in this session. "
        f"With {n_tests} test(s) at α = 0.05, the probability of at least one "
        f"false positive is approximately {p_fp * 100:.1f}%. "
        f"Consider Bonferroni correction (α* = 0.05 / {n_tests} = {bonferroni_alpha:.4f})."
    )


def _outcome_predictor_roles(
    ctx: DecisionContext,
) -> tuple[str, str, str, str]:
    """Determine which variable is the outcome and which is the predictor.

    Returns (outcome_var, predictor_var, outcome_role_key, predictor_role_key).
    Heuristic: numerical → outcome; categorical → grouping/predictor.
    If both are the same type, var_a is treated as outcome.
    """
    num_roles = {"substantive_numerical"}
    num_levels = {"interval", "ratio"}
    cat_roles = {"categorical", "boolean", "ordinal"}
    cat_levels = {"nominal", "ordinal"}

    a_is_num = ctx.role_a in num_roles or ctx.level_a in num_levels
    b_is_num = ctx.role_b in num_roles or ctx.level_b in num_levels
    a_is_cat = ctx.role_a in cat_roles or ctx.level_a in cat_levels
    b_is_cat = ctx.role_b in cat_roles or ctx.level_b in cat_levels

    if a_is_num and b_is_cat:
        return ctx.var_a, ctx.var_b, ctx.role_a, ctx.role_b
    if b_is_num and a_is_cat:
        return ctx.var_b, ctx.var_a, ctx.role_b, ctx.role_a
    # Both same type — var_a is outcome by default
    return ctx.var_a, ctx.var_b, ctx.role_a, ctx.role_b


# ============================================================================
# Main engine
# ============================================================================

def run_decision_engine(
    ctx: DecisionContext,
    df=None,
    n_tests_in_session: int = 1,
) -> DecisionResult:
    """Run the 16-step statistical decision engine.

    Parameters
    ----------
    ctx : DecisionContext
        All contextual information about the two variables.
    df : pandas.DataFrame or None
        Optional raw data.  When provided, enables automated assumption checks
        (skewness, outliers, Shapiro-Wilk).
    n_tests_in_session : int
        Number of analyses run in the current session (used for the
        multiple-testing note).  Defaults to 1.

    Returns
    -------
    DecisionResult
        Fully-annotated recommendation with step log, primary method,
        rationale, alternatives, assumption checks, and guidance notes.
    """
    log: list[str] = []
    result = DecisionResult(context=ctx)

    # ── STEP 1: Log research question ───────────────────────────────────────
    if ctx.research_question.strip():
        log.append(
            f"STEP 1 -- Research question provided: \"{ctx.research_question.strip()}\""
        )
    else:
        log.append("STEP 1 — No research question supplied.")

    # ── STEP 2: Identify outcome variable ───────────────────────────────────
    outcome_var, predictor_var, outcome_role, predictor_role = _outcome_predictor_roles(ctx)
    log.append(
        f"STEP 2 — Outcome / dependent variable identified as '{outcome_var}' "
        f"(role: {outcome_role})."
    )

    # ── STEP 3: Identify predictor / grouping variable ──────────────────────
    log.append(
        f"STEP 3 — Predictor / grouping / independent variable identified as "
        f"'{predictor_var}' (role: {predictor_role})."
    )

    # ── STEP 4: Determine measurement levels ────────────────────────────────
    log.append(
        f"STEP 4 — Measurement levels: '{ctx.var_a}' = {ctx.level_a}, "
        f"'{ctx.var_b}' = {ctx.level_b}."
    )

    # ── STEP 5: Determine analytical roles ──────────────────────────────────
    log.append(
        f"STEP 5 — Analytical roles: '{ctx.var_a}' = {ctx.role_a}, "
        f"'{ctx.var_b}' = {ctx.role_b}."
    )

    # ── STEP 6: Check n_groups for categorical variable ──────────────────────
    n_groups_effective = ctx.n_groups
    if n_groups_effective > 0:
        log.append(
            f"STEP 6 — Grouping variable has {n_groups_effective} unique "
            f"group(s)/categories."
        )
    else:
        log.append(
            "STEP 6 — n_groups = 0 (no categorical grouping variable, or not applicable)."
        )

    # ── STEP 7: Check paired / independent design ────────────────────────────
    if ctx.paired:
        log.append("STEP 7 — Design: PAIRED (repeated-measures or matched pairs).")
    else:
        log.append("STEP 7 — Design: INDEPENDENT groups.")

    # ── STEP 8: Total observations ───────────────────────────────────────────
    log.append(f"STEP 8 — Total observations (n_total): {ctx.n_total:,}.")

    # ── STEP 9: Valid paired observations ────────────────────────────────────
    log.append(
        f"STEP 9 — Valid paired observations (both variables non-missing): "
        f"{ctx.n_valid_pairs:,}."
    )

    # ── STEP 10: Missing pairs ───────────────────────────────────────────────
    log.append(
        f"STEP 10 — Missing pairs: {ctx.n_missing_pairs:,} "
        f"({_pct_label(ctx.missing_pairs_pct)} of total). "
        f"Variable-level missingness — '{ctx.var_a}': {_pct_label(ctx.missing_pct_a)}, "
        f"'{ctx.var_b}': {_pct_label(ctx.missing_pct_b)}."
    )

    # ── STEP 11: Variation / constant check ─────────────────────────────────
    constant_a = ctx.unique_a <= 1
    constant_b = ctx.unique_b <= 1
    if constant_a:
        log.append(
            f"STEP 11 — WARNING: '{ctx.var_a}' appears to be constant "
            f"(unique values = {ctx.unique_a}) — no variation to analyse."
        )
    if constant_b:
        log.append(
            f"STEP 11 — WARNING: '{ctx.var_b}' appears to be constant "
            f"(unique values = {ctx.unique_b}) — no variation to analyse."
        )
    if not constant_a and not constant_b:
        log.append(
            f"STEP 11 — Both variables show variation "
            f"(unique_a = {ctx.unique_a}, unique_b = {ctx.unique_b})."
        )

    # ── Block checks (collected before STEP 12) ─────────────────────────────

    # Blocked role
    for var_name, role in ((ctx.var_a, ctx.role_a), (ctx.var_b, ctx.role_b)):
        if role in _BLOCKED_ROLES:
            block_msg = (
                f"'{var_name}' has role '{role}', which cannot be used in "
                "statistical analysis (blocked roles: identifier, serial_number, "
                "administrative_code, constant, free_text, unknown)."
            )
            log.append(f"STEP 11 — BLOCK: {block_msg}")
            result.step_log = log
            result.blocked = True
            result.block_reason = block_msg
            result.data_quality_warnings = ctx.warnings_a + ctx.warnings_b
            result.multiple_testing_note = _build_multiple_testing_note(n_tests_in_session)
            return result

    # 100% missing
    if ctx.missing_pct_a >= 1.0:
        block_msg = f"'{ctx.var_a}' is 100% missing — no data to analyse."
        log.append(f"STEP 10 — BLOCK: {block_msg}")
        result.step_log = log
        result.blocked = True
        result.block_reason = block_msg
        result.data_quality_warnings = ctx.warnings_a + ctx.warnings_b
        result.multiple_testing_note = _build_multiple_testing_note(n_tests_in_session)
        return result

    if ctx.missing_pct_b >= 1.0:
        block_msg = f"'{ctx.var_b}' is 100% missing — no data to analyse."
        log.append(f"STEP 10 — BLOCK: {block_msg}")
        result.step_log = log
        result.blocked = True
        result.block_reason = block_msg
        result.data_quality_warnings = ctx.warnings_a + ctx.warnings_b
        result.multiple_testing_note = _build_multiple_testing_note(n_tests_in_session)
        return result

    # Constant variable
    if constant_a:
        block_msg = (
            f"'{ctx.var_a}' has no variation (unique values = {ctx.unique_a}) — "
            "statistical analysis is not possible."
        )
        log.append(f"STEP 11 — BLOCK: {block_msg}")
        result.step_log = log
        result.blocked = True
        result.block_reason = block_msg
        result.data_quality_warnings = ctx.warnings_a + ctx.warnings_b
        result.multiple_testing_note = _build_multiple_testing_note(n_tests_in_session)
        return result

    if constant_b:
        block_msg = (
            f"'{ctx.var_b}' has no variation (unique values = {ctx.unique_b}) — "
            "statistical analysis is not possible."
        )
        log.append(f"STEP 11 — BLOCK: {block_msg}")
        result.step_log = log
        result.blocked = True
        result.block_reason = block_msg
        result.data_quality_warnings = ctx.warnings_a + ctx.warnings_b
        result.multiple_testing_note = _build_multiple_testing_note(n_tests_in_session)
        return result

    # Insufficient valid pairs
    if ctx.n_valid_pairs < 5:
        block_msg = (
            f"Insufficient valid paired observations: "
            f"n_valid_pairs = {ctx.n_valid_pairs} < 5."
        )
        log.append(f"STEP 9 — BLOCK: {block_msg}")
        result.step_log = log
        result.blocked = True
        result.block_reason = "Insufficient valid observations"
        result.data_quality_warnings = ctx.warnings_a + ctx.warnings_b
        result.multiple_testing_note = _build_multiple_testing_note(n_tests_in_session)
        return result

    # ── STEP 12: Assumption checks ───────────────────────────────────────────
    if df is not None:
        assumption_checks = _check_assumptions_with_data(ctx, df, "")
        log.append(
            f"STEP 12 — Assumption checks run on supplied DataFrame "
            f"({len(assumption_checks)} check(s) performed)."
        )
    else:
        assumption_checks = _check_assumptions_without_data()
        log.append(
            "STEP 12 — No DataFrame supplied; assumption checks deferred. "
            "Pass df= to enable automated skewness, outlier, and normality checks."
        )

    # ── STEP 13: Query candidate methods ────────────────────────────────────
    # missing_pct args: stat_methods expects 0.0–1.0 fractions (not percentages)
    candidates = get_candidate_methods(
        role_a=ctx.role_a,
        level_a=ctx.level_a,
        role_b=ctx.role_b,
        level_b=ctx.level_b,
        n_groups=ctx.n_groups if ctx.n_groups > 0 else None,
        paired=ctx.paired if ctx.paired else None,
        n_obs=ctx.n_valid_pairs,
        n_valid_a=ctx.n_valid_a,
        n_valid_b=ctx.n_valid_b,
        missing_pct_a=ctx.missing_pct_a,
        missing_pct_b=ctx.missing_pct_b,
    )
    log.append(
        f"STEP 13 — get_candidate_methods() returned {len(candidates)} candidate(s)."
    )

    # Detect engine-level block from stat_methods
    if candidates and candidates[0][0] in ("__blocked__", "__insufficient_n__"):
        block_msg = candidates[0][3][0] if candidates[0][3] else "Blocked by stat_methods engine."
        log.append(f"STEP 13 — BLOCK from stat_methods: {block_msg}")
        result.step_log = log
        result.blocked = True
        result.block_reason = block_msg
        result.data_quality_warnings = ctx.warnings_a + ctx.warnings_b
        result.multiple_testing_note = _build_multiple_testing_note(n_tests_in_session)
        return result

    # ── STEP 14: Rank methods ────────────────────────────────────────────────
    # get_candidate_methods() already returns a sorted list (fewest warnings first,
    # then highest min_n). We use its order directly.
    if not candidates:
        block_msg = (
            "No suitable statistical method found for the given role/level combination."
        )
        log.append(f"STEP 14 — BLOCK: {block_msg}")
        result.step_log = log
        result.blocked = True
        result.block_reason = block_msg
        result.data_quality_warnings = ctx.warnings_a + ctx.warnings_b
        result.multiple_testing_note = _build_multiple_testing_note(n_tests_in_session)
        return result

    primary_key, primary_sm, primary_elig, primary_warns = candidates[0]
    log.append(
        f"STEP 14 — Primary method ranked: '{primary_sm.name}' (key: {primary_key}). "
        f"Eligibility: {'; '.join(primary_elig) if primary_elig else 'see rationale'}."
    )

    # ── STEP 15: Build DecisionResult ───────────────────────────────────────
    # -- Primary rationale (variable-name aware) --
    design_clause = "paired design" if ctx.paired else "independent groups"
    n_clause = (
        f"n = {ctx.n_valid_pairs:,} valid paired observations "
        f"(out of {ctx.n_total:,} total; {_pct_label(ctx.missing_pairs_pct)} missing)"
    )

    rationale_parts = [
        f"The recommended method is **{primary_sm.name}** ({primary_sm.purpose}).",
        f"",
        f"Variable overview:",
        f"  • '{ctx.var_a}': role = {ctx.role_a}, level = {ctx.level_a}, "
        f"unique values = {ctx.unique_a}, missing = {_pct_label(ctx.missing_pct_a)}.",
        f"  • '{ctx.var_b}': role = {ctx.role_b}, level = {ctx.level_b}, "
        f"unique values = {ctx.unique_b}, missing = {_pct_label(ctx.missing_pct_b)}.",
        f"",
        f"Design: {design_clause}. {n_clause}.",
        f"",
        f"Why this method fits '{ctx.var_a}' × '{ctx.var_b}':",
    ]
    for note in primary_elig:
        rationale_parts.append(f"  • {note}")
    if ctx.n_groups > 0:
        rationale_parts.append(
            f"  • The categorical variable has {ctx.n_groups} group(s), "
            f"which satisfies {primary_sm.name}'s group requirement "
            f"{primary_sm.n_groups_required}."
        )
    if primary_warns:
        rationale_parts.append("")
        rationale_parts.append("Cautions for this specific dataset:")
        for w in primary_warns:
            rationale_parts.append(f"  ⚠ {w}")

    primary_rationale = "\n".join(rationale_parts)

    # -- Alternatives --
    alternatives = []
    for key, sm, elig, warns in candidates[1:]:
        why_alt = elig[0] if elig else sm.purpose
        when_prefer_parts = []
        if sm.nonparametric_alternative is None and primary_sm.nonparametric_alternative == key:
            when_prefer_parts.append("when normality assumption fails")
        if sm.parametric_alternative is None and primary_sm.parametric_alternative == key:
            when_prefer_parts.append("when assumptions are fully met")
        if not when_prefer_parts:
            when_prefer_parts.append("as an additional robustness check")
        alternatives.append({
            "method_key": key,
            "name": sm.name,
            "why_alternative": why_alt,
            "when_prefer": "; ".join(when_prefer_parts),
        })

    # -- Data quality warnings (collated) --
    dq_warnings = list(ctx.warnings_a) + list(ctx.warnings_b)
    if ctx.missing_pairs_pct > 0.20:
        dq_warnings.append(
            f"More than 20% of row pairs are missing "
            f"({_pct_label(ctx.missing_pairs_pct)}). "
            "Results may not be representative of the full dataset."
        )
    if ctx.n_valid_pairs < 30:
        dq_warnings.append(
            f"Small effective sample (n_valid_pairs = {ctx.n_valid_pairs} < 30). "
            "Parametric tests may be unreliable; prefer non-parametric alternatives."
        )
    for w in primary_warns:
        if w not in dq_warnings:
            dq_warnings.append(w)

    # -- Re-run assumption checks with the now-known primary method, if df available --
    if df is not None:
        assumption_checks = _check_assumptions_with_data(ctx, df, primary_key)

    log.append(
        f"STEP 15 — DecisionResult assembled: primary='{primary_key}', "
        f"{len(alternatives)} alternative(s), "
        f"{len(dq_warnings)} data-quality warning(s)."
    )

    # ── STEP 16: Interpretation guidance and limitations ────────────────────
    interp_parts = [primary_sm.interpretation_guidance]
    if ctx.research_question.strip():
        interp_parts.append(
            f"\nIn the context of your research question (\"{ctx.research_question.strip()}\"), "
            f"a significant result would indicate a statistically detectable "
            f"relationship between '{ctx.var_a}' and '{ctx.var_b}'."
        )
    interp_parts.append(
        f"\nReport {_EFFECT_SIZE_MAP.get(primary_key, 'an appropriate effect-size measure')} "
        f"alongside the p-value to convey practical significance."
    )
    interpretation_guidance = " ".join(interp_parts)

    limitations = list(primary_sm.limitations)
    if ctx.n_valid_pairs < primary_sm.min_n:
        limitations.append(
            f"The effective sample (n = {ctx.n_valid_pairs}) is below the recommended "
            f"minimum of {primary_sm.min_n} for {primary_sm.name}. "
            "Interpret results with caution."
        )
    if ctx.missing_pairs_pct > 0.10:
        limitations.append(
            f"Pairwise missingness of {_pct_label(ctx.missing_pairs_pct)} may introduce "
            "bias if data are not missing completely at random (MCAR)."
        )

    log.append(
        "STEP 16 — Interpretation guidance and limitations generated."
    )

    # ── Populate result ──────────────────────────────────────────────────────
    result.step_log = log
    result.blocked = False
    result.block_reason = ""
    result.primary_method = primary_key
    result.primary_rationale = primary_rationale
    result.primary_assumptions = list(primary_sm.assumptions)
    result.alternatives = alternatives
    result.data_quality_warnings = dq_warnings
    result.assumption_check_results = assumption_checks
    result.effect_size_note = _EFFECT_SIZE_MAP.get(primary_key, "See method documentation for effect-size guidance.")
    result.ci_note = _ci_note(primary_key)
    result.multiple_testing_note = _build_multiple_testing_note(n_tests_in_session)
    result.interpretation_guidance = interpretation_guidance
    result.limitations = limitations

    return result


# ============================================================================
# Context builder
# ============================================================================

def build_context_from_profile(
    var_a: str,
    var_b: str,
    profile: dict,
    df=None,
) -> DecisionContext:
    """Build a DecisionContext from a profiler.py profile dict.

    Parameters
    ----------
    var_a, var_b : str
        Names of the two variables to analyse.
    profile : dict
        Output of ``profiler.profile_dataset()``.
    df : pandas.DataFrame or None
        Optional raw data, used to compute n_valid_pairs / n_missing_pairs
        accurately.  If not provided, these are estimated from variable-level
        missingness counts.

    Returns
    -------
    DecisionContext
    """
    col_stats = profile.get("col_stats", {})
    n_rows = profile.get("n_rows", 0)

    def _stat(var: str, key: str, default):
        return col_stats.get(var, {}).get(key, default)

    role_a = _stat(var_a, "role", "unknown")
    level_a = _stat(var_a, "level", "unknown")
    missing_pct_a_raw = _stat(var_a, "missing_pct", 0.0)
    # profiler stores missing_pct as 0–100; normalise to 0–1
    missing_pct_a = missing_pct_a_raw / 100.0 if missing_pct_a_raw > 1.0 else missing_pct_a_raw
    unique_a = _stat(var_a, "unique", 0)
    n_valid_a = n_rows - _stat(var_a, "missing", 0)
    warnings_a = _stat(var_a, "warnings", [])

    role_b = _stat(var_b, "role", "unknown")
    level_b = _stat(var_b, "level", "unknown")
    missing_pct_b_raw = _stat(var_b, "missing_pct", 0.0)
    missing_pct_b = missing_pct_b_raw / 100.0 if missing_pct_b_raw > 1.0 else missing_pct_b_raw
    unique_b = _stat(var_b, "unique", 0)
    n_valid_b = n_rows - _stat(var_b, "missing", 0)
    warnings_b = _stat(var_b, "warnings", [])

    # Pairwise missingness
    if df is not None and var_a in df.columns and var_b in df.columns:
        valid_mask = df[var_a].notna() & df[var_b].notna()
        n_valid_pairs = int(valid_mask.sum())
    else:
        # Conservative estimate: assume independent missingness
        n_valid_pairs = max(0, min(n_valid_a, n_valid_b))

    n_missing_pairs = max(0, n_rows - n_valid_pairs)
    missing_pairs_pct = n_missing_pairs / max(n_rows, 1)

    # n_groups: use unique count of the categorical/grouping variable
    cat_roles = {"categorical", "boolean", "ordinal"}
    if role_a in cat_roles:
        n_groups = unique_a
    elif role_b in cat_roles:
        n_groups = unique_b
    else:
        n_groups = 0

    return DecisionContext(
        var_a=var_a,
        role_a=role_a,
        level_a=level_a,
        missing_pct_a=missing_pct_a,
        unique_a=unique_a,
        n_valid_a=n_valid_a,
        warnings_a=list(warnings_a),
        var_b=var_b,
        role_b=role_b,
        level_b=level_b,
        missing_pct_b=missing_pct_b,
        unique_b=unique_b,
        n_valid_b=n_valid_b,
        warnings_b=list(warnings_b),
        n_total=n_rows,
        n_valid_pairs=n_valid_pairs,
        n_missing_pairs=n_missing_pairs,
        missing_pairs_pct=missing_pairs_pct,
        n_groups=n_groups,
        paired=False,  # must be set by the caller if applicable
        research_question="",
    )


# ============================================================================
# Research question heuristic parser
# ============================================================================

def interpret_research_question(question: str, profile: dict) -> dict:
    """Heuristically classify a free-text research question.

    Parses the question for variable-name hints and intent keywords.
    This is a *heuristic only* — the engine always validates against
    actual data profiles.

    Parameters
    ----------
    question : str
        Free-text research question entered by the researcher.
    profile : dict
        Output of ``profiler.profile_dataset()``, used to cross-check any
        variable name hints found in the question.

    Returns
    -------
    dict with keys:
        intent : str
            One of ``"compare"``, ``"relate"``, ``"predict"``, ``"describe"``.
        outcome_hint : str
            Variable name or keyword fragment that appears to be the outcome,
            or ``""`` if none found.
        predictor_hint : str
            Variable name or keyword fragment that appears to be the predictor,
            or ``""`` if none found.
        confidence : float
            Rough 0–1 confidence in the intent classification.
    """
    if not question or not question.strip():
        return {
            "intent": "describe",
            "outcome_hint": "",
            "predictor_hint": "",
            "confidence": 0.0,
        }

    q_lower = question.lower()
    known_cols = list(profile.get("columns", []))

    # ── Intent keywords ──────────────────────────────────────────────────────
    compare_keywords  = ["differ", "compare", "difference", "between", "versus", "vs"]
    relate_keywords   = ["relationship", "association", "associated", "related", "correlation", "and"]
    predict_keywords  = ["predict", "effect of", "influence", "impact", "regression", "explain"]

    compare_hits  = sum(1 for kw in compare_keywords  if kw in q_lower)
    relate_hits   = sum(1 for kw in relate_keywords   if kw in q_lower)
    predict_hits  = sum(1 for kw in predict_keywords  if kw in q_lower)

    max_hits = max(compare_hits, relate_hits, predict_hits)

    if max_hits == 0:
        intent = "describe"
        confidence = 0.3
    elif compare_hits >= relate_hits and compare_hits >= predict_hits:
        intent = "compare"
        confidence = min(0.5 + 0.1 * compare_hits, 0.95)
    elif predict_hits >= relate_hits:
        intent = "predict"
        confidence = min(0.5 + 0.1 * predict_hits, 0.95)
    else:
        intent = "relate"
        confidence = min(0.5 + 0.1 * relate_hits, 0.95)

    # ── Variable-name hints ──────────────────────────────────────────────────
    # Normalise question and column names to improve matching
    def _normalise(s: str) -> str:
        return s.lower().replace("_", " ").replace("-", " ")

    q_norm = _normalise(question)
    found_cols = [c for c in known_cols if _normalise(c) in q_norm]

    outcome_hint = ""
    predictor_hint = ""

    if len(found_cols) >= 2:
        # Heuristic: the variable mentioned later in the question tends to be
        # the predictor in "effect of X on Y" phrasing, otherwise first = outcome.
        positions = {c: q_norm.find(_normalise(c)) for c in found_cols}
        sorted_cols = sorted(found_cols, key=lambda c: positions[c])

        # "predict X from Y" / "effect of Y on X" → predictor is mentioned
        # near "effect of" / "predict" keywords
        if "effect of" in q_lower or "predict" in q_lower or "influence" in q_lower:
            # predictor is closer to those keywords
            trigger_positions = []
            for kw in ("effect of", "predict", "influence"):
                idx = q_lower.find(kw)
                if idx != -1:
                    trigger_positions.append(idx)
            if trigger_positions:
                trigger_pos = min(trigger_positions)
                # variable mentioned after trigger → likely predictor
                after = [c for c in sorted_cols if positions[c] > trigger_pos]
                before = [c for c in sorted_cols if positions[c] <= trigger_pos]
                predictor_hint = after[0] if after else ""
                outcome_hint = before[-1] if before else ""
        else:
            # default: first mentioned = outcome, second = predictor/grouping
            outcome_hint = sorted_cols[0]
            predictor_hint = sorted_cols[1] if len(sorted_cols) > 1 else ""

    elif len(found_cols) == 1:
        outcome_hint = found_cols[0]

    return {
        "intent": intent,
        "outcome_hint": outcome_hint,
        "predictor_hint": predictor_hint,
        "confidence": confidence,
    }
