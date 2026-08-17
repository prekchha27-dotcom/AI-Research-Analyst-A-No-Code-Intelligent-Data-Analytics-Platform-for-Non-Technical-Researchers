"""
stat_methods.py — Statistical Method Knowledge Base
====================================================
Self-contained knowledge base for a research-oriented analytics dashboard.

Rule-set additions (sourced from reference material, converted to generic auditable rules):
──────────────────────────────────────────────────────────────────────────────────────────
RULE-KB-01    Covariance entry added to METHOD_KB. Covariance measures direction only
              (positive / negative / zero). Magnitude is unit-dependent and must never be
              used to judge relationship strength. Pearson r = Cov / (SD_x * SD_y) is
              the normalised strength measure.

RULE-KB-02    Sampling methodology entries added. Probability sampling (simple random,
              stratified, systematic, cluster, multi-stage) is required for valid
              inferential statistics. Non-probability sampling (convenience, purposive)
              limits generalisability and must be disclosed as a limitation.

CONFLICT-01   Pearson r strength thresholds: the reference material uses |r| >= 0.7 =
              "strong", 0.4-0.6 = "moderate". This knowledge base retains Cohen (1988):
              |r| >= 0.5 = large, 0.3 = medium, 0.1 = small. These thresholds are NOT
              equivalent. Cohen's benchmarks are the established academic standard and are
              retained. The conflict is documented here for auditability.

Exports
-------
StatMethod
    Dataclass describing a single statistical method with its metadata,
    assumptions, effect-size measures, and guidance notes.

METHOD_KB : dict[str, StatMethod]
    Registry mapping a stable method key (e.g. ``"pearson_correlation"``) to
    its StatMethod instance.  Covers descriptive, categorical, comparison,
    correlation, regression, reliability, dimension-reduction, multivariate,
    clustering, and time-series families.

get_candidate_methods(...)
    Reasoning engine that accepts column-role / measurement-level information
    and returns a ranked list of ``(method_key, StatMethod, eligibility_notes,
    warnings)`` tuples.  Reasoning is entirely role- and level-driven; no
    column names are ever inspected.

Roles recognised
----------------
"substantive_numerical", "categorical", "ordinal", "boolean",
"identifier", "serial_number", "administrative_code", "constant",
"free_text", "datetime", "unknown"

Measurement levels recognised
------------------------------
"nominal", "ordinal", "interval", "ratio",
"identifier", "constant", "text", "datetime"

Blocked roles
-------------
identifier, serial_number, administrative_code, constant, free_text
→ get_candidate_methods returns an empty list and a blocking reason.

Dependencies
------------
Standard library only (dataclasses, typing).  No third-party packages required.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class StatMethod:
    """Full descriptor for one statistical method."""

    # --- identity -----------------------------------------------------------
    name: str
    """Human-readable display name (e.g. 'Pearson Correlation')."""

    category: str
    """Broad family: 'Descriptive', 'Categorical', 'Comparison', etc."""

    purpose: str
    """One-sentence statement of what the method achieves."""

    min_vars: int
    """Minimum number of variables the method requires."""

    # --- variable roles -----------------------------------------------------
    outcome_roles: list[str]
    """Acceptable measurement levels for the outcome / dependent variable."""

    predictor_roles: list[str]
    """Acceptable measurement levels for the predictor / independent variable."""

    measurement_levels_required: list[str]
    """Overall minimum measurement levels across all variables involved."""

    # --- design constraints -------------------------------------------------
    n_groups_required: Optional[tuple[int, int]]
    """(min_groups, max_groups) for grouping factor; None if not applicable."""

    paired_required: Optional[bool]
    """True = paired design required; False = independent; None = not applicable."""

    # --- statistical assumptions --------------------------------------------
    assumptions: list[str]
    """List of statistical assumptions that must hold for valid inference."""

    min_n: int
    """Practical minimum sample size for reliable results."""

    # --- output & interpretation --------------------------------------------
    effect_size_measures: list[str]
    """Effect-size statistics produced or recommended alongside this method."""

    ci_available: bool
    """Whether confidence intervals are routinely available for key estimates."""

    nonparametric_alternative: Optional[str]
    """Method key of the preferred non-parametric alternative, if any."""

    parametric_alternative: Optional[str]
    """Method key of the preferred parametric counterpart, if any."""

    interpretation_guidance: str
    """Plain-language guidance for interpreting results."""

    limitations: list[str]
    """Known limitations, caveats, or common misuses."""

    warning_conditions: list[str]
    """Data conditions that should trigger a caution banner in the UI."""

    source_note: str
    """Authoritative textbook / standard the method description is based on."""


# ---------------------------------------------------------------------------
# Knowledge base
# ---------------------------------------------------------------------------

METHOD_KB: dict[str, StatMethod] = {

    # ── DESCRIPTIVE ─────────────────────────────────────────────────────────

    "descriptive_stats": StatMethod(
        name="Descriptive Statistics",
        category="Descriptive",
        purpose="Summarise the central tendency, spread, and shape of a numerical variable.",
        min_vars=1,
        outcome_roles=["interval", "ratio"],
        predictor_roles=[],
        measurement_levels_required=["interval"],
        n_groups_required=None,
        paired_required=None,
        assumptions=["Observations represent the population of interest"],
        min_n=1,
        effect_size_measures=[],
        ci_available=True,
        nonparametric_alternative=None,
        parametric_alternative=None,
        interpretation_guidance=(
            "Report mean ± SD for symmetric distributions; median [IQR] for skewed "
            "distributions.  Always accompany with a distribution plot."
        ),
        limitations=["Sensitive to outliers (mean/SD)", "Does not capture multimodality well"],
        warning_conditions=["High proportion of missing values", "Extreme outliers present"],
        source_note="Field, A. (2018). Discovering Statistics Using IBM SPSS Statistics (5th ed.).",
    ),

    "frequency_table": StatMethod(
        name="Frequency Table",
        category="Descriptive",
        purpose="Tabulate counts and relative frequencies for each category of a nominal or ordinal variable.",
        min_vars=1,
        outcome_roles=["nominal", "ordinal"],
        predictor_roles=[],
        measurement_levels_required=["nominal"],
        n_groups_required=None,
        paired_required=None,
        assumptions=["Mutually exclusive, exhaustive categories"],
        min_n=1,
        effect_size_measures=[],
        ci_available=False,
        nonparametric_alternative=None,
        parametric_alternative=None,
        interpretation_guidance=(
            "Inspect modal category and any unexpected empty cells.  "
            "Report both counts and percentages."
        ),
        limitations=["Uninformative for high-cardinality variables"],
        warning_conditions=["More than 20 unique categories", "Any category has n < 5"],
        source_note="Field, A. (2018). Discovering Statistics Using IBM SPSS Statistics (5th ed.).",
    ),

    "crosstab": StatMethod(
        name="Cross-tabulation (Contingency Table)",
        category="Descriptive",
        purpose="Display the joint frequency distribution of two categorical variables.",
        min_vars=2,
        outcome_roles=["nominal", "ordinal"],
        predictor_roles=["nominal", "ordinal"],
        measurement_levels_required=["nominal"],
        n_groups_required=None,
        paired_required=None,
        assumptions=["Mutually exclusive, exhaustive categories for both variables"],
        min_n=10,
        effect_size_measures=["Cramér's V", "Phi"],
        ci_available=False,
        nonparametric_alternative=None,
        parametric_alternative=None,
        interpretation_guidance=(
            "Inspect row/column percentages and the pattern of residuals.  "
            "Combine with a chi-square test for inferential conclusions."
        ),
        limitations=["Sparse cells inflate test statistics", "Not suitable for continuous variables"],
        warning_conditions=["Any expected cell count < 5", "More than 20% of cells have expected n < 5"],
        source_note="Agresti, A. (2018). An Introduction to Categorical Data Analysis (3rd ed.).",
    ),

    # ── CATEGORICAL ──────────────────────────────────────────────────────────

    "chi_square_independence": StatMethod(
        name="Chi-Square Test of Independence",
        category="Categorical",
        purpose="Test whether two categorical variables are statistically independent.",
        min_vars=2,
        outcome_roles=["nominal", "ordinal"],
        predictor_roles=["nominal", "ordinal"],
        measurement_levels_required=["nominal"],
        n_groups_required=None,
        paired_required=False,
        assumptions=[
            "Independent observations",
            "Expected frequency ≥ 5 in ≥ 80% of cells",
            "Expected frequency ≥ 1 in all cells",
        ],
        min_n=20,
        effect_size_measures=["Cramér's V", "Phi (2×2 only)"],
        ci_available=False,
        nonparametric_alternative="fishers_exact",
        parametric_alternative=None,
        interpretation_guidance=(
            "A significant p-value indicates non-independence; pair with Cramér's V "
            "to quantify magnitude.  Inspect standardised residuals to locate the cells "
            "driving the association."
        ),
        limitations=[
            "Sensitive to sample size — trivially significant with large N",
            "Does not indicate direction of association",
        ],
        warning_conditions=["Any expected cell count < 5", "N < 20"],
        source_note="Agresti, A. (2018). An Introduction to Categorical Data Analysis (3rd ed.).",
    ),

    "chi_square_goodness_of_fit": StatMethod(
        name="Chi-Square Goodness-of-Fit Test",
        category="Categorical",
        purpose="Test whether observed category frequencies match a specified theoretical distribution.",
        min_vars=1,
        outcome_roles=["nominal", "ordinal"],
        predictor_roles=[],
        measurement_levels_required=["nominal"],
        n_groups_required=None,
        paired_required=None,
        assumptions=[
            "Independent observations",
            "Expected frequency ≥ 5 per category",
        ],
        min_n=20,
        effect_size_measures=["w (Cohen's w)"],
        ci_available=False,
        nonparametric_alternative=None,
        parametric_alternative=None,
        interpretation_guidance=(
            "Compare each observed proportion to its theoretical counterpart. "
            "Significant result means at least one category departs from expectation; "
            "use adjusted residuals to identify which."
        ),
        limitations=["Requires pre-specified theoretical proportions", "Sensitive to large N"],
        warning_conditions=["Any expected count < 5"],
        source_note="Cohen, J. (1988). Statistical Power Analysis for the Behavioral Sciences (2nd ed.).",
    ),

    "fishers_exact": StatMethod(
        name="Fisher's Exact Test",
        category="Categorical",
        purpose="Test independence between two binary/categorical variables when expected cell counts are small.",
        min_vars=2,
        outcome_roles=["nominal"],
        predictor_roles=["nominal"],
        measurement_levels_required=["nominal"],
        n_groups_required=(2, 2),
        paired_required=False,
        assumptions=["Fixed marginal totals", "Independent observations"],
        min_n=5,
        effect_size_measures=["Odds Ratio", "Phi"],
        ci_available=True,
        nonparametric_alternative=None,
        parametric_alternative="chi_square_independence",
        interpretation_guidance=(
            "Preferred over chi-square when any expected cell count < 5. "
            "Report exact p-value and odds ratio with 95% CI."
        ),
        limitations=["Computationally intensive for large tables", "Conservative in some configurations"],
        warning_conditions=["Table larger than 2×2 — use chi-square instead"],
        source_note="Fisher, R.A. (1922). On the interpretation of χ² from contingency tables.",
    ),

    "cramers_v": StatMethod(
        name="Cramér's V",
        category="Categorical",
        purpose="Measure the strength of association between two nominal variables after a significant chi-square test.",
        min_vars=2,
        outcome_roles=["nominal", "ordinal"],
        predictor_roles=["nominal", "ordinal"],
        measurement_levels_required=["nominal"],
        n_groups_required=None,
        paired_required=False,
        assumptions=["Chi-square test assumptions met"],
        min_n=20,
        effect_size_measures=["Cramér's V (0–1)"],
        ci_available=True,
        nonparametric_alternative=None,
        parametric_alternative=None,
        interpretation_guidance=(
            "Values: ~0.10 small, ~0.30 medium, ~0.50 large (adjusted for table size). "
            "Always report alongside the chi-square test result."
        ),
        limitations=["Does not indicate direction", "Affected by marginal distributions"],
        warning_conditions=["Expected cell counts < 5"],
        source_note="Cramér, H. (1946). Mathematical Methods of Statistics. Princeton University Press.",
    ),

    "odds_ratio": StatMethod(
        name="Odds Ratio",
        category="Categorical",
        purpose="Quantify the odds of an outcome event in one group relative to another.",
        min_vars=2,
        outcome_roles=["nominal"],
        predictor_roles=["nominal"],
        measurement_levels_required=["nominal"],
        n_groups_required=(2, 2),
        paired_required=False,
        assumptions=["Binary outcome", "Independent observations", "Rare outcome for RR approximation"],
        min_n=10,
        effect_size_measures=["Odds Ratio", "Risk Ratio (if prospective)"],
        ci_available=True,
        nonparametric_alternative=None,
        parametric_alternative=None,
        interpretation_guidance=(
            "OR = 1 → no association; OR > 1 → increased odds in exposed group; "
            "OR < 1 → decreased odds.  Report 95% CI; if CI excludes 1, statistically significant."
        ),
        limitations=[
            "Overestimates relative risk when outcome is not rare",
            "Not suitable for matched case-control without adjustment",
        ],
        warning_conditions=["Outcome prevalence > 10% (OR ≠ RR approximation)", "Any cell count = 0"],
        source_note="Szumilas, M. (2010). Explaining Odds Ratios. J Can Acad Child Adolesc Psychiatry.",
    ),

    # ── COMPARISON ───────────────────────────────────────────────────────────

    "independent_ttest": StatMethod(
        name="Independent Samples t-Test",
        category="Comparison",
        purpose="Compare means of a continuous variable between two independent groups.",
        min_vars=2,
        outcome_roles=["interval", "ratio"],
        predictor_roles=["nominal"],
        measurement_levels_required=["interval"],
        n_groups_required=(2, 2),
        paired_required=False,
        assumptions=[
            "Normality of outcome within each group",
            "Homogeneity of variances (Levene's test)",
            "Independence of observations",
            "Continuous outcome (interval/ratio)",
        ],
        min_n=20,
        effect_size_measures=["Cohen's d", "Hedges' g"],
        ci_available=True,
        nonparametric_alternative="mann_whitney_u",
        parametric_alternative=None,
        interpretation_guidance=(
            "Report mean difference ± 95% CI and Cohen's d.  "
            "d ≈ 0.2 small, 0.5 medium, 0.8 large.  "
            "If Levene's test significant, use Welch's t-test instead."
        ),
        limitations=["Assumes equal variances unless Welch variant used", "Sensitive to outliers"],
        warning_conditions=["n < 30 per group", "Significant Levene's test", "Non-normal residuals"],
        source_note="Cohen, J. (1988). Statistical Power Analysis for the Behavioral Sciences (2nd ed.).",
    ),

    "welch_ttest": StatMethod(
        name="Welch's t-Test",
        category="Comparison",
        purpose="Compare means of two independent groups without assuming equal variances.",
        min_vars=2,
        outcome_roles=["interval", "ratio"],
        predictor_roles=["nominal"],
        measurement_levels_required=["interval"],
        n_groups_required=(2, 2),
        paired_required=False,
        assumptions=[
            "Normality of outcome within each group",
            "Independence of observations",
            "Continuous outcome (interval/ratio)",
        ],
        min_n=20,
        effect_size_measures=["Cohen's d", "Hedges' g"],
        ci_available=True,
        nonparametric_alternative="mann_whitney_u",
        parametric_alternative="independent_ttest",
        interpretation_guidance=(
            "Preferred over equal-variance t-test when group sizes or variances differ. "
            "Interpret mean difference and 95% CI as with independent t-test."
        ),
        limitations=["Still assumes approximate normality for small samples"],
        warning_conditions=["n < 20 per group", "Heavy-tailed or highly skewed distribution"],
        source_note="Welch, B.L. (1947). The generalization of Student's problem. Biometrika.",
    ),

    "paired_ttest": StatMethod(
        name="Paired Samples t-Test",
        category="Comparison",
        purpose="Compare means of two related measurements (before/after or matched pairs).",
        min_vars=2,
        outcome_roles=["interval", "ratio"],
        predictor_roles=["nominal"],
        measurement_levels_required=["interval"],
        n_groups_required=(2, 2),
        paired_required=True,
        assumptions=[
            "Normality of the difference scores",
            "Paired/matched observations",
            "Continuous outcome (interval/ratio)",
        ],
        min_n=10,
        effect_size_measures=["Cohen's d (on differences)", "Hedges' g"],
        ci_available=True,
        nonparametric_alternative="wilcoxon_signed_rank",
        parametric_alternative=None,
        interpretation_guidance=(
            "Compute difference scores and verify their normality (Shapiro–Wilk). "
            "Report mean difference ± 95% CI and Cohen's d."
        ),
        limitations=["Requires paired structure — pairing must be meaningful", "Not valid for independent groups"],
        warning_conditions=["n < 10 pairs", "Difference scores are non-normal"],
        source_note="Field, A. (2018). Discovering Statistics Using IBM SPSS Statistics (5th ed.).",
    ),

    "mann_whitney_u": StatMethod(
        name="Mann–Whitney U Test",
        category="Comparison",
        purpose="Non-parametric comparison of distributions between two independent groups.",
        min_vars=2,
        outcome_roles=["ordinal", "interval", "ratio"],
        predictor_roles=["nominal"],
        measurement_levels_required=["ordinal"],
        n_groups_required=(2, 2),
        paired_required=False,
        assumptions=[
            "Independent observations",
            "Ordinal or continuous outcome",
            "Same distributional shape in both groups (for median interpretation)",
        ],
        min_n=10,
        effect_size_measures=["Rank-biserial correlation r", "Common Language Effect Size"],
        ci_available=True,
        nonparametric_alternative=None,
        parametric_alternative="welch_ttest",
        interpretation_guidance=(
            "Report U statistic, z, p, and rank-biserial r.  "
            "r ≈ 0.1 small, 0.3 medium, 0.5 large.  "
            "If distributions differ in shape, test concerns stochastic dominance, not medians."
        ),
        limitations=[
            "Does not test median equality unless distributions have the same shape",
            "Less powerful than t-test when normality holds",
        ],
        warning_conditions=["n < 10 per group", "Tied values — use continuity correction"],
        source_note="Mann, H.B. & Whitney, D.R. (1947). Ann. Math. Statist.",
    ),

    "wilcoxon_signed_rank": StatMethod(
        name="Wilcoxon Signed-Rank Test",
        category="Comparison",
        purpose="Non-parametric test for comparing two related samples or repeated measurements.",
        min_vars=2,
        outcome_roles=["ordinal", "interval", "ratio"],
        predictor_roles=["nominal"],
        measurement_levels_required=["ordinal"],
        n_groups_required=(2, 2),
        paired_required=True,
        assumptions=[
            "Paired/matched observations",
            "Ordinal or continuous outcome",
            "Symmetry of difference scores around median (for median interpretation)",
        ],
        min_n=10,
        effect_size_measures=["Matched-pairs rank-biserial correlation r"],
        ci_available=True,
        nonparametric_alternative=None,
        parametric_alternative="paired_ttest",
        interpretation_guidance=(
            "Use when difference scores are non-normal.  "
            "Report W, z, p, and r.  "
            "Interpret direction from medians of each condition."
        ),
        limitations=["Reduced power vs. paired t-test under normality", "Requires ≥ 10 non-zero differences"],
        warning_conditions=["Many tied difference scores", "< 10 non-zero pairs"],
        source_note="Wilcoxon, F. (1945). Individual comparisons by ranking methods. Biometrics.",
    ),

    "one_way_anova": StatMethod(
        name="One-Way ANOVA",
        category="Comparison",
        purpose="Test whether means of a continuous variable differ across three or more independent groups.",
        min_vars=2,
        outcome_roles=["interval", "ratio"],
        predictor_roles=["nominal"],
        measurement_levels_required=["interval"],
        n_groups_required=(3, None),
        paired_required=False,
        assumptions=[
            "Normality of residuals within each group",
            "Homogeneity of variances (Levene's test)",
            "Independence of observations",
            "Continuous outcome (interval/ratio)",
        ],
        min_n=20,
        effect_size_measures=["η² (eta-squared)", "ω² (omega-squared)", "Cohen's f"],
        ci_available=True,
        nonparametric_alternative="kruskal_wallis",
        parametric_alternative=None,
        interpretation_guidance=(
            "A significant F only indicates some group differs; follow with post-hoc tests "
            "(e.g., Tukey HSD) to identify which pairs differ.  "
            "η² ≈ 0.01 small, 0.06 medium, 0.14 large."
        ),
        limitations=[
            "Omnibus test — does not identify specific group differences",
            "Sensitive to violations of homogeneity with unequal group sizes",
        ],
        warning_conditions=["Significant Levene's test (use Welch ANOVA)", "n < 10 per group", "Non-normal residuals"],
        source_note="Field, A. (2018). Discovering Statistics Using IBM SPSS Statistics (5th ed.).",
    ),

    "welch_anova": StatMethod(
        name="Welch's One-Way ANOVA",
        category="Comparison",
        purpose="Test mean differences across three or more independent groups without assuming equal variances.",
        min_vars=2,
        outcome_roles=["interval", "ratio"],
        predictor_roles=["nominal"],
        measurement_levels_required=["interval"],
        n_groups_required=(3, None),
        paired_required=False,
        assumptions=[
            "Approximate normality within each group",
            "Independence of observations",
        ],
        min_n=20,
        effect_size_measures=["η²", "ω²"],
        ci_available=True,
        nonparametric_alternative="kruskal_wallis",
        parametric_alternative="one_way_anova",
        interpretation_guidance=(
            "Preferred when Levene's test is significant or group sizes are very unequal. "
            "Follow with Games-Howell post-hoc comparisons."
        ),
        limitations=["Slightly less powerful than standard ANOVA when variances are equal"],
        warning_conditions=["n < 10 per group", "Heavy-tailed distributions"],
        source_note="Welch, B.L. (1951). On the comparison of several mean values. Biometrika.",
    ),

    "kruskal_wallis": StatMethod(
        name="Kruskal–Wallis H Test",
        category="Comparison",
        purpose="Non-parametric test for differences across three or more independent groups.",
        min_vars=2,
        outcome_roles=["ordinal", "interval", "ratio"],
        predictor_roles=["nominal"],
        measurement_levels_required=["ordinal"],
        n_groups_required=(3, None),
        paired_required=False,
        assumptions=[
            "Independent observations",
            "Ordinal or continuous outcome",
            "Same distributional shape across groups (for median interpretation)",
        ],
        min_n=15,
        effect_size_measures=["η²_H (epsilon-squared)", "Rank-biserial r for pairwise"],
        ci_available=False,
        nonparametric_alternative=None,
        parametric_alternative="one_way_anova",
        interpretation_guidance=(
            "Significant H indicates at least one group differs; follow with Dunn's test "
            "(Bonferroni-adjusted) for pairwise comparisons."
        ),
        limitations=["Less powerful than ANOVA when normality holds", "Distributional shape caveat for medians"],
        warning_conditions=["n < 5 per group", "Many tied ranks"],
        source_note="Kruskal, W.H. & Wallis, W.A. (1952). J. Am. Stat. Assoc.",
    ),

    "friedman": StatMethod(
        name="Friedman Test",
        category="Comparison",
        purpose="Non-parametric test for differences across three or more related/repeated measurements.",
        min_vars=3,
        outcome_roles=["ordinal", "interval", "ratio"],
        predictor_roles=["nominal"],
        measurement_levels_required=["ordinal"],
        n_groups_required=(3, None),
        paired_required=True,
        assumptions=[
            "Repeated or matched measurements",
            "Ordinal or continuous outcome",
            "No interaction between blocks and treatment",
        ],
        min_n=10,
        effect_size_measures=["Kendall's W (concordance)", "r for pairwise"],
        ci_available=False,
        nonparametric_alternative=None,
        parametric_alternative="one_way_anova",
        interpretation_guidance=(
            "Follow significant Friedman test with Wilcoxon signed-rank pairwise tests "
            "with Bonferroni correction.  Kendall's W ranges 0–1; 1 = perfect agreement."
        ),
        limitations=["Requires complete data across all conditions", "Less power than repeated-measures ANOVA"],
        warning_conditions=["Missing values in any block", "n < 10 blocks"],
        source_note="Friedman, M. (1937). The use of ranks to avoid the assumption of normality. JASA.",
    ),

    # ── CORRELATION ──────────────────────────────────────────────────────────

    "pearson_correlation": StatMethod(
        name="Pearson Correlation",
        category="Correlation",
        purpose="Measure the linear association between two continuous variables.",
        min_vars=2,
        outcome_roles=["interval", "ratio"],
        predictor_roles=["interval", "ratio"],
        measurement_levels_required=["interval"],
        n_groups_required=None,
        paired_required=None,
        assumptions=[
            "Both variables measured at interval or ratio level",
            "Linear relationship between variables",
            "Bivariate normality (for inference)",
            "No extreme outliers",
            "Homoscedasticity",
        ],
        min_n=20,
        effect_size_measures=["r (Pearson)", "r² (coefficient of determination)"],
        ci_available=True,
        nonparametric_alternative="spearman_correlation",
        parametric_alternative=None,
        interpretation_guidance=(
            "|r| ≈ 0.10 small, 0.30 medium, 0.50 large (Cohen, 1988). "
            "Always inspect scatterplot — r only captures linear association. "
            "r² gives proportion of shared variance."
        ),
        limitations=[
            "Sensitive to outliers",
            "Captures only linear relationships",
            "Does not imply causation",
        ],
        warning_conditions=["n < 20", "Outliers detected", "Non-linear pattern in scatterplot"],
        source_note=(
            "Cohen, J. (1988). Statistical Power Analysis for the Behavioral Sciences (2nd ed.). "
            "CONFLICT-01: The reference material labels |r| >= 0.7 as 'strong' and 0.4-0.6 as "
            "'moderate'. This engine retains Cohen (1988) benchmarks: |r| >= 0.5 = large, "
            "0.3 = medium, 0.1 = small. Cohen's thresholds are the established academic standard."
        ),
    ),

    "spearman_correlation": StatMethod(
        name="Spearman Rank Correlation",
        category="Correlation",
        purpose="Measure monotonic association between two ordinal or continuous variables using rank-order.",
        min_vars=2,
        outcome_roles=["ordinal", "interval", "ratio"],
        predictor_roles=["ordinal", "interval", "ratio"],
        measurement_levels_required=["ordinal"],
        n_groups_required=None,
        paired_required=None,
        assumptions=[
            "Ordinal, interval, or ratio measurement for both variables",
            "Monotonic relationship between variables",
        ],
        min_n=10,
        effect_size_measures=["ρ (rho)", "ρ²"],
        ci_available=True,
        nonparametric_alternative=None,
        parametric_alternative="pearson_correlation",
        interpretation_guidance=(
            "Interpret ρ as Pearson's r: |ρ| ≈ 0.10 small, 0.30 medium, 0.50 large. "
            "Robust to outliers and non-normality.  "
            "Preferred when data contain ordinal scales or outliers."
        ),
        limitations=[
            "Less powerful than Pearson when linearity and normality hold",
            "Only captures monotonic, not arbitrary, non-linear relationships",
        ],
        warning_conditions=["Many tied ranks (affects p-value accuracy)", "n < 10"],
        source_note="Spearman, C. (1904). The proof and measurement of association. Am. J. Psychol.",
    ),

    "kendall_tau": StatMethod(
        name="Kendall's Tau",
        category="Correlation",
        purpose="Measure ordinal association via concordance/discordance of pairs; more robust than Spearman for small samples.",
        min_vars=2,
        outcome_roles=["ordinal", "interval", "ratio"],
        predictor_roles=["ordinal", "interval", "ratio"],
        measurement_levels_required=["ordinal"],
        n_groups_required=None,
        paired_required=None,
        assumptions=[
            "Ordinal, interval, or ratio measurement for both variables",
        ],
        min_n=10,
        effect_size_measures=["τ (tau-b or tau-c)"],
        ci_available=True,
        nonparametric_alternative=None,
        parametric_alternative="pearson_correlation",
        interpretation_guidance=(
            "τ has a different scale to ρ; |τ| ≈ 0.10 small, 0.20 medium, 0.30 large. "
            "Preferred over Spearman when N is small or there are many ties."
        ),
        limitations=["Less commonly reported in applied research than Spearman"],
        warning_conditions=["n < 10"],
        source_note="Kendall, M.G. (1938). A new measure of rank correlation. Biometrika.",
    ),

    "point_biserial": StatMethod(
        name="Point-Biserial Correlation",
        category="Correlation",
        purpose="Measure the association between a binary variable and a continuous variable.",
        min_vars=2,
        outcome_roles=["interval", "ratio"],
        predictor_roles=["nominal"],
        measurement_levels_required=["nominal"],
        n_groups_required=(2, 2),
        paired_required=False,
        assumptions=[
            "Continuous variable is approximately normally distributed within each binary group",
            "Binary variable is truly dichotomous",
            "Independence of observations",
        ],
        min_n=20,
        effect_size_measures=["r_pb", "Cohen's d (equivalent)"],
        ci_available=True,
        nonparametric_alternative="mann_whitney_u",
        parametric_alternative="independent_ttest",
        interpretation_guidance=(
            "r_pb is mathematically equivalent to Pearson r applied to a 0/1 coded variable. "
            "Interpret magnitude as Pearson r."
        ),
        limitations=["Sensitive to unequal group sizes", "Assumes normality of continuous variable"],
        warning_conditions=["Severely unequal group sizes", "Non-normal distribution within groups"],
        source_note="Field, A. (2018). Discovering Statistics Using IBM SPSS Statistics (5th ed.).",
    ),

    # ── REGRESSION ───────────────────────────────────────────────────────────

    "simple_linear_regression": StatMethod(
        name="Simple Linear Regression",
        category="Regression",
        purpose="Model the linear relationship between one continuous predictor and one continuous outcome.",
        min_vars=2,
        outcome_roles=["interval", "ratio"],
        predictor_roles=["interval", "ratio"],
        measurement_levels_required=["interval"],
        n_groups_required=None,
        paired_required=False,
        assumptions=[
            "Linearity",
            "Independence of residuals",
            "Homoscedasticity of residuals",
            "Normality of residuals",
            "No influential outliers",
        ],
        min_n=20,
        effect_size_measures=["R²", "Adjusted R²", "Cohen's f²"],
        ci_available=True,
        nonparametric_alternative="spearman_correlation",
        parametric_alternative=None,
        interpretation_guidance=(
            "Report unstandardised (B) and standardised (β) coefficients, R², and 95% CI for B. "
            "R² gives proportion of variance explained.  "
            "Examine residual plots to verify assumptions."
        ),
        limitations=[
            "Cannot capture non-linear relationships without transformation",
            "Sensitive to multicollinearity when extended to multiple predictors",
            "Extrapolation beyond data range is unreliable",
        ],
        warning_conditions=["Residuals show fan pattern (heteroscedasticity)", "n < 20", "Influential outliers"],
        source_note="Cohen, J., Cohen, P., West, S.G., & Aiken, L.S. (2003). Applied Multiple Regression (3rd ed.).",
    ),

    "multiple_linear_regression": StatMethod(
        name="Multiple Linear Regression",
        category="Regression",
        purpose="Model the relationship between multiple predictors and a continuous outcome.",
        min_vars=3,
        outcome_roles=["interval", "ratio"],
        predictor_roles=["interval", "ratio", "nominal", "ordinal"],
        measurement_levels_required=["interval"],
        n_groups_required=None,
        paired_required=False,
        assumptions=[
            "Linearity",
            "Independence of residuals",
            "Homoscedasticity of residuals",
            "Normality of residuals",
            "No perfect multicollinearity (VIF < 10)",
            "No influential outliers (Cook's D)",
        ],
        min_n=50,
        effect_size_measures=["R²", "Adjusted R²", "Cohen's f²", "Semi-partial r²"],
        ci_available=True,
        nonparametric_alternative=None,
        parametric_alternative=None,
        interpretation_guidance=(
            "Report B, β, 95% CI, R², and ΔR² for each block if hierarchical. "
            "Check VIF for multicollinearity; VIF > 10 is problematic. "
            "Rule of thumb: ≥ 10–20 observations per predictor."
        ),
        limitations=[
            "Overfitting risk with many predictors",
            "Multicollinearity inflates standard errors",
            "Assumes linearity and additivity",
        ],
        warning_conditions=["VIF > 10", "n < 10 × predictors", "Non-normal or heteroscedastic residuals"],
        source_note="Cohen, J., Cohen, P., West, S.G., & Aiken, L.S. (2003). Applied Multiple Regression (3rd ed.).",
    ),

    "logistic_regression": StatMethod(
        name="Binary Logistic Regression",
        category="Regression",
        purpose="Model the probability of a binary outcome as a function of one or more predictors.",
        min_vars=2,
        outcome_roles=["nominal"],
        predictor_roles=["interval", "ratio", "nominal", "ordinal"],
        measurement_levels_required=["nominal"],
        n_groups_required=(2, 2),
        paired_required=False,
        assumptions=[
            "Binary outcome variable",
            "Independence of observations",
            "No perfect separation",
            "No severe multicollinearity",
            "Linearity of continuous predictors with log-odds",
            "Large sample size",
        ],
        min_n=50,
        effect_size_measures=["Odds Ratio (exp(B))", "Nagelkerke R²", "AUC-ROC", "Cohen's f²"],
        ci_available=True,
        nonparametric_alternative=None,
        parametric_alternative=None,
        interpretation_guidance=(
            "Report odds ratios with 95% CI, Hosmer–Lemeshow goodness-of-fit, and AUC. "
            "Odds ratio > 1 = increased odds; < 1 = decreased odds. "
            "Rule of thumb: ≥ 10 events per predictor (EPV ≥ 10)."
        ),
        limitations=[
            "Assumes log-linear relationship between predictors and log-odds",
            "EPV < 10 leads to unstable estimates",
            "Does not handle perfect separation",
        ],
        warning_conditions=["Events per variable < 10", "Complete or quasi-complete separation", "n < 50"],
        source_note="Hosmer, D.W. & Lemeshow, S. (2000). Applied Logistic Regression (2nd ed.).",
    ),

    "ordinal_logistic": StatMethod(
        name="Ordinal Logistic Regression (Proportional Odds)",
        category="Regression",
        purpose="Model the relationship between predictors and an ordinal outcome variable.",
        min_vars=2,
        outcome_roles=["ordinal"],
        predictor_roles=["interval", "ratio", "nominal", "ordinal"],
        measurement_levels_required=["ordinal"],
        n_groups_required=None,
        paired_required=False,
        assumptions=[
            "Ordinal outcome with at least 3 ordered categories",
            "Proportional odds assumption (test with Brant test)",
            "Independence of observations",
            "No severe multicollinearity",
        ],
        min_n=50,
        effect_size_measures=["Odds Ratio (proportional)", "Nagelkerke R²", "McFadden's R²"],
        ci_available=True,
        nonparametric_alternative="spearman_correlation",
        parametric_alternative="multiple_linear_regression",
        interpretation_guidance=(
            "Test proportional odds assumption with the Brant test or score test. "
            "Interpret cumulative odds ratios; the model assumes the effect is the same across all thresholds."
        ),
        limitations=["Proportional odds assumption often violated in practice", "Requires adequate cell sizes per category"],
        warning_conditions=["Proportional odds test significant", "Sparse ordinal categories (< 5 per level)"],
        source_note="Agresti, A. (2018). An Introduction to Categorical Data Analysis (3rd ed.).",
    ),

    "poisson_regression": StatMethod(
        name="Poisson Regression",
        category="Regression",
        purpose="Model count or rate outcomes as a function of one or more predictors.",
        min_vars=2,
        outcome_roles=["ratio"],
        predictor_roles=["interval", "ratio", "nominal", "ordinal"],
        measurement_levels_required=["ratio"],
        n_groups_required=None,
        paired_required=False,
        assumptions=[
            "Count outcome (non-negative integers)",
            "Mean equals variance (equidispersion)",
            "Independence of observations",
            "Log-linear relationship between predictors and log-mean",
        ],
        min_n=30,
        effect_size_measures=["Incidence Rate Ratio (IRR = exp(B))", "Pseudo R²"],
        ci_available=True,
        nonparametric_alternative=None,
        parametric_alternative=None,
        interpretation_guidance=(
            "Report IRR with 95% CI.  Check for overdispersion (variance > mean); "
            "if present, use Negative Binomial regression.  "
            "Include offset term (log of exposure) for rate data."
        ),
        limitations=[
            "Equidispersion assumption often violated (use NB if overdispersed)",
            "Cannot model zero-inflated counts without extension",
        ],
        warning_conditions=["Overdispersion (deviance/df > 1.5)", "Excess zeros in count variable"],
        source_note="Cameron, A.C. & Trivedi, P.K. (1998). Regression Analysis of Count Data.",
    ),

    # ── RELIABILITY ──────────────────────────────────────────────────────────

    "cronbach_alpha": StatMethod(
        name="Cronbach's Alpha",
        category="Reliability",
        purpose="Estimate the internal consistency (reliability) of a set of items measuring the same construct.",
        min_vars=3,
        outcome_roles=["interval", "ratio", "ordinal"],
        predictor_roles=[],
        measurement_levels_required=["ordinal"],
        n_groups_required=None,
        paired_required=None,
        assumptions=[
            "Items are tau-equivalent (equal factor loadings) for unbiased α",
            "Essentially unidimensional scale",
            "Items measured at ordinal or higher level",
        ],
        min_n=100,
        effect_size_measures=["α (Cronbach's alpha)", "ω (McDonald's omega, preferred)"],
        ci_available=True,
        nonparametric_alternative=None,
        parametric_alternative=None,
        interpretation_guidance=(
            "α ≥ 0.70 acceptable; ≥ 0.80 good; ≥ 0.90 excellent but may indicate redundancy. "
            "Report 'if item deleted' statistics to identify weak items. "
            "Consider ω (omega) as a less assumption-dependent alternative."
        ),
        limitations=[
            "Assumes tau-equivalence — underestimates reliability if loadings differ",
            "Increases with number of items regardless of true reliability",
            "Not a measure of validity or unidimensionality",
        ],
        warning_conditions=[
            "α < 0.60 — scale may not be reliable",
            "Multidimensional scale — α is not appropriate",
            "n < 100",
        ],
        source_note="Cronbach, L.J. (1951). Coefficient alpha and the internal structure of tests. Psychometrika.",
    ),

    # ── DIMENSION REDUCTION ──────────────────────────────────────────────────

    "pca": StatMethod(
        name="Principal Component Analysis (PCA)",
        category="Dimension Reduction",
        purpose="Reduce a set of correlated continuous variables to uncorrelated principal components capturing maximum variance.",
        min_vars=3,
        outcome_roles=["interval", "ratio"],
        predictor_roles=[],
        measurement_levels_required=["interval"],
        n_groups_required=None,
        paired_required=None,
        assumptions=[
            "Continuous variables (interval/ratio)",
            "Linear relationships between variables",
            "Sufficient sample size relative to variables (≥ 5–10 per variable)",
            "Variables are correlated (KMO > 0.60, Bartlett's test significant)",
        ],
        min_n=50,
        effect_size_measures=["Eigenvalues", "% Variance explained", "Loadings"],
        ci_available=False,
        nonparametric_alternative=None,
        parametric_alternative=None,
        interpretation_guidance=(
            "Use scree plot or parallel analysis to determine number of components. "
            "Rotate components (Varimax for orthogonal, Oblimin for correlated). "
            "Interpret components by the pattern of high loadings (|λ| > 0.40)."
        ),
        limitations=[
            "Components are data-driven, not theory-driven",
            "Sensitive to variable scaling — standardise before analysis",
            "Difficult to interpret if no clear loading structure",
        ],
        warning_conditions=["KMO < 0.60", "Bartlett's test non-significant", "n < 100", "Many non-normal variables"],
        source_note="Jolliffe, I.T. (2002). Principal Component Analysis (2nd ed.). Springer.",
    ),

    "exploratory_factor_analysis": StatMethod(
        name="Exploratory Factor Analysis (EFA)",
        category="Dimension Reduction",
        purpose="Identify latent factors underlying a set of observed variables, assuming common factor structure.",
        min_vars=5,
        outcome_roles=["interval", "ratio", "ordinal"],
        predictor_roles=[],
        measurement_levels_required=["ordinal"],
        n_groups_required=None,
        paired_required=None,
        assumptions=[
            "Variables are indicators of latent factors",
            "Sufficient correlations among variables (KMO > 0.60)",
            "Adequate sample size (≥ 5–10 per variable, min 100–200)",
            "Multivariate normality (for ML extraction)",
        ],
        min_n=100,
        effect_size_measures=["Factor loadings", "Communalities (h²)", "RMSEA (for fit)"],
        ci_available=False,
        nonparametric_alternative=None,
        parametric_alternative="pca",
        interpretation_guidance=(
            "Use parallel analysis or MAP criterion to determine factors. "
            "Extract with Principal Axis Factoring or ML. "
            "Rotate with Oblimin (correlated factors expected) or Varimax. "
            "Interpret factors by loadings ≥ 0.40; cross-loadings ≥ 0.30 indicate problems."
        ),
        limitations=[
            "Results can vary with extraction method and rotation",
            "Requires theoretical interpretation of factors",
            "Not confirmatory — use CFA for theory testing",
        ],
        warning_conditions=["KMO < 0.60", "Heywood cases (communality > 1.0)", "n < 100"],
        source_note="Fabrigar, L.R. et al. (1999). Evaluating the use of EFA in psychological research. Psychol. Methods.",
    ),

    # ── MULTIVARIATE ─────────────────────────────────────────────────────────

    "manova": StatMethod(
        name="Multivariate Analysis of Variance (MANOVA)",
        category="Multivariate",
        purpose="Test whether group means differ simultaneously across two or more continuous dependent variables.",
        min_vars=3,
        outcome_roles=["interval", "ratio"],
        predictor_roles=["nominal"],
        measurement_levels_required=["interval"],
        n_groups_required=(2, None),
        paired_required=False,
        assumptions=[
            "Multivariate normality of DVs within each group",
            "Homogeneity of variance-covariance matrices (Box's M)",
            "Independence of observations",
            "No multicollinearity among DVs (but some correlation expected)",
            "Linearity among DVs",
        ],
        min_n=50,
        effect_size_measures=["Wilks' Λ", "Pillai's trace", "Partial η²", "Cohen's f²"],
        ci_available=False,
        nonparametric_alternative=None,
        parametric_alternative="one_way_anova",
        interpretation_guidance=(
            "Prefer Pillai's trace when assumptions are marginal (more robust). "
            "Follow significant MANOVA with univariate ANOVAs with Bonferroni correction. "
            "Partial η² ≈ 0.01 small, 0.06 medium, 0.14 large."
        ),
        limitations=[
            "Sensitive to violation of multivariate normality",
            "Power depends on correlational structure of DVs",
            "Box's M test is sensitive to non-normality",
        ],
        warning_conditions=["n < 20 per group per DV", "Significant Box's M (use Pillai's trace)", "DVs highly correlated (redundant)"],
        source_note="Tabachnick, B.G. & Fidell, L.S. (2019). Using Multivariate Statistics (7th ed.).",
    ),

    # ── CLUSTERING ───────────────────────────────────────────────────────────

    "kmeans": StatMethod(
        name="K-Means Clustering",
        category="Clustering",
        purpose="Partition observations into k mutually exclusive clusters based on feature similarity.",
        min_vars=2,
        outcome_roles=[],
        predictor_roles=["interval", "ratio"],
        measurement_levels_required=["interval"],
        n_groups_required=None,
        paired_required=None,
        assumptions=[
            "Variables measured at interval or ratio level",
            "Clusters are approximately spherical and equal-sized",
            "Variables are standardised before clustering",
            "k is specified a priori or selected via elbow/silhouette",
        ],
        min_n=30,
        effect_size_measures=["Silhouette coefficient", "Within-cluster sum of squares", "Calinski–Harabasz index"],
        ci_available=False,
        nonparametric_alternative="hierarchical_clustering",
        parametric_alternative=None,
        interpretation_guidance=(
            "Standardise all variables before running. "
            "Use elbow method and silhouette analysis to choose k. "
            "Validate clusters with external criteria or replication."
        ),
        limitations=[
            "Requires pre-specification of k",
            "Sensitive to initial centroid placement — run multiple times",
            "Assumes spherical clusters of similar size",
            "Not suitable for categorical variables",
        ],
        warning_conditions=["n < 30", "Un-standardised variables", "k > n/5"],
        source_note="MacQueen, J. (1967). Some methods for classification and analysis of multivariate observations.",
    ),

    "hierarchical_clustering": StatMethod(
        name="Hierarchical Clustering",
        category="Clustering",
        purpose="Build a hierarchy of clusters (dendrogram) to explore natural groupings in data.",
        min_vars=2,
        outcome_roles=[],
        predictor_roles=["interval", "ratio", "ordinal", "nominal"],
        measurement_levels_required=["ordinal"],
        n_groups_required=None,
        paired_required=None,
        assumptions=[
            "Distance/dissimilarity measure is appropriate for variable types",
            "Variables standardised for Euclidean-based methods",
        ],
        min_n=10,
        effect_size_measures=["Cophenetic correlation", "Silhouette coefficient"],
        ci_available=False,
        nonparametric_alternative=None,
        parametric_alternative="kmeans",
        interpretation_guidance=(
            "Choose linkage method based on expected cluster structure: "
            "Ward's for compact equal clusters; average for general use. "
            "Cut dendrogram at height that produces theoretically meaningful k."
        ),
        limitations=[
            "O(n²) memory and O(n² log n) time — slow for large N",
            "Merges are irreversible (no reassignment once joined)",
        ],
        warning_conditions=["n > 1000 (use k-means instead)", "Mixed-type variables without appropriate distance measure"],
        source_note="Ward, J.H. (1963). Hierarchical grouping to optimize an objective function. JASA.",
    ),

    # ── TIME SERIES ──────────────────────────────────────────────────────────

    "trend_analysis": StatMethod(
        name="Trend Analysis",
        category="Time Series",
        purpose="Detect and quantify directional change in a variable measured at regular time intervals.",
        min_vars=2,
        outcome_roles=["interval", "ratio", "ordinal"],
        predictor_roles=["datetime"],
        measurement_levels_required=["ordinal"],
        n_groups_required=None,
        paired_required=None,
        assumptions=[
            "Observations ordered by time",
            "Regular or approximately regular time intervals",
            "No seasonality confounding (or decomposed prior to test)",
        ],
        min_n=10,
        effect_size_measures=["Slope (regression coefficient)", "Mann–Kendall τ", "Sen's slope"],
        ci_available=True,
        nonparametric_alternative=None,
        parametric_alternative=None,
        interpretation_guidance=(
            "Use Mann–Kendall test (non-parametric) for monotonic trend detection. "
            "Use linear regression slope for magnitude. "
            "Decompose series into trend + seasonal + residual before testing if seasonality suspected."
        ),
        limitations=[
            "Linear trend may not capture complex temporal patterns",
            "Autocorrelated residuals violate independence — use autocorrelation-corrected tests",
        ],
        warning_conditions=["Significant autocorrelation in residuals", "Irregular time intervals", "n < 10 time points"],
        source_note="Mann, H.B. (1945). Nonparametric tests against trend. Econometrica.",
    ),

    # ── COVARIANCE (RULE-KB-01) ───────────────────────────────────────────────

    "covariance": StatMethod(
        name="Covariance",
        category="Correlation",
        purpose=(
            "Measure the directional co-movement of two numerical variables. "
            "Positive = move together; negative = move inversely; near-zero = no consistent direction."
        ),
        min_vars=2,
        outcome_roles=["interval", "ratio"],
        predictor_roles=["interval", "ratio"],
        measurement_levels_required=["interval"],
        n_groups_required=None,
        paired_required=None,
        assumptions=[
            "Both variables are numeric (interval or ratio level)",
            "Observations are independent",
            "No extreme outliers (covariance is sensitive to outliers, like the mean)",
        ],
        min_n=3,
        effect_size_measures=[
            "Not applicable — covariance magnitude is unit-dependent and not interpretable as effect size",
            "Use Pearson r = Cov(X,Y) / (SD_X * SD_Y) to obtain a standardised effect size",
        ],
        ci_available=False,
        nonparametric_alternative="spearman_correlation",
        parametric_alternative="pearson_correlation",
        interpretation_guidance=(
            "RULE-KB-01: Covariance conveys direction only. "
            "Positive covariance: both variables tend to be above (or below) their respective means together. "
            "Negative covariance: when one is above its mean the other tends to be below. "
            "CRITICAL: the magnitude of the covariance value cannot be compared across studies, variables, "
            "or measurement units. Changing km to m multiplies covariance by 1000 without changing the "
            "underlying relationship. Always normalise to Pearson r for strength interpretation."
        ),
        limitations=[
            "Magnitude is entirely unit-dependent — cannot be used to judge relationship strength",
            "Sensitive to outliers (like the mean)",
            "Does not capture non-linear relationships",
            "Does not imply causation",
        ],
        warning_conditions=[
            "Do not compare raw covariance values across different variable pairs or studies",
            "Always accompany with Pearson r for strength interpretation",
            "Outliers present — covariance may be distorted",
        ],
        source_note=(
            "RULE-KB-01 (derived from reference material). "
            "Pearson, K. (1895). Notes on regression and inheritance. Proc. R. Soc. London."
        ),
    ),
}


# ---------------------------------------------------------------------------
# Role / level classification helpers
# ---------------------------------------------------------------------------

_BLOCKED_ROLES: frozenset[str] = frozenset({
    "identifier",
    "serial_number",
    "administrative_code",
    "constant",
    "free_text",
})

_NUMERIC_LEVELS: frozenset[str] = frozenset({"interval", "ratio"})
_ORDINAL_LEVELS: frozenset[str] = frozenset({"ordinal"})
_CATEGORICAL_LEVELS: frozenset[str] = frozenset({"nominal"})

_NUMERIC_ROLES: frozenset[str] = frozenset({"substantive_numerical"})
_ORDINAL_ROLES: frozenset[str] = frozenset({"ordinal"})
_CATEGORICAL_ROLES: frozenset[str] = frozenset({"categorical", "boolean"})


def _is_numeric(role: str, level: str) -> bool:
    return role in _NUMERIC_ROLES or level in _NUMERIC_LEVELS


def _is_ordinal(role: str, level: str) -> bool:
    return role in _ORDINAL_ROLES or level in _ORDINAL_LEVELS


def _is_categorical(role: str, level: str) -> bool:
    return role in _CATEGORICAL_ROLES or level in _CATEGORICAL_LEVELS


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_candidate_methods(
    role_a: str,
    level_a: str,
    role_b: str,
    level_b: str,
    n_groups: Optional[int],
    paired: Optional[bool],
    n_obs: int,
    n_valid_a: int,
    n_valid_b: int,
    missing_pct_a: float,
    missing_pct_b: float,
) -> list[tuple[str, StatMethod, list[str], list[str]]]:
    """Return a ranked list of candidate statistical methods for a pair of variables.

    Reasoning is based entirely on variable roles and measurement levels — column
    names are never examined.  The function inspects sample-size and missingness
    thresholds to add eligibility notes and warnings.

    Parameters
    ----------
    role_a : str
        Inferred role of variable A (e.g. ``"substantive_numerical"``).
    level_a : str
        Measurement level of variable A (e.g. ``"ratio"``).
    role_b : str
        Inferred role of variable B.
    level_b : str
        Measurement level of variable B.
    n_groups : int or None
        Number of distinct groups in the grouping variable (when one variable
        is categorical).  Pass ``None`` if unknown or not applicable.
    paired : bool or None
        ``True`` if a paired/repeated-measures design; ``False`` if independent;
        ``None`` if unknown.
    n_obs : int
        Total number of observations (rows) in the dataset.
    n_valid_a : int
        Number of non-missing values in variable A.
    n_valid_b : int
        Number of non-missing values in variable B.
    missing_pct_a : float
        Proportion of missing values in variable A (0.0–1.0).
    missing_pct_b : float
        Proportion of missing values in variable B (0.0–1.0).

    Returns
    -------
    list of (method_key, StatMethod, eligibility_notes, warnings)
        Sorted by descending appropriateness.  ``eligibility_notes`` lists
        positive reasons the method applies.  ``warnings`` lists cautions.
        Returns an empty list with a blocking reason encoded as a single-element
        warning if either variable has a blocked role.
    """

    # ── 0. Block non-analytic roles ─────────────────────────────────────────
    for role, label in ((role_a, "A"), (role_b, "B")):
        if role in _BLOCKED_ROLES:
            return [
                (
                    "__blocked__",
                    StatMethod(
                        name="BLOCKED",
                        category="",
                        purpose="",
                        min_vars=0,
                        outcome_roles=[],
                        predictor_roles=[],
                        measurement_levels_required=[],
                        n_groups_required=None,
                        paired_required=None,
                        assumptions=[],
                        min_n=0,
                        effect_size_measures=[],
                        ci_available=False,
                        nonparametric_alternative=None,
                        parametric_alternative=None,
                        interpretation_guidance="",
                        limitations=[],
                        warning_conditions=[],
                        source_note="",
                    ),
                    [],
                    [
                        f"Variable {label} has role '{role}' which is not suitable for "
                        "statistical analysis.  Blocked roles: identifier, serial_number, "
                        "administrative_code, constant, free_text."
                    ],
                )
            ]

    # ── 1. Global sample-size checks ────────────────────────────────────────
    global_warnings: list[str] = []

    if n_obs < 5:
        return [
            (
                "__insufficient_n__",
                StatMethod(
                    name="INSUFFICIENT SAMPLE",
                    category="",
                    purpose="",
                    min_vars=0,
                    outcome_roles=[],
                    predictor_roles=[],
                    measurement_levels_required=[],
                    n_groups_required=None,
                    paired_required=None,
                    assumptions=[],
                    min_n=0,
                    effect_size_measures=[],
                    ci_available=False,
                    nonparametric_alternative=None,
                    parametric_alternative=None,
                    interpretation_guidance="",
                    limitations=[],
                    warning_conditions=[],
                    source_note="",
                ),
                [],
                [f"n = {n_obs} is below the absolute minimum of 5 required for any inferential test."],
            )
        ]

    if n_obs < 30:
        global_warnings.append(
            f"Small sample (n = {n_obs} < 30): parametric tests may be unreliable.  "
            "Prefer non-parametric alternatives."
        )

    if missing_pct_a > 0.50:
        global_warnings.append(
            f"Variable A has {missing_pct_a:.0%} missing values — results may be unreliable."
        )
    if missing_pct_b > 0.50:
        global_warnings.append(
            f"Variable B has {missing_pct_b:.0%} missing values — results may be unreliable."
        )

    small_n = n_obs < 30

    # ── 2. Classify both variables ──────────────────────────────────────────
    num_a = _is_numeric(role_a, level_a)
    ord_a = _is_ordinal(role_a, level_a)
    cat_a = _is_categorical(role_a, level_a)

    num_b = _is_numeric(role_b, level_b)
    ord_b = _is_ordinal(role_b, level_b)
    cat_b = _is_categorical(role_b, level_b)

    # treat boolean as categorical with 2 groups
    if role_b in ("boolean",) and n_groups is None:
        n_groups = 2
    if role_a in ("boolean",) and n_groups is None:
        n_groups = 2

    # ── 3. Build candidate list ─────────────────────────────────────────────
    # Each entry: (method_key, eligibility_notes, extra_warnings)
    raw: list[tuple[str, list[str], list[str]]] = []

    # ── NUM × NUM ────────────────────────────────────────────────────────────
    if num_a and num_b:
        if not small_n:
            raw.append((
                "pearson_correlation",
                ["Both variables are continuous (interval/ratio)", "Appropriate for linear association"],
                [],
            ))
        else:
            raw.append((
                "pearson_correlation",
                ["Both variables are continuous (interval/ratio)"],
                ["Small n — verify normality before using Pearson; consider Spearman"],
            ))

        raw.append((
            "spearman_correlation",
            ["Robust rank-based alternative for continuous or ordinal data"],
            ["Less powerful than Pearson when bivariate normality holds"] if not small_n else [],
        ))

        raw.append((
            "kendall_tau",
            ["Robust to ties and small samples"],
            [],
        ))

        raw.append((
            "simple_linear_regression",
            ["Continuous predictor and outcome — regression models predictive relationship"],
            (["Small n (< 20): regression estimates may be unstable"] if n_obs < 20 else []),
        ))

    # ── NUM × CAT ────────────────────────────────────────────────────────────
    elif (num_a and cat_b) or (cat_a and num_b):
        g = n_groups if n_groups is not None else 2  # default assumption

        if paired:
            raw.append((
                "paired_ttest",
                ["Paired/repeated design with continuous outcome"],
                (["Small n — verify normality of differences"] if small_n else []),
            ))
            raw.append((
                "wilcoxon_signed_rank",
                ["Non-parametric paired alternative"],
                [],
            ))

        elif g == 2:
            raw.append((
                "welch_ttest",
                [f"Continuous outcome, 2-group categorical predictor (n_groups = {g})",
                 "Welch variant does not require equal variances"],
                (["Small n per group"] if small_n else []),
            ))
            if not small_n:
                raw.append((
                    "independent_ttest",
                    [f"Continuous outcome, 2-group categorical predictor (n_groups = {g})"],
                    ["Assumes homogeneity of variance — verify with Levene's test"],
                ))
            raw.append((
                "mann_whitney_u",
                ["Non-parametric 2-group comparison", "Robust to non-normality and outliers"],
                [],
            ))
            raw.append((
                "point_biserial",
                ["Correlation coefficient for binary predictor × continuous outcome"],
                [],
            ))

        else:  # g >= 3
            raw.append((
                "welch_anova",
                [f"Continuous outcome, {g}-group categorical predictor", "Robust to unequal variances"],
                (["Small n per group"] if small_n else []),
            ))
            if not small_n:
                raw.append((
                    "one_way_anova",
                    [f"Continuous outcome, {g}-group categorical predictor"],
                    ["Assumes homogeneity of variance — verify with Levene's test"],
                ))
            raw.append((
                "kruskal_wallis",
                [f"Non-parametric {g}-group comparison", "Robust to non-normality"],
                [],
            ))

    # ── ORD × * ──────────────────────────────────────────────────────────────
    elif ord_a or ord_b:
        raw.append((
            "spearman_correlation",
            ["At least one variable is ordinal — rank-based correlation is appropriate"],
            [],
        ))
        raw.append((
            "kendall_tau",
            ["Robust ordinal association, preferred for small samples or many ties"],
            [],
        ))

        g = n_groups if n_groups is not None else 0
        if g == 2:
            raw.append((
                "mann_whitney_u",
                ["Ordinal outcome with 2-group categorical predictor"],
                [],
            ))
        elif g >= 3:
            raw.append((
                "kruskal_wallis",
                [f"Ordinal outcome with {g}-group categorical predictor"],
                [],
            ))

    # ── CAT × CAT ────────────────────────────────────────────────────────────
    elif cat_a and cat_b:
        raw.append((
            "chi_square_independence",
            ["Both variables are categorical — test of independence applies"],
            ["Verify expected cell frequencies ≥ 5 before relying on χ² p-value"],
        ))
        raw.append((
            "cramers_v",
            ["Effect-size measure for categorical × categorical association"],
            [],
        ))
        if n_groups == 2 or (
            level_a in ("nominal",) and level_b in ("nominal",) and n_groups is None
        ):
            raw.append((
                "fishers_exact",
                ["2×2 table — exact test appropriate when expected counts are small"],
                ["Use chi-square if all expected counts ≥ 5 and N ≥ 20"],
            ))
            raw.append((
                "odds_ratio",
                ["Binary × binary design — quantifies group odds difference"],
                [],
            ))

    # ── Fallback ─────────────────────────────────────────────────────────────
    if not raw:
        raw.append((
            "descriptive_stats",
            ["No suitable bivariate method found for role/level combination — summary statistics only"],
            ["Review variable roles and measurement levels for a more specific recommendation"],
        ))

    # ── 4. Apply n < min_n filter + per-method warnings ────────────────────
    results: list[tuple[str, StatMethod, list[str], list[str]]] = []
    for key, eligibility, extra_w in raw:
        if key not in METHOD_KB:
            continue
        method = METHOD_KB[key]
        warnings = list(global_warnings) + list(extra_w)

        # Add warning (not block) if n is below method minimum
        if n_obs < method.min_n:
            warnings.append(
                f"n = {n_obs} is below the recommended minimum of {method.min_n} "
                f"for {method.name}."
            )

        results.append((key, method, eligibility, warnings))

    # ── 5. Sort: prefer methods with no warnings first, then by min_n desc ──
    results.sort(key=lambda t: (len(t[3]), -METHOD_KB[t[0]].min_n if t[0] in METHOD_KB else 0))

    return results
