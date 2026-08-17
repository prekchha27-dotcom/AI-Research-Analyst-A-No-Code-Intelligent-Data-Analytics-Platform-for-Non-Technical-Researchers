"""
Method Recommender
Rule-based logic — no LLM calculations here.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class MethodRecommendation:
    method:      str
    stars:       int
    what:        str
    why:         str
    when:        str
    assumptions: List[str]
    alternatives: List[str]
    caution:     str = ""


def recommend_statistical_method(
    objective: str,
    dep_type:  str,
    ind_types: List[str],
    n_groups:  int = 0,
    n_obs:     int = 0,
    paired:    bool = False,
) -> List[MethodRecommendation]:

    recs = []

    # DESCRIBE
    if objective == "describe":
        recs.append(MethodRecommendation(
            method="Descriptive Statistics",
            stars=5,
            what="Summarises data using mean, median, mode, standard deviation, min, max, quartiles, and more.",
            why="Essential first step — you need to understand your data before applying any inferential tests.",
            when="Always — as the very first step of any data analysis.",
            assumptions=["No specific assumptions required."],
            alternatives=["Frequency tables for categorical variables"],
        ))
        if dep_type == "categorical":
            recs.append(MethodRecommendation(
                method="Frequency Table",
                stars=5,
                what="Counts how often each category occurs and calculates the percentage.",
                why="Categorical variables cannot be summarised by mean/median. Frequency tables are the correct approach.",
                when="When your variable has distinct categories (e.g. gender, department, programme).",
                assumptions=["No specific assumptions required."],
                alternatives=["Cross-tabulation (if relating two categorical variables)"],
            ))
        return recs

    # COMPARE — two groups
    if objective == "compare" and dep_type == "numerical" and n_groups == 2:
        if paired:
            recs.append(MethodRecommendation(
                method="Paired Samples t-Test",
                stars=5,
                what="Compares the mean of the same subjects measured twice.",
                why="Paired measurements share within-subject variance. The paired t-test correctly accounts for this.",
                when="Before/after studies, pre-test/post-test designs, same subjects under two conditions.",
                assumptions=["Differences between pairs are approximately normally distributed.",
                             "Observations are paired — same subject measured twice."],
                alternatives=["Wilcoxon Signed-Rank Test (non-parametric; if differences are clearly non-normal)"],
            ))
        else:
            recs.append(MethodRecommendation(
                method="Independent Samples t-Test",
                stars=5,
                what="Compares the means of a numerical variable between two independent groups.",
                why="Two separate groups are being compared — the independent t-test is the standard parametric method.",
                when="Comparing males vs females; treated vs control; two departments.",
                assumptions=["Independent groups (not same subjects).",
                             "Approximately normal distribution within each group (or n ≥ 30 per group).",
                             "Homogeneity of variance (tested via Levene's test — Welch correction applied if violated)."],
                alternatives=["Mann-Whitney U Test (non-parametric; if normality violated with small sample)"],
                caution="If sample is very small (< 20 per group) and clearly non-normal, consider Mann-Whitney U.",
            ))
        return recs

    # COMPARE — 3+ groups
    if objective == "compare" and dep_type == "numerical" and n_groups > 2:
        recs.append(MethodRecommendation(
            method="One-Way ANOVA",
            stars=5,
            what="Compares means across three or more independent groups simultaneously.",
            why="Running multiple t-tests inflates the Type I error rate. ANOVA controls for this while testing all groups at once.",
            when="Comparing more than two group means (e.g. three departments, four regions).",
            assumptions=["Independent groups.",
                         "Approximately normal distribution within each group.",
                         "Homogeneity of variance across groups (Levene's test)."],
            alternatives=["Kruskal-Wallis Test (non-parametric)", "Welch's ANOVA (if variances unequal)"],
            caution="If ANOVA is significant, apply post-hoc tests (Tukey HSD) to identify which specific groups differ.",
        ))
        recs.append(MethodRecommendation(
            method="Kruskal-Wallis Test",
            stars=3,
            what="Non-parametric equivalent of one-way ANOVA.",
            why="Does not assume normality — suitable when normality assumption is clearly violated.",
            when="Small samples, ordinal data, or heavily skewed distributions with 3+ groups.",
            assumptions=["Independent groups.", "At least ordinal measurement level."],
            alternatives=["One-Way ANOVA (preferred if assumptions are met — more powerful)"],
        ))
        return recs

    # RELATE — two numerical
    if objective == "relate" and dep_type == "numerical" and all(t == "numerical" for t in ind_types):
        recs.append(MethodRecommendation(
            method="Pearson Correlation",
            stars=5,
            what="Measures the strength and direction of the linear relationship between two numerical variables. Ranges from -1 to +1.",
            why="Both variables are continuous — Pearson is the standard parametric measure of linear association.",
            when="Examining whether study hours relate to scores; whether demand relates to supply.",
            assumptions=["Both variables are continuous (interval or ratio).",
                         "Approximately normal distribution.",
                         "Linear relationship.",
                         "No severe outliers."],
            alternatives=["Spearman Correlation (if normality violated, data is ordinal, or outliers are present)"],
            caution="Correlation does NOT imply causation. A significant r only indicates association, not a causal mechanism.",
        ))
        recs.append(MethodRecommendation(
            method="Spearman Rank Correlation",
            stars=4,
            what="Rank-based correlation. Measures the strength of a monotonic relationship.",
            why="More robust than Pearson when normality is violated or data contains outliers.",
            when="Ordinal data, skewed distributions, or when Pearson assumptions are not met.",
            assumptions=["Monotonic relationship.", "Ordinal, interval, or ratio data."],
            alternatives=["Pearson Correlation (if both variables are normally distributed with no severe outliers)"],
        ))
        return recs

    # RELATE — two categorical
    if objective == "relate" and dep_type == "categorical":
        recs.append(MethodRecommendation(
            method="Chi-Square Test of Independence",
            stars=5,
            what="Tests whether two categorical variables are statistically independent or associated.",
            why="The standard inferential test for examining whether a relationship exists between two categorical variables.",
            when="Is gender related to programme choice? Is department related to service availability?",
            assumptions=["Independent observations.",
                         "Expected cell frequency ≥ 5 in at least 80% of cells.",
                         "Both variables are categorical."],
            alternatives=["Fisher's Exact Test (for small 2×2 tables)", "Cramér's V (for effect size)"],
            caution="Chi-square confirms whether an association exists but not its direction or magnitude. Use Cramér's V for effect size.",
        ))
        return recs

    # PREDICT — numerical outcome
    if objective == "predict" and dep_type == "numerical":
        recs.append(MethodRecommendation(
            method="Linear Regression",
            stars=5,
            what="Models the relationship between a numerical outcome and one or more predictor variables.",
            why="Allows quantification of the relationship and prediction of the outcome variable from predictors.",
            when="Predicting exam scores from study hours; predicting demand from population.",
            assumptions=["Linear relationship between predictors and outcome.",
                         "Independence of observations.",
                         "Homoscedasticity (constant variance of residuals).",
                         "Normality of residuals.",
                         "No severe multicollinearity if multiple predictors."],
            alternatives=["Multiple Regression (more than one predictor)",
                          "Ridge/Lasso Regression (if multicollinearity is present)"],
            caution="R² describes fit but does not prove causation. Outliers can strongly influence regression coefficients.",
        ))
        return recs

    # PREDICT — binary outcome
    if objective == "predict" and dep_type in ("categorical", "binary"):
        recs.append(MethodRecommendation(
            method="Logistic Regression",
            stars=5,
            what="Predicts the probability of a binary categorical outcome from one or more predictors.",
            why="The standard method when the outcome variable is binary (yes/no, pass/fail, 0/1).",
            when="Predicting whether a student passes, whether a customer subscribes, whether an event occurs.",
            assumptions=["Binary dependent variable.",
                         "Independence of observations.",
                         "No severe multicollinearity among predictors.",
                         "Minimum of ~10 events per predictor variable."],
            alternatives=["Decision Tree (non-parametric)", "Linear Discriminant Analysis"],
        ))
        return recs

    # Default
    recs.append(MethodRecommendation(
        method="Descriptive Statistics + Exploratory Analysis",
        stars=3,
        what="General data exploration.",
        why="The provided information is insufficient for a specific recommendation. Start with descriptive statistics.",
        when="When objective or variable types are unclear.",
        assumptions=["None specific."],
        alternatives=["Specify your research question and variable types for a targeted recommendation."],
    ))
    return recs


# ── Pairwise Test Decision Engine ─────────────────────────────────────────────

from dataclasses import dataclass as _dc, field as _field
from typing import List as _List

@_dc
class PairTestDecision:
    """
    Result returned by decide_test_for_pair().
    Describes the variable profiles, the recommended test, and why.
    """
    var_a:            str
    role_a:           str
    level_a:          str
    quality_a:        dict          # {missing_pct, unique, warnings}

    var_b:            str
    role_b:           str
    level_b:          str
    quality_b:        dict

    recommended_test: str           # e.g. "Independent Samples t-Test"
    rationale:        str           # why this test fits THIS combination
    assumptions:      _List[str]
    alternatives:     _List[str]
    caution:          str
    n_groups_b:       int = 0       # populated when var_b is categorical grouping


# Role aliases
_NUM  = "substantive_numerical"
_CAT  = "categorical"
_ORD  = "ordinal"
_BOOL = "boolean"


def decide_test_for_pair(
    var_a: str,
    role_a: str,
    level_a: str,
    var_b: str,
    role_b: str,
    level_b: str,
    n_groups_b: int = 0,
    n_obs: int = 0,
    quality_a: Optional[dict] = None,
    quality_b: Optional[dict] = None,
) -> PairTestDecision:
    """
    Decide the appropriate statistical test given two variables and their
    analytically-classified roles.

    The decision is based entirely on the *combination* of roles — not on
    abstract questions about "what type is your outcome variable".

    Parameters
    ----------
    var_a, var_b     : column names
    role_a, role_b   : analytical roles (from ColumnClassification.analytical_role)
    level_a, level_b : measurement levels (nominal / ordinal / interval / ratio / …)
    n_groups_b       : number of unique categories in var_b (if categorical)
    n_obs            : total non-missing paired observations
    quality_a/b      : optional dicts with keys: missing_pct, unique, warnings
    """
    quality_a = quality_a or {}
    quality_b = quality_b or {}

    # Normalise roles into broad classes
    def _is_num(role):
        return role in (_NUM,)

    def _is_cat(role):
        return role in (_CAT, _BOOL)

    def _is_ord(role):
        return role in (_ORD,)

    def _is_any_cat(role):
        return role in (_CAT, _BOOL, _ORD)

    a_num = _is_num(role_a)
    b_num = _is_num(role_b)
    a_cat = _is_any_cat(role_a)
    b_cat = _is_any_cat(role_b)
    a_ord = _is_ord(role_a)
    b_ord = _is_ord(role_b)

    # ── CASE 1: Both numerical ─────────────────────────────────────────────────
    if a_num and b_num:
        return PairTestDecision(
            var_a=var_a, role_a=role_a, level_a=level_a, quality_a=quality_a,
            var_b=var_b, role_b=role_b, level_b=level_b, quality_b=quality_b,
            recommended_test="Pearson Correlation (or Spearman if skewed/outliers)",
            rationale=(
                f"Both **{var_a}** (numerical) and **{var_b}** (numerical) are continuous "
                f"measurement variables. Correlation measures the strength and direction of "
                f"their linear (Pearson) or monotonic (Spearman) relationship. "
                f"Use the Correlation tab → the system will automatically check skewness "
                f"and outliers and recommend Pearson vs Spearman for you."
            ),
            assumptions=[
                "Both variables are continuous (interval or ratio scale).",
                "Pearson: approximately normal distribution; no severe outliers; linear relationship.",
                "Spearman: monotonic relationship; no normality required.",
            ],
            alternatives=["Kendall's Tau (ordinal or small sample with ties)", "Linear Regression (if predicting one from the other)"],
            caution="Correlation does not imply causation.",
            n_groups_b=0,
        )

    # ── CASE 2: Numerical + Categorical (2 groups) ────────────────────────────
    if a_num and b_cat and n_groups_b == 2:
        return PairTestDecision(
            var_a=var_a, role_a=role_a, level_a=level_a, quality_a=quality_a,
            var_b=var_b, role_b=role_b, level_b=level_b, quality_b=quality_b,
            recommended_test="Independent Samples t-Test",
            rationale=(
                f"**{var_a}** is a numerical outcome and **{var_b}** is a categorical "
                f"grouping variable with exactly **2 groups**. "
                f"The independent samples t-test compares the mean of {var_a} between the "
                f"two groups of {var_b}. Levene's test will check equal-variance; "
                f"Welch's correction will be applied automatically if violated."
            ),
            assumptions=[
                f"{var_a}: approximately normal distribution within each group (or n ≥ 30 per group).",
                f"{var_b}: exactly 2 independent groups.",
                "Groups are independent (not the same subjects measured twice).",
                "Homogeneity of variance (tested automatically via Levene's test).",
            ],
            alternatives=["Mann-Whitney U (if normality clearly violated with small sample)"],
            caution=(
                "If the groups in your grouping variable represent the same subjects "
                "measured at two time points, use the Paired t-Test instead."
            ),
            n_groups_b=2,
        )

    # ── CASE 3: Numerical + Categorical (3+ groups) ───────────────────────────
    if a_num and b_cat and n_groups_b > 2:
        return PairTestDecision(
            var_a=var_a, role_a=role_a, level_a=level_a, quality_a=quality_a,
            var_b=var_b, role_b=role_b, level_b=level_b, quality_b=quality_b,
            recommended_test="One-Way ANOVA",
            rationale=(
                f"**{var_a}** is a numerical outcome and **{var_b}** is a categorical "
                f"grouping variable with **{n_groups_b} groups**. "
                f"One-Way ANOVA compares the mean of {var_a} across all {n_groups_b} groups simultaneously. "
                f"Running separate t-tests would inflate the Type I error rate — ANOVA avoids this."
            ),
            assumptions=[
                f"{var_a}: approximately normal distribution within each group.",
                f"{var_b}: {n_groups_b} independent groups.",
                "Homogeneity of variance across groups (Levene's test).",
                "Independent observations.",
            ],
            alternatives=["Kruskal-Wallis Test (if normality clearly violated)", "Welch's ANOVA (if variances differ strongly)"],
            caution="If ANOVA is significant, apply Tukey HSD post-hoc to find which specific groups differ.",
            n_groups_b=n_groups_b,
        )

    # ── CASE 4: Both categorical ───────────────────────────────────────────────
    if a_cat and b_cat:
        return PairTestDecision(
            var_a=var_a, role_a=role_a, level_a=level_a, quality_a=quality_a,
            var_b=var_b, role_b=role_b, level_b=level_b, quality_b=quality_b,
            recommended_test="Chi-Square Test of Independence",
            rationale=(
                f"Both **{var_a}** ({role_a}) and **{var_b}** ({role_b}) are categorical variables. "
                f"Chi-Square tests whether these two variables are statistically independent — "
                f"i.e. whether the distribution of one variable differs across categories of the other. "
                f"Use the Chi-Square tab."
            ),
            assumptions=[
                "Both variables are categorical.",
                "Independent observations.",
                "Expected cell frequency ≥ 5 in at least 80% of cells (checked automatically).",
            ],
            alternatives=["Fisher's Exact Test (2×2 table with small expected counts)", "Cramér's V (for effect size after Chi-Square)"],
            caution="Chi-Square confirms whether an association exists but not its direction or magnitude.",
            n_groups_b=n_groups_b,
        )

    # ── CASE 5: Ordinal + Ordinal or Ordinal + Numerical ──────────────────────
    if (a_ord and b_ord) or (a_ord and b_num) or (a_num and b_ord):
        return PairTestDecision(
            var_a=var_a, role_a=role_a, level_a=level_a, quality_a=quality_a,
            var_b=var_b, role_b=role_b, level_b=level_b, quality_b=quality_b,
            recommended_test="Spearman Rank Correlation",
            rationale=(
                f"At least one of these variables (**{var_a}**: {role_a}, **{var_b}**: {role_b}) "
                f"is ordinal (ranked categories). Spearman correlation is the correct choice because "
                f"it does not assume interval-level measurement or normality — "
                f"it works on ranks. Use the Correlation tab and select Spearman."
            ),
            assumptions=[
                "At least one variable is ordinal or the data violates normality.",
                "Monotonic relationship between variables.",
                "No requirement for normality.",
            ],
            alternatives=["Kendall's Tau-b (preferred for small samples or many ties)", "Pearson (only if both are truly interval/ratio and normal)"],
            caution="Spearman measures monotonic association, not strictly linear association.",
            n_groups_b=0,
        )

    # ── CASE 6: Categorical + Numerical (reversed arg order) ─────────────────
    if a_cat and b_num:
        n_groups_a = n_groups_b  # caller passed groups for var_b; swap perspective
        if n_groups_a == 2:
            test_name = "Independent Samples t-Test"
            rationale = (
                f"**{var_a}** ({role_a}) is a categorical grouping variable with 2 groups "
                f"and **{var_b}** (numerical) is the outcome. "
                f"The t-test compares the mean of {var_b} between the two groups of {var_a}."
            )
        elif n_groups_a > 2:
            test_name = "One-Way ANOVA"
            rationale = (
                f"**{var_a}** ({role_a}) is a categorical grouping variable with {n_groups_a} groups "
                f"and **{var_b}** (numerical) is the outcome. "
                f"ANOVA compares the mean of {var_b} across all groups of {var_a}."
            )
        else:
            test_name = "Independent Samples t-Test or One-Way ANOVA"
            rationale = (
                f"**{var_a}** ({role_a}) is categorical and **{var_b}** is numerical. "
                f"The appropriate test depends on the number of groups in {var_a}. "
                f"Please upload your data and check the number of unique values."
            )
        return PairTestDecision(
            var_a=var_a, role_a=role_a, level_a=level_a, quality_a=quality_a,
            var_b=var_b, role_b=role_b, level_b=level_b, quality_b=quality_b,
            recommended_test=test_name,
            rationale=rationale,
            assumptions=[
                f"{var_b}: approximately normal distribution within each group (or n ≥ 30).",
                f"{var_a}: independent groups.",
                "Homogeneity of variance (tested automatically via Levene's test).",
            ],
            alternatives=["Mann-Whitney U (2 groups, non-parametric)", "Kruskal-Wallis (3+ groups, non-parametric)"],
            caution="",
            n_groups_b=n_groups_a,
        )

    # ── Fallback ───────────────────────────────────────────────────────────────
    return PairTestDecision(
        var_a=var_a, role_a=role_a, level_a=level_a, quality_a=quality_a,
        var_b=var_b, role_b=role_b, level_b=level_b, quality_b=quality_b,
        recommended_test="Cannot determine — review variable roles",
        rationale=(
            f"The combination of **{var_a}** ({role_a}) and **{var_b}** ({role_b}) "
            f"does not match a standard pairwise test pattern. "
            f"This usually means one or both variables has an unusual role "
            f"(e.g. free_text, identifier, constant). "
            f"Go to **Variable Classification** and confirm the roles are correct."
        ),
        assumptions=["Variable roles must be confirmed before a test can be recommended."],
        alternatives=["Confirm variable roles in Variable Classification, then re-run."],
        caution="",
        n_groups_b=0,
    )
