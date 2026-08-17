"""
AI Interpretation Layer
Receives VERIFIED statistical results from the engine and generates
natural-language explanations. Never invents statistical values.

Principle: Statistical Result -> AI Interpretation (clearly separated)

Rule-set additions (sourced from reference material, converted to generic auditable rules):
──────────────────────────────────────────────────────────────────────────────────────────
RULE-INTERP-01  Sampling quality warning: non-probability sampling methods (convenience,
                purposive) introduce selection bias and limit generalisability. Any result
                based on a non-probability sample must carry an explicit caveat that
                findings cannot be statistically generalised to the full population.
                Probability sampling (simple random, stratified, systematic, cluster)
                is the prerequisite for valid inferential statistics.

RULE-INTERP-02  Causation vs correlation: statistical significance and effect size alone
                do not establish causation. Three conditions are required for causation:
                (a) association exists, (b) cause precedes effect in time,
                (c) alternative explanations (confounders / third variables) are ruled out.
                Observational data satisfies (a) only. Always name the hidden-third-variable
                possibility when presenting correlation or regression results.

RULE-INTERP-03  Covariance interpretation: raw covariance value is unit-dependent and
                conveys direction only (positive / negative / zero). Strength of relationship
                requires Pearson r. The chatbot must redirect covariance questions to r.

RULE-DATA-01    Types of data taxonomy: Quantitative (numerical) = Discrete (countable,
                whole numbers only) or Continuous (any value in a range, decimals valid).
                Qualitative (categorical) = Nominal (no order, labels only) or Ordinal
                (ordered categories, gaps between ranks not necessarily equal). The correct
                data type determines which statistical methods are valid.

RULE-DISP-01    Measures of dispersion: Deviation = single value minus mean (x − μ).
                Sum of raw deviations always equals zero (positives and negatives cancel).
                Variance (σ²) = mean of squared deviations; population formula divides by N,
                sample formula divides by (n−1) to correct for bias (Bessel's correction).
                Standard Deviation (σ) = √Variance — returns the unit to the original scale.
                Range = Max − Min (simple but sensitive to outliers).
                IQR = Q3 − Q1 (robust to outliers; used in Tukey fence outlier detection).

RULE-DISP-02    Z-score: z = (x − μ) / σ. Expresses how many standard deviations a single
                value is above (positive z) or below (negative z) the mean. Enables
                comparison of values from different scales. z = 0 → at the mean;
                |z| > 2 → in the outer 5% for a normal distribution.

RULE-DISP-03    MAD (Mean Absolute Deviation): average of |xi − μ| across all observations.
                Uses absolute values instead of squares to remove sign. MAD is more robust
                to extreme outliers than standard deviation. SD is preferred in inferential
                tests because it has better mathematical properties (differentiability,
                alignment with normal distribution theory). Use MAD as a descriptive
                robustness check when outliers are present.

RULE-DISP-04    Percentile: the p-th percentile is the value below which p% of observations
                fall. Q1 = 25th percentile, Q2 (median) = 50th, Q3 = 75th. Percentile is a
                measure of position, not of spread or central tendency.

RULE-INFER-01   Descriptive vs Inferential statistics: Descriptive statistics summarise the
                data actually collected (mean, SD, frequency). Inferential statistics use
                sample data to make probability-based statements about an unobserved
                population (hypothesis tests, confidence intervals, regression). Inferential
                conclusions are only as valid as the sampling method that produced the data.

RULE-INFER-02   Central Limit Theorem (CLT): as sample size increases (generally n ≥ 30),
                the sampling distribution of the sample mean approaches a normal distribution
                regardless of the shape of the original population distribution. This is the
                theoretical basis for using z-tests and t-tests even when population
                distributions are not perfectly normal. The CLT applies to the MEAN of
                repeated samples, not to individual observations.

RULE-INFER-03   Normal distribution: a continuous, symmetric, bell-shaped distribution where
                Mean = Median = Mode, all equal and located at the centre. Total area under
                the curve = 1 (representing all possible probabilities). Key property:
                ~68% of values fall within ±1σ, ~95% within ±2σ, ~99.7% within ±3σ
                (the empirical rule). Basis for z-tests, t-tests, ANOVA, and regression
                inference. A variable need not be perfectly normal for tests to be valid
                (CLT and robustness of parametric tests apply).

RULE-INFER-04   Sampling distribution: the theoretical distribution of a statistic
                (e.g. sample mean x̄) computed across all possible samples of size n from a
                population. The standard deviation of the sampling distribution of x̄ is
                the Standard Error (SE = σ/√n). A larger sample produces a narrower (more
                precise) sampling distribution. Never confuse the sampling distribution with
                the distribution of individual observations.

RULE-INFER-05   Probability distributions: Discrete — Binomial (fixed n trials, binary
                outcome per trial, constant probability p); Poisson (count of rare events
                per unit time/space, single parameter λ = mean = variance). Continuous —
                Normal/Gaussian (symmetric, bell-shaped, defined by μ and σ). Match the
                distribution to the data type: Binomial for binary count outcomes; Poisson
                for rare count outcomes; Normal for continuous measurements or sample means
                via CLT.

RULE-VAR-01     Variable vs Random Variable: a variable is any attribute that can take
                different values. A random variable maps the outcome of a random event to a
                number, with each value having an associated probability. Discrete random
                variable: finite or countably infinite values (coin flips, counts). Continuous
                random variable: any value within a range (height, weight, time). The
                distinction determines which probability model is appropriate.

CONFLICT-01     (documented in statistics.py) Pearson r strength thresholds in the source
                material differ from Cohen (1988). Cohen's benchmarks are retained throughout
                this interpreter. No change to existing threshold logic.

CONFLICT-02     Pearson r strength thresholds in the pasted reference text use:
                |r| 0.7–1.0 = Strong, 0.4–0.6 = Moderate, 0 = No relationship,
                −0.4 to −0.6 = Moderate negative, −0.7 to −1.0 = Strong negative.
                This differs from Cohen (1988): |r| ≥ 0.50 = large, 0.30 = medium,
                0.10 = small. Cohen (1988) is the established academic standard and is
                retained. The pasted text's thresholds are documented here for transparency.
                Chatbot explanations use Cohen (1988) benchmarks only.
"""

import re


# ── Result Interpreter ────────────────────────────────────────────────────────

def interpret_result(result: dict) -> str:
    """
    Take a verified statistical result dict and produce a structured interpretation.
    The interpretation is ALWAYS built from the verified values in result{}.
    Covers all tests: correlation, t-test, ANOVA, chi-square, regression,
    non-parametric, Fisher's exact, point-biserial, goodness-of-fit,
    Welch ANOVA, Friedman, PCA, Cronbach's alpha.
    """
    test = result.get("test", "Unknown test")
    interp = result.get("interpretation", "")
    assumption_note = result.get("assumption_note", "")
    warning = result.get("warning", "")

    parts = []
    parts.append(f"### 📊 Statistical Result: {test}")
    parts.append("")

    # ── Missing-data note ──────────────────────────────────────────────────────
    n_total = result.get("n_total")
    n_used  = result.get("n_used") or result.get("n")
    if n_total and n_used and n_total != n_used:
        n_missing = n_total - n_used
        pct_miss   = round(n_missing / n_total * 100, 1)
        parts.append(
            f"**Missing data:** {n_missing} of {n_total} rows excluded "
            f"({pct_miss}% pairwise missing). Analysis used n = {n_used}."
        )

    # ── Fisher's Exact Test ────────────────────────────────────────────────────
    if "Fisher" in test:
        if "odds_ratio" in result:
            or_val = result["odds_ratio"]
            or_str = "increased odds" if or_val > 1 else ("decreased odds" if or_val < 1 else "no difference in odds")
            parts.append(f"**Odds Ratio:** {or_val}  *({or_str})*")
        if "ci_95" in result:
            ci = result["ci_95"]
            if ci and ci[0] is not None:
                parts.append(f"**95% CI (Odds Ratio):** [{ci[0]}, {ci[1]}]")
        if "phi" in result:
            phi = result["phi"]
            phi_str = "strong" if abs(phi) >= 0.5 else "moderate" if abs(phi) >= 0.3 else "weak"
            parts.append(f"**Effect size (Phi):** {phi}  *({phi_str} association)*")

    # ── Point-Biserial Correlation ─────────────────────────────────────────────
    elif "Point-Biserial" in test:
        if "r_pb" in result:
            r_val = result["r_pb"]
            strength = ("very strong" if abs(r_val) >= 0.9 else "strong" if abs(r_val) >= 0.7
                        else "moderate" if abs(r_val) >= 0.5 else "weak" if abs(r_val) >= 0.3 else "very weak")
            direction = "positive" if r_val > 0 else "negative"
            parts.append(f"**Point-Biserial r:** {r_val}  *(a {strength} {direction} association)*")
        if "ci_95" in result:
            ci = result["ci_95"]
            if ci and ci[0] is not None:
                parts.append(f"**95% CI (Fisher z):** [{ci[0]}, {ci[1]}]")

    # ── PCA ───────────────────────────────────────────────────────────────────
    elif "PCA" in test or "Principal Component" in test:
        if "total_variance_pct" in result:
            parts.append(f"**Total variance explained:** {result['total_variance_pct']}%")
        if "kaiser_n_components" in result:
            parts.append(f"**Kaiser criterion suggests:** {result['kaiser_n_components']} component(s) (eigenvalue ≥ 1)")
        if "kmo_approx" in result and result["kmo_approx"] is not None:
            parts.append(f"**Approximate KMO:** {result['kmo_approx']} — {result.get('kmo_label','')}")
        if "eigenvalues" in result:
            ev_str = ", ".join(str(e) for e in result["eigenvalues"][:6])
            if len(result["eigenvalues"]) > 6:
                ev_str += "…"
            parts.append(f"**Eigenvalues:** {ev_str}")
        if "explained_variance_ratio" in result:
            evr_str = ", ".join(f"{round(v*100,1)}%" for v in result["explained_variance_ratio"][:6])
            parts.append(f"**Variance per component:** {evr_str}")

    # ── Friedman Test ─────────────────────────────────────────────────────────
    elif "Friedman" in test:
        if "statistic" in result:
            parts.append(f"**Friedman χ²:** {result['statistic']}")
        if "kendalls_w" in result:
            w = result["kendalls_w"]
            w_str = "strong" if w >= 0.7 else "moderate" if w >= 0.5 else "weak"
            parts.append(f"**Kendall's W:** {w}  *({w_str} concordance)*")
        if "n_conditions" in result:
            parts.append(f"**Conditions tested:** {result['n_conditions']}")
        if "n_subjects" in result:
            parts.append(f"**Subjects (complete blocks):** {result['n_subjects']}")
        # Condition-level summary
        if "group_summary" in result:
            gs = result["group_summary"]
            if gs:
                g_lines = []
                for cond, stats in gs.items():
                    g_lines.append(f"  - **{cond}**: n={stats['n']}, median={stats['median']}, mean={stats['mean']}")
                parts.append("**Condition summary:**")
                parts.extend(g_lines)

    # ── Welch ANOVA ───────────────────────────────────────────────────────────
    elif "Welch" in test and "ANOVA" in test:
        if "f_statistic" in result:
            parts.append(f"**Welch's F:** {result['f_statistic']}")
        if "eta_squared" in result:
            e = result["eta_squared"]
            e_str = "large" if e >= 0.14 else "medium" if e >= 0.06 else "small"
            parts.append(f"**Effect size (η²):** {e}  *({e_str} effect)*")
        if "group_summary" in result:
            gs = result["group_summary"]
            if gs:
                g_lines = []
                for grp, stats in gs.items():
                    g_lines.append(f"  - **{grp}**: n={stats['n']}, mean={stats['mean']}, sd={stats['std']}")
                parts.append("**Group means:**")
                parts.extend(g_lines)

    # ── Chi-Square Goodness-of-Fit ─────────────────────────────────────────────
    elif "Goodness-of-Fit" in test:
        if "chi2" in result:
            parts.append(f"**Chi-square statistic (χ²):** {result['chi2']}")
        if "df" in result:
            parts.append(f"**Degrees of freedom:** {result['df']}")
        if "cohens_w" in result:
            w = result["cohens_w"]
            w_str = "large" if w >= 0.5 else "medium" if w >= 0.3 else "small"
            parts.append(f"**Effect size (Cohen's w):** {w}  *({w_str} effect)*")
        if "distribution" in result:
            parts.append(f"**Expected distribution:** {result['distribution']}")

    # ── Standard statistics ────────────────────────────────────────────────────
    else:
        if "r" in result and "r_pb" not in result:
            r_val = result["r"]
            strength = ("very strong" if abs(r_val) >= 0.9 else "strong" if abs(r_val) >= 0.7
                        else "moderate" if abs(r_val) >= 0.5 else "weak" if abs(r_val) >= 0.3
                        else "very weak")
            direction = "positive" if r_val > 0 else "negative"
            coeff_label = "τ" if "Kendall" in test else "r"
            parts.append(f"**Correlation coefficient ({coeff_label}):** {r_val}  *(a {strength} {direction} association)*")
        if "chi2" in result and "Goodness" not in test:
            parts.append(f"**Chi-square statistic (χ²):** {result['chi2']}")
        if "t_statistic" in result:
            parts.append(f"**t-statistic:** {result['t_statistic']}")
        if "f_statistic" in result:
            parts.append(f"**F-statistic:** {result['f_statistic']}")
        if "r_squared" in result:
            pct = round(result['r_squared'] * 100, 1)
            parts.append(f"**R² (coefficient of determination):** {result['r_squared']}  *({pct}% variance explained)*")
        if "adj_r_squared" in result:
            parts.append(f"**Adjusted R²:** {result['adj_r_squared']}")
        if "df" in result and "chi2" in result:
            parts.append(f"**Degrees of freedom (df):** {result['df']}")
        if "cramers_v" in result:
            v = result["cramers_v"]
            v_str = "strong" if v >= 0.5 else "moderate" if v >= 0.3 else "weak" if v >= 0.1 else "negligible"
            parts.append(f"**Effect size (Cramér's V):** {v}  *({v_str} association)*")
        if "cohens_d" in result:
            d = result["cohens_d"]
            d_str = "large" if abs(d) >= 0.8 else "medium" if abs(d) >= 0.5 else "small"
            parts.append(f"**Effect size (Cohen's d):** {d}  *({d_str} effect)*")
        if "eta_squared" in result and "Welch" not in test:
            e = result["eta_squared"]
            e_str = "large" if e >= 0.14 else "medium" if e >= 0.06 else "small"
            parts.append(f"**Effect size (η²):** {e}  *({e_str} effect)*")
        if "rank_biserial_r" in result:
            rb = result["rank_biserial_r"]
            rb_str = "large" if abs(rb) >= 0.5 else "medium" if abs(rb) >= 0.3 else "small"
            parts.append(f"**Effect size (rank-biserial r):** {rb}  *({rb_str} effect)*")
        if "u_statistic" in result:
            parts.append(f"**U statistic:** {result['u_statistic']}")
        if "h_statistic" in result:
            parts.append(f"**H statistic:** {result['h_statistic']}")
            if "eta_squared_h" in result:
                eh = result["eta_squared_h"]
                eh_str = "large" if eh >= 0.14 else "medium" if eh >= 0.06 else "small"
                parts.append(f"**Effect size (η²H):** {eh}  *({eh_str} effect)*")

        # Group means for t-test
        if "mean1" in result and "mean2" in result:
            parts.append(f"**Group '{result.get('group1','G1')}' mean:** {result['mean1']}  "
                         f"**Group '{result.get('group2','G2')}' mean:** {result['mean2']}")

        # ANOVA group summary
        if "group_summary" in result and "Welch" not in test and "Friedman" not in test:
            gsummary = result["group_summary"]
            if gsummary:
                g_lines = []
                for grp, stats in gsummary.items():
                    g_lines.append(f"  - **{grp}**: n={stats['n']}, mean={stats['mean']}, sd={stats['std']}")
                parts.append("**Group means:**")
                parts.extend(g_lines)

    # ── Universal fields ───────────────────────────────────────────────────────
    if "p_value" in result and "PCA" not in test and "Principal Component" not in test:
        p = result["p_value"]
        sig_text = "✅ Statistically significant (p < 0.05)" if p < 0.05 else "❌ Not statistically significant (p ≥ 0.05)"
        parts.append(f"**p-value:** {p} — {sig_text}")
    if "n" in result:
        parts.append(f"**Sample size used (n):** {result['n']}")
    if "ci_95" in result and "Fisher" not in test and "Point-Biserial" not in test:
        ci = result.get("ci_95")
        if ci and ci[0] is not None:
            parts.append(f"**95% Confidence Interval:** [{ci[0]}, {ci[1]}]")

    parts.append("")
    parts.append(f"**Interpretation:** {interp}")
    parts.append("")

    if assumption_note:
        parts.append(f"**Assumptions:** {assumption_note}")

    if warning:
        parts.append(f"⚠️ **Warning:** {warning}")

    if result.get("posthoc_note"):
        parts.append(f"📌 **Post-hoc note:** {result['posthoc_note']}")

    parts.append("")
    parts.append("---")
    parts.append("*All values above are computed by the statistical engine and have not been modified. "
                 "p-value ≠ effect size — always report both. "
                 "Statistical significance does not imply practical importance or causation.*")

    return "\n".join(parts)


def explain_graph(chart_type: str, x_col: str, y_col: str,
                  df_stats: dict) -> str:
    """
    Explain a generated graph in plain language based on actual computed summary statistics.
    df_stats should be a dict of pre-computed descriptive values from the data.
    """
    type_labels = {
        "bar_chart":     "bar chart",
        "pie_chart":     "pie chart",
        "histogram":     "histogram",
        "line_chart":    "line chart",
        "scatter_plot":  "scatter plot",
        "box_plot":      "box plot",
        "stacked_bar":   "stacked bar chart",
        "area_chart":    "area chart",
        "heatmap":       "heatmap",
        "correlation_heatmap": "correlation heatmap",
        "frequency_bar": "frequency bar chart",
    }
    label = type_labels.get(chart_type, chart_type.replace("_", " "))

    lines = [f"### 📈 Graph Explanation — {label.title()}"]
    lines.append(f"\nThis **{label}** shows the relationship between **{x_col}** (x-axis) "
                 f"and **{y_col}** (y-axis)." if y_col and y_col != x_col
                 else f"\nThis **{label}** shows the distribution of **{x_col}**.")

    # Chart-specific guidance
    chart_guidance = {
        "bar_chart":     "Bar charts are useful for comparing values across categories. Look for the tallest/shortest bars to identify the highest and lowest values.",
        "pie_chart":     "Pie charts show proportional composition. Look for the largest slice to identify the dominant category. Use with caution for more than 5 categories.",
        "histogram":     "Histograms show the distribution shape of a numerical variable. Look for the peak (mode), symmetry, and any tails (skewness).",
        "line_chart":    "Line charts show trends over time or ordered categories. Look for the overall direction (upward/downward), peaks, and troughs.",
        "scatter_plot":  "Scatter plots show the relationship between two numerical variables. Look for the general direction and tightness of the point cloud.",
        "box_plot":      "Box plots show the median (centre line), interquartile range (box), and potential outliers (dots beyond the whiskers).",
        "correlation_heatmap": "Correlation heatmaps show the strength of relationships between multiple numerical variables. Blue/red indicate strong negative/positive correlations.",
    }
    if chart_type in chart_guidance:
        lines.append(f"\n**How to read this chart:** {chart_guidance[chart_type]}")

    if df_stats:
        if "mean" in df_stats:
            lines.append(f"\n**Key statistics for {y_col or x_col}:**")
            lines.append(f"- Mean: {df_stats.get('mean', 'N/A')}")
            lines.append(f"- Median: {df_stats.get('median', 'N/A')}")
            lines.append(f"- Range: {df_stats.get('min', 'N/A')} to {df_stats.get('max', 'N/A')}")
            skew = df_stats.get("skewness", 0)
            if abs(skew) < 0.5:
                lines.append("- Distribution: approximately symmetric")
            elif skew > 0:
                lines.append("- Distribution: positively skewed (tail extends to the right — mean > median)")
            else:
                lines.append("- Distribution: negatively skewed (tail extends to the left — mean < median)")
        if "top_values" in df_stats:
            top = df_stats["top_values"]
            if top:
                top_cat = list(top.keys())[0]
                top_n   = list(top.values())[0]
                lines.append(f"\n**Most frequent category:** '{top_cat}' ({top_n} occurrences)")

    lines.append("\n**What this graph does NOT prove:**")
    if chart_type == "scatter_plot":
        lines.append("- Correlation in a scatter plot does not establish causation.")
        lines.append("- A pattern may exist due to a third, unmeasured variable.")
        lines.append("- Run a formal Pearson or Spearman correlation test to quantify the relationship.")
    elif chart_type in ("bar_chart", "histogram", "box_plot"):
        lines.append("- Differences between groups or categories do not by themselves prove that one caused the other.")
        lines.append("- Further statistical testing (t-test, ANOVA) is needed to confirm whether differences are statistically significant.")
    elif chart_type == "line_chart":
        lines.append("- A trend line does not by itself prove causation or guarantee the trend will continue.")
        lines.append("- Consider whether the pattern could be explained by external factors.")
    elif chart_type == "correlation_heatmap":
        lines.append("- Correlation does not imply causation — a strong correlation may be coincidental.")
        lines.append("- Run individual correlation tests to obtain p-values for statistical significance.")

    return "\n".join(lines)


# ── Chatbot Response Generator ────────────────────────────────────────────────

STATISTICAL_EXPLANATIONS = {
    "r":            "The correlation coefficient (r) measures the strength and direction of a linear relationship between two numerical variables. It ranges from -1 (perfect negative relationship) to +1 (perfect positive relationship), with 0 indicating no linear relationship. Interpretation: |r| ≥ 0.9 = very strong, ≥ 0.7 = strong, ≥ 0.5 = moderate, ≥ 0.3 = weak, < 0.3 = very weak.",
    "p_value":      "The p-value tells you the probability of observing results as extreme as these if there were actually no real relationship (null hypothesis is true). A p-value < 0.05 is conventionally considered statistically significant, meaning the result is unlikely to be due to chance alone. Important: a small p-value does NOT mean the effect is large or practically important — it only means it is unlikely to be random.",
    "t_statistic":  "The t-statistic measures how many standard errors the observed difference is from zero (i.e., from no difference). A larger absolute t-value means the difference is more extreme relative to the variation in the data. The sign indicates direction: positive means Group 1 has the higher mean.",
    "f_statistic":  "The F-statistic (in ANOVA) compares the variation between groups to the variation within groups. F = (between-group variance) / (within-group variance). A larger F-value indicates that the differences between groups are larger relative to the random variation within groups.",
    "chi2":         "The chi-square statistic (χ²) measures how much the observed frequencies differ from what would be expected if the two categorical variables were completely independent. A larger chi-square value suggests a stronger association, but effect size (Cramér's V) is needed to interpret the magnitude.",
    "r_squared":    "R² (R-squared) tells you the proportion of variance in the outcome variable explained by the predictor(s). For example, R² = 0.65 means the predictor(s) explain 65% of the variation in the outcome. The remaining 35% is explained by factors not in the model.",
    "adj_r_squared":"Adjusted R² penalises for adding unnecessary predictors. It is a more reliable measure than R² when you have multiple predictors — it only increases if the new predictor genuinely improves the model.",
    "cohens_d":     "Cohen's d is an effect size measure for t-tests. It measures the standardised difference between two group means. d ≥ 0.8 is a large effect (practically meaningful difference), d ≥ 0.5 is medium, and d ≥ 0.2 is small. Effect size is important because a statistically significant result with a very small d may not be practically meaningful.",
    "cramers_v":    "Cramér's V measures the strength of association between two categorical variables (from chi-square). It ranges from 0 (no association) to 1 (perfect association). V ≥ 0.5 = strong, V ≥ 0.3 = moderate, V ≥ 0.1 = weak association.",
    "eta_squared":  "Eta-squared (η²) is an effect size for ANOVA. It represents the proportion of variance explained by the grouping variable. Values ≥ 0.14 indicate a large effect, ≥ 0.06 medium, ≥ 0.01 small.",
    "alpha":        "Cronbach's alpha (α) measures the internal consistency (reliability) of a set of survey items. Values range from 0 to 1: α ≥ 0.90 is excellent, ≥ 0.80 good, ≥ 0.70 acceptable, ≥ 0.60 questionable, < 0.60 poor. It tells you how consistently the items measure the same underlying construct. Note: alpha measures consistency, NOT validity.",
    "mean_diff":    "The mean difference is the average of all individual differences between paired measurements. A positive value means the first measurement tends to be higher; negative means the second tends to be higher.",
    "covariance":   "Covariance measures whether two numerical variables move together (positive covariance) or in opposite directions (negative covariance). IMPORTANT: the raw covariance value is unit-dependent — its magnitude tells you nothing about the strength of the relationship. A covariance of 1000 between income (£) and expenditure (£) and a covariance of 0.001 between the same variables measured in millions of £ represent exactly the same relationship. To judge strength, always use Pearson's correlation coefficient r = Cov(X,Y) / (SD_X × SD_Y), which standardises covariance onto a −1 to +1 scale.",
    "skewness":     "Skewness measures the asymmetry of a distribution. A skewness near zero means the distribution is approximately symmetric (mean ≈ median ≈ mode). Positive skewness (tail extends right) means a few very high values pull the mean above the median — the mean overstates the typical value. Negative skewness (tail extends left) means a few very low values drag the mean below the median. Rule: |skew| < 0.5 = approximately symmetric; 0.5–1.0 = moderate skew; > 1.0 = severe skew. When |skew| ≥ 0.5, prefer the median over the mean as the central tendency measure.",
    "sampling":     "Sampling is the process of selecting a subset of individuals from a population for study. Probability sampling (simple random, stratified, systematic, cluster) gives every individual a known non-zero chance of selection and is the gold standard for unbiased statistical inference. Non-probability sampling (convenience, purposive) is faster but introduces selection bias — results cannot be statistically generalised to the full population. Sampling error is the unavoidable gap between a sample statistic and the true population parameter; larger samples reduce it but cannot eliminate it.",
    "population":   "The population (N) is the entire set of individuals relevant to a research question. A parameter is a descriptive measure of the population (e.g. population mean μ). Because studying entire populations is usually impractical, we study a sample (n) and compute statistics to estimate population parameters. The validity of this inference depends critically on how the sample was selected — a non-representative sample produces biased estimates regardless of how large it is.",
    "five_number":  "The five-number summary describes a distribution with five values: Minimum, Q1 (25th percentile), Median (Q2, 50th percentile), Q3 (75th percentile), and Maximum. Together they define the IQR = Q3 − Q1 and enable Tukey outlier detection: values below Q1 − 1.5×IQR or above Q3 + 1.5×IQR are flagged as potential outliers. The five-number summary is non-parametric and valid for any distribution shape.",
    "central_tendency": "Central tendency summarises an entire dataset with a single representative value. Mean = sum / n (sensitive to outliers; best for symmetric data). Median = middle value when sorted (resistant to outliers; best when |skewness| ≥ 0.5 or outliers present). Mode = most frequent value (only valid central tendency measure for nominal/categorical variables). Rule: symmetric data with no outliers → use Mean; skewed data or outliers present → use Median; categorical data → use Mode.",

    # ── RULE-DATA-01 ──────────────────────────────────────────────────────────
    "data_types":   (
        "Data types in statistics fall into two main branches:\n\n"
        "**Quantitative (Numerical) data** — values are numbers with arithmetic meaning.\n"
        "- **Discrete:** can only take specific, countable values (usually whole numbers). "
        "Result of counting. E.g. number of students, number of cars. No meaningful decimals.\n"
        "- **Continuous:** can take any value within a range — decimals are valid and meaningful. "
        "Result of measuring. E.g. height, weight, temperature, time.\n\n"
        "**Qualitative (Categorical) data** — values are labels or categories, not numbers.\n"
        "- **Nominal:** categories with no natural order. E.g. gender, eye colour, country. "
        "Only valid operations: counting, mode, chi-square test.\n"
        "- **Ordinal:** categories with a meaningful order, but the gaps between ranks are not "
        "necessarily equal. E.g. satisfaction ratings (low/medium/high), education level. "
        "Valid operations: median, mode, Spearman correlation, Mann-Whitney.\n\n"
        "**Why this matters:** the data type controls which statistical methods are valid. "
        "Calculating a mean on nominal data (e.g. average of {male=1, female=2}) is "
        "arithmetically possible but statistically meaningless."
    ),

    # ── RULE-DISP-01 ──────────────────────────────────────────────────────────
    "deviation":    (
        "Deviation is how far a single value sits from the mean of the dataset.\n\n"
        "**Formula:** Deviation = x − μ\n\n"
        "A positive deviation means the value is above the mean; negative means below. "
        "The critical insight: **the sum of all deviations in a dataset always equals zero** — "
        "the positives and negatives cancel perfectly. This is why you cannot use raw "
        "deviations to measure overall spread.\n\n"
        "**The chain that follows:**\n"
        "1. Deviation (x − μ) — measures distance from the mean, but sums to zero.\n"
        "2. Squared Deviation (x − μ)² — removes negatives; all values now positive.\n"
        "3. Variance σ² = Σ(x − μ)² / N — average of squared deviations. Unit is squared "
        "(e.g. k²), which is hard to interpret.\n"
        "4. Standard Deviation σ = √σ² — square root returns the unit to the original scale "
        "(e.g. k), making it directly interpretable alongside the mean.\n\n"
        "**Population vs sample formula:**\n"
        "- Population variance: σ² = Σ(xi − μ)² / N\n"
        "- Sample variance: s² = Σ(xi − x̄)² / (n − 1)\n"
        "The sample formula divides by (n − 1), not n — this is Bessel's correction, which "
        "compensates for the fact that a sample underestimates the true population spread."
    ),
    "variance":     (
        "Variance (σ² for a population, s² for a sample) measures how spread out data values "
        "are from their mean. It is the average of all squared deviations.\n\n"
        "**Why squared?** Raw deviations (x − μ) always sum to zero — squaring removes the "
        "negative signs so values no longer cancel each other out.\n\n"
        "**Formulas:**\n"
        "- Population: σ² = Σ(xi − μ)² / N\n"
        "- Sample: s² = Σ(xi − x̄)² / (n − 1)   ← divides by n−1 (Bessel's correction)\n\n"
        "**Interpreting size:** a larger variance means data is more spread out; smaller "
        "variance means values cluster tightly around the mean.\n\n"
        "**The unit problem:** variance is expressed in squared units (e.g. k²), which is "
        "not directly interpretable. Take the square root to get the Standard Deviation, "
        "which restores the original unit and is human-readable."
    ),
    "standard_deviation": (
        "Standard Deviation (SD) is the square root of Variance. It measures the typical "
        "distance of data values from the mean, in the **same unit as the original data**.\n\n"
        "**Formula:** σ = √[Σ(xi − μ)² / N]   (population)\n"
        "             s = √[Σ(xi − x̄)² / (n−1)]   (sample)\n\n"
        "**Interpretation:**\n"
        "- A small SD means values cluster tightly around the mean.\n"
        "- A large SD means values are spread widely.\n"
        "- For a normal distribution: ~68% of values lie within ±1 SD of the mean, "
        "~95% within ±2 SD, ~99.7% within ±3 SD.\n\n"
        "**SD vs variance:** SD is preferred for reporting because its unit matches the "
        "data (e.g. 'salaries vary by ±18k', not '±340k²').\n\n"
        "**SD vs MAD:** SD gives more weight to extreme values (because deviations are "
        "squared before averaging). MAD (Mean Absolute Deviation) is more robust to outliers "
        "but is less used in inferential statistics."
    ),
    "range":        (
        "Range is the simplest measure of dispersion: the difference between the maximum "
        "and minimum values in a dataset.\n\n"
        "**Formula:** Range = Max − Min\n\n"
        "**Advantage:** Quick and easy to calculate and explain.\n\n"
        "**Limitation:** Entirely determined by just two values — the extremes. A single "
        "outlier dramatically changes the range, making it a misleading summary of typical "
        "spread. For this reason, the IQR (Q3 − Q1) is preferred when outliers are present."
    ),

    # ── RULE-DISP-02 ──────────────────────────────────────────────────────────
    "z_score":      (
        "A Z-score (standard score) measures how many standard deviations a single value is "
        "above or below the mean of its dataset.\n\n"
        "**Formula:** z = (x − μ) / σ\n\n"
        "**Interpretation:**\n"
        "- z = 0 → the value equals the mean exactly.\n"
        "- z = +2 → the value is 2 standard deviations above the mean.\n"
        "- z = −1.5 → the value is 1.5 standard deviations below the mean.\n"
        "- |z| > 2 → the value is in the outer ~5% of a normal distribution (potential outlier).\n"
        "- |z| > 3 → the value is in the outer ~0.3% (strong outlier candidate).\n\n"
        "**Key uses:**\n"
        "1. **Outlier detection:** values with |z| > 3 are commonly flagged as outliers.\n"
        "2. **Comparison across scales:** z-scores allow comparison of values measured on "
        "different scales (e.g. comparing a student's maths score to their English score).\n"
        "3. **Standardisation before PCA or regression:** z-scoring (subtracting mean, "
        "dividing by SD) puts all variables on the same scale so no single variable "
        "dominates due to its unit size."
    ),

    # ── RULE-DISP-03 ──────────────────────────────────────────────────────────
    "mad":          (
        "MAD (Mean Absolute Deviation) measures the average absolute distance of each data "
        "point from the mean, ignoring the direction of the difference.\n\n"
        "**Formula:** MAD = Σ|xi − μ| / n\n\n"
        "**Key difference from Standard Deviation:**\n"
        "- MAD uses absolute values (|xi − μ|); SD uses squared deviations ((xi − μ)²).\n"
        "- Squaring in SD gives heavy weight to extreme values (outliers); absolute values "
        "in MAD treat all deviations proportionally.\n"
        "- Result: MAD is **more robust to outliers** than SD.\n\n"
        "**When to use MAD vs SD:**\n"
        "- Use SD for inferential statistics (t-tests, ANOVA, regression) because SD has "
        "better mathematical properties and aligns with normal distribution theory.\n"
        "- Use MAD as a descriptive robustness check when your data contains extreme "
        "outliers that you believe are genuine observations, not errors."
    ),

    # ── RULE-DISP-04 ──────────────────────────────────────────────────────────
    "percentile":   (
        "A percentile tells you the value below which a given percentage of observations fall.\n\n"
        "**Examples:**\n"
        "- 25th percentile (Q1): 25% of values are below this point.\n"
        "- 50th percentile (Q2/Median): 50% of values are below this point.\n"
        "- 75th percentile (Q3): 75% of values are below this point.\n"
        "- 90th percentile: 90% of values are below this point.\n\n"
        "**Percentile vs Percentage:** Percentage describes a proportion of a total "
        "(e.g. 80% correct). Percentile describes position relative to other values "
        "(e.g. scoring at the 80th percentile means you scored higher than 80% of the group).\n\n"
        "**Practical use:** Percentiles are used in growth charts (child height/weight), "
        "standardised test scores, income distribution analysis, and the five-number summary "
        "(Min, Q1, Median, Q3, Max)."
    ),

    # ── RULE-INFER-01 ─────────────────────────────────────────────────────────
    "descriptive_vs_inferential": (
        "Statistics divides into two major branches:\n\n"
        "**Descriptive Statistics** — summarise and describe the data you actually have.\n"
        "Examples: mean, median, mode, standard deviation, frequency table, histogram.\n"
        "Descriptive statistics make no claims beyond the data collected. "
        "They answer: *What does this dataset look like?*\n\n"
        "**Inferential Statistics** — use sample data to make probability-based statements "
        "about a larger, unobserved population.\n"
        "Examples: t-test, ANOVA, chi-square, regression, confidence intervals.\n"
        "Inferential statistics answer: *What can we conclude about the population based on "
        "this sample?*\n\n"
        "**Key inferential tasks:**\n"
        "1. **Parameter estimation** — using a sample statistic to estimate a population "
        "parameter (e.g. sample mean x̄ estimates population mean μ).\n"
        "2. **Hypothesis testing** — testing whether a claim about the population is "
        "supported by the sample evidence (e.g. 'Does the new teaching method improve scores?').\n\n"
        "**Critical constraint (RULE-INTERP-01):** inferential conclusions are only as valid "
        "as the sampling method that produced the data. Non-probability samples "
        "(convenience, purposive) cannot support population-level inference."
    ),

    # ── RULE-INFER-02 ─────────────────────────────────────────────────────────
    "central_limit_theorem": (
        "The Central Limit Theorem (CLT) is one of the most important principles in "
        "inferential statistics.\n\n"
        "**Statement:** As sample size increases (generally n ≥ 30), the distribution of "
        "**sample means** (the sampling distribution of x̄) approaches a normal distribution, "
        "regardless of the shape of the original population distribution.\n\n"
        "**Why it matters:**\n"
        "- It is the theoretical foundation that allows t-tests, z-tests, ANOVA, and "
        "regression to be used even when the original data is not perfectly normally distributed.\n"
        "- It explains why larger samples give more reliable results.\n\n"
        "**Critical distinction:** the CLT applies to the distribution of SAMPLE MEANS, "
        "not to the distribution of individual observations. "
        "A large sample does not make non-normal raw data become normally distributed — "
        "it makes the *average of many samples* normally distributed.\n\n"
        "**Practical implication:** with n ≥ 30, you can generally proceed with parametric "
        "tests even if the data is moderately skewed, because the sampling distribution "
        "of the mean is approximately normal."
    ),

    # ── RULE-INFER-03 ─────────────────────────────────────────────────────────
    "normal_distribution": (
        "The Normal Distribution (also called Gaussian distribution or bell curve) is the "
        "most important probability distribution in statistics.\n\n"
        "**Key properties:**\n"
        "- Symmetric and bell-shaped around the mean.\n"
        "- Mean = Median = Mode — all three are equal and located at the centre.\n"
        "- Total area under the curve = 1 (sum of all probabilities = 100%).\n"
        "- Defined by two parameters: μ (mean, determines location) and σ (SD, determines width).\n\n"
        "**The empirical rule (68-95-99.7 rule):**\n"
        "- ~68% of values fall within μ ± 1σ\n"
        "- ~95% of values fall within μ ± 2σ\n"
        "- ~99.7% of values fall within μ ± 3σ\n\n"
        "**When do data follow a normal distribution?** Many natural measurements "
        "(height, weight, IQ) approximate normality. However, skewed data (income, house prices) "
        "and count data (number of events) do not. "
        "Always check normality with a Shapiro-Wilk test (n ≤ 50) or "
        "D'Agostino-Pearson test (n > 50) before applying parametric methods.\n\n"
        "**Important:** you do not need perfectly normal data for parametric tests to be valid. "
        "The CLT ensures that parametric tests are robust for moderately non-normal data "
        "with n ≥ 30."
    ),

    # ── RULE-INFER-04 ─────────────────────────────────────────────────────────
    "sampling_distribution": (
        "A sampling distribution is the theoretical distribution of a statistic "
        "(e.g. the sample mean x̄) computed across all possible samples of size n "
        "drawn from a population.\n\n"
        "**How to think about it:** imagine drawing 1,000 random samples of 30 students "
        "from a university and computing the average exam score for each sample. "
        "The distribution of those 1,000 averages is the sampling distribution of the mean.\n\n"
        "**Key property — Standard Error (SE):**\n"
        "SE = σ / √n\n"
        "The SE is the standard deviation of the sampling distribution. "
        "A larger sample (bigger n) produces a smaller SE — meaning individual sample "
        "means cluster more tightly around the true population mean.\n\n"
        "**Why it matters:** hypothesis tests (t-test, z-test) and confidence intervals "
        "are based on the sampling distribution. The p-value is the probability of "
        "observing a result as extreme as yours *under the sampling distribution* if H₀ were true.\n\n"
        "**Critical distinction:** the sampling distribution is NOT the distribution of "
        "individual observations — it is the distribution of a statistic (e.g. mean) "
        "computed from many samples."
    ),

    # ── RULE-INFER-05 ─────────────────────────────────────────────────────────
    "probability_distributions": (
        "A probability distribution describes all possible values of a random variable "
        "and the probability associated with each value.\n\n"
        "**Discrete probability distributions** — for countable outcomes:\n"
        "- **Binomial:** models the number of successes in n independent trials where each "
        "trial has exactly two outcomes (success/failure) with constant probability p. "
        "E.g. number of heads in 10 coin flips.\n"
        "- **Poisson:** models the count of rare events occurring in a fixed time/space "
        "interval. Defined by a single parameter λ (lambda) = mean = variance. "
        "E.g. number of customer calls per hour.\n\n"
        "**Continuous probability distributions** — for measured outcomes:\n"
        "- **Normal (Gaussian):** symmetric, bell-shaped, defined by μ and σ. "
        "Most common in nature and the basis for parametric inference.\n\n"
        "**How to choose the right distribution:**\n"
        "- Binary outcome (yes/no, pass/fail) counted over n trials → Binomial\n"
        "- Rare event count per unit time/space → Poisson\n"
        "- Continuous measurement (height, score, income) → Normal (or check skewness)\n"
        "- Sample mean of any distribution with n ≥ 30 → approximately Normal (CLT)"
    ),

    # ── RULE-VAR-01 ───────────────────────────────────────────────────────────
    "random_variable": (
        "A **variable** is any attribute that can take different values across observations "
        "(e.g. a person's age, score, or gender).\n\n"
        "A **random variable** is a variable whose value is determined by the outcome of a "
        "random process — each possible value has an associated probability.\n\n"
        "**Two types of random variables:**\n\n"
        "- **Discrete random variable:** can only take a countable number of specific values "
        "(often whole numbers). Arises from counting. "
        "Examples: number of heads in coin flips (0, 1, 2…); number of defective items in a "
        "batch; number of customers arriving per hour.\n\n"
        "- **Continuous random variable:** can take any value within a range — decimals are "
        "valid and meaningful. Arises from measuring. "
        "Examples: a person's exact height, weight, reaction time, or temperature.\n\n"
        "**Why the distinction matters:** discrete variables use discrete probability "
        "distributions (Binomial, Poisson); continuous variables use continuous "
        "distributions (Normal, t, F). Mixing them up leads to incorrect probability "
        "calculations and invalid statistical tests."
    ),

    # ── RULE-DISP-01 (variation concept) ─────────────────────────────────────
    "variation":    (
        "Variation is the general concept that values in a dataset differ from each other. "
        "It is not a formula — it is a plain-language description of spread.\n\n"
        "'There is a lot of variation in salaries' simply means salaries differ widely. "
        "It answers the question: *How different are these values from each other?*\n\n"
        "**Formal measures of variation (most to least sensitive to outliers):**\n"
        "1. **Range** = Max − Min (simplest, most sensitive to outliers)\n"
        "2. **Variance** σ² = Σ(xi − μ)² / N (average squared deviation)\n"
        "3. **Standard Deviation** σ = √Variance (same unit as data; most commonly reported)\n"
        "4. **IQR** = Q3 − Q1 (middle 50% spread; most robust to outliers)\n"
        "5. **MAD** = Σ|xi − μ| / n (absolute deviation; robust alternative to SD)\n\n"
        "**Rule of thumb:** report SD alongside the mean for symmetric data; "
        "report IQR alongside the median for skewed data or data with outliers."
    ),
}

METHOD_EXPLANATIONS = {
    "pearson":      "**Why Pearson correlation?**\n\nPearson correlation was used because both variables are numerical (continuous) and the relationship between them is expected to be linear. Pearson measures the linear association directly and produces a result (r) that is easily interpretable.\n\n**When Pearson is preferred over Spearman:** When both variables are continuous and approximately normally distributed without severe outliers.\n\n**Assumption:** Both variables should be approximately normally distributed and linearly related.",
    "spearman":     "**Why Spearman correlation?**\n\nSpearman correlation was used because the data may not meet the normality assumption required for Pearson, or because one or both variables are ordinal (ranked). Spearman ranks all observations first and then calculates the correlation on the ranks, making it more robust to non-normality and outliers.\n\n**When Spearman is preferred over Pearson:** Ordinal data, skewed distributions, presence of outliers, or non-linear but monotonic relationships.",
    "chi_square":   "**Why Chi-Square Test?**\n\nChi-square was used because both variables are categorical (nominal or ordinal). It tests whether there is a statistically significant association between the two categorical variables by comparing observed frequencies with expected frequencies under the assumption of independence.\n\n**Important:** Chi-square only tells you WHETHER an association exists — use Cramér's V to understand the STRENGTH of the association.",
    "independent_ttest": "**Why Independent Samples t-Test?**\n\nThe independent samples t-test was used to compare the mean of a numerical variable between two separate (independent) groups. It determines whether the observed difference in means is statistically significant or could be due to chance.\n\n**Why not ANOVA?** ANOVA is used for 3 or more groups. When there are exactly two groups, the independent t-test is the standard parametric test (it is equivalent to ANOVA for two groups).\n\n**Levene's test** was applied to check whether the two groups have equal variance. If violated, Welch's correction was automatically applied.",
    "paired_ttest": "**Why Paired Samples t-Test?**\n\nThe paired samples t-test was used because the two measurements come from the same subjects (e.g., before and after an intervention). This accounts for within-subject variation, making the test more sensitive than an independent t-test for this design.\n\n**Why not independent t-test?** The independent t-test ignores the natural pairing between measurements, which would waste information and reduce statistical power.",
    "anova":        "**Why One-Way ANOVA?**\n\nOne-way ANOVA was used because you are comparing a numerical variable across more than two groups. Running multiple pairwise t-tests instead of ANOVA would artificially inflate the Type I error rate (false positive rate). For example, with 5 groups, there are 10 possible pairwise comparisons — the probability of at least one false positive would be much higher than 5%.\n\nANOVA handles this by testing all groups simultaneously with a single F-statistic.\n\n**Post-hoc tests:** If ANOVA is significant (p < 0.05), Tukey's HSD post-hoc test was applied to identify which specific pairs of groups differ significantly.",
    "welch_anova":  "**Why Welch's One-Way ANOVA?**\n\nWelch's ANOVA was used instead of standard one-way ANOVA because Levene's test indicated unequal variances across groups (violation of homogeneity of variance). Welch's ANOVA uses a modified F-statistic and Welch-Satterthwaite degrees of freedom that remain valid when group variances differ.\n\n**When Welch ANOVA is preferred over standard ANOVA:** When group sizes are unequal AND Levene's test is significant (p < 0.05).\n\n**Post-hoc tests:** If Welch ANOVA is significant, use Games-Howell post-hoc comparisons (not Tukey HSD) because Games-Howell does not assume equal variances.",
    "fishers_exact": "**Why Fisher's Exact Test?**\n\nFisher's Exact Test was used instead of chi-square because at least one expected cell count in the 2×2 contingency table was below 5. Chi-square relies on an asymptotic approximation that becomes unreliable with small expected frequencies. Fisher's Exact Test calculates the exact hypergeometric probability of observing a table as extreme or more extreme than the one observed, without relying on any approximation.\n\n**When Fisher's is preferred over chi-square:** Any expected cell count < 5 in a 2×2 table. For larger tables, use chi-square with a warning.\n\n**Effect size:** Report the Odds Ratio (OR) with its 95% confidence interval and the Phi coefficient.",
    "point_biserial": "**Why Point-Biserial Correlation?**\n\nPoint-biserial correlation was used because the research question involves the relationship between one continuous variable and one genuinely dichotomous binary variable. It is mathematically equivalent to Pearson r applied to a 0/1 coded binary variable, and directly produces an effect size (r_pb) interpretable on the same scale as Pearson r.\n\n**Why not a t-test?** An independent t-test would also be valid for this variable configuration. Point-biserial r is preferred when the goal is to quantify the strength of association rather than test a mean difference.\n\n**Assumption:** The continuous variable should be approximately normally distributed within each binary group.",
    "goodness_of_fit": "**Why Chi-Square Goodness-of-Fit Test?**\n\nThe goodness-of-fit test was used to test whether the observed frequency distribution of a single categorical variable matches a specified theoretical distribution (usually uniform equal proportions, or a theoretically derived set of proportions).\n\n**Why not chi-square test of independence?** The independence test requires two categorical variables and tests association between them. Goodness-of-fit tests one variable against a theoretical distribution — a different question.\n\n**Effect size:** Cohen's w measures the magnitude of departure from the expected distribution.",
    "friedman_test": "**Why Friedman Test?**\n\nThe Friedman test was used as a non-parametric alternative to repeated-measures ANOVA. It is appropriate when: (1) the same subjects are measured across three or more conditions, and (2) the normality assumption required for parametric repeated-measures ANOVA cannot be satisfied.\n\n**Why not repeated-measures ANOVA?** Repeated-measures ANOVA assumes normality of the data within each condition. Friedman uses only the ranks, making it robust to non-normality.\n\n**Effect size:** Kendall's W measures the degree of concordance across conditions (0 = no agreement, 1 = perfect agreement).\n\n**Follow-up:** Pairwise Wilcoxon signed-rank tests with Bonferroni/Holm correction to identify which specific condition pairs differ.",
    "pca_method":   "**Why Principal Component Analysis (PCA)?**\n\nPCA was used to reduce the dimensionality of a set of inter-correlated continuous variables into a smaller number of uncorrelated components that capture the maximum possible variance. It helps:\n1. Identify the underlying structure of a set of variables\n2. Remove redundancy from highly correlated predictors before regression\n3. Explore which variables tend to co-vary together\n\n**Assumptions:** Variables should be continuous (interval/ratio), approximately linearly related, and the correlation structure should be adequate (KMO ≥ 0.60). Variables were standardised to zero mean and unit variance before PCA to ensure comparability.\n\n**Criterion for retaining components:** Kaiser's rule (eigenvalue ≥ 1) was used — components explaining more variance than a single original variable are retained.",
    "linear_reg":   "**Why Linear Regression?**\n\nLinear regression was used to quantify how a numerical outcome variable changes as a function of one or more predictor variables. It produces:\n- **Coefficients** showing the expected change in the outcome per unit change in each predictor\n- **R²** showing how much variance in the outcome is explained\n- **p-values** for each predictor showing whether its contribution is statistically significant\n\n**Assumption:** The relationship between predictors and outcome should be approximately linear.",
    "logistic_reg": "**Why Logistic Regression?**\n\nLogistic regression was used because the outcome variable is binary (only two possible values). Standard linear regression is not appropriate for binary outcomes because it can produce probabilities outside [0, 1] and violates key assumptions. Logistic regression models the log-odds of the outcome and is the standard method for binary classification with predictors.",
}


def answer_research_question(question: str, analysis_context: dict) -> str:
    """
    Generate a contextual answer to a research/methodology question.
    analysis_context contains: last_result, dataset_profile, cleaning_log
    """
    q = question.lower()
    last    = analysis_context.get("last_result", {})
    profile = analysis_context.get("dataset_profile") or {}
    cleaning_log = analysis_context.get("cleaning_log") or []
    num_cols = profile.get("analysis_ready_numerical", profile.get("numerical_cols", []))
    cat_cols = profile.get("analysis_ready_categorical", profile.get("categorical_cols", []))
    n_rows   = profile.get("n_rows", None)

    # ── Cronbach's alpha specific ──────────────────────────────────────────────
    if "cronbach" in q or ("alpha" in q and ("reliab" in q or "consistency" in q)):
        response = f"**What is Cronbach's Alpha?**\n\n{STATISTICAL_EXPLANATIONS['alpha']}"
        if "alpha" in last:
            alpha_val = last["alpha"]
            reliability = last.get("reliability", "")
            n_items = last.get("n_items", "?")
            n_obs = last.get("n", "?")
            response += (f"\n\n**In your analysis:** α = **{alpha_val}** — {reliability}. "
                         f"Computed across **{n_items} items** and **{n_obs} respondents**.\n\n"
                         f"**Interpretation:** {last.get('interpretation', '')}")
        return response

    # ── Pre-loop specific overrides (must come BEFORE the generic word loop) ──
    # These questions contain fragments that would be caught by the generic loop
    # (e.g. "population" in STATISTICAL_EXPLANATIONS, "deviation" in the loop),
    # so they must be checked first with more precise patterns.
    import re as _re
    if _re.search(r'\bpopulation variance\b', q) or _re.search(r'\bsample variance\b', q) or \
       _re.search(r'\bn-1\b', q) or "bessel" in q or "variance formula" in q or \
       "how to calculate variance" in q:
        return f"**Variance (RULE-DISP-01)**\n\n{STATISTICAL_EXPLANATIONS['variance']}"

    if _re.search(r'\bmean absolute deviation\b', q) or _re.search(r'\babsolute deviation\b', q) or \
       "mad vs sd" in q or "mad vs standard deviation" in q or q.strip() == "what is mad":
        return f"**Mean Absolute Deviation — MAD (RULE-DISP-03)**\n\n{STATISTICAL_EXPLANATIONS['mad']}"

    if _re.search(r'\bcentral limit theorem\b', q) or "clt says" in q or "clt means" in q or \
       "why normal distribution for large samples" in q or "sampling distribution of the mean" in q:
        return (f"**Central Limit Theorem (RULE-INFER-02)**\n\n"
                f"{STATISTICAL_EXPLANATIONS['central_limit_theorem']}")

    # "sampling distribution" must be checked before the broader "sampling" keyword catch
    if _re.search(r'\bsampling distribution\b', q) or "distribution of sample means" in q or \
       "standard error of the mean" in q or "what is standard error" in q:
        return (f"**Sampling Distribution and Standard Error (RULE-INFER-04)**\n\n"
                f"{STATISTICAL_EXPLANATIONS['sampling_distribution']}")

    # ── Statistical values — word-boundary match ───────────────────────────────
    for term, explanation in STATISTICAL_EXPLANATIONS.items():
        term_word = term.replace("_", " ")
        pattern = r'\b' + _re.escape(term_word) + r'\b'
        if _re.search(pattern, q) or _re.search(r'\b' + _re.escape(term) + r'\b', q):
            response = f"**What does {term_word} mean?**\n\n{explanation}"
            if term in last:
                val = last[term]
                response += f"\n\n**In your analysis:** {term_word} = **{val}**"
                # Add contextual interpretation
                if term == "r" and "p_value" in last:
                    p = last["p_value"]
                    strength = ("very strong" if abs(val) >= 0.9 else "strong" if abs(val) >= 0.7
                                else "moderate" if abs(val) >= 0.5 else "weak" if abs(val) >= 0.3 else "very weak")
                    direction = "positive" if val > 0 else "negative"
                    sig = "statistically significant (p < 0.05)" if p < 0.05 else "not statistically significant (p ≥ 0.05)"
                    response += (f" — this indicates a **{strength} {direction}** association. "
                                 f"The result is **{sig}** (p = {p}).")
                elif term == "p_value":
                    response += (" — " + ("This IS statistically significant at the 5% level."
                                          if val < 0.05 else
                                          "This is NOT statistically significant at the 5% level."))
                elif term == "r_squared":
                    response += f" — the predictor(s) explain **{round(val*100,1)}%** of the variance in the outcome."
                elif term == "cohens_d":
                    size = "large" if abs(val) >= 0.8 else "medium" if abs(val) >= 0.5 else "small"
                    response += f" — this is a **{size}** effect."
                elif term == "eta_squared":
                    size = "large" if val >= 0.14 else "medium" if val >= 0.06 else "small"
                    response += f" — this is a **{size}** effect."
                elif term == "cramers_v":
                    strength = "strong" if val >= 0.5 else "moderate" if val >= 0.3 else "weak"
                    response += f" — this indicates a **{strength}** association."
                elif term == "alpha" and "reliability" in last:
                    response += f" — {last['reliability']}"
            return response

    # ── Method justification ───────────────────────────────────────────────────
    if "why" in q and "point" in q and ("biserial" in q or "biseri" in q):
        return METHOD_EXPLANATIONS.get("point_biserial", "")
    if "why" in q and any(m in q for m in ["pearson", "correlation"]) and "spearman" not in q and "biserial" not in q:
        return METHOD_EXPLANATIONS.get("pearson", "")
    if "why" in q and "spearman" in q:
        return METHOD_EXPLANATIONS.get("spearman", "")
    if ("pearson" in q and "spearman" in q) or ("versus" in q and "correlation" in q) or ("vs" in q and "correlation" in q):
        return ("**Pearson vs. Spearman Correlation:**\n\n"
                "- **Pearson** measures the *linear* relationship between two continuous variables. "
                "It assumes approximate normality and is sensitive to outliers.\n"
                "- **Spearman** measures the *monotonic* relationship using ranked data. "
                "It is more robust to non-normality and outliers.\n\n"
                "**When to prefer Spearman:** If your data is clearly non-normal (heavily skewed), "
                "contains outliers, or if at least one variable is ordinal (ranked categories).\n\n"
                "**When to prefer Pearson:** If both variables are continuous and approximately normally distributed.")
    if "why" in q and ("chi" in q or "chi-square" in q) and "goodness" not in q:
        return METHOD_EXPLANATIONS.get("chi_square", "")
    if "why" in q and ("goodness" in q or "goodness-of-fit" in q or "gof" in q):
        return METHOD_EXPLANATIONS.get("goodness_of_fit", "")
    if "why" in q and "fisher" in q:
        return METHOD_EXPLANATIONS.get("fishers_exact", "")
    if "why" in q and "point" in q and ("biserial" in q or "biseri" in q):
        return METHOD_EXPLANATIONS.get("point_biserial", "")
    if "why" in q and "welch" in q and "anova" in q:
        return METHOD_EXPLANATIONS.get("welch_anova", "")
    if "why" in q and "friedman" in q:
        return METHOD_EXPLANATIONS.get("friedman_test", "")
    if "why" in q and ("pca" in q or "principal component" in q or "factor" in q):
        return METHOD_EXPLANATIONS.get("pca_method", "")
    if "why" in q and "paired" in q and "t-test" in q:
        return METHOD_EXPLANATIONS.get("paired_ttest", "")
    if "why" in q and ("t-test" in q or "t test" in q) and "anova" not in q:
        return METHOD_EXPLANATIONS.get("independent_ttest", "")
    if "why" in q and "welch" in q and "anova" not in q:
        return METHOD_EXPLANATIONS.get("welch_anova", "")
    if "why" in q and "anova" in q:
        return METHOD_EXPLANATIONS.get("welch_anova" if "welch" in q else "anova", "")
    if "why" in q and ("logistic" in q):
        return METHOD_EXPLANATIONS.get("logistic_reg", "")
    if "why" in q and ("regression" in q or "linear" in q):
        return METHOD_EXPLANATIONS.get("linear_reg", "")

    # ── RULE-DATA-01: Types of data ───────────────────────────────────────────
    if any(kw in q for kw in ["type of data", "types of data", "data type",
                               "discrete", "continuous data", "nominal data",
                               "ordinal data", "qualitative data", "quantitative data",
                               "categorical data type", "numerical data type"]):
        return f"**Types of Data (RULE-DATA-01)**\n\n{STATISTICAL_EXPLANATIONS['data_types']}"

    # ── RULE-DISP-01: Deviation, Variance, Standard Deviation ────────────────
    # Note: "deviation" is checked BEFORE the generic STATISTICAL_EXPLANATIONS word loop
    if any(kw in q for kw in ["sum of deviations", "why square",
                               "squared deviation", "bessel",
                               "what is deviation", "how to calculate deviation"]):
        return f"**Deviation and Why We Square It (RULE-DISP-01)**\n\n{STATISTICAL_EXPLANATIONS['deviation']}"

    if any(kw in q for kw in ["population variance", "sample variance",
                               "n-1", "n minus 1", "bessel's correction",
                               "what is variance", "how to calculate variance",
                               "variance formula"]):
        return f"**Variance (RULE-DISP-01)**\n\n{STATISTICAL_EXPLANATIONS['variance']}"

    if any(kw in q for kw in ["standard deviation", "what is sd", "what is σ",
                               "interpret sd", "sd formula", "68 95 99", "empirical rule"]):
        return f"**Standard Deviation (RULE-DISP-01)**\n\n{STATISTICAL_EXPLANATIONS['standard_deviation']}"

    if any(kw in q for kw in ["what is range", "range formula", "max min",
                               "range vs iqr", "iqr vs range"]):
        return f"**Range as a Measure of Dispersion (RULE-DISP-01)**\n\n{STATISTICAL_EXPLANATIONS['range']}"

    if any(kw in q for kw in ["variation", "spread of data", "measures of dispersion",
                               "dispersion", "how spread"]) and "normal" not in q:
        return f"**Variation and Measures of Spread (RULE-DISP-01)**\n\n{STATISTICAL_EXPLANATIONS['variation']}"

    # ── RULE-DISP-02: Z-score ─────────────────────────────────────────────────
    if any(kw in q for kw in ["z-score", "z score", "zscore", "standard score",
                               "z = ", "standardise", "standardize",
                               "standard deviations above", "standard deviations below"]):
        return f"**Z-Score (RULE-DISP-02)**\n\n{STATISTICAL_EXPLANATIONS['z_score']}"

    # ── RULE-DISP-03: MAD ─────────────────────────────────────────────────────
    if any(kw in q for kw in ["mean absolute deviation", "absolute deviation",
                               "mad vs sd", "mad vs standard deviation",
                               "what is mad"]):
        return f"**Mean Absolute Deviation — MAD (RULE-DISP-03)**\n\n{STATISTICAL_EXPLANATIONS['mad']}"

    # ── RULE-DISP-04: Percentile ──────────────────────────────────────────────
    if any(kw in q for kw in ["percentile", "percentile vs percentage",
                               "nth percentile", "90th percentile", "what is percentile"]):
        return f"**Percentiles (RULE-DISP-04)**\n\n{STATISTICAL_EXPLANATIONS['percentile']}"

    # ── RULE-INFER-01: Descriptive vs Inferential ─────────────────────────────
    if any(kw in q for kw in ["descriptive vs inferential", "inferential vs descriptive",
                               "difference between descriptive and inferential",
                               "inferential statistics", "descriptive statistics",
                               "parameter estimation", "hypothesis testing"]):
        return (f"**Descriptive vs Inferential Statistics (RULE-INFER-01)**\n\n"
                f"{STATISTICAL_EXPLANATIONS['descriptive_vs_inferential']}")

    # ── RULE-INFER-02: Central Limit Theorem ──────────────────────────────────
    if any(kw in q for kw in ["central limit theorem", "what is clt", "central limit",
                               "sampling distribution of the mean",
                               "why normal distribution for large samples",
                               "clt says", "clt means"]):
        return (f"**Central Limit Theorem (RULE-INFER-02)**\n\n"
                f"{STATISTICAL_EXPLANATIONS['central_limit_theorem']}")

    # ── RULE-INFER-03: Normal distribution ────────────────────────────────────
    if any(kw in q for kw in ["normal distribution", "bell curve", "gaussian",
                               "68 95 99", "empirical rule", "bell shaped",
                               "mean equals median equals mode"]):
        return (f"**Normal Distribution (RULE-INFER-03)**\n\n"
                f"{STATISTICAL_EXPLANATIONS['normal_distribution']}")

    # ── RULE-INFER-04: Sampling distribution / standard error ─────────────────
    if any(kw in q for kw in ["sampling distribution", "standard error of",
                               "se formula", "distribution of sample means",
                               "standard error of the mean",
                               "what is standard error"]):
        return (f"**Sampling Distribution and Standard Error (RULE-INFER-04)**\n\n"
                f"{STATISTICAL_EXPLANATIONS['sampling_distribution']}")

    # ── RULE-INFER-05: Probability distributions ──────────────────────────────
    if any(kw in q for kw in ["probability distribution", "binomial distribution",
                               "poisson distribution", "binomial", "poisson",
                               "discrete distribution", "continuous distribution",
                               "what distribution", "which distribution"]):
        return (f"**Probability Distributions (RULE-INFER-05)**\n\n"
                f"{STATISTICAL_EXPLANATIONS['probability_distributions']}")

    # ── RULE-VAR-01: Random variable ──────────────────────────────────────────
    if any(kw in q for kw in ["random variable", "discrete random variable",
                               "continuous random variable", "what is a variable",
                               "variable vs random variable"]):
        return (f"**Variables and Random Variables (RULE-VAR-01)**\n\n"
                f"{STATISTICAL_EXPLANATIONS['random_variable']}")

    # ── t-test vs ANOVA question ───────────────────────────────────────────────
    if ("t-test" in q or "t test" in q) and "anova" in q:
        return ("**When to use t-test vs ANOVA:**\n\n"
                "- **t-Test** — use when comparing a numerical variable across **exactly 2 groups**.\n"
                "- **ANOVA** — use when comparing a numerical variable across **3 or more groups**.\n\n"
                "**Why not run multiple t-tests?** If you compare 3 groups with three separate t-tests "
                "(A vs B, A vs C, B vs C), each test has a 5% chance of a false positive. "
                "The combined probability of at least one false positive exceeds 14%. "
                "ANOVA controls this by testing all groups simultaneously with a single F-statistic.\n\n"
                "**Rule of thumb:** 2 groups → t-test. 3+ groups → ANOVA + post-hoc test (Tukey HSD).")

    # ── Sampling methodology (RULE-INTERP-01) ────────────────────────────────
    if any(kw in q for kw in ["sampling", "sample method", "random sample",
                               "stratified", "convenience sample", "purposive",
                               "cluster sample", "systematic sample", "probability sample",
                               "non-probability", "sampling bias", "sampling error"]):
        return (
            "**Sampling Methods and Their Impact on Inference (RULE-INTERP-01)**\n\n"
            "**Probability Sampling** — every individual has a known, non-zero chance of selection. "
            "Required for valid statistical inference to the full population.\n\n"
            "| Method | How it works | When to use |\n"
            "|---|---|---|\n"
            "| Simple Random Sampling | Every unit has an equal selection chance | Gold standard; no natural subgroups |\n"
            "| Stratified Sampling | Divide into subgroups first; sample randomly within each | When subgroups must be representated proportionally |\n"
            "| Systematic Sampling | Select every k-th unit from an ordered list | Large ordered populations |\n"
            "| Cluster Sampling | Sample whole clusters randomly, then all within chosen clusters | Geographically dispersed populations |\n"
            "| Multi-Stage Sampling | Combine methods in stages | Large-scale national studies |\n\n"
            "**Non-Probability Sampling** — individuals are not selected by chance. "
            "Results **cannot be statistically generalised** to the full population.\n\n"
            "| Method | Risk |\n"
            "|---|---|\n"
            "| Convenience Sampling | High selection bias — only accessible people included |\n"
            "| Purposive (Judgement) Sampling | Introduces researcher judgement bias; not generalisable |\n\n"
            "**Sampling Error:** Even perfect random sampling produces a gap between the sample statistic "
            "and the true population parameter. This is unavoidable — larger samples reduce it but "
            "cannot eliminate it. Always report sample size and sampling method so readers can judge generalisability.\n\n"
            "**In your report, you must state:** which sampling method was used, whether it was probability or "
            "non-probability, and acknowledge any limitations this places on generalisability."
        )

    # ── Covariance (RULE-INTERP-03) ───────────────────────────────────────────
    if "covariance" in q or "covariation" in q or "cov(" in q:
        return (
            "**What is Covariance? (RULE-INTERP-03)**\n\n"
            + STATISTICAL_EXPLANATIONS["covariance"]
            + "\n\n**Direction of covariance:**\n"
            "- **Positive covariance** → both variables tend to move in the same direction. "
            "When one is above its mean, the other tends to be above its mean too.\n"
            "- **Negative covariance** → variables move in opposite directions. "
            "When one is above its mean, the other tends to be below its mean.\n"
            "- **Near-zero covariance** → no consistent directional relationship.\n\n"
            "**⚠️ Critical rule:** Never use the raw covariance number to judge *how strong* a "
            "relationship is. The number changes completely when you change units (e.g. £ to £000). "
            "Always compute Pearson's r = Cov(X,Y) / (SD_X × SD_Y) to get a standardised "
            "strength measure on the −1 to +1 scale."
        )

    # ── Skewness / distribution shape (RULE-STAT-01) ─────────────────────────
    if any(kw in q for kw in ["skew", "skewness", "positive skew", "negative skew",
                               "distribution shape", "tail"]):
        skew_text = STATISTICAL_EXPLANATIONS["skewness"]
        if last.get("skewness") is not None:
            sv = last["skewness"]
            sd = last.get("skew_direction", "")
            skew_text += f"\n\n**In your analysis:** skewness = **{sv}**"
            if sd:
                skew_text += f" — {sd}"
        return f"**What is Skewness?**\n\n{skew_text}"

    # ── Central tendency (RULE-STAT-02) ──────────────────────────────────────
    if any(kw in q for kw in ["central tendency", "mean or median", "median or mean",
                               "when to use mean", "when to use median",
                               "mean vs median", "median vs mean"]):
        return (
            "**Which Measure of Central Tendency Should You Use? (RULE-STAT-02)**\n\n"
            + STATISTICAL_EXPLANATIONS["central_tendency"]
            + "\n\n**Quick decision rule:**\n"
            "1. Is the variable **categorical/nominal**? → Use **Mode** only.\n"
            "2. Is |skewness| < 0.5 AND no IQR outliers? → Use **Mean** (± SD).\n"
            "3. Otherwise (skewed OR outliers present) → Use **Median** ([Q1–Q3]).\n\n"
            "**Why does it matter?** In a positively skewed distribution (e.g. income, house prices), "
            "a single very high value drags the mean far above what most observations actually are. "
            "A model or report that uses the mean in this situation will consistently over-predict "
            "the typical value for the majority of the population."
        )

    # ── Five-number summary ───────────────────────────────────────────────────
    if any(kw in q for kw in ["five number", "5 number", "five-number", "box plot",
                               "boxplot", "quartile", "q1", "q3", "iqr fence",
                               "tukey fence", "outlier fence"]):
        return (
            "**Five-Number Summary and Tukey Outlier Fences**\n\n"
            + STATISTICAL_EXPLANATIONS["five_number"]
            + "\n\n**Box plot components:**\n"
            "- The **box** spans Q1 to Q3 (the IQR).\n"
            "- The line inside the box is the **median**.\n"
            "- The **whiskers** extend to the Tukey fences (Q1 − 1.5×IQR and Q3 + 1.5×IQR).\n"
            "- Points **beyond the whiskers** are flagged as potential outliers.\n\n"
            "Always investigate outliers before removing them — they may be legitimate extreme values "
            "or data-entry errors. Report your decision in the methods section."
        )

    # ── Population vs sample ──────────────────────────────────────────────────
    if any(kw in q for kw in ["population", "parameter", "population vs sample",
                               "sample vs population", "inference to population"]):
        return (
            "**Population vs Sample (RULE-INTERP-01)**\n\n"
            + STATISTICAL_EXPLANATIONS["population"]
            + "\n\n**Key terms:**\n"
            "- **Population (N):** The entire group you want to draw conclusions about.\n"
            "- **Sample (n):** The subset you actually measure.\n"
            "- **Parameter:** A descriptive measure of the *population* (e.g., μ, σ²).\n"
            "- **Statistic:** A descriptive measure of the *sample* (e.g., x̄, s²).\n\n"
            "**Why it matters for your analysis:** If your sample was not randomly drawn from "
            "the population, your statistics estimate the *sample* accurately but may not "
            "generalise to the full population. Always report your sampling method."
        )

    # ── Causation question (RULE-INTERP-02 — enriched with third-variable rule) ────
    if "caus" in q or "confound" in q or "third variable" in q or "spurious" in q:
        return (
            "**Does statistical significance prove causation? (RULE-INTERP-02)**\n\n"
            "No. Statistical significance tells you that a relationship or difference is unlikely "
            "to be due to chance alone — it does **not** prove that one variable *caused* the other.\n\n"
            "**Three conditions required for causation:**\n"
            "1. **Association exists** — a statistical relationship is present. *(Observational data can show this.)*\n"
            "2. **Temporal precedence** — the cause must precede the effect in time.\n"
            "3. **Elimination of confounders** — alternative explanations, including hidden third variables, "
            "must be ruled out.\n\n"
            "Observational data (surveys, administrative records, datasets) typically satisfies condition "
            "(1) only. To satisfy (2) and (3) you need a controlled experimental design.\n\n"
            "**The hidden third-variable trap (RULE-INTERP-02):**\n"
            "Two variables can show a strong, statistically significant correlation purely because "
            "a third, unmeasured variable is driving both of them. Classic examples:\n"
            "- Ice cream sales and drowning rates both rise in summer — the third variable is *hot weather*.\n"
            "- Shoe size and reading ability correlate in children — the third variable is *age*.\n"
            "- Hospital admission rates and mortality rates correlate — the third variable is *illness severity*.\n\n"
            "**What to do:** Before claiming any causal link, explicitly ask: *'Could a third variable "
            "explain this relationship?'* List plausible confounders in your limitations section.\n\n"
            "**Use cautious language:** *'associated with'*, *'related to'*, *'the analysis suggests'* — "
            "never *'causes'* or *'proves'* unless you have a controlled experimental design."
        )

    # ── p-value question ───────────────────────────────────────────────────────
    if "p-value" in q or "p value" in q or ("significance" in q and "practical" not in q):
        p_text = STATISTICAL_EXPLANATIONS["p_value"]
        if "p_value" in last:
            p_val = last["p_value"]
            p_text += f"\n\n**In your analysis:** p = **{p_val}**\n\n"
            if p_val < 0.05:
                p_text += (f"This IS statistically significant (p < 0.05). "
                           f"There is strong evidence against the null hypothesis. "
                           f"The probability of observing this result by chance alone (if H₀ were true) "
                           f"is {p_val:.4f} ({round(p_val*100, 2)}%).")
            else:
                p_text += (f"This is NOT statistically significant (p ≥ 0.05). "
                           f"There is insufficient evidence to reject the null hypothesis at the 5% level. "
                           f"This does NOT mean the null hypothesis is true — only that the data "
                           f"do not provide strong enough evidence against it.")
        return p_text

    # ── Statistical vs practical significance ─────────────────────────────────
    if "practical" in q and "significance" in q:
        return ("**Statistical Significance vs. Practical Significance:**\n\n"
                "- **Statistical significance** (p < 0.05) tells you whether the result is likely real "
                "or whether it could be due to chance. It depends heavily on sample size — with a very "
                "large sample, even tiny differences become statistically significant.\n\n"
                "- **Practical significance** (effect size) tells you whether the effect is *large enough "
                "to matter in the real world*. A statistically significant result may have a very small "
                "effect size and therefore little practical importance.\n\n"
                "**Example:** A study of 10,000 people might find that a new drug reduces blood pressure "
                "by 0.5 mmHg (p = 0.001). The result is statistically significant, but a 0.5 mmHg "
                "reduction has no clinical relevance.\n\n"
                "**How to report both:** Always report the effect size (Cohen's d, η², Cramér's V, R²) "
                "alongside the p-value to give the full picture.")

    # ── Assumption questions ───────────────────────────────────────────────────
    if "assumption" in q:
        if "assumption_note" in last:
            test_name = last.get("test", "this test")
            return (f"**Assumptions for {test_name}:**\n\n{last['assumption_note']}\n\n"
                    f"**Why assumptions matter:** Statistical tests are designed to work correctly only "
                    f"when their assumptions are met. Violated assumptions can produce unreliable results, "
                    f"including inflated or deflated p-values and biased effect size estimates.")
        return ("Statistical tests rely on assumptions about the data. Common assumptions include:\n\n"
                "- **Independence of observations:** Each data point is from a different, unrelated subject.\n"
                "- **Approximate normality:** The variable(s) are not severely skewed (especially important for small samples).\n"
                "- **Homogeneity of variance:** Groups have similar variances (for t-test and ANOVA).\n"
                "- **Expected cell frequencies ≥ 5:** Required for chi-square test.\n"
                "- **Linearity:** For regression and Pearson correlation, the relationship should be roughly linear.\n\n"
                "The system automatically checks key assumptions and displays warnings when they may be violated.")

    # ── Post-hoc tests question ────────────────────────────────────────────────
    if "post" in q and "hoc" in q or "tukey" in q:
        return ("**Post-Hoc Tests (after ANOVA):**\n\n"
                "When ANOVA is significant (p < 0.05), it tells you that at least one group mean differs "
                "significantly from the others — but it does NOT tell you *which* specific groups differ.\n\n"
                "**Tukey's HSD (Honestly Significant Difference)** is the most commonly used post-hoc test. "
                "It compares all possible pairs of groups while controlling the overall false positive rate.\n\n"
                "**How to report:** 'One-way ANOVA revealed a significant difference between groups "
                "[F(df₁, df₂) = x.xx, p = .xxx]. Post-hoc Tukey HSD tests showed that Group A "
                "differed significantly from Group C (p = .xxx).'")

    # ── Viva question ──────────────────────────────────────────────────────────
    if "viva" in q or "defend" in q or "justify" in q:
        test = last.get("test", "the statistical method used")
        response = (f"**How to justify {test} in your viva:**\n\n"
                    f"1. **State your research question** — what were you trying to investigate?\n"
                    f"2. **Describe your variables** — numerical, categorical, ordinal? How many groups?\n"
                    f"3. **Justify the method** — explain why this method is appropriate for those variable types.\n"
                    f"4. **Address assumptions** — state which assumptions you checked and whether they were satisfied.\n"
                    f"5. **Report the result** — statistic, degrees of freedom, p-value, effect size.\n"
                    f"6. **Interpret the result** — what does it mean in plain language?\n"
                    f"7. **Acknowledge limitations** — sample size, potential confounders, generalisability.\n\n"
                    f"**If asked 'Why not another test?':**\n"
                    f"Explain what the alternative is, what conditions it requires, and why your data's "
                    f"characteristics made your chosen test more appropriate.")
        if last.get("p_value") is not None:
            response += (f"\n\n**In your specific analysis ({test}):**\n"
                         f"- Result: p = {last['p_value']} "
                         f"({'significant' if last['p_value'] < 0.05 else 'not significant'})\n")
            if "r" in last:
                response += f"- r = {last['r']} (effect magnitude)\n"
            if "cohens_d" in last:
                response += f"- Cohen's d = {last['cohens_d']} (effect size)\n"
        return response

    # ── Interpretation question ────────────────────────────────────────────────
    if "interpret" in q or ("what" in q and "mean" in q) or "explain" in q:
        if last.get("interpretation"):
            return (f"**Result Interpretation:**\n\n{last['interpretation']}\n\n"
                    f"*This interpretation is based on the computed statistical values from your dataset.*")

    # ── Missing value questions ────────────────────────────────────────────────
    if "missing" in q or ("imputation" in q) or ("clean" in q and "data" in q):
        if cleaning_log:
            imputation_steps = [e for e in cleaning_log if "missing" in e.get("operation", "").lower()
                                or "impute" in e.get("operation", "").lower()
                                or "imputation" in e.get("method", "").lower()]
            if imputation_steps:
                lines = ["**How missing values were handled in your dataset:**\n"]
                for step in imputation_steps:
                    lines.append(f"- **{step.get('column', '?')}**: {step.get('method', '?')} imputation "
                                 f"— {step.get('values_changed', 0)} value(s) changed. "
                                 f"Reason: {step.get('detail', '')}")
                lines.append("\n**Why this matters for your research:**")
                lines.append("- The original dataset was preserved unchanged. All cleaning was on a working copy.")
                lines.append("- Imputation introduces assumptions — results may vary slightly from a complete-case analysis.")
                lines.append("- In your report, acknowledge which variables had missing values and how they were handled.")
                return "\n".join(lines)
        return ("**Handling missing values in research:**\n\n"
                "Missing data can bias results if not handled appropriately. Common methods:\n\n"
                "- **Mean/Median imputation:** Replace with average value (for numerical data). "
                "Mean is used when data is symmetric; median is preferred when skewed.\n"
                "- **Mode imputation:** Replace with most frequent value (for categorical data).\n"
                "- **KNN imputation:** Uses similar observations to estimate missing values (more sophisticated).\n"
                "- **Drop rows:** Remove observations with missing values (use only if missingness is minimal and random).\n\n"
                "**Important principle:** Never silently modify the original data. Always document cleaning decisions.")

    # ── Limitations ────────────────────────────────────────────────────────────
    if "limitation" in q:
        lines = ["**Common limitations to acknowledge in research:**\n"]
        # Add dataset-specific limitations if available
        if n_rows and n_rows < 30:
            lines.append(f"⚠️ **Small sample size (n = {n_rows}):** Results should be interpreted with caution. "
                         "Statistical tests have lower power and results may not generalise.")
        if profile.get("total_missing", 0) > 0:
            lines.append(f"⚠️ **Missing data ({profile.get('total_missing', 0)} values):** Imputation methods "
                         "introduce assumptions that may affect results.")
        lines.append("1. **Sample size:** A small sample reduces statistical power and generalisability.")
        lines.append("2. **Self-selection bias:** If participants were not randomly selected, results may not generalise.")
        lines.append("3. **Measurement error:** Variables may not perfectly capture the concept being studied.")
        lines.append("4. **Missing data:** Imputation methods introduce assumptions that may affect results.")
        lines.append("5. **Confounding variables:** An unmeasured third variable may explain the observed relationship.")
        lines.append("6. **Correlation ≠ causation:** Observational designs cannot establish causal claims.")
        lines.append("7. **Multiple testing:** Running many tests increases the chance of false positives (Type I error).")
        return "\n".join(lines)

    # ── How to report result ───────────────────────────────────────────────────
    if "report" in q or "write" in q or "apa" in q or "thesis" in q:
        if last.get("test"):
            test = last["test"]
            response = f"**How to report {test} in your thesis/paper (APA style):**\n\n"
            if "Pearson" in test or "Spearman" in test:
                method_abbr = "r" if "Pearson" in test else "r_s"
                response += (f"*'A {test} was conducted to examine the relationship between "
                             f"[Variable 1] and [Variable 2]. "
                             f"The analysis revealed {'a significant' if last.get('significant') else 'no significant'} "
                             f"{'positive' if last.get('r',0) > 0 else 'negative'} relationship, "
                             f"{method_abbr}({last.get('n',0)-2}) = {last.get('r','?')}, "
                             f"p = {last.get('p_value','?')}.'*")
            elif "t-Test" in test:
                response += (f"*'An independent samples t-test was conducted to compare [outcome] "
                             f"for [{last.get('group1','Group 1')}] and [{last.get('group2','Group 2')}]. "
                             f"{'A significant' if last.get('significant') else 'No significant'} difference was found, "
                             f"t({last.get('n',0)-2}) = {last.get('t_statistic','?')}, "
                             f"p = {last.get('p_value','?')}, d = {last.get('cohens_d','?')}.'*")
            elif "ANOVA" in test:
                response += (f"*'A one-way ANOVA was conducted to compare [outcome] across {last.get('n_groups','?')} groups. "
                             f"{'A significant' if last.get('significant') else 'No significant'} difference was found, "
                             f"F({last.get('n_groups',1)-1}, {last.get('n',0)-last.get('n_groups',1)}) = {last.get('f_statistic','?')}, "
                             f"p = {last.get('p_value','?')}, η² = {last.get('eta_squared','?')}.'*")
            elif "Fisher" in test:
                response += (f"*'Fisher's Exact Test was conducted to examine the association between "
                             f"[Variable 1] and [Variable 2]. "
                             f"The test revealed {'a statistically significant' if last.get('significant') else 'no statistically significant'} association, "
                             f"p = {last.get('p_value','?')}, OR = {last.get('odds_ratio','?')}."
                             + (f" 95% CI [{last['ci_95'][0]}, {last['ci_95'][1]}]." if last.get('ci_95') and last['ci_95'][0] else "")
                             + "'*")
            elif "Point-Biserial" in test:
                response += (f"*'A point-biserial correlation was conducted to examine the association between "
                             f"[continuous variable] and [binary variable]. "
                             f"{'A significant' if last.get('significant') else 'No significant'} association was found, "
                             f"r_pb({last.get('n',0)-2}) = {last.get('r_pb','?')}, "
                             f"p = {last.get('p_value','?')}.'*")
            elif "Goodness-of-Fit" in test:
                response += (f"*'A chi-square goodness-of-fit test was conducted to examine whether "
                             f"[variable] follows a [distribution] distribution. "
                             f"{'A significant' if last.get('significant') else 'No significant'} departure was found, "
                             f"χ²({last.get('df','?')}) = {last.get('chi2','?')}, "
                             f"p = {last.get('p_value','?')}, w = {last.get('cohens_w','?')}.'*")
            elif "Welch" in test and "ANOVA" in test:
                response += (f"*'Welch's one-way ANOVA was conducted to compare [outcome] across {last.get('n_groups','?')} groups. "
                             f"{'A significant' if last.get('significant') else 'No significant'} difference was found, "
                             f"F = {last.get('f_statistic','?')}, "
                             f"p = {last.get('p_value','?')}, η² = {last.get('eta_squared','?')}.'*")
            elif "Friedman" in test:
                response += (f"*'A Friedman test was conducted to compare [outcome] across {last.get('n_conditions','?')} conditions. "
                             f"{'A significant' if last.get('significant') else 'No significant'} difference was found, "
                             f"χ²({last.get('n_conditions',1)-1}) = {last.get('statistic','?')}, "
                             f"p = {last.get('p_value','?')}, Kendall's W = {last.get('kendalls_w','?')}.'*")
            elif "Mann-Whitney" in test:
                response += (f"*'A Mann-Whitney U test was conducted to compare [outcome] between "
                             f"[{last.get('group1','Group 1')}] and [{last.get('group2','Group 2')}]. "
                             f"{'A significant' if last.get('significant') else 'No significant'} difference was found, "
                             f"U = {last.get('u_statistic','?')}, p = {last.get('p_value','?')}, "
                             f"rank-biserial r = {last.get('rank_biserial_r','?')}.'*")
            elif "Wilcoxon" in test:
                response += (f"*'A Wilcoxon signed-rank test was conducted to compare [outcome] "
                             f"between two related conditions. "
                             f"{'A significant' if last.get('significant') else 'No significant'} difference was found, "
                             f"W = {last.get('statistic','?')}, p = {last.get('p_value','?')}, "
                             f"rank-biserial r = {last.get('rank_biserial_r','?')}.'*")
            elif "Kruskal" in test:
                response += (f"*'A Kruskal-Wallis H test was conducted to compare [outcome] across {last.get('n_groups','?')} groups. "
                             f"{'A significant' if last.get('significant') else 'No significant'} difference was found, "
                             f"H({last.get('n_groups',1)-1}) = {last.get('h_statistic','?')}, "
                             f"p = {last.get('p_value','?')}, η²H = {last.get('eta_squared_h','?')}.'*")
            elif "Chi-Square" in test:
                response += (f"*'A chi-square test of independence was performed to examine the relationship between "
                             f"[Variable 1] and [Variable 2]. "
                             f"{'A significant' if last.get('significant') else 'No significant'} association was found, "
                             f"χ²({last.get('df','?')}) = {last.get('chi2','?')}, "
                             f"p = {last.get('p_value','?')}, V = {last.get('cramers_v','?')}.'*")
            elif "Regression" in test:
                response += (f"*'A linear regression analysis was conducted to predict [outcome] from [predictors]. "
                             f"The model was {'statistically significant' if last.get('significant') else 'not statistically significant'}, "
                             f"F = {last.get('f_statistic','?')}, p = {last.get('p_value','?')}, "
                             f"R² = {last.get('r_squared','?')} (Adj. R² = {last.get('adj_r_squared','?')}).'*")
            return response
        return ("**How to report statistical results in APA style:**\n\n"
                "Run a statistical analysis first, then return here for a formatted result statement.")

    # ── Dataset-aware generic response ─────────────────────────────────────────
    context_lines = []
    if last.get("test"):
        context_lines.append(f"Your most recent analysis was: **{last['test']}**.")
        if last.get("p_value") is not None:
            p = last["p_value"]
            context_lines.append(
                f"Your last p-value was **{p}** — "
                f"{'statistically significant (p < 0.05)' if p < 0.05 else 'not statistically significant (p ≥ 0.05)'}."
            )
        if "r" in last:
            context_lines.append(f"Correlation: r = {last['r']}")
        if "cohens_d" in last:
            context_lines.append(f"Effect size: Cohen's d = {last['cohens_d']}")
    if n_rows:
        context_lines.append(f"Your dataset has **{n_rows:,} rows**.")
    if num_cols:
        context_lines.append(f"Numerical variables: {', '.join(num_cols[:6])}{'…' if len(num_cols) > 6 else ''}.")
    if cat_cols:
        context_lines.append(f"Categorical variables: {', '.join(cat_cols[:6])}{'…' if len(cat_cols) > 6 else ''}.")

    context_section = ("\n\n**Context from your analysis:**\n" + "\n".join(f"- {l}" for l in context_lines)
                       if context_lines else "")

    # ── Fallback ──────────────────────────────────────────────────────────────
    return ("I can help you understand your statistical results, the reasoning behind the methods used, "
            "how to interpret specific values, and how to justify your choices in a viva. "
            "Try asking things like:\n\n"
            "**Understanding Results:**\n"
            "- *What does p = 0.003 mean?*\n"
            "- *What does r = 0.72 mean?*\n"
            "- *What does R-squared mean?*\n"
            "- *What does Cohen's d mean?*\n\n"
            "**Statistics Basics:**\n"
            "- *What are the types of data?*\n"
            "- *What is the difference between discrete and continuous data?*\n"
            "- *What is variance and why do we square deviations?*\n"
            "- *What is standard deviation?*\n"
            "- *What is a z-score?*\n"
            "- *What is MAD (mean absolute deviation)?*\n"
            "- *What is a percentile?*\n"
            "- *What is skewness and which direction is it?*\n"
            "- *Should I use mean or median for this variable?*\n"
            "- *What is the five-number summary?*\n"
            "- *What is variation?*\n\n"
            "**Distributions & Inference:**\n"
            "- *What is the normal distribution?*\n"
            "- *What is the Central Limit Theorem?*\n"
            "- *What is a sampling distribution?*\n"
            "- *What is the difference between descriptive and inferential statistics?*\n"
            "- *What is a random variable?*\n"
            "- *What is a binomial distribution?*\n"
            "- *What is a Poisson distribution?*\n\n"
            "**Method Justification:**\n"
            "- *Why was ANOVA used instead of a t-test?*\n"
            "- *Why was Pearson used instead of Spearman?*\n"
            "- *Does this result prove causation?*\n"
            "- *What is covariance and how does it differ from correlation?*\n"
            "- *What sampling method should I use?*\n\n"
            "**Reporting & Viva:**\n"
            "- *How do I defend this method in my viva?*\n"
            "- *How do I report this result in APA format?*\n"
            "- *What are the limitations of my analysis?*\n"
            "- *What is the difference between statistical and practical significance?*\n"
            "- *What is the difference between population and sample?*\n"
            "- *Could a third variable explain my correlation?*"
            + context_section)


# ── Viva Preparation ──────────────────────────────────────────────────────────

def generate_viva_questions(analysis_history: list) -> list:
    """
    Generate viva questions based on the actual analyses performed.
    Returns a list of {question, hint, category} dicts.
    """
    questions = []
    seen_tests = set()

    for result in analysis_history:
        test = result.get("test", "")
        if test in seen_tests:
            continue
        seen_tests.add(test)

        if "Point-Biserial" in test:
            # Handled separately below — skip here to avoid duplicate via Correlation fallthrough
            questions += [
                {"question": "Why did you use Point-Biserial correlation?",
                 "hint": "Point-biserial correlation is used when one variable is continuous and the other is genuinely dichotomous (binary). It is mathematically equivalent to Pearson r applied to a 0/1 coded binary variable.",
                 "category": "Method Justification"},
                {"question": f"Your r_pb = {result.get('r_pb', '?')}, p = {result.get('p_value', '?')}. What does this mean?",
                 "hint": "Interpret the strength (weak/moderate/strong), direction (positive = higher continuous values associated with the '1' category), and whether the result is statistically significant.",
                 "category": "Result Interpretation"},
                {"question": "What assumption does Point-Biserial correlation have about the binary variable?",
                 "hint": "The binary variable should be a genuine dichotomy (e.g. pass/fail, yes/no), not an arbitrarily cut continuous variable. The continuous variable should be approximately normal within each binary group.",
                 "category": "Assumptions"},
                {"question": "Why might you prefer Point-Biserial correlation over an independent t-test for this design?",
                 "hint": "Both are mathematically equivalent for a continuous vs. binary comparison. Point-biserial r directly gives a standardised effect size in the familiar r scale, making it easier to report strength of association alongside the p-value.",
                 "category": "Critical Understanding"},
            ]

        elif "Correlation" in test:
            method = "Pearson" if "Pearson" in test else "Spearman"
            questions += [
                {"question": f"Why did you choose {test}?",
                 "hint": "Discuss the variable types (both numerical), distribution characteristics, and the research objective (relationship analysis).",
                 "category": "Method Justification"},
                {"question": f"Your r = {result.get('r', '?')}. What does this mean?",
                 "hint": "Discuss the strength (weak/moderate/strong), direction (positive/negative), the p-value, and what 'significant' means.",
                 "category": "Result Interpretation"},
                {"question": f"Why did you choose {method} rather than {'Spearman' if method == 'Pearson' else 'Pearson'} correlation?",
                 "hint": ("Pearson assumes normality and linearity. Spearman is non-parametric and more robust to outliers and non-normal distributions."
                          if method == "Pearson" else
                          "Spearman is appropriate when normality cannot be assumed, data is ordinal, or outliers are present."),
                 "category": "Method Justification"},
                {"question": "Does your correlation result prove that one variable causes the other?",
                 "hint": "No — correlation only shows association. Causation requires experimental design with controlled variables.",
                 "category": "Critical Understanding"},
                {"question": "What assumptions does this test require? Were they satisfied in your data?",
                 "hint": f"{result.get('assumption_note', 'Discuss the normality and linearity assumptions.')}",
                 "category": "Assumptions"},
            ]

        elif "t-Test" in test:
            is_paired = "Paired" in test
            questions += [
                {"question": f"Why did you use a {'paired' if is_paired else 'independent'} t-test?",
                 "hint": ("The measurements come from the same subjects (e.g. before/after). The paired t-test accounts for within-subject variation."
                          if is_paired else
                          "You had exactly two independent groups. The independent t-test is the standard parametric test for comparing two group means."),
                 "category": "Method Justification"},
                {"question": f"Why did you use a t-test rather than ANOVA?",
                 "hint": "Explain that you had exactly two groups — a t-test is appropriate. ANOVA is used for three or more groups.",
                 "category": "Method Justification"},
                {"question": f"Your p-value = {result.get('p_value', '?')}. Is this result significant?",
                 "hint": "Compare to α = 0.05. Discuss what rejecting the null hypothesis means in your context.",
                 "category": "Result Interpretation"},
                {"question": "What is Cohen's d and what does your value indicate?",
                 "hint": f"d = {result.get('cohens_d', '?')}. Small (≥ 0.2), medium (≥ 0.5), large (≥ 0.8). Effect size tells you practical significance, not just statistical.",
                 "category": "Effect Size"},
                {"question": "What does Levene's test check, and was it satisfied?",
                 "hint": "Levene's test checks whether the two groups have equal variance (homogeneity of variance). If violated, Welch's correction is applied.",
                 "category": "Assumptions"},
            ]

        elif "Welch" in test and "ANOVA" in test:
            questions += [
                {"question": "Why did you use Welch's ANOVA instead of standard one-way ANOVA?",
                 "hint": "Welch's ANOVA is used when Levene's test for homogeneity of variance is significant (p < 0.05). Standard ANOVA assumes equal group variances; Welch's ANOVA corrects the degrees of freedom using the Welch-Satterthwaite method to remain valid.",
                 "category": "Method Justification"},
                {"question": f"Your Welch's F = {result.get('f_statistic', '?')}, p = {result.get('p_value', '?')}. What does this mean?",
                 "hint": "Discuss whether the group means differ significantly. If p < 0.05, at least one group's mean differs from the others. Report effect size η² alongside the F-statistic.",
                 "category": "Result Interpretation"},
                {"question": "Which post-hoc test should follow a significant Welch's ANOVA, and why?",
                 "hint": "Games-Howell post-hoc test is recommended after Welch's ANOVA because it does not assume equal variances (unlike Tukey's HSD). It adjusts degrees of freedom pairwise for each comparison.",
                 "category": "Follow-up Analysis"},
                {"question": f"What does η² = {result.get('eta_squared', '?')} tell you about the practical importance of your result?",
                 "hint": "Eta-squared is the proportion of total variance explained by the group factor. ≥ 0.14 = large, ≥ 0.06 = medium, ≥ 0.01 = small. A statistically significant result with a small η² may not be practically important.",
                 "category": "Effect Size"},
                {"question": "When would you choose Welch ANOVA over Kruskal-Wallis?",
                 "hint": "Welch ANOVA is appropriate when the data is approximately normal within groups but variances are unequal. Kruskal-Wallis is better when normality cannot be assumed (small samples, clearly skewed distributions).",
                 "category": "Critical Understanding"},
            ]

        elif "ANOVA" in test:
            n_groups = result.get("n_groups", "?")
            questions += [
                {"question": "Why did you use ANOVA instead of multiple t-tests?",
                 "hint": "Multiple t-tests inflate the Type I error (false positive) rate. ANOVA tests all groups simultaneously while controlling the error rate at 5%.",
                 "category": "Method Justification"},
                {"question": f"Your F = {result.get('f_statistic', '?')}, p = {result.get('p_value', '?')}. What does this tell you?",
                 "hint": "Discuss whether the overall difference across groups is significant and what eta-squared indicates about effect size.",
                 "category": "Result Interpretation"},
                {"question": "If ANOVA was significant, which post-hoc test did you use and why?",
                 "hint": "Tukey's HSD is standard for comparing all pairs of groups after a significant ANOVA — it controls the family-wise error rate.",
                 "category": "Follow-up Analysis"},
                {"question": f"What does η² = {result.get('eta_squared','?')} indicate?",
                 "hint": "Eta-squared represents the proportion of variance explained by the grouping variable. ≥ 0.14 = large, ≥ 0.06 = medium, ≥ 0.01 = small.",
                 "category": "Effect Size"},
            ]

        elif "Goodness-of-Fit" in test:
            questions += [
                {"question": "What does the Chi-Square Goodness-of-Fit test examine?",
                 "hint": "It tests whether the observed distribution of a categorical variable significantly differs from a specified (often uniform) theoretical distribution.",
                 "category": "Method Justification"},
                {"question": f"Your χ²({result.get('df', '?')}) = {result.get('chi2', '?')}, p = {result.get('p_value', '?')}. What does this tell you?",
                 "hint": "If p < 0.05, the observed frequencies significantly depart from the expected distribution. Cohen's w measures the effect size: ≥ 0.5 = large, ≥ 0.3 = medium, < 0.3 = small.",
                 "category": "Result Interpretation"},
                {"question": "How did you specify the expected distribution in this test?",
                 "hint": "The default is a uniform (equal) distribution. You can also specify theoretical proportions based on prior research, population data, or a null hypothesis. Justify your choice.",
                 "category": "Critical Understanding"},
                {"question": "How is Chi-Square Goodness-of-Fit different from Chi-Square Test of Independence?",
                 "hint": "Goodness-of-Fit uses one categorical variable and tests it against a theoretical distribution. Test of Independence uses two categorical variables and tests whether they are associated. They use the same chi-square statistic but answer different research questions.",
                 "category": "Critical Understanding"},
            ]

        elif "Chi-Square" in test:
            questions += [
                {"question": "Why did you use the Chi-square test?",
                 "hint": "Both variables are categorical. Chi-square tests whether there is a statistically significant association between two categorical variables.",
                 "category": "Method Justification"},
                {"question": "What does Cramér's V tell you beyond the p-value?",
                 "hint": f"V = {result.get('cramers_v', '?')}. Cramér's V measures the effect size (strength of association), not just whether it is significant.",
                 "category": "Effect Size"},
                {"question": "What was the assumption about expected cell frequencies? Was it satisfied?",
                 "hint": "Chi-square requires expected cell frequencies ≥ 5 in at least 80% of cells. If violated, Fisher's exact test should be considered.",
                 "category": "Assumptions"},
                {"question": "Can you use chi-square to prove causation?",
                 "hint": "No — chi-square tests association, not causation. It only tells you whether the pattern of frequencies is unlikely to be random.",
                 "category": "Critical Understanding"},
            ]

        elif "Regression" in test:
            is_logistic = "Logistic" in test
            questions += [
                {"question": f"What does R² = {result.get('r_squared', result.get('pseudo_r2', '?'))} mean in your regression?",
                 "hint": "R² represents the proportion of variance in the outcome explained by the predictor(s). The remaining proportion is unexplained variance.",
                 "category": "Result Interpretation"},
                {"question": "How do you interpret the regression coefficients?",
                 "hint": ("Each coefficient shows the expected change in the log-odds of the outcome for a one-unit increase in that predictor."
                          if is_logistic else
                          "Each coefficient shows the expected change in the outcome for a one-unit increase in that predictor, holding all other predictors constant."),
                 "category": "Result Interpretation"},
                {"question": "What are the key assumptions of regression analysis?",
                 "hint": ("Linearity, independence of observations, no severe multicollinearity, adequate event rate (10 events per predictor)."
                          if is_logistic else
                          "Linearity, independence, homoscedasticity (constant variance of residuals), normality of residuals, no severe multicollinearity."),
                 "category": "Assumptions"},
                {"question": "What are the limitations of using regression with your data?",
                 "hint": "Discuss assumptions, potential confounders, possible overfitting if many predictors, and the observational (non-causal) nature of the analysis.",
                 "category": "Limitations"},
            ]

        elif "Cronbach" in test or "Reliability" in test:
            questions += [
                {"question": f"What does your Cronbach's alpha value of {result.get('alpha', '?')} indicate?",
                 "hint": "Discuss the reliability threshold (α ≥ 0.70 is generally acceptable) and what it means for the internal consistency of your scale.",
                 "category": "Result Interpretation"},
                {"question": "Why did you use Cronbach's alpha?",
                 "hint": "Explain that you used a multi-item scale (e.g. Likert questions) and wanted to verify it consistently measures the same underlying construct.",
                 "category": "Method Justification"},
                {"question": "What would you do if one of your items had a low item-total correlation?",
                 "hint": "A corrected item-total correlation below 0.30 suggests the item may not be measuring the same construct. You would consider revising or removing it and re-computing alpha.",
                 "category": "Critical Understanding"},
                {"question": "What is the difference between reliability and validity?",
                 "hint": "Reliability (Cronbach's alpha) measures consistency — whether items give consistent answers. Validity measures whether you are actually measuring the correct concept.",
                 "category": "Critical Understanding"},
            ]

        elif "Fisher" in test and "Exact" in test:
            questions += [
                {"question": "Why did you use Fisher's Exact Test instead of Chi-Square?",
                 "hint": "Fisher's Exact Test is preferred when expected cell frequencies fall below 5 in a 2×2 table. It calculates the exact probability rather than relying on a chi-square approximation.",
                 "category": "Method Justification"},
                {"question": f"Your p-value = {result.get('p_value', '?')} and Odds Ratio = {result.get('odds_ratio', '?')}. What do these mean?",
                 "hint": "p-value tests whether the association is statistically significant. An odds ratio > 1 means the event is more likely in one group; < 1 means less likely. Always report with the 95% CI.",
                 "category": "Result Interpretation"},
                {"question": "What is an odds ratio and how is it different from a relative risk?",
                 "hint": "An odds ratio compares the odds of an outcome between two groups. Relative risk compares the probabilities. They give similar values when the outcome is rare, but diverge for common outcomes.",
                 "category": "Critical Understanding"},
                {"question": f"What does the Phi coefficient of {result.get('phi', '?')} tell you?",
                 "hint": "Phi measures the effect size for a 2×2 contingency table, equivalent to Pearson r for binary variables. |Phi| ≥ 0.5 = strong, ≥ 0.3 = moderate, < 0.3 = weak association.",
                 "category": "Effect Size"},
            ]

        elif "Friedman" in test:
            questions += [
                {"question": "Why did you use the Friedman test?",
                 "hint": "The Friedman test is a non-parametric alternative to repeated-measures ANOVA. You use it when the same subjects are measured across 3+ conditions and normality cannot be assumed.",
                 "category": "Method Justification"},
                {"question": f"Your χ²({result.get('n_conditions', '?')}) = {result.get('statistic', '?')}, p = {result.get('p_value', '?')}. What does this mean?",
                 "hint": "If p < 0.05, there is a statistically significant difference in at least one condition. Use Kendall's W to assess effect size and follow up with pairwise Wilcoxon tests.",
                 "category": "Result Interpretation"},
                {"question": f"What does Kendall's W = {result.get('kendalls_w', '?')} represent?",
                 "hint": "Kendall's W measures the degree of agreement (concordance) across conditions/raters. W = 0 means no agreement; W = 1 means perfect agreement. Values ≥ 0.7 indicate strong concordance.",
                 "category": "Effect Size"},
                {"question": "What follow-up tests would you perform if the Friedman test is significant?",
                 "hint": "Conduct pairwise Wilcoxon signed-rank tests between all condition pairs. Apply Bonferroni or Holm-Bonferroni correction to control the family-wise error rate.",
                 "category": "Follow-up Analysis"},
            ]

        elif "Principal Component" in test or "PCA" in test:
            questions += [
                {"question": "Why did you use PCA?",
                 "hint": "PCA reduces the dimensionality of a dataset with many inter-correlated variables into fewer uncorrelated components that capture most of the variance. It is useful for identifying underlying patterns.",
                 "category": "Method Justification"},
                {"question": f"What does the KMO = {result.get('kmo_approx', '?')} tell you about your data?",
                 "hint": "KMO (Kaiser-Meyer-Olkin) measures the proportion of variance among variables that might be common variance. Values ≥ 0.60 are acceptable for PCA. Lower values suggest the variables are not sufficiently correlated.",
                 "category": "Assumptions"},
                {"question": f"How many components did the Kaiser criterion suggest retaining, and why?",
                 "hint": f"Kaiser's criterion retains components with eigenvalues ≥ 1 (they explain more variance than a single standardised variable). Your analysis suggested {result.get('kaiser_n_components', '?')} component(s).",
                 "category": "Result Interpretation"},
                {"question": "What are the assumptions of PCA?",
                 "hint": "PCA assumes: continuous (interval/ratio) variables, linear relationships, sufficient sample size (≥ 5–10 per variable), and meaningful correlations between variables (check KMO and Bartlett's test).",
                 "category": "Assumptions"},
            ]

        elif "Mann-Whitney" in test:
            questions += [
                {"question": "Why did you use the Mann-Whitney U test?",
                 "hint": "Mann-Whitney is a non-parametric alternative to the independent t-test. Use it when normality cannot be assumed, sample sizes are small, or data is ordinal.",
                 "category": "Method Justification"},
                {"question": f"Your U = {result.get('u_statistic', '?')}, p = {result.get('p_value', '?')}. What does this indicate?",
                 "hint": "If p < 0.05, the distributions differ significantly between groups. The rank-biserial r measures effect size: |r| ≥ 0.5 = large, ≥ 0.3 = medium, < 0.3 = small.",
                 "category": "Result Interpretation"},
                {"question": "What does the rank-biserial r measure?",
                 "hint": f"Rank-biserial r = {result.get('rank_biserial_r', '?')} is the effect size for Mann-Whitney. It estimates the probability that a randomly selected value from one group exceeds one from the other.",
                 "category": "Effect Size"},
            ]

        elif "Wilcoxon" in test:
            questions += [
                {"question": "Why did you use the Wilcoxon Signed-Rank test instead of the paired t-test?",
                 "hint": "Wilcoxon is used when the differences between paired measurements are not normally distributed. It ranks the absolute differences and tests whether the median difference equals zero.",
                 "category": "Method Justification"},
                {"question": f"Your W = {result.get('statistic', '?')}, p = {result.get('p_value', '?')}. What does this mean?",
                 "hint": "If p < 0.05, there is a statistically significant difference between the two related conditions. Interpret the direction and use rank-biserial r for effect size.",
                 "category": "Result Interpretation"},
            ]

        elif "Kruskal" in test:
            questions += [
                {"question": "Why did you use the Kruskal-Wallis test instead of one-way ANOVA?",
                 "hint": "Kruskal-Wallis is the non-parametric alternative to one-way ANOVA. It is appropriate when normality cannot be assumed or group sizes are very unequal.",
                 "category": "Method Justification"},
                {"question": f"Your H({result.get('n_groups', 1)-1}) = {result.get('h_statistic', '?')}, p = {result.get('p_value', '?')}. What does this tell you?",
                 "hint": "H tests whether at least one group's rank distribution differs from the others. If significant, follow up with pairwise Mann-Whitney tests with Bonferroni/Holm correction.",
                 "category": "Result Interpretation"},
                {"question": "If the Kruskal-Wallis result is significant, what are your next steps?",
                 "hint": "Conduct pairwise Mann-Whitney U tests between groups. Apply Bonferroni or Holm-Bonferroni correction to control the family-wise error rate across multiple comparisons.",
                 "category": "Follow-up Analysis"},
            ]

        elif "Holm-Bonferroni" in test or "Bonferroni" in test:
            questions += [
                {"question": "Why did you apply a multiple testing correction?",
                 "hint": "When running many statistical tests simultaneously, the probability of at least one false positive increases. Correction controls the family-wise error rate (FWER).",
                 "category": "Method Justification"},
                {"question": "Why did you use Holm-Bonferroni rather than standard Bonferroni?",
                 "hint": "Holm-Bonferroni is uniformly more powerful than standard Bonferroni while still controlling the FWER at the same level. It is preferred unless maximum simplicity is required.",
                 "category": "Method Justification"},
                {"question": "What is the family-wise error rate (FWER) and why does it matter?",
                 "hint": "FWER is the probability of making at least one Type I error (false positive) across a family of tests. Without correction, running 20 tests at α = 0.05 gives a ~64% chance of at least one false positive.",
                 "category": "Critical Understanding"},
            ]

    # Always add general questions
    questions += [
        {"question": "How did you handle missing values in your dataset?",
         "hint": "Describe the method used (mean/median/mode/KNN), why it was chosen, how many values were imputed, and how the original dataset was preserved.",
         "category": "Data Cleaning"},
        {"question": "What are the main limitations of your analysis?",
         "hint": "Discuss sample size, potential bias, assumption violations, generalisability, and possible confounding variables.",
         "category": "Limitations"},
        {"question": "What is the difference between statistical significance and practical significance?",
         "hint": "Statistical significance (p < 0.05) tells you the result is unlikely due to chance. Practical significance (effect size: Cohen's d, η², R², Cramér's V) tells you whether the effect is large enough to matter.",
         "category": "Critical Understanding"},
        {"question": "How did you decide which statistical test to use?",
         "hint": "Explain the decision process: consider the research question, the type of variables (numerical/categorical), the number of groups, and whether the data met test assumptions.",
         "category": "Method Justification"},
        {"question": "Could any of your results be explained by a confounding variable?",
         "hint": "A confounding variable is one that is related to both your independent and dependent variable. Even a significant result may be partly or fully explained by an unmeasured confounder. (RULE-INTERP-02: the hidden third-variable trap — two variables may correlate not because one causes the other, but because a third variable drives both.)",
         "category": "Critical Understanding"},
        # RULE-INTERP-01: sampling methodology viva questions
        {"question": "What sampling method was used to collect your data, and why does this matter?",
         "hint": "Distinguish probability sampling (simple random, stratified, systematic, cluster — allows statistical inference to the population) from non-probability sampling (convenience, purposive — introduces bias, limits generalisability). Sampling error is unavoidable but is reduced by larger samples.",
         "category": "Research Design"},
        {"question": "Can you generalise your findings to the wider population?",
         "hint": "Generalisation requires: (1) a probability sample from the target population, and (2) adequate sample size. If convenience or purposive sampling was used, explicitly acknowledge that results describe the sample only and cannot be statistically extended to the full population.",
         "category": "Research Design"},
        # RULE-STAT-01 / RULE-STAT-02: distribution and central tendency viva questions
        {"question": "How did you decide whether to report the mean or the median for your numerical variables?",
         "hint": "Use Mean (± SD) when distribution is approximately symmetric (|skewness| < 0.5) with no extreme outliers. Use Median ([Q1–Q3]) when skewness is present (|skewness| ≥ 0.5) or IQR outliers are detected. In a positively skewed distribution the mean is pulled above the median by high-end values — it overstates what is typical for most observations.",
         "category": "Descriptive Statistics"},
        {"question": "What is skewness and how did it affect your choice of method?",
         "hint": "Skewness measures distributional asymmetry. Positive skew (tail right): mean > median; few very high values pull the mean up. Negative skew (tail left): mean < median. When |skew| ≥ 0.5 the mean is misleading — prefer median. When |skew| ≥ 1.0, parametric assumptions are questionable for small samples — prefer non-parametric methods.",
         "category": "Assumptions"},
    ]
    return questions
