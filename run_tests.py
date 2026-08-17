"""
Validation test — runs all engines against the test dataset.
"""
import sys
import io
sys.path.insert(0, '.')
# Force UTF-8 output on Windows to avoid UnicodeEncodeError in PowerShell
if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import pandas as pd
import numpy as np

df = pd.read_excel('test_data.xlsx')
print(f"Loaded: {len(df)} rows, {len(df.columns)} columns")

# ── Test Profiler ─────────────────────────────────────────────────────────────
from engine.profiler import profile_dataset, generate_plain_summary
profile = profile_dataset(df)
assert profile['n_rows'] == len(df)
assert profile['total_missing'] > 0
assert profile['duplicate_rows'] >= 3
print(f"PROFILER OK — Numerical: {profile['numerical_cols']}, Categorical: {profile['categorical_cols']}")
summary = generate_plain_summary(profile)
print(summary[:200])  # truncate to avoid encoding issues in terminal

# ── Test Cleaner ──────────────────────────────────────────────────────────────
from engine.cleaner import recommend_imputation_method, impute_column, remove_duplicates
col = 'Exam_Score'
rec = recommend_imputation_method(col, df[col], 'numerical')
assert rec['n_missing'] > 0
print(f"CLEANER OK — Recommended method: {rec['method']}, reason: {rec['reason'][:60]}")

df_clean, log1 = impute_column(df, col, rec['method'])
assert df_clean[col].isnull().sum() == 0
print(f"IMPUTATION OK — {log1['values_changed']} values changed in {col}")

df_dedup, log2 = remove_duplicates(df_clean)
print(f"DEDUP OK — {log2['values_changed']} duplicates removed")

# ── Test Statistics ───────────────────────────────────────────────────────────
from engine.statistics import (
    descriptive_stats, frequency_table, crosstab,
    pearson_correlation, spearman_correlation, correlation_matrix,
    chi_square_test, independent_ttest, one_way_anova,
    linear_regression
)

desc = descriptive_stats(df_clean, ['Exam_Score', 'Study_Hours_Per_Week'])
assert len(desc) == 2
print(f"DESCRIPTIVE STATS OK — {desc[['Variable','Mean','Median']].to_string(index=False)}")

freq = frequency_table(df_clean, 'Gender')
assert len(freq) >= 2
print(f"FREQUENCY OK — {len(freq)} categories")

corr = pearson_correlation(df_clean, 'Study_Hours_Per_Week', 'Exam_Score')
assert 'r' in corr
print(f"PEARSON OK — r={corr['r']}, p={corr['p_value']}")

# t-test: need a clean 2-group column
df_clean['Gender_Clean'] = df_clean['Gender'].str.strip().str.title()
df_2grp = df_clean[df_clean['Gender_Clean'].isin(['Male', 'Female'])]
ttest = independent_ttest(df_2grp, 'Exam_Score', 'Gender_Clean')
assert 't_statistic' in ttest
print(f"T-TEST OK — t={ttest['t_statistic']}, p={ttest['p_value']}")

anova = one_way_anova(df_clean, 'Exam_Score', 'Department')
assert 'f_statistic' in anova
print(f"ANOVA OK — F={anova['f_statistic']}, p={anova['p_value']}, groups={anova['n_groups']}")

chi2 = chi_square_test(df_clean, 'Gender_Clean', 'Part_Time_Job')
assert 'chi2' in chi2
print(f"CHI-SQUARE OK — chi2={chi2['chi2']}, p={chi2['p_value']}, V={chi2['cramers_v']}")

lr = linear_regression(df_clean, 'Exam_Score', ['Study_Hours_Per_Week', 'Attendance_Pct'])
assert 'r_squared' in lr
print(f"LINEAR REG OK — R2={lr['r_squared']}, F={lr['f_statistic']}")

# ── Test Interpreter ──────────────────────────────────────────────────────────
from engine.interpreter import interpret_result, answer_research_question, generate_viva_questions
interp = interpret_result(corr)
assert 'Pearson' in interp
print("INTERPRETER OK")

answer = answer_research_question("What does p-value mean?", {"last_result": corr})
assert len(answer) > 50
print("CHATBOT OK")

viva_qs = generate_viva_questions([corr, ttest, anova])
assert len(viva_qs) >= 3
print(f"VIVA QUESTIONS OK — {len(viva_qs)} questions generated")

# ── Test Visualizer ───────────────────────────────────────────────────────────
from engine.visualizer import build_chart, recommend_visualization, correlation_heatmap
from engine.statistics import correlation_matrix

fig = build_chart(df_clean, 'bar_chart', 'Department', 'Exam_Score')
assert fig is not None
print("BAR CHART OK")

fig2 = build_chart(df_clean, 'scatter_plot', 'Study_Hours_Per_Week', 'Exam_Score')
assert fig2 is not None
print("SCATTER PLOT OK")

corr_mat = correlation_matrix(df_clean, ['Study_Hours_Per_Week', 'Exam_Score', 'Attendance_Pct'])
fig3 = correlation_heatmap(corr_mat)
assert fig3 is not None
print("CORRELATION HEATMAP OK")

recs = recommend_visualization('numerical', 'numerical', 'Study_Hours_Per_Week', 'Exam_Score')
assert recs[0]['stars'] == 5
assert recs[0]['chart_type'] == 'scatter_plot'
print(f"VIZ RECOMMENDATION OK — Top: {recs[0]['chart_type']} ({recs[0]['star_label']})")

# ── Test Recommender ──────────────────────────────────────────────────────────
from engine.recommender import recommend_statistical_method
recs_compare = recommend_statistical_method('compare', 'numerical', ['categorical'], n_groups=2)
assert recs_compare[0].method == 'Independent Samples t-Test'
print(f"RECOMMENDER OK — {recs_compare[0].method}")

recs_anova = recommend_statistical_method('compare', 'numerical', ['categorical'], n_groups=4)
assert recs_anova[0].method == 'One-Way ANOVA'
print(f"RECOMMENDER OK — {recs_anova[0].method}")

# ── Test Reporter ─────────────────────────────────────────────────────────────
from engine.reporter import generate_report_markdown
report = generate_report_markdown('test_data.xlsx', profile, [log1, log2], [corr, ttest], ['Test finding'])
assert 'Dataset Description' in report
assert 'Statistical Analysis' in report
print("REPORT GENERATOR OK")

print()
print("=" * 50)
print("ALL TESTS PASSED SUCCESSFULLY")
print("=" * 50)
