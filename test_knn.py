"""Test the fixed KNN imputation against the exact failure scenario."""
import sys
sys.path.insert(0, '.')
import pandas as pd
import numpy as np
from engine.cleaner import impute_column, remove_duplicates

df = pd.read_excel('test_data.xlsx')
print(f"Original: {len(df)} rows, {df.isnull().sum().sum()} total missing")

# Simulate the exact failure: deduplicate first (leaves non-contiguous index), then KNN
df_dedup, _ = remove_duplicates(df)
print(f"After dedup: {len(df_dedup)} rows, index max={df_dedup.index.max()}")

col = 'Study_Hours_Per_Week'
n_miss = int(df_dedup[col].isnull().sum())
print(f"Missing in '{col}' after dedup: {n_miss}")

df_fixed, log = impute_column(df_dedup, col, 'knn')
assert df_fixed[col].isnull().sum() == 0, "Still has missing values after KNN!"
print(f"KNN on deduped df: changed={log['values_changed']}, detail={log['detail']}")

# Also test on fresh df (no prior operations)
df_fresh, log2 = impute_column(df, 'Exam_Score', 'knn')
assert df_fresh['Exam_Score'].isnull().sum() == 0, "Still missing after KNN on fresh df!"
print(f"KNN on fresh df: changed={log2['values_changed']}, detail={log2['detail']}")

# Test multiple sequential operations (most complex path)
df_work = df.copy()
df_work, _ = remove_duplicates(df_work)
df_work, _ = impute_column(df_work, 'Study_Hours_Per_Week', 'median')
df_work, _ = impute_column(df_work, 'Exam_Score', 'knn')
df_work, _ = impute_column(df_work, 'Attendance_Pct', 'knn')
remaining = df_work[['Study_Hours_Per_Week','Exam_Score','Attendance_Pct']].isnull().sum().sum()
assert remaining == 0, f"Still {remaining} missing after sequential operations!"
print(f"Sequential dedup+median+knn+knn: all passed, remaining missing={remaining}")

print()
print("=" * 50)
print("ALL KNN TESTS PASSED")
print("=" * 50)
