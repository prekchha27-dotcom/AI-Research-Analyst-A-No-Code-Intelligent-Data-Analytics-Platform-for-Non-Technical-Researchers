# -*- coding: utf-8 -*-
"""Smoke test for the new analytical_roles + profiler integration."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np

PASS = FAIL = 0

def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        print(f"  PASS  {name}")
        PASS += 1
    else:
        print(f"  FAIL  {name}" + (f" — {detail}" if detail else ""))
        FAIL += 1

print("\n[A] Analytical roles — import and basic classification")
from engine.analytical_roles import classify_column, classify_dataset, recommend_imputation_for_column

# Serial number column (integers 1..100)
df_ser = pd.DataFrame({
    'SL_NO':    list(range(1, 101)),
    'Score':    np.random.randint(40, 100, 100),
    'Gender':   np.random.choice(['Male', 'Female'], 100),
})
cl_sl = classify_column(df_ser, 'SL_NO')
check("serial: role is serial or identifier",
      cl_sl.analytical_role in ('serial_number', 'identifier'))
check("serial: no suitable analyses",   len(cl_sl.suitable_analyses) == 0)
check("serial: imputation_role is identifier_preserve",
      cl_sl.imputation_role == 'identifier_preserve')
check("serial: has warning",            len(cl_sl.warnings) > 0)

cl_sc = classify_column(df_ser, 'Score')
check("score: role is substantive_numerical", cl_sc.analytical_role == 'substantive_numerical')
check("score: has suitable analyses",         len(cl_sc.suitable_analyses) > 0)
check("score: imputation_role is imputable",  cl_sc.imputation_role == 'imputable')

cl_gen = classify_column(df_ser, 'Gender')
check("gender: role is categorical",     cl_gen.analytical_role == 'categorical')
check("gender: suitable analyses has frequency", 'frequency' in cl_gen.suitable_analyses)

print("\n[B] Profiler with role classification")
from engine.profiler import profile_dataset, generate_plain_summary

df_mix = pd.DataFrame({
    'ID':          list(range(1, 51)),
    'Age':         np.random.randint(18, 65, 50),
    'Score':       np.random.uniform(50, 100, 50),
    'Department':  np.random.choice(['CS', 'Math', 'Physics'], 50),
    'Passed':      np.random.choice(['Yes', 'No'], 50),
})
df_mix.loc[5, 'Age'] = np.nan
df_mix.loc[10, 'Score'] = np.nan

profile = profile_dataset(df_mix)
check("profiler: identifier excluded from numerical_cols",
      'ID' not in profile['numerical_cols'])
check("profiler: Age in numerical_cols",
      'Age' in profile['numerical_cols'] or 'Age' in profile.get('analysis_ready_numerical', []))
check("profiler: ID in identifier_cols",
      'ID' in profile.get('identifier_cols', []))
check("profiler: classifications dict present",
      'classifications' in profile and len(profile['classifications']) == 5)
check("profiler: imputation_recommendations present",
      'imputation_recommendations' in profile)
# ID has no missing values in df_mix — correct method is "none"
# But imputation_role in the classification should be "identifier_preserve"
check("profiler: ID classification imputation_role is identifier_preserve",
      profile['classifications']['ID']['imputation_role'] == 'identifier_preserve')
check("profiler: generate_plain_summary works",
      isinstance(generate_plain_summary(profile), str))

print("\n[C] Imputation recommendation — role-aware")
from engine.cleaner import recommend_imputation_method

# ID column
id_series = pd.Series(list(range(1, 51)), name='ID')
id_series[5] = np.nan
rec_id = recommend_imputation_method('ID', id_series, 'numerical', analytical_role='identifier')
check("ID imputation: do_not_impute",  rec_id['method'] == 'do_not_impute')
check("ID imputation: has reason",     len(rec_id.get('reason', '')) > 0)

# Score column (skewed)
from scipy.stats import skewnorm
score_series = pd.Series(np.concatenate([np.random.normal(60, 10, 40), [np.nan]*10]))
rec_score = recommend_imputation_method('Score', score_series, 'numerical',
                                         analytical_role='substantive_numerical')
check("Score imputation: not do_not_impute", rec_score['method'] != 'do_not_impute')
check("Score imputation: has why_appropriate", len(rec_score.get('why_appropriate', '')) > 0)

# Categorical column
cat_series = pd.Series(['Male', 'Female', None, 'Male', 'Female'] * 10)
rec_cat = recommend_imputation_method('Gender', cat_series, 'categorical')
check("Gender imputation: mode",       rec_cat['method'] == 'mode')
check("Gender imputation: why_appropriate present", 'why_appropriate' in rec_cat)

print("\n[D] Constant column detection")
df_const = pd.DataFrame({
    'Country': ['Nigeria'] * 30,
    'Score': np.random.randint(40, 90, 30),
})
cl_const = classify_column(df_const, 'Country')
check("constant: role is constant",     cl_const.analytical_role == 'constant')
check("constant: no suitable analyses", len(cl_const.suitable_analyses) == 0)

print("\n[E] High-uniqueness string column (ID-like)")
import string, random
df_str = pd.DataFrame({
    'RegNo': ['REG' + ''.join(random.choices(string.digits, k=5)) for _ in range(50)],
    'Name': [f'Student {i}' for i in range(50)],
    'Score': np.random.randint(40, 100, 50),
})
cl_reg = classify_column(df_str, 'RegNo')
check("RegNo: role is identifier or admin_code",
      cl_reg.analytical_role in ('identifier', 'administrative_code'))

print("\n[F] User overrides applied through profiler")
profile_ovr = profile_dataset(df_mix, user_overrides={
    'ID': {'analytical_role': 'substantive_numerical'}
})
check("override: ID now in numerical_cols",
      'ID' in profile_ovr['numerical_cols'])
check("override: evidence has override marker",
      any('manually overridden' in e.lower() or 'override' in e.lower()
          for e in profile_ovr['classifications']['ID']['evidence']))

print("\n" + "=" * 60)
print(f"  PASSED : {PASS}")
print(f"  FAILED : {FAIL}")
print("=" * 60)
if FAIL == 0:
    print("  ALL ENGINE SMOKE TESTS PASSED")
else:
    print(f"  {FAIL} CHECK(S) FAILED")
print("=" * 60)
