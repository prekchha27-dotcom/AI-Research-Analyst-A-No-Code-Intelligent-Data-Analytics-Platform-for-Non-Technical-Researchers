"""
Generate a realistic test dataset for the AI Research Analyst platform.
"""

import pandas as pd
import numpy as np

np.random.seed(42)
N = 120

data = {
    "Student_ID": range(1001, 1001 + N),
    "Gender": np.random.choice(["Male", "Female", "male", "FEMALE"], N, p=[0.40, 0.40, 0.10, 0.10]),
    "Department": np.random.choice(["Science", "Arts", "Engineering", "Management", "Science "], N,
                                    p=[0.25, 0.20, 0.30, 0.20, 0.05]),
    "Study_Hours_Per_Week": np.round(np.random.normal(20, 6, N), 1).clip(0, 50),
    "Exam_Score": np.round(np.random.normal(65, 12, N), 1).clip(20, 100),
    "Attendance_Pct": np.round(np.random.uniform(50, 100, N), 1),
    "Part_Time_Job": np.random.choice(["Yes", "No"], N, p=[0.35, 0.65]),
    "Scholarship": np.random.choice(["Yes", "No"], N, p=[0.30, 0.70]),
    "Passed": np.random.choice([1, 0], N, p=[0.75, 0.25]),
    "Year_of_Study": np.random.choice([1, 2, 3, 4], N),
    "Monthly_Living_Cost": np.round(np.random.normal(800, 150, N), 0).clip(300, 1500),
    "Distance_km": np.round(np.random.exponential(15, N), 1),
}

df = pd.DataFrame(data)

# Introduce missing values
np.random.seed(7)
for col, pct in [("Study_Hours_Per_Week", 0.08), ("Exam_Score", 0.05),
                  ("Attendance_Pct", 0.12), ("Monthly_Living_Cost", 0.07)]:
    idx = np.random.choice(df.index, size=int(N * pct), replace=False)
    df.loc[idx, col] = np.nan

# Outliers
df.loc[[5, 22, 67], "Exam_Score"] = [3.0, 2.5, 4.0]

# Duplicates
df = pd.concat([df, df.iloc[[10, 25, 40]]], ignore_index=True)

df.to_excel("test_data.xlsx", index=False)
df.to_csv("test_data.csv", index=False)
print(f"Created: {len(df)} rows, {len(df.columns)} columns")
print(f"Missing values total: {df.isnull().sum().sum()}")
print(f"Duplicate rows: {df.duplicated().sum()}")
