import pandas as pd
from db_connect import run_query
import os

os.makedirs("powerbi_data", exist_ok=True)

print("Exporting data for Power BI...")

# ── 1. Main Patient + Admission Data ─────────────────────────────────
df_main = pd.read_csv("final_data_with_clusters.csv")
df_main.to_csv("powerbi_data/01_main_data.csv", index=False)
print("1. Main data exported!")

# ── 2. Mortality by Diagnosis ─────────────────────────────────────────
mortality_diag = run_query("""
    SELECT diagnosis,
           COUNT(*) as total_cases,
           SUM(hospital_expire_flag) as deaths,
           ROUND(100.0 * SUM(hospital_expire_flag) / COUNT(*), 2) as mortality_rate,
           ROUND(AVG(DATEDIFF(dischtime, admittime)), 2) as avg_los_days
    FROM admissions
    WHERE dischtime IS NOT NULL
    GROUP BY diagnosis
    ORDER BY total_cases DESC
    LIMIT 20
""")
mortality_diag.to_csv("powerbi_data/02_mortality_by_diagnosis.csv", index=False)
print("2. Mortality by diagnosis exported!")

# ── 3. Monthly Admission Trend ────────────────────────────────────────
monthly = run_query("""
    SELECT DATE_FORMAT(admittime, '%Y-%m') as admission_month,
           COUNT(*) as admissions,
           SUM(hospital_expire_flag) as deaths,
           ROUND(AVG(DATEDIFF(dischtime, admittime)), 2) as avg_los
    FROM admissions
    WHERE dischtime IS NOT NULL
    GROUP BY admission_month
    ORDER BY admission_month
""")
monthly.to_csv("powerbi_data/03_monthly_trends.csv", index=False)
print("3. Monthly trends exported!")

# ── 4. Insurance Analytics ────────────────────────────────────────────
insurance = run_query("""
    SELECT insurance,
           COUNT(*) as admissions,
           SUM(hospital_expire_flag) as deaths,
           ROUND(100.0 * SUM(hospital_expire_flag) / COUNT(*), 2) as mortality_pct,
           ROUND(AVG(DATEDIFF(dischtime, admittime)), 2) as avg_los
    FROM admissions
    WHERE dischtime IS NOT NULL
    GROUP BY insurance
""")
insurance.to_csv("powerbi_data/04_insurance_analytics.csv", index=False)
print("4. Insurance analytics exported!")

# ── 5. Lab Analytics ──────────────────────────────────────────────────
lab = run_query("""
    SELECT d.label as test_name,
           d.category,
           COUNT(l.row_id) as total_tests,
           SUM(IF(l.flag='abnormal', 1, 0)) as abnormal_count,
           ROUND(100.0 * SUM(IF(l.flag='abnormal',1,0)) / COUNT(l.row_id), 2) as abnormal_pct
    FROM labevents l
    JOIN d_labitems d ON l.itemid = d.itemid
    GROUP BY d.label, d.category
    ORDER BY total_tests DESC
    LIMIT 30
""")
lab.to_csv("powerbi_data/05_lab_analytics.csv", index=False)
print("5. Lab analytics exported!")

# ── 6. Admission Type Summary ─────────────────────────────────────────
adm_type = run_query("""
    SELECT admission_type,
           COUNT(*) as total,
           SUM(hospital_expire_flag) as deaths,
           ROUND(100.0 * SUM(hospital_expire_flag) / COUNT(*), 2) as mortality_pct,
           ROUND(AVG(DATEDIFF(dischtime, admittime)), 2) as avg_los
    FROM admissions
    WHERE dischtime IS NOT NULL
    GROUP BY admission_type
""")
adm_type.to_csv("powerbi_data/06_admission_type.csv", index=False)
print("6. Admission type exported!")

# ── 7. Patient Severity Clusters ──────────────────────────────────────
severity = df_main.groupby("severity").agg(
    patients        = ("subject_id",           "count"),
    avg_age         = ("age",                  "mean"),
    avg_los         = ("los_days",             "mean"),
    avg_abnormal    = ("abnormal_pct",         "mean"),
    mortality_rate  = ("hospital_expire_flag", "mean")
).round(2).reset_index()
severity["mortality_rate"] = (severity["mortality_rate"] * 100).round(2)
severity.to_csv("powerbi_data/07_severity_clusters.csv", index=False)
print("7. Severity clusters exported!")

# ── 8. LOS Distribution ───────────────────────────────────────────────
los = run_query("""
    SELECT
        CASE
            WHEN DATEDIFF(dischtime, admittime) <= 1  THEN '0-1 days'
            WHEN DATEDIFF(dischtime, admittime) <= 3  THEN '2-3 days'
            WHEN DATEDIFF(dischtime, admittime) <= 7  THEN '4-7 days'
            WHEN DATEDIFF(dischtime, admittime) <= 14 THEN '8-14 days'
            ELSE '15+ days'
        END as los_bucket,
        COUNT(*) as patients
    FROM admissions
    WHERE dischtime IS NOT NULL
    GROUP BY los_bucket
""")
los.to_csv("powerbi_data/08_los_distribution.csv", index=False)
print("8. LOS distribution exported!")

print("\nAll files exported to powerbi_data/ folder!")
print("Ready to load into Power BI!")