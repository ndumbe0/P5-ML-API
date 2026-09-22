# Power BI Dashboard — P5 Sepsis Prediction

Built from `p5_sepsis_patients.csv` in this folder (599 ICU patients,
34.7% sepsis-positive). The raw file in `data/` had CR-only line endings and
BOM; this one is fixed, renamed to readable columns, and physiologically
impossible zeros (glucose/BP/skin/insulin/BMI = 0) are set to blank so averages
are honest.

## 1. Load the data

1. **Get data → Text/CSV** → `p5_sepsis_patients.csv` → **Transform Data**.
2. Types: `age`, `pregnancies`, `has_insurance`, `sepsis_positive` → Whole
   Number; `glucose`, `bp`, `skin_thickness`, `insulin`, `bmi`,
   `diabetes_pedigree` → Decimal; rest Text.
3. Rename the query to `Patients`. Close & Apply.
4. (Optional) Load `p5_sepsis_summary.csv` as `PatientSummary`.

## 2. DAX measures

```dax
Sepsis Rate = AVERAGE(Patients[sepsis_positive])

Patients Count = COUNTROWS(Patients)

Positives = SUM(Patients[sepsis_positive])

Avg Glucose = AVERAGE(Patients[glucose])

Avg BMI = AVERAGE(Patients[bmi])

Sepsis Rate Insured = CALCULATE([Sepsis Rate], Patients[has_insurance] = 1)

Sepsis Rate Uninsured = CALCULATE([Sepsis Rate], Patients[has_insurance] = 0)
```

## 3. Pages and visuals

### Page 1 — Overview
| Visual | Fields |
|---|---|
| **Card** | `Sepsis Rate` (%) |
| **Card** | `Patients Count` |
| **Donut** | Legend: `sepsis` • Values: `Patients Count` |
| **Line** | Axis: `age_group` (order <=25…51+) • Values: `Sepsis Rate` |
| **Bar** | Axis: `bmi_category` • Values: `Sepsis Rate` |

### Page 2 — Clinical drivers
| Visual | Fields |
|---|---|
| **Scatter** | X: `glucose` • Y: `bmi` • Legend: `sepsis` • Details: `patient_id` |
| **Clustered bar** | Axis: `has_insurance` • Values: `Sepsis Rate` |
| **Decomposition tree** | Explain `Positives` by `age_group` → `bmi_category` → `glucose` |
| **Table** | `v_high_risk_patients` equivalent: filter `sepsis_positive = 1`, columns age/glucose/bmi, sort by glucose desc |

### Page 3 — Model context
Show `model_metrics.json` numbers as cards or a small pasted table:
Logistic Regression best — accuracy 71.7%, ROC-AUC 79.8%, recall 71.4%
(recall matters most for sepsis screening).

## 4. Optional: SQL Server

Run `p5_sepsis.sql` → database `P5_SepsisPrediction`, table
`p5_sepsis_patients` + views `v_sepsis_by_age_group`, `v_sepsis_by_bmi`,
`v_sepsis_by_insurance`, `v_glucose_bmi_profile`, `v_high_risk_patients`.

## 5. Tableau

Open `p5_sepsis_dashboard.twb` (expects `p5_sepsis_patients.csv` alongside).
5 sheets: Sepsis Rate by Age Group / BMI / Insurance, Avg Glucose by Outcome,
Patients by Outcome.
