import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
import joblib
import numpy as np


BMI_THRESHOLDS = {
    'underweight': 18.5,      # < 18.5 = Underweight
    'normal': 25,             # 18.5-24.9 = Normal
    'overweight': 30,         # 25-29.9 = Overweight
    'obese_class_1': 35,      # 30-34.9 = Obese Class I
    'obese_class_2': 40,      # 35-39.9 = Obese Class II
    # >= 40 = Severe Obesity
}


FBS_THRESHOLDS = {
    'hypoglycemia': 70,       # < 70 = Hypoglycemia (Low)
    'normal': 100,            # 70-99 = Normal
    'prediabetes': 126,       # 100-125 = Prediabetes
    # >= 126 = Diabetes
}

HBA1C_THRESHOLDS = {
    'normal': 5.7,            # < 5.7% = Normal
    'prediabetes': 6.5,       # 5.7-6.4% = Prediabetes
    # >= 6.5% = Diabetes
}

HDL_THRESHOLDS = {
    'male_low': 40,           # Male: < 40 = Low (Risk)
    'male_protective': 60,    # Male: >= 60 = Protective
    'female_low': 50,         # Female: < 50 = Low (Risk)
    'female_protective': 60,  # Female: >= 60 = Protective
    # Between low and protective = Normal
}

LDL_THRESHOLDS = {
    'optimal': 100,           # < 100 = Optimal
    'near_optimal': 130,      # 100-129 = Near Optimal
    'borderline': 160,        # 130-159 = Borderline High
    'high': 190,              # 160-189 = High
    # >= 190 = Very High
}

TG_THRESHOLDS = {
    'normal': 150,            # < 150 = Normal
    'borderline': 200,        # 150-199 = Borderline High
    'high': 500,              # 200-499 = High
    # >= 500 = Very High
}

# --- CONFIGURATION ---

FILE_NAME = 'dataset.csv'
TARGET_COLUMN = 'bs2hr1_4'  # 2-hour post-prandial glucose
TARGET_THRESHOLD = 140       # mg/dL - Prediabetes/IGT threshold

# Feature Selection
FEATURES = [
    'ageyr14_4',    # Age
    'sex_4',        # Sex (1=Male, 2=Female)
    'BMI_4',        # Body Mass Index
    'fbs1_4',       # Fasting Blood Sugar
    'tg1_4',        # Triglycerides
    'hdl1_4',       # HDL Cholesterol
    'ldl1_4',       # LDL Cholesterol
    'kcal_4',       # Calorie intake
    'weight_4',     # Weight
    'height_4',     # Height
    'waist_4',      # Waist circumference
    'physical_activity',  # Physical activity level
]


# --- 1. DATA LOADING ---

print("\n[STEP 1] Loading data...")
try:
    df = pd.read_csv(FILE_NAME)
    print(f" Loaded {len(df)} rows, {len(df.columns)} columns")
except FileNotFoundError:
    print(f" Error: File '{FILE_NAME}' not found.")
    exit()

# Verify required columns
missing_cols = [col for col in FEATURES + [TARGET_COLUMN] if col not in df.columns]
if missing_cols:
    print(f"✗ Missing columns: {missing_cols}")
    exit()

# Select features
df = df[FEATURES + [TARGET_COLUMN]].copy()

# Convert to numeric
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

print(f" Selected {len(FEATURES)} features")
print(f" Missing values: {df.isnull().sum().sum()} total")

# Drop rows with missing values
df_clean = df.dropna().reset_index(drop=True)
print(f" Clean dataset: {len(df_clean)} rows (removed {len(df) - len(df_clean)} rows)")

# --- 2. FEATURE ENGINEERING WITH EXACT THRESHOLDS ---

print("\n[STEP 2] Feature engineering with exact thresholds...")

# Sex binary features
df_clean['sex_is_male'] = (df_clean['sex_4'] == 1).astype(int)
df_clean['sex_is_female'] = (df_clean['sex_4'] == 2).astype(int)


df_clean['bmi_underweight'] = (df_clean['BMI_4'] < BMI_THRESHOLDS['underweight']).astype(int)
df_clean['bmi_normal'] = ((df_clean['BMI_4'] >= BMI_THRESHOLDS['underweight']) & 
                           (df_clean['BMI_4'] < BMI_THRESHOLDS['normal'])).astype(int)
df_clean['bmi_overweight'] = ((df_clean['BMI_4'] >= BMI_THRESHOLDS['normal']) & 
                               (df_clean['BMI_4'] < BMI_THRESHOLDS['overweight'])).astype(int)
df_clean['bmi_obese_1'] = ((df_clean['BMI_4'] >= BMI_THRESHOLDS['overweight']) & 
                            (df_clean['BMI_4'] < BMI_THRESHOLDS['obese_class_1'])).astype(int)
df_clean['bmi_obese_2'] = ((df_clean['BMI_4'] >= BMI_THRESHOLDS['obese_class_1']) & 
                            (df_clean['BMI_4'] < BMI_THRESHOLDS['obese_class_2'])).astype(int)
df_clean['bmi_severe_obesity'] = (df_clean['BMI_4'] >= BMI_THRESHOLDS['obese_class_2']).astype(int)


df_clean['fbs_hypoglycemia'] = (df_clean['fbs1_4'] < FBS_THRESHOLDS['hypoglycemia']).astype(int)
df_clean['fbs_normal'] = ((df_clean['fbs1_4'] >= FBS_THRESHOLDS['hypoglycemia']) & 
                           (df_clean['fbs1_4'] < FBS_THRESHOLDS['normal'])).astype(int)
df_clean['fbs_prediabetes'] = ((df_clean['fbs1_4'] >= FBS_THRESHOLDS['normal']) & 
                                (df_clean['fbs1_4'] < FBS_THRESHOLDS['prediabetes'])).astype(int)
df_clean['fbs_diabetes'] = (df_clean['fbs1_4'] >= FBS_THRESHOLDS['prediabetes']).astype(int)


# For males
df_clean['hdl_low_male'] = ((df_clean['sex_is_male'] == 1) & 
                             (df_clean['hdl1_4'] < HDL_THRESHOLDS['male_low'])).astype(int)
df_clean['hdl_protective_male'] = ((df_clean['sex_is_male'] == 1) & 
                                    (df_clean['hdl1_4'] >= HDL_THRESHOLDS['male_protective'])).astype(int)

# For females
df_clean['hdl_low_female'] = ((df_clean['sex_is_female'] == 1) & 
                               (df_clean['hdl1_4'] < HDL_THRESHOLDS['female_low'])).astype(int)
df_clean['hdl_protective_female'] = ((df_clean['sex_is_female'] == 1) & 
                                      (df_clean['hdl1_4'] >= HDL_THRESHOLDS['female_protective'])).astype(int)

# Combined HDL features
df_clean['hdl_low'] = (df_clean['hdl_low_male'] | df_clean['hdl_low_female']).astype(int)
df_clean['hdl_protective'] = (df_clean['hdl_protective_male'] | df_clean['hdl_protective_female']).astype(int)


df_clean['ldl_optimal'] = (df_clean['ldl1_4'] < LDL_THRESHOLDS['optimal']).astype(int)
df_clean['ldl_near_optimal'] = ((df_clean['ldl1_4'] >= LDL_THRESHOLDS['optimal']) & 
                                 (df_clean['ldl1_4'] < LDL_THRESHOLDS['near_optimal'])).astype(int)
df_clean['ldl_borderline'] = ((df_clean['ldl1_4'] >= LDL_THRESHOLDS['near_optimal']) & 
                               (df_clean['ldl1_4'] < LDL_THRESHOLDS['borderline'])).astype(int)
df_clean['ldl_high'] = ((df_clean['ldl1_4'] >= LDL_THRESHOLDS['borderline']) & 
                         (df_clean['ldl1_4'] < LDL_THRESHOLDS['high'])).astype(int)
df_clean['ldl_very_high'] = (df_clean['ldl1_4'] >= LDL_THRESHOLDS['high']).astype(int)


df_clean['tg_normal'] = (df_clean['tg1_4'] < TG_THRESHOLDS['normal']).astype(int)
df_clean['tg_borderline'] = ((df_clean['tg1_4'] >= TG_THRESHOLDS['normal']) & 
                              (df_clean['tg1_4'] < TG_THRESHOLDS['borderline'])).astype(int)
df_clean['tg_high'] = ((df_clean['tg1_4'] >= TG_THRESHOLDS['borderline']) & 
                        (df_clean['tg1_4'] < TG_THRESHOLDS['high'])).astype(int)
df_clean['tg_very_high'] = (df_clean['tg1_4'] >= TG_THRESHOLDS['high']).astype(int)

# === METABOLIC SYNDROME RISK SCORE ===
# Using: FBS ≥100, TG ≥150, HDL low (sex-specific), BMI ≥30
df_clean['metabolic_syndrome_score'] = (
    (df_clean['fbs1_4'] >= FBS_THRESHOLDS['normal']).astype(int) +
    (df_clean['tg1_4'] >= TG_THRESHOLDS['normal']).astype(int) +
    df_clean['hdl_low'] +
    (df_clean['BMI_4'] >= BMI_THRESHOLDS['overweight']).astype(int)
)
df_clean['metabolic_syndrome_risk'] = (df_clean['metabolic_syndrome_score'] >= 3).astype(int)



# --- 3. TARGET VARIABLE ---

print("\n[STEP 3] Defining target variable...")
y = (df_clean[TARGET_COLUMN] > TARGET_THRESHOLD).astype(int)
print(f" Target distribution:")
print(f"  - High Risk (1): {y.sum()} ({y.sum()/len(y)*100:.1f}%)")
print(f"  - Low Risk (0): {(~y.astype(bool)).sum()} ({(~y.astype(bool)).sum()/len(y)*100:.1f}%)")

# --- 4. PREPARE FEATURES ---

# Drop original categorical column and target
X = df_clean.drop(columns=['sex_4', TARGET_COLUMN])
feature_names = X.columns.tolist()

print(f"\n Final feature set: {len(feature_names)} features")

# --- 5. TRAIN-TEST SPLIT ---

print("\n[STEP 4] Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f" Training: {len(X_train)} samples")
print(f" Test: {len(X_test)} samples")

# Save processed data
X_train.to_csv('X_train.csv', index=False)
y_train.to_csv('y_train.csv', index=False)
X_test.to_csv('X_test.csv', index=False)
y_test.to_csv('y_test.csv', index=False)

with open('feature_names.txt', 'w') as f:
    f.write('\n'.join(feature_names))

print(" Saved processed data")

# --- 6. MODEL TRAINING ---

print("\n[STEP 5] Training XGBoost model...")
model = XGBClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=6,
    min_child_weight=1,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0.1,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42,
    scale_pos_weight=(~y_train.astype(bool)).sum()/y_train.sum()
)

model.fit(X_train, y_train)
print(" Model trained")

# --- 7. EVALUATION ---

print("\n[STEP 6] Evaluating model...")

# Cross-validation
cv = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc')
print(f" 5-Fold CV AUC: {cv_scores.mean():.4f} (± {cv_scores.std():.4f})")

# Test set
test_probas = model.predict_proba(X_test)[:, 1]
test_predictions = model.predict(X_test)
test_auc = roc_auc_score(y_test, test_probas)
print(f" Test Set AUC: {test_auc:.4f}")

print(f"\nClassification Report:")
print(classification_report(y_test, test_predictions, 
                          target_names=['Low Risk', 'High Risk']))

# --- 8. FEATURE IMPORTANCE ---

print("\n[STEP 7] Feature importance analysis...")
feature_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\nTop 20 Most Important Features:")
print(feature_importance.head(20).to_string(index=False))

feature_importance.to_csv('feature_importance.csv', index=False)

# --- 9. SAVE MODEL AND RESULTS ---

print("\n[STEP 8] Saving model and results...")
joblib.dump(model, 'trained_model.pkl')

with open('model_results.txt', 'w') as f:
    f.write("="*80 + "\n")
    f.write("NUTRITION ML MODEL - RESULTS\n")
    f.write("="*80 + "\n\n")
    f.write(f"Dataset: {FILE_NAME}\n")
    f.write(f"Total samples: {len(df_clean)}\n")
    f.write(f"Features: {len(feature_names)}\n\n")
    f.write(f"5-Fold CV AUC: {cv_scores.mean():.4f} (± {cv_scores.std():.4f})\n")
    f.write(f"Test Set AUC: {test_auc:.4f}\n\n")
    f.write("Classification Report:\n")
    f.write(classification_report(y_test, test_predictions, 
                                 target_names=['Low Risk', 'High Risk']))
    f.write("\n\nTop 20 Features:\n")
    f.write(feature_importance.head(20).to_string(index=False))

print(" Saved: trained_model.pkl")
print(" Saved: model_results.txt")
print(" Saved: feature_importance.csv")

print("\n" + "="*80)
print(" TRAINING COMPLETE!")
print("="*80)
print(f" Performance: CV AUC = {cv_scores.mean():.4f}, Test AUC = {test_auc:.4f}")
print("="*80)
