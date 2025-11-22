

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
import numpy as np
import os

app = Flask(__name__)
CORS(app)


BMI_THRESHOLDS = {
    'underweight': 18.5,
    'normal': 25,
    'overweight': 30,
    'obese_class_1': 35,
    'obese_class_2': 40
}


FBS_THRESHOLDS = {
    'hypoglycemia': 70,
    'normal': 100,
    'prediabetes': 126
}


HBA1C_THRESHOLDS = {
    'normal': 5.7,
    'prediabetes': 6.5
}


HDL_THRESHOLDS = {
    'male_low': 40,
    'male_protective': 60,
    'female_low': 50,
    'female_protective': 60
}


LDL_THRESHOLDS = {
    'optimal': 100,
    'near_optimal': 130,
    'borderline': 160,
    'high': 190
}


TG_THRESHOLDS = {
    'normal': 150,
    'borderline': 200,
    'high': 500
}

# Load model
try:
    # Try new improved model first
    if os.path.exists('models/trained_model_1.pkl'):
        model = joblib.load('models/trained_model_1.pkl')
        print("✅ Model loaded successfully! (trained_model_1.pkl)")
    elif os.path.exists('trained_model_1.pkl'):
        model = joblib.load('trained_model_1.pkl')
        print("✅ Model loaded successfully! (trained_model_1.pkl from current dir)")
    else:
        model = None
        print("⚠️  Model not found. Using rule-based predictions.")
    
    # Try to load feature names (handle both .txt and .json)
    expected_features = None
    if os.path.exists('models/feature_names.txt'):
        with open('models/feature_names.txt', 'r') as f:
            expected_features = [line.strip() for line in f.readlines()]
    elif os.path.exists('feature_names.txt'):
        with open('feature_names.txt', 'r') as f:
            expected_features = [line.strip() for line in f.readlines()]
    
except Exception as e:
    print(f"⚠️  Error loading model: {e}")
    model = None
    expected_features = None

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "thresholds": {
            "BMI": BMI_THRESHOLDS,
            "FBS": FBS_THRESHOLDS,
            "HDL": HDL_THRESHOLDS,
            "LDL": LDL_THRESHOLDS,
            "TG": TG_THRESHOLDS
        }
    })

def categorize_bmi(bmi):
    
    if bmi < BMI_THRESHOLDS['underweight']:
        return "Underweight", "🔵"
    elif bmi < BMI_THRESHOLDS['normal']:
        return "Normal", "🟢"
    elif bmi < BMI_THRESHOLDS['overweight']:
        return "Overweight", "🟡"
    elif bmi < BMI_THRESHOLDS['obese_class_1']:
        return "Obese Class I", "🟠"
    elif bmi < BMI_THRESHOLDS['obese_class_2']:
        return "Obese Class II", "🔴"
    else:
        return "Severe Obesity", "⚫"

def categorize_fbs(fbs):
    
    if fbs < FBS_THRESHOLDS['hypoglycemia']:
        return "Hypoglycemia (Low)", "🔵", 0.15
    elif fbs < FBS_THRESHOLDS['normal']:
        return "Normal", "🟢", 0
    elif fbs < FBS_THRESHOLDS['prediabetes']:
        return "Prediabetes", "🟡", 0.30
    else:
        return "Diabetes", "🔴", 0.45

def categorize_hdl(hdl, sex):
    
    if sex.lower() == 'male':
        if hdl < HDL_THRESHOLDS['male_low']:
            return "Low (Risk)", "🔴", 0.20
        elif hdl >= HDL_THRESHOLDS['male_protective']:
            return "Protective", "🟢", -0.15
        else:
            return "Normal", "🟡", 0
    else:  # female
        if hdl < HDL_THRESHOLDS['female_low']:
            return "Low (Risk)", "🔴", 0.20
        elif hdl >= HDL_THRESHOLDS['female_protective']:
            return "Protective", "🟢", -0.15
        else:
            return "Normal", "🟡", 0

def categorize_ldl(ldl):
    
    if ldl < LDL_THRESHOLDS['optimal']:
        return "Optimal", "🟢", 0
    elif ldl < LDL_THRESHOLDS['near_optimal']:
        return "Near Optimal", "🟡", 0.05
    elif ldl < LDL_THRESHOLDS['borderline']:
        return "Borderline High", "🟠", 0.12
    elif ldl < LDL_THRESHOLDS['high']:
        return "High", "🔴", 0.18
    else:
        return "Very High", "⚫", 0.25

def categorize_tg(tg):
    
    if tg < TG_THRESHOLDS['normal']:
        return "Normal", "🟢", 0
    elif tg < TG_THRESHOLDS['borderline']:
        return "Borderline High", "🟡", 0.15
    elif tg < TG_THRESHOLDS['high']:
        return "High", "🟠", 0.25
    else:
        return "Very High", "🔴", 0.35

def calculate_metabolic_syndrome_score(fbs, tg, hdl, bmi, sex):
    
    score = 0
    
    # FBS >= 100
    if fbs >= FBS_THRESHOLDS['normal']:
        score += 1
    
    # TG >= 150
    if tg >= TG_THRESHOLDS['normal']:
        score += 1
    
    # HDL low (sex-specific)
    if sex.lower() == 'male':
        if hdl < HDL_THRESHOLDS['male_low']:
            score += 1
    else:
        if hdl < HDL_THRESHOLDS['female_low']:
            score += 1
    
    # BMI >= 30 (using obesity as proxy for waist)
    if bmi >= BMI_THRESHOLDS['overweight']:
        score += 1
    
    return score

@app.route('/predict', methods=['POST'])
def predict_risk():
    
    try:
        data = request.json
        
        # Calculate BMI
        height_m = float(data['height']) / 100
        bmi = float(data['weight']) / (height_m ** 2)
        
        # Handle unknown values
        fbs = 95 if str(data.get('fbs', '')).lower() == 'unknown' else float(data['fbs'])
        hdl = 50 if str(data.get('hdl', '')).lower() == 'unknown' else float(data['hdl'])
        ldl = 130 if str(data.get('ldl', '')).lower() == 'unknown' else float(data['ldl'])
        tg = 150 if str(data.get('tg', '')).lower() == 'unknown' else float(data['tg'])
        waist = data.get('waist', 90)
        physical_activity = data.get('physical_activity', 465.78)
        
        sex = data['sex']
        age = float(data['age'])
        
        # Categorize all biomarkers
        bmi_cat, bmi_icon = categorize_bmi(bmi)
        fbs_cat, fbs_icon, fbs_risk = categorize_fbs(fbs)
        hdl_cat, hdl_icon, hdl_risk = categorize_hdl(hdl, sex)
        ldl_cat, ldl_icon, ldl_risk = categorize_ldl(ldl)
        tg_cat, tg_icon, tg_risk = categorize_tg(tg)
        
        # Calculate metabolic syndrome score
        metabolic_score = calculate_metabolic_syndrome_score(fbs, tg, hdl, bmi, sex)
        metabolic_risk = metabolic_score >= 3
        
        # Prepare features for ML model
        sex_is_male = 1 if sex.lower() == 'male' else 0
        sex_is_female = 1 - sex_is_male
        
        # All categorical features (matching training)
        features = pd.DataFrame({
            'ageyr14_4': [age],
            'BMI_4': [bmi],
            'fbs1_4': [fbs],
            'tg1_4': [tg],
            'hdl1_4': [hdl],
            'ldl1_4': [ldl],
            'kcal_4': [2000.0],
            'weight_4': [float(data['weight'])],
            'height_4': [float(data['height'])],
            'waist_4': [waist],
            'physical_activity': [physical_activity],
            'sex_is_male': [sex_is_male],
            'sex_is_female': [sex_is_female],
            
            # BMI categories
            'bmi_underweight': [int(bmi < BMI_THRESHOLDS['underweight'])],
            'bmi_normal': [int((bmi >= BMI_THRESHOLDS['underweight']) & (bmi < BMI_THRESHOLDS['normal']))],
            'bmi_overweight': [int((bmi >= BMI_THRESHOLDS['normal']) & (bmi < BMI_THRESHOLDS['overweight']))],
            'bmi_obese_1': [int((bmi >= BMI_THRESHOLDS['overweight']) & (bmi < BMI_THRESHOLDS['obese_class_1']))],
            'bmi_obese_2': [int((bmi >= BMI_THRESHOLDS['obese_class_1']) & (bmi < BMI_THRESHOLDS['obese_class_2']))],
            'bmi_severe_obesity': [int(bmi >= BMI_THRESHOLDS['obese_class_2'])],
            
            # FBS categories
            'fbs_hypoglycemia': [int(fbs < FBS_THRESHOLDS['hypoglycemia'])],
            'fbs_normal': [int((fbs >= FBS_THRESHOLDS['hypoglycemia']) & (fbs < FBS_THRESHOLDS['normal']))],
            'fbs_prediabetes': [int((fbs >= FBS_THRESHOLDS['normal']) & (fbs < FBS_THRESHOLDS['prediabetes']))],
            'fbs_diabetes': [int(fbs >= FBS_THRESHOLDS['prediabetes'])],
            
            # HDL categories
            'hdl_low_male': [int((sex_is_male) & (hdl < HDL_THRESHOLDS['male_low']))],
            'hdl_protective_male': [int((sex_is_male) & (hdl >= HDL_THRESHOLDS['male_protective']))],
            'hdl_low_female': [int((sex_is_female) & (hdl < HDL_THRESHOLDS['female_low']))],
            'hdl_protective_female': [int((sex_is_female) & (hdl >= HDL_THRESHOLDS['female_protective']))],
            'hdl_low': [int(((sex_is_male) & (hdl < HDL_THRESHOLDS['male_low'])) | ((sex_is_female) & (hdl < HDL_THRESHOLDS['female_low'])))],
            'hdl_protective': [int(((sex_is_male) & (hdl >= HDL_THRESHOLDS['male_protective'])) | ((sex_is_female) & (hdl >= HDL_THRESHOLDS['female_protective'])))],
            
            # LDL categories
            'ldl_optimal': [int(ldl < LDL_THRESHOLDS['optimal'])],
            'ldl_near_optimal': [int((ldl >= LDL_THRESHOLDS['optimal']) & (ldl < LDL_THRESHOLDS['near_optimal']))],
            'ldl_borderline': [int((ldl >= LDL_THRESHOLDS['near_optimal']) & (ldl < LDL_THRESHOLDS['borderline']))],
            'ldl_high': [int((ldl >= LDL_THRESHOLDS['borderline']) & (ldl < LDL_THRESHOLDS['high']))],
            'ldl_very_high': [int(ldl >= LDL_THRESHOLDS['high'])],
            
            # TG categories
            'tg_normal': [int(tg < TG_THRESHOLDS['normal'])],
            'tg_borderline': [int((tg >= TG_THRESHOLDS['normal']) & (tg < TG_THRESHOLDS['borderline']))],
            'tg_high': [int((tg >= TG_THRESHOLDS['borderline']) & (tg < TG_THRESHOLDS['high']))],
            'tg_very_high': [int(tg >= TG_THRESHOLDS['high'])],
            
            # Metabolic syndrome
            'metabolic_syndrome_score': [metabolic_score],
            'metabolic_syndrome_risk': [int(metabolic_risk)]
        })
        
        # Predict with ML model
        if model is not None:
            try:
                risk_probability = model.predict_proba(features)[0, 1]
            except:
                # Fallback to rule-based
                risk_probability = fbs_risk + hdl_risk + ldl_risk + tg_risk + (0.35 if metabolic_risk else 0)
                risk_probability = min(risk_probability, 1.0)
        else:
            risk_probability = fbs_risk + hdl_risk + ldl_risk + tg_risk + (0.35 if metabolic_risk else 0)
            risk_probability = min(risk_probability, 1.0)
        
        # Response
        response = {
            "risk_score": float(risk_probability * 100),
            "bmi": round(bmi, 1),
            "metabolic_syndrome_score": metabolic_score,
            "metabolic_syndrome_risk": metabolic_risk,
            
            "clinical_categories": {
                "bmi": {"category": bmi_cat, "icon": bmi_icon, "value": round(bmi, 1)},
                "fbs": {"category": fbs_cat, "icon": fbs_icon, "value": fbs},
                "hdl": {"category": hdl_cat, "icon": hdl_icon, "value": hdl, "sex": sex},
                "ldl": {"category": ldl_cat, "icon": ldl_icon, "value": ldl},
                "tg": {"category": tg_cat, "icon": tg_icon, "value": tg}
            },
            
            "feature_importance": generate_feature_importance(
                bmi, bmi_cat, fbs, fbs_cat, hdl, hdl_cat, ldl, ldl_cat, tg, tg_cat, age, metabolic_score
            ),
            
            "counterfactuals": generate_counterfactuals(
                bmi, fbs, hdl, ldl, tg, sex, risk_probability, metabolic_score
            ),
            
            "lab_values": {"fbs": fbs, "hdl": hdl, "ldl": ldl, "tg": tg}
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 400

def generate_feature_importance(bmi, bmi_cat, fbs, fbs_cat, hdl, hdl_cat, ldl, ldl_cat, tg, tg_cat, age, metabolic_score):
    """Generate feature importance explanation"""
    features = []
    
    # Metabolic Syndrome Score (Highest Priority!)
    if metabolic_score >= 3:
        features.append({
            "name": "⚠️ METABOLIC SYNDROME RISK",
            "value": f"{metabolic_score}/4 risk factors present",
            "contribution": 0.40,
            "direction": "increases",
            "severity": "critical"
        })
    elif metabolic_score > 0:
        features.append({
            "name": "Metabolic Syndrome Score",
            "value": f"{metabolic_score}/4 risk factors",
            "contribution": metabolic_score * 0.10,
            "direction": "increases" if metabolic_score >= 2 else "normal",
            "severity": "moderate" if metabolic_score >= 2 else "low"
        })
    
    # FBS
    if "Diabetes" in fbs_cat:
        contrib, severity = 0.35, "critical"
    elif "Prediabetes" in fbs_cat:
        contrib, severity = 0.25, "high"
    elif "Hypoglycemia" in fbs_cat:
        contrib, severity = 0.10, "moderate"
    else:
        contrib, severity = 0, "normal"
    
    features.append({
        "name": "Fasting Blood Sugar",
        "value": f"{fbs:.0f} mg/dL ({fbs_cat})",
        "contribution": contrib,
        "direction": "increases" if contrib > 0 else "normal",
        "severity": severity
    })
    
    # BMI
    if "Severe" in bmi_cat or "Class II" in bmi_cat:
        contrib, severity = 0.30, "critical"
    elif "Class I" in bmi_cat:
        contrib, severity = 0.22, "high"
    elif "Overweight" in bmi_cat:
        contrib, severity = 0.15, "moderate"
    else:
        contrib, severity = 0, "normal"
    
    features.append({
        "name": "BMI",
        "value": f"{bmi:.1f} ({bmi_cat})",
        "contribution": contrib,
        "direction": "increases" if contrib > 0 else "normal",
        "severity": severity
    })
    
    # TG
    if "Very High" in tg_cat:
        contrib, severity = 0.28, "critical"
    elif "High" in tg_cat:
        contrib, severity = 0.20, "high"
    elif "Borderline" in tg_cat:
        contrib, severity = 0.12, "moderate"
    else:
        contrib, severity = 0, "normal"
    
    features.append({
        "name": "Triglycerides",
        "value": f"{tg:.0f} mg/dL ({tg_cat})",
        "contribution": contrib,
        "direction": "increases" if contrib > 0 else "normal",
        "severity": severity
    })
    
    # HDL
    if "Low" in hdl_cat:
        contrib, severity, direction = 0.18, "high", "increases"
    elif "Protective" in hdl_cat:
        contrib, severity, direction = 0.15, "beneficial", "protective"
    else:
        contrib, severity, direction = 0, "normal", "normal"
    
    features.append({
        "name": "HDL Cholesterol",
        "value": f"{hdl:.0f} mg/dL ({hdl_cat})",
        "contribution": abs(contrib),
        "direction": direction,
        "severity": severity
    })
    
    # LDL
    if "Very High" in ldl_cat:
        contrib, severity = 0.20, "critical"
    elif "High" in ldl_cat:
        contrib, severity = 0.15, "high"
    elif "Borderline" in ldl_cat:
        contrib, severity = 0.10, "moderate"
    else:
        contrib, severity = 0, "normal"
    
    features.append({
        "name": "LDL Cholesterol",
        "value": f"{ldl:.0f} mg/dL ({ldl_cat})",
        "contribution": contrib,
        "direction": "increases" if contrib > 0 else "normal",
        "severity": severity
    })
    
    # Age
    if age >= 45:
        features.append({
            "name": "Age",
            "value": f"{age:.0f} years",
            "contribution": 0.12,
            "direction": "increases",
            "severity": "moderate"
        })
    
    # Sort by contribution
    features.sort(key=lambda x: x['contribution'], reverse=True)
    return features

def generate_counterfactuals(bmi, fbs, hdl, ldl, tg, sex, current_risk, metabolic_score):
    """Generate personalized recommendations"""
    counterfactuals = []
    
    # Metabolic Syndrome - Highest Priority
    if metabolic_score >= 3:
        counterfactuals.append({
            "priority": "🔴 CRITICAL",
            "change": f"Reduce Metabolic Syndrome Score from {metabolic_score} to <3",
            "method": "Address multiple risk factors simultaneously: lower blood sugar, improve cholesterol, reduce weight",
            "impact": "Dramatically reduces diabetes and cardiovascular disease risk",
            "timeframe": "3-6 months with intensive lifestyle intervention"
        })
    
    # FBS
    if fbs >= FBS_THRESHOLDS['prediabetes']:
        counterfactuals.append({
            "priority": "🔴 HIGH",
            "change": f"Lower FBS from {fbs:.0f} to <{FBS_THRESHOLDS['normal']} mg/dL",
            "method": "Eliminate added sugars, eat low-GI foods, increase fiber to 35g/day, exercise after meals",
            "impact": f"Risk ↓ ~30% (from {current_risk*100:.0f}% to {(current_risk-0.30)*100:.0f}%)",
            "timeframe": "2-3 months"
        })
    elif fbs >= FBS_THRESHOLDS['normal']:
        counterfactuals.append({
            "priority": "🟡 MODERATE",
            "change": f"Lower FBS from {fbs:.0f} to <{FBS_THRESHOLDS['normal']} mg/dL",
            "method": "Reduce refined carbs, choose whole grains, increase vegetables and lean protein",
            "impact": f"Risk ↓ ~20% (from {current_risk*100:.0f}% to {(current_risk-0.20)*100:.0f}%)",
            "timeframe": "1-2 months"
        })
    
    # BMI
    if bmi >= BMI_THRESHOLDS['obese_class_2']:
        weight_loss = (bmi - 29.9) * 10
        counterfactuals.append({
            "priority": "🔴 HIGH",
            "change": f"Reduce BMI from {bmi:.1f} to <{BMI_THRESHOLDS['overweight']} (lose ~{weight_loss:.0f}kg)",
            "method": "600-800 kcal deficit/day + 200min exercise/week + medical supervision",
            "impact": f"Risk ↓ ~25% (from {current_risk*100:.0f}% to {(current_risk-0.25)*100:.0f}%)",
            "timeframe": "6-12 months"
        })
    elif bmi >= BMI_THRESHOLDS['overweight']:
        weight_loss = (bmi - 24.9) * 8
        counterfactuals.append({
            "priority": "🟡 MODERATE",
            "change": f"Reduce BMI from {bmi:.1f} to <{BMI_THRESHOLDS['overweight']} (lose ~{weight_loss:.0f}kg)",
            "method": "500 kcal deficit/day + 150min moderate exercise/week",
            "impact": f"Risk ↓ ~18% (from {current_risk*100:.0f}% to {(current_risk-0.18)*100:.0f}%)",
            "timeframe": "4-6 months"
        })
    
    # Triglycerides
    if tg >= TG_THRESHOLDS['borderline']:
        counterfactuals.append({
            "priority": "🟡 MODERATE",
            "change": f"Lower Triglycerides from {tg:.0f} to <{TG_THRESHOLDS['normal']} mg/dL",
            "method": "Reduce simple carbs & alcohol, eat fatty fish 3x/week, add omega-3 supplements",
            "impact": f"Risk ↓ ~15% (from {current_risk*100:.0f}% to {(current_risk-0.15)*100:.0f}%)",
            "timeframe": "2-3 months"
        })
    
    # HDL
    hdl_threshold = HDL_THRESHOLDS['female_low'] if sex.lower() == 'female' else HDL_THRESHOLDS['male_low']
    if hdl < hdl_threshold:
        counterfactuals.append({
            "priority": "🟡 MODERATE",
            "change": f"Increase HDL from {hdl:.0f} to ≥{hdl_threshold} mg/dL",
            "method": "30min aerobic exercise 5x/week, add healthy fats (nuts, avocado, olive oil), quit smoking",
            "impact": f"Risk ↓ ~15% (from {current_risk*100:.0f}% to {(current_risk-0.15)*100:.0f}%)",
            "timeframe": "3-4 months"
        })
    
    # LDL
    if ldl >= LDL_THRESHOLDS['borderline']:
        counterfactuals.append({
            "priority": "🟡 MODERATE",
            "change": f"Lower LDL from {ldl:.0f} to <{LDL_THRESHOLDS['near_optimal']} mg/dL",
            "method": "Reduce saturated fat, increase soluble fiber (oats, beans), add plant sterols",
            "impact": f"Risk ↓ ~12% (from {current_risk*100:.0f}% to {(current_risk-0.12)*100:.0f}%)",
            "timeframe": "2-3 months"
        })
    
    # Sort by priority
    priority_order = {"🔴 CRITICAL": 0, "🔴 HIGH": 1, "🟡 MODERATE": 2, "🟢 LOW": 3}
    counterfactuals.sort(key=lambda x: priority_order.get(x['priority'], 999))
    
    return counterfactuals

@app.route('/calculate-calories', methods=['POST'])
def calculate_calories():
    """Calculate calorie needs"""
    try:
        data = request.json
        age = float(data['age'])
        weight = float(data['weight'])
        height = float(data['height'])
        sex = data['sex']
        activity_level = data['activity_level']
        goal = data['goal']
        
        # Mifflin-St Jeor
        if sex.lower() == 'male':
            bmr = 10 * weight + 6.25 * height - 5 * age + 5
        else:
            bmr = 10 * weight + 6.25 * height - 5 * age - 161
        
        activity_multipliers = {
            'Sedentary': 1.2,
            'Lightly Active': 1.375,
            'Moderately Active': 1.55,
            'Very Active': 1.725
        }
        
        tdee = bmr * activity_multipliers.get(activity_level, 1.2)
        
        if 'lose' in goal.lower():
            tdee -= 500
        elif 'gain' in goal.lower():
            tdee += 500
        
        protein_percent = 0.30 if 'lose' in goal.lower() else 0.25
        protein_g = round((tdee * protein_percent) / 4)
        carbs_g = round((tdee * 0.45) / 4)
        fat_g = round((tdee * (1 - protein_percent - 0.45)) / 9)
        
        return jsonify({
            "calories": round(tdee),
            "macros": {"protein": protein_g, "carbs": carbs_g, "fat": fat_g}
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/thresholds', methods=['GET'])
def get_thresholds():
    """Get all clinical thresholds"""
    return jsonify({
        "BMI": {
            "Underweight": f"< {BMI_THRESHOLDS['underweight']}",
            "Normal": f"{BMI_THRESHOLDS['underweight']}-{BMI_THRESHOLDS['normal']-0.1}",
            "Overweight": f"{BMI_THRESHOLDS['normal']}-{BMI_THRESHOLDS['overweight']-0.1}",
            "Obese Class I": f"{BMI_THRESHOLDS['overweight']}-{BMI_THRESHOLDS['obese_class_1']-0.1}",
            "Obese Class II": f"{BMI_THRESHOLDS['obese_class_1']}-{BMI_THRESHOLDS['obese_class_2']-0.1}",
            "Severe Obesity": f"≥ {BMI_THRESHOLDS['obese_class_2']}"
        },
        "FBS (mg/dL)": {
            "Hypoglycemia": f"< {FBS_THRESHOLDS['hypoglycemia']}",
            "Normal": f"{FBS_THRESHOLDS['hypoglycemia']}-{FBS_THRESHOLDS['normal']-1}",
            "Prediabetes": f"{FBS_THRESHOLDS['normal']}-{FBS_THRESHOLDS['prediabetes']-1}",
            "Diabetes": f"≥ {FBS_THRESHOLDS['prediabetes']}"
        },
        "HDL (mg/dL)": {
            "Male": {
                "Low (Risk)": f"< {HDL_THRESHOLDS['male_low']}",
                "Normal": f"{HDL_THRESHOLDS['male_low']}-{HDL_THRESHOLDS['male_protective']-1}",
                "Protective": f"≥ {HDL_THRESHOLDS['male_protective']}"
            },
            "Female": {
                "Low (Risk)": f"< {HDL_THRESHOLDS['female_low']}",
                "Normal": f"{HDL_THRESHOLDS['female_low']}-{HDL_THRESHOLDS['female_protective']-1}",
                "Protective": f"≥ {HDL_THRESHOLDS['female_protective']}"
            }
        },
        "LDL (mg/dL)": {
            "Optimal": f"< {LDL_THRESHOLDS['optimal']}",
            "Near Optimal": f"{LDL_THRESHOLDS['optimal']}-{LDL_THRESHOLDS['near_optimal']-1}",
            "Borderline High": f"{LDL_THRESHOLDS['near_optimal']}-{LDL_THRESHOLDS['borderline']-1}",
            "High": f"{LDL_THRESHOLDS['borderline']}-{LDL_THRESHOLDS['high']-1}",
            "Very High": f"≥ {LDL_THRESHOLDS['high']}"
        },
        "Triglycerides (mg/dL)": {
            "Normal": f"< {TG_THRESHOLDS['normal']}",
            "Borderline High": f"{TG_THRESHOLDS['normal']}-{TG_THRESHOLDS['borderline']-1}",
            "High": f"{TG_THRESHOLDS['borderline']}-{TG_THRESHOLDS['high']-1}",
            "Very High": f"≥ {TG_THRESHOLDS['high']}"
        }
    })

if __name__ == '__main__':
    print("🚀 ML API Server (Exact Thresholds)")
    print("="*60)
    app.run(host='0.0.0.0', port=5000, debug=True)
