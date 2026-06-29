"""
Train ML models for disease prediction using synthetic data
modeled after real UCI/Kaggle datasets distributions.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import joblib
import os

np.random.seed(42)
N = 2000

def generate_diabetes_data():
    """Based on Pima Indians Diabetes Dataset (UCI)"""
    age = np.random.randint(21, 81, N)
    bmi = np.random.normal(32, 8, N).clip(15, 60)
    glucose = np.random.normal(120, 32, N).clip(50, 250)
    blood_pressure = np.random.normal(70, 12, N).clip(40, 120)
    skin_thickness = np.random.normal(23, 10, N).clip(0, 60)
    insulin = np.random.normal(80, 115, N).clip(0, 500)
    pregnancies = np.random.randint(0, 15, N)
    dpf = np.random.exponential(0.5, N).clip(0.08, 2.5)

    risk_score = (
        0.3 * (glucose / 200) +
        0.25 * (bmi / 50) +
        0.2 * (age / 80) +
        0.15 * dpf +
        0.1 * (pregnancies / 14)
    )
    noise = np.random.normal(0, 0.08, N)
    outcome = (risk_score + noise > 0.38).astype(int)

    return pd.DataFrame({
        'Pregnancies': pregnancies,
        'Glucose': glucose.astype(int),
        'BloodPressure': blood_pressure.astype(int),
        'SkinThickness': skin_thickness.astype(int),
        'Insulin': insulin.astype(int),
        'BMI': bmi.round(1),
        'DiabetesPedigreeFunction': dpf.round(3),
        'Age': age,
        'Outcome': outcome
    })

def generate_heart_disease_data():
    """Based on Cleveland Heart Disease Dataset (UCI)"""
    age = np.random.randint(29, 78, N)
    sex = np.random.randint(0, 2, N)
    cp = np.random.randint(0, 4, N)
    trestbps = np.random.normal(131, 17, N).clip(90, 200).astype(int)
    chol = np.random.normal(246, 51, N).clip(120, 400).astype(int)
    fbs = (np.random.normal(120, 30, N) > 120).astype(int)
    restecg = np.random.randint(0, 3, N)
    thalach = np.random.normal(149, 22, N).clip(70, 200).astype(int)
    exang = np.random.randint(0, 2, N)
    oldpeak = np.random.exponential(1.0, N).clip(0, 5).round(1)
    slope = np.random.randint(0, 3, N)
    ca = np.random.randint(0, 4, N)
    thal = np.random.choice([0, 1, 2, 3], N)

    risk_score = (
        0.25 * (age / 77) +
        0.15 * (chol / 400) +
        0.2 * (1 - thalach / 200) +
        0.15 * (cp / 3) +
        0.15 * (ca / 3) +
        0.1 * exang
    )
    noise = np.random.normal(0, 0.08, N)
    target = (risk_score + noise > 0.42).astype(int)

    return pd.DataFrame({
        'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps,
        'chol': chol, 'fbs': fbs, 'restecg': restecg, 'thalach': thalach,
        'exang': exang, 'oldpeak': oldpeak, 'slope': slope, 'ca': ca,
        'thal': thal, 'target': target
    })

def generate_liver_disease_data():
    """Based on Indian Liver Patient Dataset (UCI)"""
    age = np.random.randint(4, 90, N)
    gender = np.random.randint(0, 2, N)
    total_bilirubin = np.random.exponential(3, N).clip(0.4, 50).round(1)
    direct_bilirubin = (total_bilirubin * np.random.uniform(0.2, 0.6, N)).round(1)
    alkaline_phosphotase = np.random.lognormal(5.5, 0.8, N).clip(60, 2000).astype(int)
    alamine_aminotransferase = np.random.lognormal(4.5, 1.2, N).clip(10, 2000).astype(int)
    aspartate_aminotransferase = np.random.lognormal(4.5, 1.2, N).clip(10, 3000).astype(int)
    total_proteins = np.random.normal(6.5, 1.0, N).clip(2, 9).round(1)
    albumin = np.random.normal(3.1, 0.8, N).clip(0.9, 5.5).round(1)
    albumin_globulin_ratio = np.random.normal(1.0, 0.4, N).clip(0.3, 2.8).round(2)

    risk_score = (
        0.3 * (total_bilirubin / 50) +
        0.2 * np.log1p(alamine_aminotransferase) / np.log1p(2000) +
        0.2 * (1 - albumin / 5.5) +
        0.15 * (age / 90) +
        0.15 * np.log1p(alkaline_phosphotase) / np.log1p(2000)
    )
    noise = np.random.normal(0, 0.08, N)
    dataset = (risk_score + noise > 0.38).astype(int)

    return pd.DataFrame({
        'Age': age, 'Gender': gender, 'Total_Bilirubin': total_bilirubin,
        'Direct_Bilirubin': direct_bilirubin, 'Alkaline_Phosphotase': alkaline_phosphotase,
        'Alamine_Aminotransferase': alamine_aminotransferase,
        'Aspartate_Aminotransferase': aspartate_aminotransferase,
        'Total_Protiens': total_proteins, 'Albumin': albumin,
        'Albumin_and_Globulin_Ratio': albumin_globulin_ratio, 'Dataset': dataset
    })

def generate_kidney_disease_data():
    """Based on Chronic Kidney Disease Dataset (UCI)"""
    age = np.random.randint(2, 90, N)
    blood_pressure = np.random.normal(76, 14, N).clip(50, 120).astype(int)
    specific_gravity = np.random.choice([1.005, 1.010, 1.015, 1.020, 1.025], N)
    albumin = np.random.randint(0, 6, N)
    sugar = np.random.randint(0, 6, N)
    blood_glucose = np.random.normal(148, 79, N).clip(70, 500).astype(int)
    blood_urea = np.random.normal(57, 50, N).clip(1.5, 300).round(1)
    serum_creatinine = np.random.exponential(3, N).clip(0.4, 50).round(1)
    sodium = np.random.normal(137, 10, N).clip(100, 170).round(1)
    potassium = np.random.normal(4.6, 1.5, N).clip(2.5, 10).round(1)
    haemoglobin = np.random.normal(12.5, 2.5, N).clip(3.1, 17.8).round(1)
    hypertension = np.random.randint(0, 2, N)
    diabetes_mellitus = np.random.randint(0, 2, N)
    coronary_artery = np.random.randint(0, 2, N)
    anemia = np.random.randint(0, 2, N)

    risk_score = (
        0.25 * (serum_creatinine / 50) +
        0.2 * (albumin / 5) +
        0.2 * (blood_urea / 300) +
        0.15 * (1 - haemoglobin / 17.8) +
        0.1 * hypertension +
        0.1 * diabetes_mellitus
    )
    noise = np.random.normal(0, 0.08, N)
    classification = (risk_score + noise > 0.22).astype(int)

    return pd.DataFrame({
        'age': age, 'bp': blood_pressure, 'sg': specific_gravity, 'al': albumin,
        'su': sugar, 'bgr': blood_glucose, 'bu': blood_urea, 'sc': serum_creatinine,
        'sod': sodium, 'pot': potassium, 'hemo': haemoglobin,
        'htn': hypertension, 'dm': diabetes_mellitus, 'cad': coronary_artery,
        'ane': anemia, 'classification': classification
    })

def train_and_save_model(X, y, name, model_type='rf'):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if model_type == 'rf':
        model = RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42, class_weight='balanced')
    else:
        model = GradientBoostingClassifier(n_estimators=150, max_depth=4, random_state=42)

    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    proba = model.predict_proba(X_test_scaled)[:, 1]
    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, proba)
    print(f"{name}: Accuracy={acc:.3f}, AUC={auc:.3f}")

    os.makedirs('models', exist_ok=True)
    joblib.dump(model, f'models/{name}_model.pkl')
    joblib.dump(scaler, f'models/{name}_scaler.pkl')
    joblib.dump(list(X.columns), f'models/{name}_features.pkl')
    return acc, auc

if __name__ == '__main__':
    print("Generating data and training models...")

    df_d = generate_diabetes_data()
    X_d = df_d.drop('Outcome', axis=1)
    y_d = df_d['Outcome']
    train_and_save_model(X_d, y_d, 'diabetes', 'rf')

    df_h = generate_heart_disease_data()
    X_h = df_h.drop('target', axis=1)
    y_h = df_h['target']
    train_and_save_model(X_h, y_h, 'heart', 'gb')

    df_l = generate_liver_disease_data()
    X_l = df_l.drop('Dataset', axis=1)
    y_l = df_l['Dataset']
    train_and_save_model(X_l, y_l, 'liver', 'rf')

    df_k = generate_kidney_disease_data()
    X_k = df_k.drop('classification', axis=1)
    y_k = df_k['classification']
    train_and_save_model(X_k, y_k, 'kidney', 'gb')

    print("\nAll models trained and saved!")
