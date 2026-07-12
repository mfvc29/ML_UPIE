"""
Entrena el modelo Gradient Boosting Classifier y exporta los resultados
para que app_streamlit.py los lea directamente sin necesidad de sklearn.
"""
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    accuracy_score, f1_score
)

print("Cargando datos...")
df = pd.read_csv("3.3. UPIE_dataset.csv")

leakage_cols = [
    'V022_EstadoMatricula', 'V001_StudentID', 'V002_MatriculaID',
    'V077_SegmentoRetencion', 'V078_TutoriasRecomendadas'
]
target_col = 'V075_DesercionBinario'

drop_cols = [col for col in leakage_cols if col in df.columns] + [target_col]
X = df.drop(columns=drop_cols)
y = df[target_col]

print("Separando sets de entrenamiento y prueba...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

num_features = X_train.select_dtypes(include=['int64', 'float64']).columns
cat_features = X_train.select_dtypes(include=['object', 'category', 'str']).columns

num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='Desconocido')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=True))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', num_transformer, num_features),
    ('cat', cat_transformer, cat_features)
])

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', GradientBoostingClassifier(
        n_estimators=150, learning_rate=0.1, max_depth=3, random_state=42
    ))
])

print("Entrenando Gradient Boosting Classifier...")
model.fit(X_train, y_train)

print("Calculando predicciones y metricas...")
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_val = auc(fpr, tpr)
f1_0 = f1_score(y_test, y_pred, pos_label=0)
f1_1 = f1_score(y_test, y_pred, pos_label=1)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)

# Feature importance
gb_clf = model.named_steps['classifier']
feat_names = model.named_steps['preprocessor'].get_feature_names_out()
importances = gb_clf.feature_importances_
top_idx = np.argsort(importances)[-15:]
top_features = [feat_names[i] for i in top_idx]
top_importances = [float(importances[i]) for i in top_idx]

results = {
    'accuracy': float(acc),
    'roc_auc': float(roc_val),
    'f1_0': float(f1_0),
    'f1_1': float(f1_1),
    'fpr': fpr.tolist(),
    'tpr': tpr.tolist(),
    'cm': cm.tolist(),
    'report': report,
    'n_test': int(len(y_test)),
    'top_features': top_features,
    'top_importances': top_importances,
}

print("Guardando resultados en model_results.joblib ...")
joblib.dump(results, "model_results.joblib", compress=3, protocol=4)

print(f"Accuracy: {acc*100:.1f}%")
print(f"ROC-AUC:  {roc_val:.4f}")
print(f"F1 Activos:    {f1_0:.2f}")
print(f"F1 Desertores: {f1_1:.2f}")
print("Listo!")
