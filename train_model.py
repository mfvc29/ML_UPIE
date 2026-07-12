import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import GradientBoostingClassifier
import pickle

print("Cargando datos...")
df = pd.read_csv("3.3. UPIE_dataset.csv")

leakage_cols = ['V022_EstadoMatricula', 'V001_StudentID', 'V002_MatriculaID', 'V077_SegmentoRetencion', 'V078_TutoriasRecomendadas']
target_col = 'V075_DesercionBinario'

drop_cols = [col for col in leakage_cols if col in df.columns] + [target_col]
X = df.drop(columns=drop_cols)
y = df[target_col]

print("Separando sets de entrenamiento y prueba...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

num_features = X.select_dtypes(include=['int64', 'float64']).columns
cat_features = X.select_dtypes(include=['object', 'category']).columns

num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='Desconocido')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', num_transformer, num_features),
    ('cat', cat_transformer, cat_features)
])

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, random_state=42))
])

print("Entrenando Gradient Boosting Classifier...")
model.fit(X_train, y_train)

print("Guardando modelo en model_gb.joblib ...")
# Utilizar protocolo 4 para asegurar máxima compatibilidad con versiones antiguas de Python en la nube
joblib.dump(model, "model_gb.joblib", protocol=4)
print("Modelo guardado exitosamente.")
