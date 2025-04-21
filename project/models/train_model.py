import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import os
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt
import seaborn as sns

# Create directories if they don't exist
os.makedirs('project/models', exist_ok=True)

# Load data using scikit-learn's built-in dataset
print("Loading Wisconsin Breast Cancer Dataset...")
cancer = load_breast_cancer()
data = pd.DataFrame(cancer.data, columns=cancer.feature_names)
data['diagnosis'] = cancer.target

print(f"Dataset loaded with shape: {data.shape}")
print(f"Features: {cancer.feature_names}")
print(f"Target names: {cancer.target_names}")  # 0=malignant, 1=benign

# Extract features and target
X = data.drop('diagnosis', axis=1)
y = data['diagnosis']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize base models
rf = RandomForestClassifier(n_estimators=100, random_state=42)
gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
svc = SVC(probability=True, random_state=42)
lr = LogisticRegression(max_iter=1000, random_state=42)

# Train individual models
print("\nTraining individual models...")
models = {
    'Random Forest': rf,
    'Gradient Boosting': gb,
    'SVM': svc,
    'Logistic Regression': lr
}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {accuracy:.4f}")

# Create and train the voting classifier (ensemble)
print("\nTraining Soft Voting Ensemble...")
ensemble = VotingClassifier(
    estimators=[
        ('rf', rf),
        ('gb', gb),
        ('svc', svc),
        ('lr', lr)
    ],
    voting='soft'
)

ensemble.fit(X_train_scaled, y_train)
ensemble_pred = ensemble.predict(X_test_scaled)
ensemble_accuracy = accuracy_score(y_test, ensemble_pred)
print(f"Ensemble Accuracy: {ensemble_accuracy:.4f}")

# Calculate feature importances from Random Forest
feature_importances = pd.DataFrame({
    'feature': X.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 Feature Importances:")
print(feature_importances.head(10))

# Create an object containing all necessary components
model_package = {
    'ensemble': ensemble,
    'scaler': scaler,
    'feature_names': X.columns.tolist(),
    'features_to_use': feature_importances['feature'].head(10).tolist(),
    'feature_importances': dict(zip(feature_importances['feature'], feature_importances['importance'])),
    'accuracy': ensemble_accuracy
}

# Save the model package
joblib.dump(model_package, 'project/models/ensemble_model.pkl')
print("\nModel package saved as project/models/ensemble_model.pkl")
