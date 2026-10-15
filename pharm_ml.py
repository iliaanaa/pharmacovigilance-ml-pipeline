import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import warnings

# Suppress runtime warnings for clean stdout
warnings.filterwarnings('ignore')

print("[INFO] Initializing Machine Learning pipeline and data preparation...")

# 1. Load cleaned clinical dataset
df = pd.read_csv("openfda_clean_data.csv")

# 2. Define Feature Matrix (X) and Target Vector (y)
X = df[['patient_age', 'patient_sex', 'drug_name']]
y = df['serious']

# 3. Perform One-Hot Encoding on categorical predictors
X_encoded = pd.get_dummies(X, columns=['patient_sex', 'drug_name'], drop_first=True)

# 4. Perform Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.3, random_state=42)

# 5. Apply SMOTE for class rebalancing strictly on the training set
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
print(f"[INFO] Resampled training dataset size post-SMOTE: {X_train_smote.shape[0]} samples.")

# 6. Hyperparameter Tuning via Cross-Validated GridSearchCV
print("\n[INFO] Executing GridSearchCV hyperparameter optimization. Processing...")
param_grid = {
    'n_estimators': [100, 200, 300],        # Number of decision trees
    'max_depth': [5, 10, 15, None],         # Maximum depth of each tree
    'min_samples_split': [2, 5, 10]         # Minimum samples required to split an internal node
}

rf_base = RandomForestClassifier(random_state=42)

# Optimize hyperparameters targeting the F1-score metric
grid_search = GridSearchCV(estimator=rf_base, param_grid=param_grid, cv=5, scoring='f1', n_jobs=-1)
grid_search.fit(X_train_smote, y_train_smote)

print(f"[SUCCESS] Optimal hyperparameter configuration identified: {grid_search.best_params_}")

# 7. Model evaluation and inference using the best estimator
best_rf_model = grid_search.best_estimator_
y_pred = best_rf_model.predict(X_test)

print("\n=== Model Evaluation Performance Metrics (Tuned Model) ===")
print(classification_report(y_test, y_pred, zero_division=0))

print("=== Confusion Matrix ===")
print(confusion_matrix(y_test, y_pred))
