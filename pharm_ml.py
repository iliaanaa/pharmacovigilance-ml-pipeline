import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import warnings

# Απόκρυψη προειδοποιήσεων
warnings.filterwarnings('ignore')

print("Ξεκινάει η προετοιμασία δεδομένων...")

# 1. Φόρτωση δεδομένων
df = pd.read_csv("openfda_clean_data.csv")

# 2. Επιλογή Χαρακτηριστικών και Target
X = df[['patient_age', 'patient_sex', 'drug_name']]
y = df['serious']

# 3. One-Hot Encoding
X_encoded = pd.get_dummies(X, columns=['patient_sex', 'drug_name'], drop_first=True)

# 4. Διαχωρισμός Train/Test
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.3, random_state=42)

# 5. Εφαρμογή SMOTE
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
print(f"Εκπαίδευση σε {X_train_smote.shape[0]} εξισορροπημένα δείγματα (SMOTE).")

# 6. Hyperparameter Tuning με GridSearchCV
print("\nΕκκίνηση GridSearchCV για εύρεση βέλτιστων υπερπαραμέτρων. Παρακαλώ περιμένετε...")
param_grid = {
    'n_estimators': [100, 200, 300],        # Αριθμός δέντρων στο δάσος
    'max_depth': [5, 10, 15, None],         # Μέγιστο βάθος κάθε δέντρου
    'min_samples_split': [2, 5, 10]         # Ελάχιστα δείγματα για να σπάσει ένας κόμβος
}

rf_base = RandomForestClassifier(random_state=42)

# Ψάχνουμε τον συνδυασμό που μεγιστοποιεί το f1-score (την ισορροπία Precision και Recall)
grid_search = GridSearchCV(estimator=rf_base, param_grid=param_grid, cv=5, scoring='f1', n_jobs=-1)
grid_search.fit(X_train_smote, y_train_smote)

print(f"✅ Βρέθηκαν οι βέλτιστες ρυθμίσεις: {grid_search.best_params_}")

# 7. Χρήση του καλύτερου μοντέλου για πρόβλεψη
best_rf_model = grid_search.best_estimator_
y_pred = best_rf_model.predict(X_test)

print("\n--- Τελικά Αποτελέσματα Αξιολόγησης (Tuned Model) ---")
print(classification_report(y_test, y_pred, zero_division=0))

print("--- Τελικός Πίνακας Σύγχυσης ---")
print(confusion_matrix(y_test, y_pred))