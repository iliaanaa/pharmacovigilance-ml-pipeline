import requests
import pandas as pd
import numpy as np
import ast
import os
import time  # Προσθήκη για την παύση μεταξύ των κλήσεων

print("--- ΣΤΑΔΙΟ 1: DATA INGESTION (Ανάκτηση με Σελιδοποίηση) ---")
url = "https://api.fda.gov/drug/event.json"
all_results = []
api_success = False

# Τραβάμε 10 σελίδες από 100 εγγραφές (σύνολο 1000 εγγραφές)
for i in range(10):
    skip_value = i * 100
    params = {
        "search": "receivedate:[20230101 TO 20240101]",
        "limit": 100,
        "skip": skip_value # Λέει στο API πόσες εγγραφές να προσπεράσει
    }
    
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        api_success = True
        data = response.json()
        all_results.extend(data['results'])
        print(f"Επιτυχία: Κατέβηκε η σελίδα {i+1}/10 (Εγγραφές {skip_value} έως {skip_value + 100})")
    else:
        print(f"Σφάλμα {response.status_code} στη σελίδα {i+1}. Διακοπή άντλησης.")
        break  # Αν το API μας κόψει, σταματάμε τη λούπα για να μην φάμε ban
        
    time.sleep(1) # Παύση 1 δευτερολέπτου για να είμαστε "ευγενικοί" με τον server του FDA

# Αν κατεβάσαμε έστω και μία σελίδα, φτιάχνουμε το Dataframe
if api_success and len(all_results) > 0:
    df = pd.json_normalize(all_results)
    print(f"\nΣυνολικά φορτώθηκαν {df.shape[0]} ακατέργαστες εγγραφές από το API.")
    df.to_csv("openfda_raw_data.csv", index=False)
    print("Τα ακατέργαστα δεδομένα αποθηκεύτηκαν στο 'openfda_raw_data.csv'\n")
else:
    print("\nΣφάλμα στο API. Φόρτωση από τοπικό CSV...")
    df = pd.read_csv("openfda_raw_data.csv")

print("--- ΣΤΑΔΙΟ 2: DATA CLEANING (Καθαρισμός) ---")
# 1. Επιλογή στηλών
cols_to_keep = [
    'safetyreportid', 'serious', 'patient.patientsex', 
    'patient.patientonsetage', 'patient.drug', 'patient.reaction'
]
# Κρατάμε μόνο όσες στήλες υπάρχουν όντως στο df για αποφυγή σφαλμάτων
cols_to_keep = [col for col in cols_to_keep if col in df.columns]
df_clean = df[cols_to_keep].copy()

# 2. Κωδικοποίηση Σοβαρότητας (1 = Σοβαρό, 0 = Μη σοβαρό)
if 'serious' in df_clean.columns:
    df_clean['serious'] = df_clean['serious'].astype(str).map({'1': 1, '2': 0, '1.0': 1, '2.0': 0})

# 3. Κωδικοποίηση Φύλου
if 'patient.patientsex' in df_clean.columns:
    df_clean['patient_sex'] = df_clean['patient.patientsex'].astype(str).map({'1': 'Male', '2': 'Female', '1.0': 'Male', '2.0': 'Female'})
    df_clean.drop('patient.patientsex', axis=1, inplace=True)

# 4. Διαχείριση Ηλικίας (Median Imputation)
if 'patient.patientonsetage' in df_clean.columns:
    df_clean['patient_age'] = pd.to_numeric(df_clean['patient.patientonsetage'], errors='coerce')
    median_age = df_clean['patient_age'].median()
    df_clean['patient_age'] = df_clean['patient_age'].fillna(median_age)
    df_clean.drop('patient.patientonsetage', axis=1, inplace=True)

# 5. Συναρτήσεις Εξαγωγής Φαρμάκου & Παρενέργειας
def extract_drug(x):
    try:
        if pd.isna(x): return 'Unknown'
        if isinstance(x, str): x = ast.literal_eval(x)
        return x[0].get('medicinalproduct', 'Unknown')
    except:
        return 'Unknown'

def extract_reaction(x):
    try:
        if pd.isna(x): return 'Unknown'
        if isinstance(x, str): x = ast.literal_eval(x)
        return x[0].get('reactionmeddrapt', 'Unknown')
    except:
        return 'Unknown'

# Εφαρμογή των συναρτήσεων
if 'patient.drug' in df_clean.columns:
    df_clean['drug_name'] = df_clean['patient.drug'].apply(extract_drug)
    df_clean.drop('patient.drug', axis=1, inplace=True)

if 'patient.reaction' in df_clean.columns:
    df_clean['adverse_reaction'] = df_clean['patient.reaction'].apply(extract_reaction)
    df_clean.drop('patient.reaction', axis=1, inplace=True)

# 6. Τελικό Φιλτράρισμα
df_clean = df_clean[(df_clean.get('drug_name', '') != 'Unknown') & (df_clean.get('adverse_reaction', '') != 'Unknown')]

# 7. Αποθήκευση καθαρού dataset
df_clean.to_csv("openfda_clean_data.csv", index=False)

print("Ο καθαρισμός ολοκληρώθηκε!")
print(f"Τελικό μέγεθος καθαρού dataset: {df_clean.shape[0]} εγγραφές, {df_clean.shape[1]} χαρακτηριστικά.")
print("\n--- ΔΕΙΓΜΑ ΔΕΔΟΜΕΝΩΝ ---")
print(df_clean.head())