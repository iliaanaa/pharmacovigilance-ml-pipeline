import requests
import pandas as pd
import numpy as np
import ast
import os
import time  # Latency control between consecutive API requests

print("[INFO] Initiating Data Ingestion Stage via OpenFDA API with Pagination...")
url = "https://api.fda.gov/drug/event.json"
all_results = []
api_success = False

# Paginate through 10 pages at 100 records per request (total target: 1,000 records)
for i in range(10):
    skip_value = i * 100
    params = {
        "search": "receivedate:[20230101 TO 20240101]",
        "limit": 100,
        "skip": skip_value  # Defines offset for server pagination
    }
    
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        api_success = True
        data = response.json()
        all_results.extend(data['results'])
        print(f"[INFO] Successfully fetched page {i+1}/10 (Records {skip_value} to {skip_value + 100})")
    else:
        print(f"[WARNING] HTTP Status {response.status_code} encountered on page {i+1}. Aborting pagination loop.")
        break  # Prevent rate-limiting bans by breaking on error
        
    time.sleep(1)  # 1-second pause to adhere to API rate limits

# Construct DataFrame upon successful request processing
if api_success and len(all_results) > 0:
    df = pd.json_normalize(all_results)
    print(f"\n[INFO] Total raw records ingested from API: {df.shape[0]}")
    df.to_csv("openfda_raw_data.csv", index=False)
    print("[INFO] Raw dataset serialized to 'openfda_raw_data.csv'\n")
else:
    print("\n[WARNING] API connection unviable. Falling back to local cache 'openfda_raw_data.csv'...")
    df = pd.read_csv("openfda_raw_data.csv")

print("[INFO] Initiating Clinical Data Engineering and Preprocessing Stage...")

# 1. Feature selection of clinically relevant attributes
cols_to_keep = [
    'safetyreportid', 'serious', 'patient.patientsex', 
    'patient.patientonsetage', 'patient.drug', 'patient.reaction'
]
# Retain existing columns to prevent KeyError exceptions
cols_to_keep = [col for col in cols_to_keep if col in df.columns]
df_clean = df[cols_to_keep].copy()

# 2. Target Variable Encoding (1 = Serious Event, 0 = Non-Serious Event)
if 'serious' in df_clean.columns:
    df_clean['serious'] = df_clean['serious'].astype(str).map({'1': 1, '2': 0, '1.0': 1, '2.0': 0})

# 3. Demographic Gender Encoding
if 'patient.patientsex' in df_clean.columns:
    df_clean['patient_sex'] = df_clean['patient.patientsex'].astype(str).map({'1': 'Male', '2': 'Female', '1.0': 'Male', '2.0': 'Female
