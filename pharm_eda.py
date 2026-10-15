import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print("[INFO] Initializing Exploratory Data Analysis (EDA) pipeline...")

# 1. Load cleaned clinical dataset
df = pd.read_csv("openfda_clean_data.csv")

# Set visualization style aesthetics
sns.set_theme(style="whitegrid")

# 2. Figure 1: Adverse Event Severity Class Distribution
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='serious', palette='Set2')
plt.title('Clinical Severity Distribution (0 = Non-Serious, 1 = Serious)')
plt.xlabel('Severity Class Index')
plt.ylabel('Report Volume')
plt.tight_layout()
plt.savefig('eda_1_severity_distribution.png')
plt.close()
print("[INFO] Artifact generated: 'eda_1_severity_distribution.png'")

# 3. Figure 2: Patient Demographic Age Distribution
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x='patient_age', bins=10, kde=True, color='skyblue')
plt.title('Patient Demographic Age Distribution in Safety Reports')
plt.xlabel('Patient Age (Years)')
plt.ylabel('Observation Frequency')
plt.tight_layout()
plt.savefig('eda_2_age_distribution.png')
plt.close()
print("[INFO] Artifact generated: 'eda_2_age_distribution.png'")

# 4. Figure 3: High-Frequency Medicinal Products
plt.figure(figsize=(10, 6))
top_drugs = df['drug_name'].value_counts().nlargest(5)
sns.barplot(y=top_drugs.index, x=top_drugs.values, palette='viridis')
plt.title('Top 5 Medicinal Products by Safety Report Frequency')
plt.xlabel('Total Adverse Event Reports')
plt.ylabel('Medicinal Product Name')
plt.tight_layout()
plt.savefig('eda_3_top_drugs.png')
plt.close()
print("[INFO] Artifact generated: 'eda_3_top_drugs.png'")

print("[SUCCESS] Exploratory Data Analysis execution complete. Artifacts successfully written to storage.")
