import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print("Ξεκινάει η Εξερευνητική Ανάλυση Δεδομένων (EDA)...")

# 1. Φόρτωση των καθαρών δεδομένων
df = pd.read_csv("openfda_clean_data.csv")

# Ρύθμιση του αισθητικού στυλ των γραφημάτων
sns.set_theme(style="whitegrid")

# 2. Γράφημα 1: Κατανομή Σοβαρότητας Παρενεργειών
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='serious', palette='Set2')
plt.title('Severity of Adverse Events (0 = Non-Serious, 1 = Serious)')
plt.xlabel('Serious')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('eda_1_severity_distribution.png')
plt.close()
print("- Δημιουργήθηκε το eda_1_severity_distribution.png")

# 3. Γράφημα 2: Ηλικιακή Κατανομή των Ασθενών
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x='patient_age', bins=10, kde=True, color='skyblue')
plt.title('Patient Age Distribution in Reported Events')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('eda_2_age_distribution.png')
plt.close()
print("- Δημιουργήθηκε το eda_2_age_distribution.png")

# 4. Γράφημα 3: Top Φάρμακα με τις περισσότερες αναφορές
plt.figure(figsize=(10, 6))
top_drugs = df['drug_name'].value_counts().nlargest(5)
sns.barplot(y=top_drugs.index, x=top_drugs.values, palette='viridis')
plt.title('Top 5 Drugs by Number of Adverse Event Reports')
plt.xlabel('Number of Reports')
plt.ylabel('Drug Name')
plt.tight_layout()
plt.savefig('eda_3_top_drugs.png')
plt.close()
print("- Δημιουργήθηκε το eda_3_top_drugs.png")

print("Η ανάλυση ολοκληρώθηκε! Άνοιξε τον φάκελο 'pharmacovigilance_pipeline' για να δεις τις εικόνες.")