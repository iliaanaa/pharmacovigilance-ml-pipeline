# Real-World Pharmacovigilance: Automated Safety Signal Detection

## Business Context
Manual review of post-marketing Adverse Event (AE) reports is resource-intensive and prone to delayed detection of critical clinical risks. Pharmacovigilance and Real-World Evidence (RWE) teams require automated, robust pipelines to continuously monitor drug safety profiles post-market approval, identifying critical safety signals amidst vast amounts of noisy data.

## Solution Architecture
An end-to-end Machine Learning pipeline designed to ingest, clean, and analyze Real-World Data (RWD) directly from the FDA Adverse Event Reporting System (FAERS). The system automatically identifies underlying patterns and predicts the clinical severity of adverse reactions based on patient demographics and pharmacological interactions.

## Technical Pipeline
1. **Automated Data Ingestion:** Utilizes the OpenFDA API with custom pagination logic to extract large-scale, real-world adverse event JSON records while respecting server rate limits.
2. **Clinical Data Engineering:** 
   * Flattens deeply nested JSON structures.
   * Performs imputation on missing clinical values (e.g., median patient age).
   * Extracts specific MedDRA terms for adverse reactions and active substances safely.
3. **Exploratory Data Analysis (EDA):** Generates automated distribution matrices highlighting severity ratios, demographic spreads, and high-frequency medicinal products.
4. **Machine Learning Optimization:**
   * **Algorithm:** Random Forest Classifier.
   * **Class Imbalance Resolution:** Implemented **SMOTE** (Synthetic Minority Over-sampling Technique) to resolve the inherent imbalance between rare serious events and common mild events.
   * **Hyperparameter Tuning:** Applied **GridSearchCV** with Cross-Validation to optimize `max_depth`, `n_estimators`, and `min_samples_split`. This rigorous tuning prevented overfitting on synthetic data and established a robust, generalizable model for safety signal triage.

## Tech Stack
* **Language:** Python 3.10
* **Data Manipulation:** Pandas, NumPy
* **Machine Learning:** Scikit-learn, Imbalanced-learn
* **Visualization:** Matplotlib, Seaborn
* **API Integration:** Requests

## How to Run
1. Clone the repository and install dependencies: `pip install -r requirements.txt`
2. Run data extraction and cleaning: `python pipeline.py`
3. Generate clinical visualizations: `python pharm_eda.py`
4. Train and optimize the ML model: `python pharm_ml.py`