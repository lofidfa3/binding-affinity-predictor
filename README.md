# ML-Driven Binding Affinity Predictor for EGFR

A complete machine learning pipeline that predicts ligand binding affinity against **EGFR (Epidermal Growth Factor Receptor)** — a key oncology drug target — using experimental bioactivity data from [ChEMBL](https://www.ebi.ac.uk/chembl/). The project includes data collection, molecular featurization, model training and comparison, SHAP-based interpretation, virtual screening, and an interactive Streamlit web app.

## Results Summary

Trained on **10,413 unique compounds** with pChEMBL values (standardized -log₁₀ of IC50/Ki/Kd):

| Model | R² | RMSE | Pearson r |
|-------|-----|------|-----------|
| Random Forest | 0.711 | 0.699 | 0.850 |
| **XGBoost** | **0.736** | **0.669** | **0.858** |
| SVR | 0.687 | 0.728 | 0.829 |

**Best model: XGBoost** with R² = 0.736 on the held-out test set.

### Generalization Assessment (Notebook 03b)

The random split above can inflate performance due to analog bias. We evaluated the XGBoost model under rigorous splitting strategies:

| Split | Train Size | Test Size | R² | RMSE | Pearson r |
|-------|-----------|----------|-----|------|-----------|
| Random | 8,330 | 2,083 | 0.736 | 0.669 | 0.858 |
| Scaffold | 8,333 | 2,080 | 0.466 | 0.909 | 0.696 |
| Temporal (post-2020) | 7,564 | 2,810 | -0.768 | 1.562 | 0.027 |

- **Scaffold split** groups compounds by Murcko scaffold, preventing structural leakage. The R² drop (0.736 → 0.466) confirms analog bias in the random split.
- **Temporal split** trains on pre-2020 data and tests on post-2020 compounds. The negative R² reflects a severe covariate shift from a large 2021 screening batch (1,497 compounds).

## Project Structure

```
binding-affinity-predictor/
├── environment.yml                        # Conda environment specification
├── README.md
├── notebooks/
│   ├── 01_data_collection.ipynb           # ChEMBL data query & cleaning
│   ├── 02_molecular_features.ipynb        # RDKit descriptors & Morgan fingerprints
│   ├── 03_model_training.ipynb            # RF, XGBoost, SVR training & comparison
│   ├── 03b_scaffold_temporal_split.ipynb  # Scaffold & temporal split evaluation
│   ├── 04_model_interpretation.ipynb      # SHAP analysis & feature importance
│   └── 05_virtual_screening.ipynb         # Score & rank novel compounds
├── app/
│   └── streamlit_app.py                   # Interactive prediction dashboard
├── data/                                  # Generated at runtime (gitignored)
│   ├── raw/                               # Raw ChEMBL bioactivity data
│   └── processed/                         # Feature matrices & screening results
└── models/                                # Saved models & plots (gitignored)
```

## Pipeline Overview

### Step 1: Data Collection (`01_data_collection.ipynb`)
- Queries ChEMBL for all compounds tested against human EGFR (CHEMBL203)
- Filters for binding assays (IC50, Ki, Kd) with valid pChEMBL values
- Removes duplicates by averaging multiple measurements per compound
- Removes outliers (pChEMBL < 3 or > 12)
- **Output:** 10,413 unique compounds with pChEMBL range 4.00–11.00

### Step 2: Molecular Representation (`02_molecular_features.ipynb`)
- Converts SMILES to RDKit Mol objects
- Computes **12 physicochemical descriptors:** Molecular Weight, LogP, HBA, HBD, TPSA, Rotatable Bonds, Aromatic Rings, Heavy Atoms, Ring Count, Fraction CSP3, Heteroatom Count, Molar Refractivity
- Computes **2048-bit Morgan fingerprints** (circular fingerprints, radius 2)
- Generates correlation analysis between descriptors and binding affinity
- **Output:** 2,060-dimensional feature matrix

### Step 3: Model Training (`03_model_training.ipynb`)
- Stratified 80/20 train/test split (8,330 train / 2,083 test)
- Trains three models with `RandomizedSearchCV` (5-fold CV):
  - **Random Forest** — 30 hyperparameter combinations
  - **XGBoost** — 30 hyperparameter combinations
  - **SVR** (RBF kernel) — 20 hyperparameter combinations
- Evaluates on held-out test set with R², RMSE, and Pearson correlation
- Generates predicted vs. actual scatter plots
- **Output:** Saved models (`.pkl`), scaler, and metadata

### Step 3b: Scaffold & Temporal Split Evaluation (`03b_scaffold_temporal_split.ipynb`)
- Loads the trained XGBoost model (same hyperparameters, no new search) and refits on each split's training set
- **Murcko scaffold split:** Groups compounds by Bemis–Murcko generic scaffold; assigns least-frequent scaffolds to test set (~20%); zero scaffold overlap between train/test
- **Temporal split:** Fetches `document_year` from ChEMBL; trains on compounds published ≤2020, tests on post-2020
- Produces a 3-panel predicted vs. actual scatter plot and a UMAP chemical space visualization
- Includes a ready-to-copy results paragraph for manuscript use
- **Output:** `models/split_comparison.png`, `models/chemical_space_splits.png`, `models/split_comparison.csv`

### Step 4: Model Interpretation (`04_model_interpretation.ipynb`)
- Feature importance bar plots for Random Forest and XGBoost
- SHAP (SHapley Additive exPlanations) analysis:
  - Beeswarm summary plot (top 20 features)
  - Mean absolute SHAP bar plot
  - Dependence plots for top physicochemical descriptors
- Biological context: links molecular features to EGFR binding pocket characteristics
- Compares contribution of physicochemical descriptors vs. fingerprint bits

### Step 5: Virtual Screening (`05_virtual_screening.ipynb`)
- Queries ChEMBL for **3,064 approved small molecule drugs** (max_phase=4), excluding compounds in the EGFR training set
- Computes 2,060-dim features and predicts binding affinity using the best XGBoost model
- Filters by Lipinski Rule of 5 → **2,620 drug-like candidates** (85.5%)
- Identifies **16 strong predicted binders** (pChEMBL ≥ 7), including Sotorasib, Hydralazine, and known kinase inhibitors
- Visualizes predicted affinity distribution and top 10 candidate structures

### Step 6: Streamlit Dashboard (`app/streamlit_app.py`)
- Input any SMILES string and get an instant prediction
- Displays:
  - 2D molecular structure
  - Predicted pChEMBL value and estimated IC50 (nM)
  - Color-coded binding strength (strong/moderate/weak)
  - Lipinski Rule of 5 compliance check
  - Full molecular property table
  - Generalization assessment (scaffold & temporal R²) in sidebar
- Includes example molecules (Erlotinib, Gefitinib, Lapatinib)

## Getting Started

### Prerequisites
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda

### Installation

```bash
git clone https://github.com/lofidfa3/binding-affinity-predictor.git
cd binding-affinity-predictor

# Create and activate the conda environment
conda env create -f environment.yml
conda activate binding-affinity
```

### Running the Pipeline

Run the notebooks sequentially (each depends on the previous):

```bash
# Execute all notebooks in order
jupyter nbconvert --to notebook --execute notebooks/01_data_collection.ipynb --output 01_data_collection.ipynb
jupyter nbconvert --to notebook --execute notebooks/02_molecular_features.ipynb --output 02_molecular_features.ipynb
jupyter nbconvert --to notebook --execute notebooks/03_model_training.ipynb --output 03_model_training.ipynb --ExecutePreprocessor.timeout=1200
jupyter nbconvert --to notebook --execute notebooks/03b_scaffold_temporal_split.ipynb --output 03b_scaffold_temporal_split.ipynb --ExecutePreprocessor.timeout=600
jupyter nbconvert --to notebook --execute notebooks/04_model_interpretation.ipynb --output 04_model_interpretation.ipynb
jupyter nbconvert --to notebook --execute notebooks/05_virtual_screening.ipynb --output 05_virtual_screening.ipynb
```

Or open them interactively:

```bash
jupyter notebook
```

### Launching the Dashboard

After running notebooks 01–03 (to generate the trained model):

```bash
streamlit run app/streamlit_app.py
```

## Key Dependencies

| Package | Purpose |
|---------|---------|
| `chembl_webresource_client` | Query ChEMBL bioactivity database |
| `rdkit` | Molecular descriptors & fingerprints |
| `scikit-learn` | Random Forest, SVR, preprocessing |
| `xgboost` | Gradient boosting regression |
| `shap` | Model interpretation |
| `umap-learn` | Chemical space visualization |
| `streamlit` | Interactive web dashboard |
| `matplotlib` / `seaborn` | Visualization |

## Technologies Used

- **Data Source:** ChEMBL (European Bioinformatics Institute)
- **Cheminformatics:** RDKit — Morgan fingerprints, physicochemical descriptors
- **Machine Learning:** scikit-learn, XGBoost
- **Interpretability:** SHAP (TreeExplainer)
- **Web App:** Streamlit
- **Target:** EGFR (CHEMBL203) — Epidermal Growth Factor Receptor, a validated cancer drug target

## License

This project is for educational and research purposes.
