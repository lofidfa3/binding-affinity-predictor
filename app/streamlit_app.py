"""
EGFR Binding Affinity Predictor - Streamlit Dashboard

An interactive web app that predicts binding affinity for novel compounds
against EGFR using a trained ML model.

Usage:
    conda activate binding-affinity
    streamlit run app/streamlit_app.py
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, Draw
from rdkit import DataStructs
from io import BytesIO

# Page config
st.set_page_config(
    page_title="EGFR Binding Affinity Predictor",
    page_icon="🧬",
    layout="wide"
)

st.title("EGFR Binding Affinity Predictor")
st.markdown(
    "Predict the binding affinity (pChEMBL) of compounds against **EGFR** "
    "(Epidermal Growth Factor Receptor) using a machine learning model trained "
    "on ChEMBL bioactivity data."
)


@st.cache_resource
def load_model():
    """Load the trained model, scaler, and metadata."""
    model = joblib.load("models/best_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    metadata = joblib.load("models/metadata.pkl")
    return model, scaler, metadata


def compute_features(smiles):
    """Compute the full feature vector for a SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None

    # Physicochemical descriptors
    desc = {
        "MW": Descriptors.MolWt(mol),
        "LogP": Descriptors.MolLogP(mol),
        "HBA": Descriptors.NumHAcceptors(mol),
        "HBD": Descriptors.NumHDonors(mol),
        "TPSA": Descriptors.TPSA(mol),
        "RotBonds": Descriptors.NumRotatableBonds(mol),
        "AromaticRings": Descriptors.NumAromaticRings(mol),
        "HeavyAtoms": Descriptors.HeavyAtomCount(mol),
        "RingCount": Descriptors.RingCount(mol),
        "FractionCSP3": Descriptors.FractionCSP3(mol),
        "NumHeteroatoms": Descriptors.NumHeteroatoms(mol),
        "MolRefractivity": Descriptors.MolMR(mol),
    }

    # Morgan fingerprint
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    fp_arr = np.zeros(2048, dtype=int)
    DataStructs.ConvertToNumpyArray(fp, fp_arr)

    for i in range(2048):
        desc[f"FP_{i}"] = fp_arr[i]

    return pd.DataFrame([desc]), mol


def check_lipinski(desc):
    """Check Lipinski Rule of 5."""
    rules = {
        "MW <= 500": desc["MW"] <= 500,
        "LogP <= 5": desc["LogP"] <= 5,
        "HBA <= 10": desc["HBA"] <= 10,
        "HBD <= 5": desc["HBD"] <= 5,
    }
    return rules


def mol_to_image(mol, size=(400, 300)):
    """Convert RDKit mol to image bytes."""
    img = Draw.MolToImage(mol, size=size)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# Load model
try:
    model, scaler, metadata = load_model()
    model_loaded = True
except FileNotFoundError:
    model_loaded = False
    st.error(
        "Model files not found. Please run notebooks 01-03 first to train the model."
    )

if model_loaded:
    st.sidebar.header("Model Info")
    st.sidebar.write(f"**Algorithm:** {metadata['best_model']}")
    st.sidebar.write(f"**Test R²:** {metadata['best_r2']:.4f}")
    st.sidebar.write(f"**Test RMSE:** {metadata['best_rmse']:.4f}")
    st.sidebar.write(f"**Training samples:** {metadata['n_train']}")
    st.sidebar.write(f"**Target:** EGFR (CHEMBL203)")

    # Split comparison (from notebook 03b)
    split_csv = "models/split_comparison.csv"
    if os.path.exists(split_csv):
        st.sidebar.header("Generalization Assessment")
        split_df = pd.read_csv(split_csv)
        for _, row in split_df.iterrows():
            label = row["Split"].replace(" (existing)", "").replace(" (new)", "")
            r2 = row["R²"]
            rmse = row["RMSE"]
            st.sidebar.write(f"**{label}:** R²={r2:.3f}, RMSE={rmse:.3f}")
        st.sidebar.caption(
            "Scaffold and temporal splits remove analog bias. "
            "See notebook 03b for details."
        )

    # Example molecules
    st.sidebar.header("Example SMILES")
    examples = {
        "Erlotinib": "C=CC(=O)Nc1cc2c(Nc3ccc(F)c(Cl)c3)ncnc2cc1OC",
        "Gefitinib": "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1",
        "Lapatinib": "CS(=O)(=O)CCNCc1ccc(-c2ccc3ncnc(Nc4ccc(OCc5cccc(F)c5)c(Cl)c4)c3c2)o1",
        "Simple aniline": "NC1=CC=CC=C1",
    }

    selected_example = st.sidebar.selectbox("Try an example:", [""] + list(examples.keys()))

    # Main input
    st.header("Enter a Compound")

    default_smiles = examples.get(selected_example, "")
    smiles_input = st.text_input(
        "SMILES string:",
        value=default_smiles,
        placeholder="e.g., C1=CC=C2C(=C1)C=NC(=N2)NC3=CC=CC=C3",
    )

    if smiles_input:
        features_df, mol = compute_features(smiles_input)

        if features_df is None:
            st.error("Invalid SMILES string. Please check the input.")
        else:
            # Make prediction
            if metadata["needs_scaling"]:
                prediction = model.predict(scaler.transform(features_df))[0]
            else:
                prediction = model.predict(features_df)[0]

            # Layout: two columns
            col1, col2 = st.columns([1, 1])

            with col1:
                st.subheader("Molecular Structure")
                img_bytes = mol_to_image(mol, size=(400, 300))
                st.image(img_bytes, use_container_width=True)

            with col2:
                st.subheader("Prediction Results")

                # Predicted affinity with color coding
                if prediction >= 7:
                    color = "green"
                    label = "Strong binder"
                elif prediction >= 5:
                    color = "orange"
                    label = "Moderate binder"
                else:
                    color = "red"
                    label = "Weak binder"

                st.markdown(
                    f"### Predicted pChEMBL: "
                    f"<span style='color:{color}'>{prediction:.2f}</span> "
                    f"({label})",
                    unsafe_allow_html=True,
                )

                # Convert to IC50
                ic50_nm = 10 ** (9 - prediction)
                st.write(f"**Estimated IC50:** {ic50_nm:.1f} nM")

            # Drug-likeness assessment
            st.subheader("Drug-Likeness Assessment (Lipinski Rule of 5)")

            desc = features_df.iloc[0]
            lipinski = check_lipinski(desc)

            lip_col1, lip_col2, lip_col3, lip_col4 = st.columns(4)

            properties = [
                ("MW", desc["MW"], "MW <= 500", lip_col1),
                ("LogP", desc["LogP"], "LogP <= 5", lip_col2),
                ("HBA", desc["HBA"], "HBA <= 10", lip_col3),
                ("HBD", desc["HBD"], "HBD <= 5", lip_col4),
            ]

            for name, value, rule, col in properties:
                with col:
                    passed = lipinski[rule]
                    icon = "✅" if passed else "❌"
                    st.metric(label=f"{icon} {name}", value=f"{value:.1f}")

            violations = sum(1 for v in lipinski.values() if not v)
            if violations == 0:
                st.success("Passes all Lipinski rules - good drug-likeness!")
            elif violations == 1:
                st.warning(f"{violations} Lipinski violation - borderline drug-likeness.")
            else:
                st.error(f"{violations} Lipinski violations - poor drug-likeness.")

            # Additional properties table
            st.subheader("Molecular Properties")
            props_df = pd.DataFrame(
                {
                    "Property": [
                        "Molecular Weight",
                        "LogP",
                        "H-Bond Acceptors",
                        "H-Bond Donors",
                        "TPSA",
                        "Rotatable Bonds",
                        "Aromatic Rings",
                        "Heavy Atoms",
                    ],
                    "Value": [
                        f"{desc['MW']:.2f}",
                        f"{desc['LogP']:.2f}",
                        f"{int(desc['HBA'])}",
                        f"{int(desc['HBD'])}",
                        f"{desc['TPSA']:.2f} Å²",
                        f"{int(desc['RotBonds'])}",
                        f"{int(desc['AromaticRings'])}",
                        f"{int(desc['HeavyAtoms'])}",
                    ],
                }
            )
            st.table(props_df)

st.markdown("---")
st.markdown(
    "*Built with ChEMBL data, RDKit, and scikit-learn/XGBoost. "
    "Model trained on EGFR (CHEMBL203) bioactivity data.*"
)
