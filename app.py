import os
import re
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Protein Structure Predictor",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.main {
    background-color: #f8fafc;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
}

.hero {
    padding: 2rem;
    border-radius: 20px;
    background: linear-gradient(
        135deg,
        #0f172a,
        #1e3a8a
    );
    color: white;
    margin-bottom: 2rem;
}

.hero h1 {
    font-size: 42px;
    margin-bottom: 8px;
}

.hero p {
    font-size: 18px;
    opacity: 0.9;
}

.metric-card {
    background: white;
    padding: 1.2rem;
    border-radius: 15px;
    border: 1px solid #e2e8f0;
    box-shadow: 0 4px 15px rgba(0,0,0,0.05);
}

.result-card {
    background: #ffffff;
    padding: 1.5rem;
    border-radius: 18px;
    border: 1px solid #e2e8f0;
    margin-top: 1rem;
}

.structure-box {
    padding: 10px;
    border-radius: 10px;
    background: #f1f5f9;
    font-family: monospace;
    font-size: 18px;
    word-break: break-all;
    line-height: 2;
}

.footer {
    text-align: center;
    color: #64748b;
    padding: 2rem;
    margin-top: 3rem;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.path.join(
    BASE_DIR,
    "Protein_Secondary_Structure_Models"
)


# ============================================================
# MODEL FILES
# ============================================================

ML_MODELS = {
    "Logistic Regression":
        "logistic_regression.pkl",

    "Random Forest":
        "random_forest.pkl",

    "SVM":
        "support_vector_machine.pkl",

    "KNN":
        "k_nearest_neighbors.pkl",

    "XGBoost":
        "xgboost.pkl"
}

DL_MODELS = {
    "ANN":
        "ANN_Protein_Secondary_Structure.keras",

    "1D CNN":
        "CNN_Protein_Secondary_Structure.keras",

    "BiLSTM":
        "BiLSTM_Protein_Secondary_Structure.keras"
}


# ============================================================
# LOAD CONFIGURATION
# ============================================================

@st.cache_resource
def load_configuration():

    config_path = os.path.join(
        MODEL_DIR,
        "model_configuration.pkl"
    )

    if os.path.exists(config_path):

        return joblib.load(config_path)

    return {
        "classes": ["C", "E", "H"]
    }


# ============================================================
# LOAD ML MODEL
# ============================================================

@st.cache_resource
def load_ml_model(model_name):

    filename = ML_MODELS[model_name]

    path = os.path.join(
        MODEL_DIR,
        filename
    )

    if not os.path.exists(path):
        return None

    return joblib.load(path)


# ============================================================
# LOAD DL MODEL
# ============================================================

@st.cache_resource
def load_dl_model(model_name):

    try:

        import tensorflow as tf

        filename = DL_MODELS[model_name]

        path = os.path.join(
            MODEL_DIR,
            filename
        )

        if not os.path.exists(path):
            return None

        return tf.keras.models.load_model(path)

    except Exception:
        return None


# ============================================================
# LOAD LABEL ENCODER
# ============================================================

@st.cache_resource
def load_encoder():

    path = os.path.join(
        MODEL_DIR,
        "ml_label_encoder.pkl"
    )

    if os.path.exists(path):

        return joblib.load(path)

    return None


# ============================================================
# AMINO ACID VOCABULARY
# ============================================================

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"

AA_TO_ID = {
    aa: i + 1
    for i, aa in enumerate(AMINO_ACIDS)
}

UNKNOWN_ID = 0


# ============================================================
# SEQUENCE CLEANING
# ============================================================

def clean_sequence(sequence):

    sequence = sequence.upper()

    # Remove spaces and line breaks
    sequence = re.sub(
        r"\s+",
        "",
        sequence
    )

    # Remove FASTA header
    if sequence.startswith(">"):

        sequence = sequence.split("\n", 1)[-1]

        sequence = re.sub(
            r"\s+",
            "",
            sequence
        )

    return sequence


# ============================================================
# VALIDATE SEQUENCE
# ============================================================

def validate_sequence(sequence):

    if not sequence:

        return False, "Please enter a protein sequence."

    invalid = sorted(
        set(sequence) - set(AMINO_ACIDS)
    )

    if invalid:

        return False, (
            "Invalid amino-acid symbols detected: "
            + ", ".join(invalid)
        )

    return True, ""


# ============================================================
# SEQUENCE → NUMERICAL
# ============================================================

def encode_sequence(sequence):

    return np.array([
        AA_TO_ID.get(
            aa,
            UNKNOWN_ID
        )
        for aa in sequence
    ])


# ============================================================
# CREATE WINDOWS FOR ML
# ============================================================

def create_windows(
    encoded_sequence,
    window_size=9
):

    half = window_size // 2

    padded = np.pad(
        encoded_sequence,
        (half, half),
        mode="constant",
        constant_values=UNKNOWN_ID
    )

    windows = []

    for i in range(
        len(encoded_sequence)
    ):

        window = padded[
            i:i + window_size
        ]

        windows.append(window)

    return np.array(windows)


# ============================================================
# ML PREDICTION
# ============================================================

def predict_ml(
    model,
    sequence,
    window_size=9
):

    encoded = encode_sequence(
        sequence
    )

    windows = create_windows(
        encoded,
        window_size
    )

    predictions = model.predict(
        windows
    )

    # Handle probability output
    if len(predictions.shape) > 1:

        predictions = np.argmax(
            predictions,
            axis=1
        )

    predictions = predictions.astype(int)

    encoder = load_encoder()

    if encoder is not None:

        try:

            labels = encoder.inverse_transform(
                predictions
            )

            return "".join(labels)

        except Exception:
            pass

    mapping = {
        0: "C",
        1: "E",
        2: "H"
    }

    return "".join(
        mapping.get(
            int(x),
            "C"
        )
        for x in predictions
    )


# ============================================================
# DL SEQUENCE ENCODING
# ============================================================

def prepare_dl_sequence(
    sequence,
    max_length
):

    encoded = encode_sequence(
        sequence
    )

    # Truncate
    encoded = encoded[:max_length]

    # Padding
    padded = np.pad(
        encoded,
        (
            0,
            max(
                0,
                max_length - len(encoded)
            )
        ),
        mode="constant"
    )

    return padded


# ============================================================
# DL PREDICTION
# ============================================================

def predict_dl(
    model,
    sequence,
    max_length
):

    encoded = prepare_dl_sequence(
        sequence,
        max_length
    )

    # Add batch dimension
    X = np.expand_dims(
        encoded,
        axis=0
    )

    prediction = model.predict(
        X,
        verbose=0
    )

    prediction = np.array(
        prediction
    )

    # Case 1:
    # Output shape = (1, sequence_length, classes)
    if prediction.ndim == 3:

        class_ids = np.argmax(
            prediction[0],
            axis=-1
        )

    # Case 2:
    # Output shape = (1, classes)
    elif prediction.ndim == 2:

        class_ids = np.argmax(
            prediction,
            axis=-1
        )

    else:

        class_ids = prediction.flatten()

    mapping = {
        0: "C",
        1: "E",
        2: "H"
    }

    labels = [
        mapping.get(
            int(x),
            "C"
        )
        for x in class_ids
    ]

    return "".join(
        labels[:len(sequence)]
    )


# ============================================================
# STRUCTURE STATISTICS
# ============================================================

def structure_statistics(
    structure
):

    counts = Counter(
        structure
    )

    total = len(structure)

    data = []

    for label, name in [
        ("H", "Alpha Helix"),
        ("E", "Beta Strand"),
        ("C", "Coil")
    ]:

        count = counts.get(
            label,
            0
        )

        percentage = (
            count / total * 100
            if total > 0
            else 0
        )

        data.append({
            "Structure": name,
            "Code": label,
            "Count": count,
            "Percentage": round(
                percentage,
                2
            )
        })

    return pd.DataFrame(data)


# ============================================================
# STRUCTURE VISUALIZATION
# ============================================================

def plot_structure(
    structure
):

    mapping = {
        "H": 3,
        "E": 2,
        "C": 1
    }

    values = [
        mapping.get(
            x,
            0
        )
        for x in structure
    ]

    fig, ax = plt.subplots(
        figsize=(14, 2.5)
    )

    ax.plot(
        range(len(values)),
        values,
        linewidth=2
    )

    ax.set_yticks(
        [1, 2, 3]
    )

    ax.set_yticklabels([
        "Coil (C)",
        "Beta Strand (E)",
        "Alpha Helix (H)"
    ])

    ax.set_xlabel(
        "Residue Position"
    )

    ax.set_title(
        "Predicted Secondary Structure"
    )

    ax.grid(
        alpha=0.25
    )

    return fig


# ============================================================
# HERO SECTION
# ============================================================

st.markdown("""
<div class="hero">

<h1>🧬 Protein Secondary Structure Predictor</h1>

<p>
Machine Learning & Deep Learning based prediction of
protein secondary structures.
</p>

<p>
Predict <b>Alpha Helix (H)</b>,
<b>Beta Strand (E)</b>, and
<b>Coil (C)</b> directly from an amino-acid sequence.
</p>

</div>
""", unsafe_allow_html=True)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "🧬 Prediction Settings"
)

model_type = st.sidebar.radio(
    "Select Model Type",
    [
        "Machine Learning",
        "Deep Learning"
    ]
)

if model_type == "Machine Learning":

    model_name = st.sidebar.selectbox(
        "Select ML Model",
        list(ML_MODELS.keys())
    )

else:

    model_name = st.sidebar.selectbox(
        "Select Deep Learning Model",
        list(DL_MODELS.keys())
    )


# ============================================================
# MAIN INPUT
# ============================================================

st.subheader(
    "🔬 Enter Protein Sequence"
)

st.write(
    "Enter a protein sequence using the standard "
    "20 amino-acid symbols."
)

sequence_input = st.text_area(
    "Protein Sequence",
    height=180,
    placeholder=(
        "Example:\n"
        "MKTIIALSYIFCLVFADYKDDDDK"
    )
)

predict_button = st.button(
    "🚀 Predict Secondary Structure",
    use_container_width=True
)


# ============================================================
# PREDICTION
# ============================================================

if predict_button:

    sequence = clean_sequence(
        sequence_input
    )

    valid, error = validate_sequence(
        sequence
    )

    if not valid:

        st.error(error)

    else:

        st.success(
            f"Valid protein sequence detected — "
            f"{len(sequence)} residues."
        )

        with st.spinner(
            "Running prediction..."
        ):

            try:

                # --------------------------------------------
                # MACHINE LEARNING
                # --------------------------------------------

                if model_type == "Machine Learning":

                    model = load_ml_model(
                        model_name
                    )

                    if model is None:

                        st.error(
                            f"{model_name} model file "
                            "was not found."
                        )

                        st.stop()

                    config = load_configuration()

                    window_size = config.get(
                        "window_size",
                        9
                    )

                    predicted_structure = predict_ml(
                        model,
                        sequence,
                        window_size
                    )

                # --------------------------------------------
                # DEEP LEARNING
                # --------------------------------------------

                else:

                    model = load_dl_model(
                        model_name
                    )

                    if model is None:

                        st.error(
                            f"{model_name} model could not "
                            "be loaded."
                        )

                        st.stop()

                    config = load_configuration()

                    max_length = config.get(
                        "max_sequence_length",
                        512
                    )

                    predicted_structure = predict_dl(
                        model,
                        sequence,
                        max_length
                    )

            except Exception as e:

                st.error(
                    "Prediction failed."
                )

                st.exception(e)

                st.stop()


        # ====================================================
        # RESULTS
        # ====================================================

        st.markdown(
            "## 🎯 Prediction Results"
        )

        # Metrics
        stats = structure_statistics(
            predicted_structure
        )

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Sequence Length",
            len(sequence)
        )

        col2.metric(
            "Alpha Helix (H)",
            f"{stats.loc[stats['Code'] == 'H', 'Percentage'].iloc[0]}%"
        )

        col3.metric(
            "Beta Strand (E)",
            f"{stats.loc[stats['Code'] == 'E', 'Percentage'].iloc[0]}%"
        )

        col4.metric(
            "Coil (C)",
            f"{stats.loc[stats['Code'] == 'C', 'Percentage'].iloc[0]}%"
        )


        # ====================================================
        # SEQUENCE
        # ====================================================

        st.markdown(
            "### 🧬 Input Sequence"
        )

        st.code(
            sequence,
            language="text"
        )


        # ====================================================
        # PREDICTED STRUCTURE
        # ====================================================

        st.markdown(
            "### 🔮 Predicted Structure"
        )

        st.code(
            predicted_structure,
            language="text"
        )


        # ====================================================
        # STRUCTURE TABLE
        # ====================================================

        st.markdown(
            "### 📊 Structure Composition"
        )

        st.dataframe(
            stats,
            use_container_width=True,
            hide_index=True
        )


        # ====================================================
        # GRAPH
        # ====================================================

        st.markdown(
            "### 📈 Structure Visualization"
        )

        figure = plot_structure(
            predicted_structure
        )

        st.pyplot(
            figure,
            use_container_width=True
        )

        plt.close(
            figure
        )


        # ====================================================
        # DOWNLOAD RESULTS
        # ====================================================

        result_df = pd.DataFrame({
            "Position": range(
                1,
                len(sequence) + 1
            ),
            "Amino_Acid": list(
                sequence
            ),
            "Predicted_Structure": list(
                predicted_structure
            )
        })

        csv_data = result_df.to_csv(
            index=False
        )

        st.download_button(
            label="⬇️ Download Prediction CSV",
            data=csv_data,
            file_name=(
                "protein_secondary_structure_prediction.csv"
            ),
            mime="text/csv",
            use_container_width=True
        )


# ============================================================
# ABOUT SECTION
# ============================================================

st.markdown("---")

st.markdown(
    "## 📚 About the Project"
)

col1, col2, col3 = st.columns(3)

with col1:

    st.markdown("""
    ### 🤖 Machine Learning

    - Logistic Regression
    - Random Forest
    - SVM
    - KNN
    - XGBoost
    """)

with col2:

    st.markdown("""
    ### 🧠 Deep Learning

    - ANN
    - 1D CNN
    - BiLSTM
    """)

with col3:

    st.markdown("""
    ### 🧬 Structures

    **H** — Alpha Helix

    **E** — Beta Strand

    **C** — Coil
    """)


# ============================================================
# FOOTER
# ============================================================

st.markdown("""
<div class="footer">

🧬 <b>Protein Secondary Structure Prediction</b>

<br><br>

Machine Learning + Deep Learning

<br>

Kaggle Dataset | Streamlit Application

</div>
""", unsafe_allow_html=True)