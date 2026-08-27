import os
import re
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from collections import Counter

import sys

print("PYTHON USED BY APP:")
print(sys.executable)

import tensorflow as tf

print("TENSORFLOW VERSION:")
print(tf.__version__)
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
    padding: 2.2rem;
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
    background: white;
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

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

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
# AMINO ACID CONFIGURATION
# ============================================================

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"

AA_TO_INDEX = {
    aa: i
    for i, aa in enumerate(AMINO_ACIDS)
}

NUM_AA_CLASSES = 21

UNKNOWN_INDEX = 20

DEFAULT_WINDOW_SIZE = 9

DEFAULT_MAX_SEQUENCE_LENGTH = 512


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

        try:

            return joblib.load(
                config_path
            )

        except Exception as e:

            st.warning(
                f"Could not load model configuration: {e}"
            )

    return {
        "classes": [
            "C",
            "E",
            "H"
        ],
        "window_size":
            DEFAULT_WINDOW_SIZE,
        "max_sequence_length":
            DEFAULT_MAX_SEQUENCE_LENGTH
    }


# ============================================================
# LOAD ML MODEL
# ============================================================

@st.cache_resource
def load_ml_model(model_name):

    if model_name not in ML_MODELS:

        return None

    filename = ML_MODELS[
        model_name
    ]

    path = os.path.join(
        MODEL_DIR,
        filename
    )

    if not os.path.exists(path):

        return None

    return joblib.load(
        path
    )


# ============================================================
# LOAD DEEP LEARNING MODEL
# ============================================================

@st.cache_resource
def load_dl_model(model_name):

    try:

        if model_name not in DL_MODELS:
            return None

        filename = DL_MODELS[
            model_name
        ]

        path = os.path.join(
            MODEL_DIR,
            filename
        )

        if not os.path.exists(path):

            st.error(
                f"Model file not found: {path}"
            )

            return None

        model = tf.keras.models.load_model(
            path
        )

        return model

    except Exception as e:

        st.error(
            f"Error loading {model_name}: {e}"
        )

        st.exception(e)

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

    if not os.path.exists(path):

        return None

    try:

        return joblib.load(
            path
        )

    except Exception:

        return None


# ============================================================
# CHECK MODEL DIRECTORY
# ============================================================

def check_model_directory():

    if not os.path.exists(
        MODEL_DIR
    ):

        return False

    return True


# ============================================================
# CLEAN PROTEIN SEQUENCE
# ============================================================

def clean_sequence(sequence):

    if sequence is None:

        return ""

    sequence = sequence.strip()

    # Handle FASTA format
    if sequence.startswith(">"):

        lines = sequence.splitlines()

        # Remove FASTA header
        lines = [
            line.strip()
            for line in lines[1:]
        ]

        sequence = "".join(
            lines
        )

    else:

        # Remove whitespace
        sequence = re.sub(
            r"\s+",
            "",
            sequence
        )

    return sequence.upper()


# ============================================================
# VALIDATE PROTEIN SEQUENCE
# ============================================================

def validate_sequence(sequence):

    if not sequence:

        return (
            False,
            "Please enter a protein sequence."
        )

    invalid = sorted(
        set(sequence)
        - set(AMINO_ACIDS)
    )

    if invalid:

        return (
            False,
            "Invalid amino-acid symbols detected: "
            + ", ".join(invalid)
            + "."
        )

    return (
        True,
        ""
    )


# ============================================================
# ONE-HOT ENCODE ONE AMINO ACID
# ============================================================

def one_hot_encode_amino_acid(
    amino_acid
):

    vector = np.zeros(
        NUM_AA_CLASSES,
        dtype=np.float32
    )

    if amino_acid in AA_TO_INDEX:

        vector[
            AA_TO_INDEX[amino_acid]
        ] = 1.0

    else:

        vector[
            UNKNOWN_INDEX
        ] = 1.0

    return vector


# ============================================================
# CREATE 189-FEATURE WINDOWS
# ============================================================

def create_ml_windows(
    sequence,
    window_size=DEFAULT_WINDOW_SIZE
):

    if window_size % 2 == 0:

        raise ValueError(
            "Window size must be odd."
        )

    half = window_size // 2

    # Unknown/padding represented by X
    padded_sequence = (
        "X" * half
        + sequence
        + "X" * half
    )

    windows = []

    for i in range(
        len(sequence)
    ):

        window = padded_sequence[
            i:i + window_size
        ]

        encoded_window = np.array([
            one_hot_encode_amino_acid(
                amino_acid
            )
            for amino_acid in window
        ])

        flattened_window = (
            encoded_window.flatten()
        )

        windows.append(
            flattened_window
        )

    X = np.array(
        windows,
        dtype=np.float32
    )

    return X


# ============================================================
# GET EXPECTED FEATURES FROM MODEL
# ============================================================

def get_expected_features(model):

    # Pipeline itself
    if hasattr(
        model,
        "n_features_in_"
    ):

        return model.n_features_in_

    # Search pipeline steps
    if hasattr(
        model,
        "named_steps"
    ):

        for name, step in (
            model.named_steps.items()
        ):

            if hasattr(
                step,
                "n_features_in_"
            ):

                return step.n_features_in_

    return None


# ============================================================
# ML PREDICTION
# ============================================================

def predict_ml(
    model,
    sequence,
    window_size=DEFAULT_WINDOW_SIZE
):

    # --------------------------------------------------------
    # Create one-hot encoded windows
    # --------------------------------------------------------

    X = create_ml_windows(
        sequence,
        window_size
    )

    # --------------------------------------------------------
    # Validate feature count
    # --------------------------------------------------------

    expected_features = (
        get_expected_features(
            model
        )
    )

    if expected_features is not None:

        if X.shape[1] != expected_features:

            raise ValueError(
                f"Preprocessing generated "
                f"{X.shape[1]} features, but the "
                f"trained model expects "
                f"{expected_features} features."
            )

    # --------------------------------------------------------
    # Predict
    # --------------------------------------------------------

    predictions = model.predict(
        X
    )

    predictions = np.asarray(
        predictions
    )

    # --------------------------------------------------------
    # Convert probabilities to class IDs
    # --------------------------------------------------------

    if predictions.ndim > 1:

        # Binary probability
        if (
            predictions.shape[1] == 1
        ):

            predictions = (
                predictions.ravel() >= 0.5
            ).astype(int)

        else:

            predictions = np.argmax(
                predictions,
                axis=1
            )

    predictions = predictions.astype(
        int
    )

    # --------------------------------------------------------
    # Decode using saved encoder
    # --------------------------------------------------------

    encoder = load_encoder()

    if encoder is not None:

        try:

            labels = encoder.inverse_transform(
                predictions
            )

            return "".join(
                labels
            )

        except Exception:

            pass

    # --------------------------------------------------------
    # Fallback mapping
    # --------------------------------------------------------

    mapping = {
        0: "C",
        1: "E",
        2: "H"
    }

    return "".join(
        mapping.get(
            int(prediction),
            "C"
        )
        for prediction in predictions
    )


# ============================================================
# DL SEQUENCE ENCODING
# ============================================================

def prepare_dl_sequence(
    sequence,
    max_length
):

    encoded = np.array([
        AA_TO_INDEX.get(
            amino_acid,
            UNKNOWN_INDEX
        )
        for amino_acid in sequence
    ])

    # Truncate
    encoded = encoded[
        :max_length
    ]

    # Pad
    if len(encoded) < max_length:

        encoded = np.pad(
            encoded,
            (
                0,
                max_length - len(encoded)
            ),
            mode="constant",
            constant_values=UNKNOWN_INDEX
        )

    return encoded


# ============================================================
# GET DL INPUT INFORMATION
# ============================================================

def get_dl_input_shape(model):

    try:

        return model.input_shape

    except Exception:

        return None


# ============================================================
# DL PREDICTION
# ============================================================

def predict_dl(
    model,
    sequence,
    max_length
):

    # --------------------------------------------------------
    # Check model input shape
    # --------------------------------------------------------

    input_shape = get_dl_input_shape(
        model
    )

    # --------------------------------------------------------
    # Determine sequence length
    # --------------------------------------------------------

    actual_max_length = max_length

    if (
        input_shape is not None
        and len(input_shape) >= 2
        and isinstance(
            input_shape[1],
            int
        )
    ):

        actual_max_length = (
            input_shape[1]
        )

    # --------------------------------------------------------
    # Encode sequence
    # --------------------------------------------------------

    encoded = prepare_dl_sequence(
        sequence,
        actual_max_length
    )

    # --------------------------------------------------------
    # Create batch
    # --------------------------------------------------------

    X = np.expand_dims(
        encoded,
        axis=0
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    prediction = model.predict(
        X,
        verbose=0
    )

    prediction = np.asarray(
        prediction
    )

    # --------------------------------------------------------
    # Sequence-to-sequence output
    # Shape:
    # (1, sequence_length, classes)
    # --------------------------------------------------------

    if prediction.ndim == 3:

        class_ids = np.argmax(
            prediction[0],
            axis=-1
        )

        mapping = {
            0: "C",
            1: "E",
            2: "H"
        }

        labels = [
            mapping.get(
                int(class_id),
                "C"
            )
            for class_id in class_ids
        ]

        return "".join(
            labels[:len(sequence)]
        )

    # --------------------------------------------------------
    # Classification output
    # Shape:
    # (1, classes)
    # --------------------------------------------------------

    elif prediction.ndim == 2:

        class_id = np.argmax(
            prediction[0]
        )

        mapping = {
            0: "C",
            1: "E",
            2: "H"
        }

        predicted_class = mapping.get(
            int(class_id),
            "C"
        )

        # If the DL model predicts one
        # class for the entire sequence,
        # repeat it for visualization.
        return (
            predicted_class
            * len(sequence)
        )

    else:

        raise ValueError(
            "Unexpected Deep Learning "
            f"output shape: {prediction.shape}"
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

    total = len(
        structure
    )

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

            "Percentage":
                round(
                    percentage,
                    2
                )
        })

    return pd.DataFrame(
        data
    )


# ============================================================
# STRUCTURE VISUALIZATION
# ============================================================

def plot_structure(
    structure
):

    mapping = {

        "C": 1,

        "E": 2,

        "H": 3
    }

    values = [

        mapping.get(
            amino_acid,
            0
        )

        for amino_acid
        in structure
    ]

    fig, ax = plt.subplots(
        figsize=(14, 3)
    )

    ax.plot(
        range(
            1,
            len(values) + 1
        ),
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

    ax.set_ylabel(
        "Structure"
    )

    ax.set_title(
        "Predicted Secondary Structure",
        fontsize=15,
        fontweight="bold"
    )

    ax.grid(
        alpha=0.25
    )

    plt.tight_layout()

    return fig


# ============================================================
# HERO SECTION
# ============================================================

st.markdown("""
<div class="hero">

<h1>🧬 Protein Secondary Structure Predictor</h1>

<p>
Machine Learning & Deep Learning based prediction
of protein secondary structures.
</p>

<p>
Predict <b>Alpha Helix (H)</b>,
<b>Beta Strand (E)</b>, and
<b>Coil (C)</b> directly from an amino-acid sequence.
</p>

</div>
""", unsafe_allow_html=True)


# ============================================================
# MODEL DIRECTORY STATUS
# ============================================================

if not check_model_directory():

    st.error(
        "❌ Model folder not found."
    )

    st.info(
        "Make sure the folder "
        "'Protein_Secondary_Structure_Models' "
        "is located in the same directory as app.py."
    )

else:

    # ========================================================
    # SIDEBAR
    # ========================================================

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
            list(
                ML_MODELS.keys()
            )
        )

    else:

        model_name = st.sidebar.selectbox(
            "Select Deep Learning Model",
            list(
                DL_MODELS.keys()
            )
        )


    # ========================================================
    # MODEL STATUS
    # ========================================================

    st.sidebar.markdown("---")

    if model_type == "Machine Learning":

        selected_file = os.path.join(
            MODEL_DIR,
            ML_MODELS[model_name]
        )

    else:

        selected_file = os.path.join(
            MODEL_DIR,
            DL_MODELS[model_name]
        )

    if os.path.exists(
        selected_file
    ):

        st.sidebar.success(
            "✅ Model file found"
        )

    else:

        st.sidebar.error(
            "❌ Model file not found"
        )

        st.sidebar.caption(
            os.path.basename(
                selected_file
            )
        )


    # ========================================================
    # MAIN INPUT
    # ========================================================

    st.subheader(
        "🔬 Enter Protein Sequence"
    )

    st.write(
        "Enter a protein sequence using the standard "
        "20 amino-acid symbols. FASTA format is also supported."
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


    # ========================================================
    # PREDICTION
    # ========================================================

    if predict_button:

        # ----------------------------------------------------
        # Clean input
        # ----------------------------------------------------

        sequence = clean_sequence(
            sequence_input
        )

        # ----------------------------------------------------
        # Validate
        # ----------------------------------------------------

        valid, error = validate_sequence(
            sequence
        )

        if not valid:

            st.error(
                error
            )

        else:

            st.success(
                f"Valid protein sequence detected — "
                f"{len(sequence)} residues."
            )

            # ------------------------------------------------
            # Run model
            # ------------------------------------------------

            with st.spinner(
                f"Running {model_name} prediction..."
            ):

                try:

                    # ========================================
                    # MACHINE LEARNING
                    # ========================================

                    if (
                        model_type
                        == "Machine Learning"
                    ):

                        model = load_ml_model(
                            model_name
                        )

                        if model is None:

                            st.error(
                                f"{model_name} model "
                                "could not be loaded."
                            )

                            st.stop()

                        config = (
                            load_configuration()
                        )

                        window_size = config.get(
                            "window_size",
                            DEFAULT_WINDOW_SIZE
                        )

                        predicted_structure = (
                            predict_ml(
                                model,
                                sequence,
                                window_size
                            )
                        )

                    # ========================================
                    # DEEP LEARNING
                    # ========================================

                    else:

                        model = load_dl_model(
                            model_name
                        )

                        if model is None:

                            st.error(
                                f"{model_name} model "
                                "could not be loaded."
                            )

                            st.stop()

                        config = (
                            load_configuration()
                        )

                        max_length = config.get(
                            "max_sequence_length",
                            DEFAULT_MAX_SEQUENCE_LENGTH
                        )

                        predicted_structure = (
                            predict_dl(
                                model,
                                sequence,
                                max_length
                            )
                        )

                except Exception as e:

                    st.error(
                        "❌ Prediction failed."
                    )

                    st.exception(
                        e
                    )

                    st.stop()


            # =================================================
            # RESULTS
            # =================================================

            st.markdown(
                "## 🎯 Prediction Results"
            )

            stats = structure_statistics(
                predicted_structure
            )


            # =================================================
            # METRICS
            # =================================================

            col1, col2, col3, col4 = st.columns(
                4
            )

            col1.metric(
                "Sequence Length",
                len(sequence)
            )

            helix_percentage = stats.loc[
                stats["Code"] == "H",
                "Percentage"
            ].iloc[0]

            strand_percentage = stats.loc[
                stats["Code"] == "E",
                "Percentage"
            ].iloc[0]

            coil_percentage = stats.loc[
                stats["Code"] == "C",
                "Percentage"
            ].iloc[0]

            col2.metric(
                "Alpha Helix (H)",
                f"{helix_percentage}%"
            )

            col3.metric(
                "Beta Strand (E)",
                f"{strand_percentage}%"
            )

            col4.metric(
                "Coil (C)",
                f"{coil_percentage}%"
            )


            # =================================================
            # INPUT SEQUENCE
            # =================================================

            st.markdown(
                "### 🧬 Input Sequence"
            )

            st.code(
                sequence,
                language="text"
            )


            # =================================================
            # PREDICTED STRUCTURE
            # =================================================

            st.markdown(
                "### 🔮 Predicted Structure"
            )

            st.code(
                predicted_structure,
                language="text"
            )


            # =================================================
            # STRUCTURE TABLE
            # =================================================

            st.markdown(
                "### 📊 Structure Composition"
            )

            st.dataframe(
                stats,
                use_container_width=True,
                hide_index=True
            )


            # =================================================
            # VISUALIZATION
            # =================================================

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


            # =================================================
            # RESIDUE-BY-RESIDUE RESULTS
            # =================================================

            result_df = pd.DataFrame({

                "Position":
                    range(
                        1,
                        len(sequence) + 1
                    ),

                "Amino_Acid":
                    list(sequence),

                "Predicted_Structure":
                    list(
                        predicted_structure
                    )
            })

            st.markdown(
                "### 🔬 Residue-Level Prediction"
            )

            st.dataframe(
                result_df,
                use_container_width=True,
                hide_index=True
            )


            # =================================================
            # DOWNLOAD RESULTS
            # =================================================

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

col1, col2, col3 = st.columns(
    3
)


with col1:

    st.markdown("""
    ### 🤖 Machine Learning

    - Logistic Regression
    - Random Forest
    - SVM
    - KNN
    - XGBoost

    **Input:** 9-residue amino-acid
    windows with one-hot encoding.
    """)


with col2:

    st.markdown("""
    ### 🧠 Deep Learning

    - ANN
    - 1D CNN
    - BiLSTM

    Models operate on encoded
    protein sequence data.
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

Kaggle Protein Secondary Structure Dataset

</div>
""", unsafe_allow_html=True)