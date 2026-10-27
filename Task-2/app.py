"""
AI-Based Iris Flower Classification Using K-Nearest Neighbors
DecodeLabs Artificial Intelligence Project 2 — Streamlit Application
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from train import (
    DEFAULT_K_NEIGHBORS,
    DEFAULT_RANDOM_STATE,
    DEFAULT_TEST_SIZE,
    FEATURE_NAMES,
    FEATURE_RANGES,
    TARGET_NAMES,
    compute_k_elbow_curve,
    evaluate_model,
    load_iris_data,
    load_model,
    predict_sample,
    run_training_experiment,
    split_data,
    train_model,
    validate_input_measurements,
)

# Page configuration
st.set_page_config(
    page_title="Iris AI Classification | DecodeLabs P2",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-header {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
        padding: 24px 28px;
        border-radius: 14px;
        margin-bottom: 24px;
        border: 1px solid #334155;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
    }
    .badge {
        display: inline-block;
        background-color: #38BDF8;
        color: #0F172A;
        font-weight: 700;
        font-size: 0.75rem;
        padding: 4px 10px;
        border-radius: 9999px;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin-bottom: 8px;
    }
    .title-text {
        font-size: 1.85rem;
        font-weight: 800;
        color: #F8FAFC;
        margin: 0;
        line-height: 1.25;
    }
    .subtitle-text {
        color: #94A3B8;
        font-size: 0.95rem;
        margin-top: 6px;
        margin-bottom: 0;
    }
    .card-metric {
        background: #1E293B;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
    }
    .metric-value {
        font-size: 1.75rem;
        font-weight: 700;
        color: #38BDF8;
    }
    .metric-label {
        font-size: 0.82rem;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .prediction-box {
        background: linear-gradient(135deg, #064E3B 0%, #065F46 100%);
        border: 1px solid #10B981;
        border-radius: 12px;
        padding: 20px;
        color: #ECFDF5;
        margin-top: 15px;
    }
    .warning-box {
        background: #422006;
        border-left: 4px solid #F59E0B;
        padding: 12px 16px;
        border-radius: 6px;
        color: #FEF3C7;
        font-size: 0.88rem;
        margin-top: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Training model pipeline...")
def get_model_and_metrics():
    """Load or train pipeline and evaluate on the 80:20 split."""
    try:
        pipeline = load_model()
        X, y, _, _ = load_iris_data()
        X_train, X_test, y_train, y_test = split_data(
            X, y, test_size=DEFAULT_TEST_SIZE, random_state=DEFAULT_RANDOM_STATE, shuffle=True
        )
        metrics = evaluate_model(pipeline, X_test, y_test)
    except Exception:
        pipeline, metrics, X_train, X_test = run_training_experiment(
            test_size=DEFAULT_TEST_SIZE,
            random_state=DEFAULT_RANDOM_STATE,
            n_neighbors=DEFAULT_K_NEIGHBORS,
            save_artifact=True,
        )
        X, y, _, _ = load_iris_data()
        _, _, y_train, y_test = split_data(
            X, y, test_size=DEFAULT_TEST_SIZE, random_state=DEFAULT_RANDOM_STATE, shuffle=True
        )

    return pipeline, metrics, X_train, X_test, y_train, y_test


# Header banner
st.markdown(
    """
    <div class="main-header">
        <span class="badge">DecodeLabs • Artificial Intelligence Track</span>
        <h1 class="title-text">AI-Based Iris Flower Classification Using K-Nearest Neighbors</h1>
        <p class="subtitle-text">
            Supervised Machine Learning pipeline featuring scikit-learn Pipeline feature scaling,
            leakage prevention, and multi-class diagnostic evaluation.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Load pipeline and baseline metrics
pipeline, baseline_metrics, X_train, X_test, y_train, y_test = get_model_and_metrics()

# Navigation tabs
tab_pred, tab_metrics, tab_arch, tab_tuning = st.tabs(
    [
        "🌸 Species Classifier",
        "📊 Evaluation & Confusion Matrix",
        "📐 Architecture & IPO Blueprint",
        "⚙️ K-Tuning & Diagnostics",
    ]
)

# ==============================================================================
# TAB 1: SPECIES CLASSIFIER (INFERENCE)
# ==============================================================================
with tab_pred:
    st.subheader("Flower Measurements Input")
    st.write(
        "Enter botanical dimensions in centimeters (cm) or select a reference preset to observe classification."
    )

    # Preset selection helpers
    preset_cols = st.columns([1, 1, 1, 3])
    with preset_cols[0]:
        if st.button("Preset: Iris Setosa", use_container_width=True):
            st.session_state["sepal_len"] = 5.0
            st.session_state["sepal_wid"] = 3.5
            st.session_state["petal_len"] = 1.4
            st.session_state["petal_wid"] = 0.2
    with preset_cols[1]:
        if st.button("Preset: Iris Versicolor", use_container_width=True):
            st.session_state["sepal_len"] = 6.0
            st.session_state["sepal_wid"] = 2.8
            st.session_state["petal_len"] = 4.3
            st.session_state["petal_wid"] = 1.3
    with preset_cols[2]:
        if st.button("Preset: Iris Virginica", use_container_width=True):
            st.session_state["sepal_len"] = 6.8
            st.session_state["sepal_wid"] = 3.1
            st.session_state["petal_len"] = 5.6
            st.session_state["petal_wid"] = 2.1

    # Form with inputs
    input_c1, input_c2 = st.columns(2)

    with input_c1:
        sepal_length = st.number_input(
            "Sepal Length (cm)",
            min_value=0.1,
            max_value=25.0,
            value=float(st.session_state.get("sepal_len", 5.4)),
            step=0.1,
            help="Length of the outer flower sepals (typical range: 4.0 - 8.0 cm)",
        )
        sepal_width = st.number_input(
            "Sepal Width (cm)",
            min_value=0.1,
            max_value=25.0,
            value=float(st.session_state.get("sepal_wid", 3.4)),
            step=0.1,
            help="Width of the outer flower sepals (typical range: 2.0 - 4.5 cm)",
        )

    with input_c2:
        petal_length = st.number_input(
            "Petal Length (cm)",
            min_value=0.1,
            max_value=25.0,
            value=float(st.session_state.get("petal_len", 1.5)),
            step=0.1,
            help="Length of the inner flower petals (typical range: 1.0 - 7.0 cm)",
        )
        petal_width = st.number_input(
            "Petal Width (cm)",
            min_value=0.1,
            max_value=25.0,
            value=float(st.session_state.get("petal_wid", 0.4)),
            step=0.1,
            help="Width of the inner flower petals (typical range: 0.1 - 2.6 cm)",
        )

    predict_btn = st.button("🔍 Classify Flower Sample", type="primary", use_container_width=True)

    # Perform prediction
    sample_values = [sepal_length, sepal_width, petal_length, petal_width]

    try:
        validated = validate_input_measurements(sample_values)
        res = predict_sample(pipeline, validated)
        pred_species = res["class_name"].capitalize()
        pred_idx = res["class_index"]
        probs = res["probabilities"]

        species_icons = {
            "Setosa": "🌱 Iris setosa",
            "Versicolor": "🌿 Iris versicolor",
            "Virginica": "🌺 Iris virginica",
        }

        species_descriptions = {
            "Setosa": "Known for compact petal dimensions, distinctively smaller petal length & width.",
            "Versicolor": "Intermediate morphological features between Setosa and Virginica.",
            "Virginica": "Characterized by prominent, expansive petals and elongated sepals.",
        }

        st.markdown("---")
        res_col1, res_col2 = st.columns([1.2, 1])

        with res_col1:
            st.markdown(
                f"""
                <div class="prediction-box">
                    <span style="font-size:0.85rem; text-transform:uppercase; letter-spacing:0.05em; opacity:0.85;">Model Classification Result</span>
                    <h2 style="margin:4px 0 10px 0; color:#FFFFFF;">{species_icons.get(pred_species, pred_species)}</h2>
                    <p style="margin:0; font-size:0.95rem; opacity:0.95;">{species_descriptions.get(pred_species, '')}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Check if input is outside typical botanical bounds
            anomalies = []
            for name, val in zip(FEATURE_NAMES, sample_values):
                low, high = FEATURE_RANGES[name]
                if val < low or val > high:
                    anomalies.append(f"{name} ({val} cm is outside typical botanical range {low}-{high} cm)")

            if anomalies:
                st.markdown(
                    f"""
                    <div class="warning-box">
                        <strong>⚠️ Out-of-Distribution Warning:</strong><br>
                        {'<br>'.join(anomalies)}<br>
                        <em>KNN classifies based on nearest neighbors in feature space, but extreme inputs represent extrapolation.</em>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        with res_col2:
            st.markdown("#### Class Confidence Probabilities")
            prob_df = pd.DataFrame(
                {
                    "Species": [c.capitalize() for c in probs.keys()],
                    "Probability": list(probs.values()),
                }
            )
            st.bar_chart(prob_df.set_index("Species"), y="Probability", color="#38BDF8")

    except ValueError as e:
        st.error(f"Input Validation Error: {e}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

# ==============================================================================
# TAB 2: EVALUATION & CONFUSION MATRIX
# ==============================================================================
with tab_metrics:
    st.subheader("Model Validation & Evaluation Metrics")
    st.caption(
        "Evaluated on a held-out test split (80:20 ratio, 30 test samples, fixed seed = 42, shuffled)."
    )

    # Top summary metrics
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(
            f"""
            <div class="card-metric">
                <div class="metric-value">{baseline_metrics['accuracy'] * 100:.1f}%</div>
                <div class="metric-label">Test Accuracy</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            f"""
            <div class="card-metric">
                <div class="metric-value">{baseline_metrics['macro_precision']:.2f}</div>
                <div class="metric-label">Macro Precision</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m3:
        st.markdown(
            f"""
            <div class="card-metric">
                <div class="metric-value">{baseline_metrics['macro_recall']:.2f}</div>
                <div class="metric-label">Macro Recall</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m4:
        st.markdown(
            f"""
            <div class="card-metric">
                <div class="metric-value">{baseline_metrics['macro_f1']:.2f}</div>
                <div class="metric-label">Macro F1-Score</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    c_diag1, c_diag2 = st.columns([1.1, 1.2])

    with c_diag1:
        st.markdown("#### Diagnostic Tool: Confusion Matrix")
        fig_cm, ax_cm = plt.subplots(figsize=(5, 4.2))
        fig_cm.patch.set_facecolor("#0F172A")
        ax_cm.set_facecolor("#0F172A")

        sns.heatmap(
            baseline_metrics["confusion_matrix"],
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=[c.capitalize() for c in TARGET_NAMES],
            yticklabels=[c.capitalize() for c in TARGET_NAMES],
            cbar=False,
            ax=ax_cm,
            annot_kws={"size": 14, "weight": "bold", "color": "#0F172A"},
        )
        ax_cm.set_xlabel("Predicted Class", color="#F8FAFC", fontsize=11, labelpad=8)
        ax_cm.set_ylabel("Actual Class", color="#F8FAFC", fontsize=11, labelpad=8)
        ax_cm.tick_params(colors="#94A3B8")
        st.pyplot(fig_cm)
        plt.close(fig_cm)

    with c_diag2:
        st.markdown("#### Per-Class Diagnostic Report")
        report_rows = []
        for cls_name, vals in baseline_metrics["per_class"].items():
            report_rows.append(
                {
                    "Species": cls_name.capitalize(),
                    "Precision": f"{vals['precision']:.3f}",
                    "Recall": f"{vals['recall']:.3f}",
                    "F1-Score": f"{vals['f1_score']:.3f}",
                    "Support (N)": vals["support"],
                }
            )
        st.dataframe(pd.DataFrame(report_rows), use_container_width=True, hide_index=True)

        st.markdown(
            """
            <div class="warning-box">
                <strong>⚠️ Architectural Note — The Accuracy Mirage & Configuration Dependency:</strong><br>
                1. <strong>Split Sensitivity:</strong> The 100% accuracy achieved on this test set is a function of the 
                specific 80:20 random seed (42). Iris is a clean, well-separated benchmark dataset.<br>
                2. <strong>Real-World Generalization:</strong> Never claim a model is "perfect" or "production-grade" solely from a small benchmark test split.<br>
                3. <strong>Metric Trade-Offs:</strong> In production datasets with class imbalance or overlap, high accuracy often hides missed detections (False Negatives). Always inspect per-class Precision, Recall, and the harmonic mean (F1-score).
            </div>
            """,
            unsafe_allow_html=True,
        )

# ==============================================================================
# TAB 3: ARCHITECTURE & IPO BLUEPRINT
# ==============================================================================
with tab_arch:
    st.subheader("Architectural Paradigm: The IPO Framework")
    st.write(
        "DecodeLabs Project 2 bridges the transition from manual heuristic rules to algorithmic Supervised Learning."
    )

    ipo1, ipo2, ipo3 = st.columns(3)
    with ipo1:
        st.markdown(
            """
            <div class="card-metric" style="text-align:left; min-height: 240px;">
                <span class="badge" style="background:#F97316; color:#FFFFFF;">1. INPUT</span>
                <h4 style="color:#F8FAFC; margin-top:8px;">Iris Benchmark Domain</h4>
                <p style="color:#94A3B8; font-size:0.88rem;">
                    • 150 Balanced Samples<br>
                    • 3 Classes (Setosa, Versicolor, Virginica)<br>
                    • 4 Continuous Features (Sepal/Petal L&W)<br>
                    • <strong>StandardScaler:</strong> Zero mean & unit variance scaling.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with ipo2:
        st.markdown(
            """
            <div class="card-metric" style="text-align:left; min-height: 240px;">
                <span class="badge" style="background:#38BDF8; color:#0F172A;">2. PROCESS</span>
                <h4 style="color:#F8FAFC; margin-top:8px;">Supervised Pipeline</h4>
                <p style="color:#94A3B8; font-size:0.88rem;">
                    • <strong>Shuffle & Split:</strong> 80% Train, 20% Locked Test.<br>
                    • <strong>The Gatekeeper Rule:</strong> StandardScaler fitted exclusively on training data inside Pipeline.<br>
                    • <strong>KNN Engine:</strong> K=5 Minkowski metric (p=2 Euclidean distance).
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with ipo3:
        st.markdown(
            """
            <div class="card-metric" style="text-align:left; min-height: 240px;">
                <span class="badge" style="background:#10B981; color:#0F172A;">3. OUTPUT</span>
                <h4 style="color:#F8FAFC; margin-top:8px;">Validation & Inference</h4>
                <p style="color:#94A3B8; font-size:0.88rem;">
                    • <strong>Confusion Matrix:</strong> Tracks True Positives & False Alarms.<br>
                    • <strong>Precision & Recall:</strong> Balancing trust vs sensitivity.<br>
                    • <strong>F1 Score:</strong> Harmonic mean indicator.<br>
                    • Calibrated class probabilities.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("#### Prevention of Data Leakage (The Gatekeeper Rule)")
    st.info(
        """
        **Why is fitting the scaler on raw data a critical flaw?**
        If `StandardScaler.fit()` is called on the entire dataset prior to splitting, the test set's mean and 
        variance leak into the training representations. 
        In DecodeLabs Project 2, scikit-learn's `Pipeline([('scaler', StandardScaler()), ('classifier', KNeighborsClassifier(n_neighbors=5))])` 
        ensures test data remains strictly quarantined behind a lock until inference time.
        """
    )

# ==============================================================================
# TAB 4: K-TUNING & DIAGNOSTICS (ELBOW CURVE)
# ==============================================================================
with tab_tuning:
    st.subheader("Tuning the Engine: Choosing 'K'")
    st.write(
        "Understand the trade-off between noise sensitivity (overfitting at low K) and generic underfitting (high K)."
    )

    tune_col1, tune_col2 = st.columns([1.5, 1])

    with tune_col1:
        max_k = st.slider("Select Maximum K range to evaluate:", min_value=5, max_value=30, value=20)
        elbow_df = compute_k_elbow_curve(
            X_train, y_train, X_test, y_test, k_range=range(1, max_k + 1)
        )

        fig_k, ax_k = plt.subplots(figsize=(7, 4))
        fig_k.patch.set_facecolor("#0F172A")
        ax_k.set_facecolor("#0F172A")

        ax_k.plot(
            elbow_df["k"],
            elbow_df["error_rate"],
            color="#38BDF8",
            marker="o",
            linestyle="-",
            linewidth=2,
            label="Error Rate (1 - Accuracy)",
        )
        ax_k.axvline(x=5, color="#F59E0B", linestyle="--", label="Project Default (K=5)")

        ax_k.set_title("K-Value vs Error Rate (Elbow Diagnostic)", color="#F8FAFC", fontsize=12)
        ax_k.set_xlabel("Value of K (Number of Neighbors)", color="#F8FAFC", fontsize=10)
        ax_k.set_ylabel("Test Error Rate", color="#F8FAFC", fontsize=10)
        ax_k.tick_params(colors="#94A3B8")
        ax_k.grid(True, color="#334155", linestyle=":", alpha=0.6)
        ax_k.legend(facecolor="#1E293B", edgecolor="#334155", labelcolor="#F8FAFC")

        st.pyplot(fig_k)
        plt.close(fig_k)

    with tune_col2:
        st.markdown("#### The Proximity Principle")
        st.markdown(
            """
            - **K = 1 (Overfitting):** The decision boundary tightly hugs every training point, making it highly sensitive to outliers and measurement noise.
            - **K = 5 (Optimal Sweet Spot):** Smooths out individual point anomalies while retaining crisp boundaries between separable species.
            - **High K (Underfitting):** Over-averages neighboring votes, diluting minority or boundary nuances and causing underfitting.
            """
        )

st.markdown("---")
st.caption(
    "DecodeLabs Industrial Training Kit 2026 • AI Project 2 • Built strictly adhering to the technical specification."
)
