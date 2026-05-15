import os
import threading
import subprocess
from typing import Optional

import streamlit as st

import sys

# ----------------------------------------------------------------------
# Repository layout:
#   REPO_ROOT/                  (project root)
#       frontend/
#           app.py             (this file)
#       models/                 (trained models go here)
#       outputs/
#           analysis_outputs/   (plots go here)
#       src/                    (prediction/training code)
# ----------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))      # frontend/
REPO_ROOT = os.path.dirname(PROJECT_ROOT)                      # repo root
sys.path.insert(0, REPO_ROOT)                                  # so `src.*` imports work

from src.predict import load_model, predict_from_csv

# ----------------------------------------------------------------------
# Model candidates – check both frontend/ and repo root
# ----------------------------------------------------------------------
MODEL_CANDIDATES = [
    # Under frontend/ (if someone placed them there)
    os.path.join(PROJECT_ROOT, "models", "global_model.pth"),
    os.path.join(PROJECT_ROOT, "models", "global_model.pkl"),
    os.path.join(PROJECT_ROOT, "global_model.pth"),
    os.path.join(PROJECT_ROOT, "global_model.pkl"),
    # Under repo root (the expected location after training)
    os.path.join(REPO_ROOT, "models", "global_model.pth"),
    os.path.join(REPO_ROOT, "models", "global_model.pkl"),
    os.path.join(REPO_ROOT, "global_model.pth"),
    os.path.join(REPO_ROOT, "global_model.pkl"),
]


def resolve_model_path() -> Optional[str]:
    """Return the first existing model file from the candidate list."""
    for p in MODEL_CANDIDATES:
        if os.path.exists(p):
            return p
    return None


# ----------------------------------------------------------------------
# Plot candidates (same dual‑location logic)
# ----------------------------------------------------------------------
PLOT_CANDIDATES = [
    # Repo root first (most likely)
    os.path.join(REPO_ROOT, "analysis_outputs", "training_results.png"),
    os.path.join(REPO_ROOT, "analysis_outputs", "federated_accuracy_plot.png"),
    os.path.join(REPO_ROOT, "analysis_outputs", "federated_metrics_plot.png"),
    os.path.join(REPO_ROOT, "outputs", "analysis_outputs", "training_results.png"),
    os.path.join(REPO_ROOT, "outputs", "analysis_outputs", "federated_accuracy_plot.png"),
    os.path.join(REPO_ROOT, "outputs", "analysis_outputs", "federated_metrics_plot.png"),
    # Fallback: frontend/ (unlikely but harmless)
    os.path.join(PROJECT_ROOT, "analysis_outputs", "training_results.png"),
    os.path.join(PROJECT_ROOT, "analysis_outputs", "federated_accuracy_plot.png"),
    os.path.join(PROJECT_ROOT, "analysis_outputs", "federated_metrics_plot.png"),
    os.path.join(PROJECT_ROOT, "outputs", "analysis_outputs", "training_results.png"),
    os.path.join(PROJECT_ROOT, "outputs", "analysis_outputs", "federated_accuracy_plot.png"),
    os.path.join(PROJECT_ROOT, "outputs", "analysis_outputs", "federated_metrics_plot.png"),
]


def find_first_existing(paths: list[str]) -> Optional[str]:
    """Return the first path from the list that exists on disk."""
    for p in paths:
        if os.path.exists(p):
            return p
    return None


# ----------------------------------------------------------------------
# Subprocess helper – run from REPO_ROOT so relative paths in training
# scripts resolve correctly (e.g., saving to models/ or outputs/).
# Also set PYTHONPATH so that `src.*` can be imported inside the subprocess.
# ----------------------------------------------------------------------
def _run_subprocess(cmd: list[str]) -> str:
    env = os.environ.copy()
    # Ensure the repo root is on PYTHONPATH for the subprocess
    env["PYTHONPATH"] = REPO_ROOT + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.Popen(
        cmd,
        cwd=REPO_ROOT,            # <-- run from repo root, not frontend/
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
        env=env,
    )
    out_lines: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        out_lines.append(line)
    proc.wait()
    return "".join(out_lines)


# ----------------------------------------------------------------------
# Artifact existence checks
# ----------------------------------------------------------------------
def get_artifact_debug() -> dict[str, list[str]]:
    return {
        "model_candidates": MODEL_CANDIDATES,
        "plot_candidates": PLOT_CANDIDATES,
    }


@st.cache_resource(show_spinner=False)
def backend_artifacts_exist() -> bool:
    model_ok = resolve_model_path() is not None
    plots_ok = any(os.path.exists(p) for p in PLOT_CANDIDATES)
    return model_ok and plots_ok


# ----------------------------------------------------------------------
# Backend launch (training)
# ----------------------------------------------------------------------
def ensure_model_and_run_training(run_mode: str, num_rounds: int) -> tuple[bool, str]:
    """Kick off backend processes (best‑effort)."""
    if run_mode == "Use existing model":
        ok = resolve_model_path() is not None
        return ok, "Using existing trained model artifacts."

    logs: list[str] = []

    if run_mode == "Centralized (baseline)":
        try:
            logs.append(_run_subprocess(["python", "src/centralized.py"]))
        except Exception as e:
            logs.append(str(e))
        ok = resolve_model_path() is not None
        return ok, "\n".join(logs)

    if run_mode == "Federated (FL)":
        env_rounds = str(num_rounds)

        def run_server():
            os.environ["FL_NUM_ROUNDS"] = env_rounds
            logs.append(_run_subprocess(["python", "src/server.py"]))

        def run_client(client_id: str):
            os.environ["FL_NUM_ROUNDS"] = env_rounds
            logs.append(_run_subprocess(["python", "src/client.py", client_id]))

        st.info("Starting Federated Learning (may take a few minutes)...")

        t1 = threading.Thread(target=run_server, daemon=True)
        t2 = threading.Thread(target=run_client, args=("0",), daemon=True)
        t3 = threading.Thread(target=run_client, args=("1",), daemon=True)
        for t in (t1, t2, t3):
            t.start()
        for t in (t1, t2, t3):
            t.join()

        ok = resolve_model_path() is not None
        return ok, "\n".join(logs)

    return False, "Unknown run mode."


# ----------------------------------------------------------------------
# Streamlit UI
# ----------------------------------------------------------------------
st.set_page_config(page_title="Diabetes FL Dashboard", layout="wide")

st.title("Diabetes Prediction — Federated Learning")

with st.sidebar:
    st.header("Backend")

    run_mode = st.selectbox(
        "When app runs, what should it do?",
        options=["Use existing model", "Centralized (baseline)", "Federated (FL)"],
        index=0,
    )

    num_rounds = st.slider("Federated rounds (best‑effort)", min_value=1, max_value=20, value=10, step=1)

    run_backend_btn = st.button("Run backend processes", type="primary")

    st.divider()

    uploaded_csv = st.file_uploader(
        "Upload a CSV for prediction (must contain 8 feature columns)",
        type=["csv"],
        accept_multiple_files=False,
    )

    threshold = st.slider("Prediction threshold", min_value=0.01, max_value=0.99, value=0.35, step=0.01)
    include_labels = st.checkbox("CSV has Outcome labels (evaluation only; optional)", value=False)

# ----------------------------------------------------------------------
# Tabs
# ----------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["Dashboard", "Single sample", "CSV / Batch"])

# ----- Tab 1: Dashboard -----
with tab1:
    st.subheader("Training artifacts")

    if backend_artifacts_exist():
        st.success("Model artifacts and plots found.")
    else:
        dbg = get_artifact_debug()
        st.warning("Artifacts not found. Showing attempted artifact paths.")
        st.code("Model candidates:\n" + "\n".join(dbg["model_candidates"]), language="text")
        st.code("Plot candidates:\n" + "\n".join(dbg["plot_candidates"]), language="text")

    colA, colB, colC = st.columns(3)

    with colA:
        png1 = find_first_existing([
            os.path.join(REPO_ROOT, "analysis_outputs", "federated_accuracy_plot.png"),
            os.path.join(REPO_ROOT, "outputs", "analysis_outputs", "federated_accuracy_plot.png"),
            os.path.join(PROJECT_ROOT, "analysis_outputs", "federated_accuracy_plot.png"),
            os.path.join(PROJECT_ROOT, "outputs", "analysis_outputs", "federated_accuracy_plot.png"),
        ])
        if png1:
            st.image(png1, use_container_width=True)
        else:
            st.caption("federated_accuracy_plot.png not found")

    with colB:
        png2 = find_first_existing([
            os.path.join(REPO_ROOT, "analysis_outputs", "federated_metrics_plot.png"),
            os.path.join(REPO_ROOT, "outputs", "analysis_outputs", "federated_metrics_plot.png"),
            os.path.join(PROJECT_ROOT, "analysis_outputs", "federated_metrics_plot.png"),
            os.path.join(PROJECT_ROOT, "outputs", "analysis_outputs", "federated_metrics_plot.png"),
        ])
        if png2:
            st.image(png2, use_container_width=True)
        else:
            st.caption("federated_metrics_plot.png not found")

    with colC:
        png3 = find_first_existing([
            os.path.join(REPO_ROOT, "analysis_outputs", "training_results.png"),
            os.path.join(REPO_ROOT, "outputs", "analysis_outputs", "training_results.png"),
            os.path.join(PROJECT_ROOT, "analysis_outputs", "training_results.png"),
            os.path.join(PROJECT_ROOT, "outputs", "analysis_outputs", "training_results.png"),
        ])
        if png3:
            st.image(png3, use_container_width=True)
        else:
            st.caption("training_results.png not found")

    st.divider()

    st.subheader("Run backend")
    if run_backend_btn:
        st.info("Running backend...")
        ok, backend_logs = ensure_model_and_run_training(run_mode, num_rounds)
        st.success("Backend run complete." if ok else "Backend run complete, but model artifacts were not detected.")
        st.text(backend_logs[-4000:])

# ----- Tab 2: Single prediction -----
with tab2:
    st.subheader("Single prediction")
    st.caption("Enter feature values in the PIMA diabetes format (8 features).")

    features = {
        "Pregnancies": st.number_input("Pregnancies", value=6.0),
        "Glucose": st.number_input("Glucose", value=148.0),
        "BloodPressure": st.number_input("BloodPressure", value=72.0),
        "SkinThickness": st.number_input("SkinThickness", value=35.0),
        "Insulin": st.number_input("Insulin", value=0.0),
        "BMI": st.number_input("BMI", value=33.6),
        "DiabetesPedigreeFunction": st.number_input("DiabetesPedigreeFunction", value=0.627),
        "Age": st.number_input("Age", value=50.0),
    }

    predict_btn = st.button("Predict", type="primary")

    if predict_btn:
        model_path = resolve_model_path()
        if model_path is None:
            st.error("Model not found. Run backend training first.")
        else:
            model = load_model(model_path)
            from src.predict import predict_single

            ordered = [
                float(features["Pregnancies"]),
                float(features["Glucose"]),
                float(features["BloodPressure"]),
                float(features["SkinThickness"]),
                float(features["Insulin"]),
                float(features["BMI"]),
                float(features["DiabetesPedigreeFunction"]),
                float(features["Age"]),
            ]
            result = predict_single(model, ordered, threshold=threshold)
            st.metric("Class", result["class"])
            st.metric("Probability", f"{result['probability']:.4f}")

# ----- Tab 3: CSV predictions -----
with tab3:
    st.subheader("CSV predictions")
    st.caption("Upload a CSV that contains the 8 PIMA feature columns (+ optional Outcome).")

    do_predict_btn = st.button("Run predictions", type="primary")

    if do_predict_btn:
        if uploaded_csv is None:
            st.error("Upload a CSV in the sidebar first.")
        else:
            model_path = resolve_model_path()
            if model_path is None:
                st.error("Model not found. Run backend training first.")
            else:
                model = load_model(model_path)
                tmp_path = os.path.join(REPO_ROOT, "outputs", "uploaded_prediction_input.csv")
                os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
                with open(tmp_path, "wb") as f:
                    f.write(uploaded_csv.getbuffer())

                try:
                    if include_labels:
                        from src.predict import predict_from_csv_with_analysis
                        df_out = predict_from_csv_with_analysis(
                            model, tmp_path, threshold=threshold, has_labels=True
                        )
                    else:
                        df_out = predict_from_csv(model, tmp_path, threshold=threshold)

                    st.dataframe(df_out.head(200), use_container_width=True)
                    st.success("Prediction complete.")
                except Exception as e:
                    st.exception(e)

st.caption("Tip: for federated learning, server + clients are separate processes; interactive prompts may block clients.")