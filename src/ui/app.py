import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# Add project root to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.processing.extractor import extract_signal
from src.processing.integrator import (
    calculate_auc_simpson,
    calculate_auc_spline,
    calculate_auc_trapezoidal,
)

st.set_page_config(page_title="Spectroscopy AUC Extractor", layout="wide")

# Initialize session state for results
if "results" not in st.session_state:
    st.session_state.results = []
if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()

st.title("🔬 Spectroscopy AUC Extractor")

# Sidebar settings
with st.sidebar:
    st.header("⚙️ Configuration")

    auc_method = st.selectbox(
        "AUC Method",
        ["Spline", "Trapezoidal", "Simpson"],
        index=0,
        help="Mathematical method for integration",
    )

    col1, col2 = st.columns(2)
    with col1:
        x_scaling = st.number_input(
            "X Scale", value=1.0, step=0.1, help="Physical unit per pixel"
        )
    with col2:
        y_scaling = st.number_input(
            "Y Scale", value=1.0, step=0.1, help="Intensity multiplier"
        )

    auc_unit = st.text_input(
        "AUC Unit", value="cm²", help="Unit to display for AUC results"
    )

    st.divider()
    if st.button("Clear Results", type="secondary"):
        st.session_state.results = []
        st.session_state.processed_files = set()
        st.rerun()

# Main processing area
st.subheader("📁 Data Input")
input_col, action_col = st.columns([3, 1])

with input_col:
    input_mode = st.radio(
        "Selection Mode", ["File Upload", "Folder Path"], horizontal=True
    )

    files_to_process = []
    if input_mode == "File Upload":
        uploaded_files = st.file_uploader(
            "Choose TIFF files",
            type=["tif"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )
        if uploaded_files:
            files_to_process = uploaded_files
    else:
        folder_path = st.text_input(
            "Enter Folder Path", value="data", label_visibility="collapsed"
        )
        if os.path.isdir(folder_path):
            files_to_process = [
                os.path.join(folder_path, f)
                for f in os.listdir(folder_path)
                if f.endswith(".tif")
            ]
            st.caption(f"Found {len(files_to_process)} .tif files")
        else:
            st.error("Invalid folder path")

with action_col:
    st.write(" ")  # Padding
    st.write(" ")
    st.markdown(
        """
        <style>
        div.stButton > button:first-child {
            background: linear-gradient(135deg, #4caf50 0%, #2e7d32 100%);
            color: white;
            border: none;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            font-weight: bold;
            transition: all 0.3s ease;
        }
        div.stButton > button:first-child:hover {
            box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
            transform: translateY(-2px);
        }
        </style>
    """,
        unsafe_allow_html=True,
    )
    process_btn = st.button(
        "🚀 Process Files", type="primary", use_container_width=True
    )


def process_single_file(file_obj, x_scale, y_scale, method):
    is_upload = hasattr(file_obj, "name")
    if is_upload:  # UploadedFile
        processed_path = file_obj
        display_name = file_obj.name
    else:
        processed_path = file_obj
        display_name = os.path.basename(file_obj)

    try:
        # Seek stream to start before passing natively to PIL
        if is_upload:
            processed_path.seek(0)

        with Image.open(processed_path) as img:
            img_arr = np.array(img)

        # Reset stream for extractor
        if is_upload:
            processed_path.seek(0)

        x, y = extract_signal(processed_path, x_scaling=x_scale, y_scaling=y_scale)

        if method == "Trapezoidal":
            auc = calculate_auc_trapezoidal(x, y)
        elif method == "Simpson":
            auc = calculate_auc_simpson(x, y)
        else:
            auc = calculate_auc_spline(x, y)

        return {"Filename": display_name, "AUC": auc, "x": x, "y": y, "image": img_arr}
    except Exception as e:
        st.error(f"Error processing {display_name}: {e}")
        return None


if process_btn and files_to_process:
    new_results = []
    progress_bar = st.progress(0)

    for i, f in enumerate(files_to_process):
        res = process_single_file(f, x_scaling, y_scaling, auc_method)
        if res:
            new_results.append(res)
        progress_bar.progress((i + 1) / len(files_to_process))

    st.session_state.results = new_results
    progress_bar.empty()

# Display Results
if st.session_state.results:
    res_df = pd.DataFrame(st.session_state.results)[["Filename", "AUC"]]
    res_df.columns = ["Filename", f"AUC ({auc_unit})"]

    main_col, viz_col = st.columns([1, 2])

    with main_col:
        st.subheader("📊 Results")
        st.dataframe(res_df, use_container_width=True, hide_index=True)

        csv = res_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "💾 Download CSV",
            csv,
            "auc_results.csv",
            "text/csv",
            use_container_width=True,
        )

    with viz_col:
        st.subheader("📈 Visualization")
        selected_file = st.selectbox(
            "Select file to visualize",
            [r["Filename"] for r in st.session_state.results],
            label_visibility="collapsed",
        )

        selected_res = next(
            r for r in st.session_state.results if r["Filename"] == selected_file
        )

        st.metric(
            label="Area Under Curve (AUC)",
            value=f"{selected_res['AUC']:.4f} {auc_unit}",
        )

        img_col, plot_col = st.columns([1, 2])

        with img_col:
            st.markdown("**Original Image**")
            st.image(selected_res["image"], use_container_width=True, clamp=True)

        with plot_col:
            st.markdown("**Signal Profile**")
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(selected_res["x"], selected_res["y"], color="#2e7d32", linewidth=2)
            ax.fill_between(
                selected_res["x"], selected_res["y"], alpha=0.3, color="#4caf50"
            )
            ax.set_title(f"Signal: {selected_file}", fontsize=12)
            ax.set_xlabel("Physical Coordinate", fontsize=10)
            ax.set_ylabel("Intensity", fontsize=10)

            # Set explicit scales
            if len(selected_res["x"]) > 0:
                ax.set_xlim(0, np.max(selected_res["x"]))
            if len(selected_res["y"]) > 0:
                ax.set_ylim(0, np.max(selected_res["y"]) * 1.1)

            ax.grid(True, linestyle="--", alpha=0.6)

            # Style improvements
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            st.pyplot(fig)
else:
    if process_btn and not files_to_process:
        st.warning("No files selected to process.")
