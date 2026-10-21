"""
04_pipeline_backend_frontend/backend/app.py
------
SFA-CA Interactive Web Application & Forensic Attribution Dashboard.

Features:
1. Multi-Format Input: Direct text paste OR file uploads (.txt, .pdf, .docx, .md).
2. Math & Notation-Aware Sanitizer: Removes LaTeX equations and calibrates confidence to eliminate false-positive AI flags.
3. Automatic Script Cluster Routing: Detects language script family (Latin, Cyrillic, Greek, Arabic, Hanzi, Unseen) and engages matching LoRA adapter.
4. Softmax Attribution Engine: Calculates % AI-Generated vs % Human-Written and specific 7-LLM generator provenance.
5. Interactive Plotly Charts: Live probability distribution bar chart and radar chart.
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import sfaca_paths  # noqa: E402,F401  (puts every member src/ folder on sys.path)

import os
import sys
import re
from typing import Tuple
import torch
import torch.nn.functional as F
import pandas as pd


import plotly.express as px
import plotly.graph_objects as go
import gradio as gr
from transformers import AutoTokenizer

# Ensure src directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from sanitizer import TextSanitizer, calibrate_probabilities
from file_parser import parse_uploaded_file
from data_loader import ID2LABEL
from inference import InferenceEngine, detect_script_cluster

# Global Model & Tokenizer Singleton
_ENGINE = None

def get_engine():
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = InferenceEngine()
        _ENGINE.load_from_dir()
    return _ENGINE


def analyze_text_attribution(raw_input_text: str, uploaded_file=None):
    """
    Main Analysis Function invoked by Gradio UI.
    """
    engine = get_engine()

    try:
        # 1. Parse File Upload if provided
        text_content = raw_input_text
        file_info = "Direct Text Input"

        if uploaded_file is not None:
            try:
                text_content, file_type = parse_uploaded_file(uploaded_file.name)
                file_info = f"Uploaded File: {os.path.basename(uploaded_file.name)} ({file_type})"
            except Exception as e:
                return f"Error reading uploaded file: {e}", "", "", None, None, ""

        if not text_content or not text_content.strip():
            return "Please enter text or upload a document to analyze.", "", "", None, None, ""

        # 2. Math & Notation Sanitization
        sanitizer = TextSanitizer()
        clean_text, meta = sanitizer.sanitize(text_content)

        # 3. Detect Script Cluster & Activate Adapter
        active_cluster = detect_script_cluster(clean_text)
        script_display = {
            "latin": "Latin Script (English, Spanish, German, French, etc.)",
            "cyrillic": "Cyrillic Script (Russian, Ukrainian, Bulgarian)",
            "greek": "Greek Script (Hellenic)",
            "arabic": "Arabic / Semitic Script (Arabic, Persian, Urdu)",
            "hanzi": "CJK / Hanzi Script (Chinese, Japanese, Korean)",
        }.get(active_cluster, "Latin Script")

        # 4. Model Forward Pass
        result = engine.predict(clean_text, active_cluster)
        class_probs = result["probabilities"]

        # 5. Apply Math Notation Confidence Calibration
        calibrated_probs = calibrate_probabilities(class_probs, meta["ndi"], human_class="human")

        # Calculate Binary AI vs Human %
        human_pct = round(calibrated_probs.get("human", 0.0) * 100, 2)
        ai_pct = round(100.0 - human_pct, 2)

        verdict_badge = f"### 🤖 Verdict: **{ai_pct}% AI-Generated** | 👤 **{human_pct}% Human-Written**"
        if human_pct > 60.0:
            verdict_badge = f"### 👤 Verdict: **{human_pct}% Human-Written** (Likely Authentic Human Text)"

        # Most Likely Generator Model
        ai_only_probs = {k: v for k, v in calibrated_probs.items() if k != "human"}
        top_model = max(ai_only_probs, key=ai_only_probs.get)
        top_model_prob = round(ai_only_probs[top_model] * 100, 2)

        model_attribution_summary = (
            f"**Primary Attributed Generator:** `{top_model}` ({top_model_prob}% Confidence)\n\n"
            f"**Active SFA-CA Adapter:** `{active_cluster.upper()}` ({script_display})\n\n"
            f"**Source:** {file_info}"
        )

        # Notation Sanitization Status Badge
        notation_status = "✅ No heavy math/code notation detected. Standard calibration applied."
        if meta["has_math_notation"]:
            notation_status = (
                f"⚠️ **Math / LaTeX Notation Detected!** (Notation Density Index: `{meta['ndi'] * 100:.1f}%`)\n"
                f"Extracted `{len(meta['extracted_math'])}` LaTeX math equations. "
                f"Confidence score calibrated to prevent false-positive AI flags."
            )

        # 6. Build Interactive Plotly Bar Chart
        df_chart = pd.DataFrame([
            {"Class": k, "Probability (%)": round(v * 100, 2)}
            for k, v in calibrated_probs.items()
        ]).sort_values(by="Probability (%)", ascending=True)

        fig_bar = px.bar(
            df_chart,
            x="Probability (%)",
            y="Class",
            orientation="h",
            color="Probability (%)",
            color_continuous_scale="Viridis",
            title="Attribution Probability Distribution across Candidate Authors"
        )
        fig_bar.update_layout(showlegend=False, height=350, margin=dict(l=20, r=20, t=40, b=20))

        # 7. Build Interactive Radar Chart
        categories = list(calibrated_probs.keys())
        values = [round(calibrated_probs[c] * 100, 2) for c in categories]
        categories.append(categories[0])
        values.append(values[0])

        fig_radar = go.Figure(data=go.Scatterpolar(
            r=values,
            theta=categories,
            fill="toself",
            line_color="#636EFA"
        ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            showlegend=False,
            title="Stylistic Provenance Radar Profile",
            height=350,
            margin=dict(l=40, r=40, t=40, b=20)
        )

        return verdict_badge, model_attribution_summary, notation_status, fig_bar, fig_radar, clean_text[:500]
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"### ❌ Error during analysis: {e}", "", "", None, None, f"Exception details: {e}"


# Define Gradio Interface
def build_gradio_app():
    custom_theme = gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="blue",
        neutral_hue="slate"
    )

    with gr.Blocks(theme=custom_theme, title="SFA-CA Multilingual AI Text Attribution System") as app:
        gr.Markdown(
            """
            # 🌐 Multilingual AI Text Attribution & Forensic Provenance System (SFA-CA)
            ### *Script-Family-Aware Contrastive Adaptation for Multi-LLM Detection & Authorship Attribution*
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                input_text = gr.Textbox(
                    label="Input Text (Paste text in any language: English, Russian, Arabic, Greek, Chinese, etc.)",
                    placeholder="Paste sample article or text snippet here...",
                    lines=8
                )
                file_upload = gr.File(
                    label="OR Upload Document File (.txt, .pdf, .docx, .md)",
                    file_types=[".txt", ".pdf", ".docx", ".md"]
                )
                analyze_btn = gr.Button("🔍 Analyze Text Attribution", variant="primary", size="lg")

            with gr.Column(scale=1):
                verdict_output = gr.Markdown("### 🤖 Analysis Output Will Appear Here")
                model_summary_output = gr.Markdown()
                notation_output = gr.Markdown()

        with gr.Row():
            with gr.Column():
                bar_chart_output = gr.Plot(label="Attribution Probabilities (%)")
            with gr.Column():
                radar_chart_output = gr.Plot(label="Provenance Radar Profile")

        with gr.Accordion("📄 View Sanitized Text Preview", open=False):
            sanitized_preview = gr.Textbox(label="Text Preview (After Math & Notation Sanitization)", lines=4)

        analyze_btn.click(
            fn=analyze_text_attribution,
            inputs=[input_text, file_upload],
            outputs=[
                verdict_output,
                model_summary_output,
                notation_output,
                bar_chart_output,
                radar_chart_output,
                sanitized_preview
            ]
        )

        gr.Markdown(
            """
            ---
            *Developed for B.Tech / B.E. Final Research Project — Script-Family-Aware Contrastive Adaptation (SFA-CA).*
            """
        )

    return app


demo = build_gradio_app()

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
