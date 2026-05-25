import base64
import io
import webbrowser
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import precision_recall_curve, PrecisionRecallDisplay

# ---------- CONFIGURATION ----------
METRICS_CSV = Path("outputs/tables/model_metrics_gold.csv")
SHAP_IMP_CSV = Path("outputs/tables/xgboost_shap_importance.csv")
BEESWARM_PNG = Path("outputs/figures/xgboost_shap_beeswarm.png")
GOLD_TEST_CSV = Path("data/interim/gold_test.csv")
XGBOOST_MODEL = Path("models/xgboost.joblib")
REPORT_FILE = Path("outputs/reports/evaluation_report.html")

# Professional minimal color palette
PRIMARY_COLOR = "#2c3e50"
BAR_COLORS = ["#2c3e50", "#455a64", "#78909c", "#b0bec5", "#cfd8dc"]
BG_COLOR = "#ffffff"
TEXT_COLOR = "#333333"

# Setup Seaborn styling for professional look
sns.set_theme(style="whitegrid", rc={
    "axes.edgecolor": "#e0e0e0",
    "axes.facecolor": BG_COLOR,
    "grid.color": "#f0f0f0",
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "Roboto", "Helvetica Neue", "Arial", "sans-serif"],
    "text.color": TEXT_COLOR,
    "axes.labelcolor": TEXT_COLOR,
    "xtick.color": TEXT_COLOR,
    "ytick.color": TEXT_COLOR
})

def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight", transparent=False, facecolor=BG_COLOR)
    buf.seek(0)
    img = base64.b64encode(buf.read()).decode()
    buf.close()
    plt.close(fig)
    return img

def plot_hbar(metrics, metric_col, label):
    sub = metrics.sort_values(metric_col, ascending=True)
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    
    bars = ax.barh(
        sub["Model"], sub[metric_col],
        color=BAR_COLORS[: len(sub)], 
        height=0.5,
        edgecolor="none"
    )
    
    # Remove top and right spines
    sns.despine(left=True, bottom=False)
    
    # Add values at the end of bars
    for bar, val in zip(bars, sub[metric_col]):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                f"{val*100:.1f}%", va="center", ha="left", 
                fontsize=10, fontweight="500", color=PRIMARY_COLOR)
                
    ax.set_title(label, fontsize=14, fontweight="600", color=PRIMARY_COLOR, pad=15)
    ax.set_xlim(0, max(sub[metric_col]) * 1.15)
    ax.tick_params(labelsize=11, left=False)
    ax.grid(axis='y') # Remove horizontal grid lines
    
    plt.tight_layout()
    return fig_to_base64(fig)

def plot_confusion_matrix(cm, model_name):
    fig, ax = plt.subplots(figsize=(4, 4))
    
    # Custom minimal colormap
    cmap = sns.light_palette(PRIMARY_COLOR, as_cmap=True)
    
    sns.heatmap(
        cm, annot=True, fmt="d", cmap=cmap, cbar=False, ax=ax,
        xticklabels=["Neg", "Pos"], yticklabels=["Neg", "Pos"],
        annot_kws={"fontsize": 14, "fontweight": "600"},
        linewidths=1, linecolor='white'
    )
    
    ax.set_title(model_name, fontweight="600", fontsize=13, color=PRIMARY_COLOR, pad=15)
    ax.set_xlabel("Predicted", fontsize=11, color="#666")
    ax.set_ylabel("Actual", fontsize=11, color="#666")
    ax.tick_params(labelsize=11, length=0)
    
    plt.tight_layout()
    return fig_to_base64(fig)

def build_html_report():
    metrics = pd.read_csv(METRICS_CSV)
    ranked = metrics.sort_values("F1-Score", ascending=False).reset_index(drop=True)

    # ---------- ranking rows ----------
    rank_rows = ""
    for i, row in ranked.iterrows():
        is_best = i == 0
        cls = ' class="best"' if is_best else ""
        rank_rows += (
            f"<tr{cls}>"
            f"<td>{i+1}</td>"
            f"<td class='model'>{row['Model']}</td>"
            f"<td>{row['F1-Score']*100:.1f}%</td>"
            f"<td>{row['Precision']*100:.1f}%</td>"
            f"<td>{row['Recall']*100:.1f}%</td>"
            f"</tr>"
        )

    # ---------- metric bar charts ----------
    chart_img = {}
    for col, name in [("F1-Score", "F1-Score"),
                       ("Precision", "Precision"),
                       ("Recall", "Recall")]:
        chart_img[col] = plot_hbar(ranked, col, name)

    # ---------- confusion matrices ----------
    cm_html = ""
    for _, row in ranked.iterrows():
        cm = np.array([[int(row["TN"]), int(row["FP"])],
                       [int(row["FN"]), int(row["TP"])]])
        img = plot_confusion_matrix(cm, row["Model"])
        cm_html += (
            f"<div class='figure'>"
            f"<img src='data:image/png;base64,{img}' alt='Confusion Matrix {row['Model']}'>"
            f"<div class='caption'>{row['Model']}</div>"
            f"</div>"
        )

    # ---------- SHAP ----------
    shap_imp_img = ""
    if SHAP_IMP_CSV.exists():
        imp = pd.read_csv(SHAP_IMP_CSV).sort_values("mean_abs_shap", ascending=True)
        fig, ax = plt.subplots(figsize=(7, 5))
        
        ax.barh(imp["feature"], imp["mean_abs_shap"],
                color=PRIMARY_COLOR, height=0.6, edgecolor="none")
        
        sns.despine(left=True, bottom=False)
        ax.set_title("XGBoost Global Feature Importance", fontsize=14, fontweight="600", color=PRIMARY_COLOR, pad=15)
        ax.tick_params(labelsize=11, left=False)
        ax.set_xlabel("Mean Absolute SHAP Value", fontsize=11, color="#666")
        ax.grid(axis='y')
        
        plt.tight_layout()
        shap_imp_img = fig_to_base64(fig)

    beeswarm_img = ""
    if BEESWARM_PNG.exists():
        beeswarm_img = base64.b64encode(BEESWARM_PNG.read_bytes()).decode()

    # ---------- Precision-Recall curve ----------
    pr_curve_img = ""
    if GOLD_TEST_CSV.exists() and XGBOOST_MODEL.exists():
        gold = pd.read_csv(GOLD_TEST_CSV)
        model = joblib.load(XGBOOST_MODEL)
        
        # FIX: Dynamically fetch the exact feature names the model was trained on
        if hasattr(model, 'feature_names_in_'):
            expected_cols = model.feature_names_in_
        else:
            expected_cols = model.get_booster().feature_names
            
        X_g = gold[expected_cols].astype(float)
        y_g = gold["label"].astype(int).values
        
        proba = model.predict_proba(X_g)[:, 1]
        precision, recall, _ = precision_recall_curve(y_g, proba)

        fig, ax = plt.subplots(figsize=(8, 5))
        
        display = PrecisionRecallDisplay(precision=precision, recall=recall)
        display.plot(ax=ax, color=PRIMARY_COLOR, linewidth=2.5)
        
        sns.despine()
        ax.set_title("Precision-Recall Curve (XGBoost)", fontsize=14, fontweight="600", color=PRIMARY_COLOR, pad=15)
        ax.tick_params(labelsize=11)
        ax.fill_between(recall, precision, alpha=0.1, color=PRIMARY_COLOR)
        
        plt.tight_layout()
        pr_curve_img = fig_to_base64(fig)

    # ---------- Build HTML ----------
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>BOLA Risk Prediction - Evaluation Report</title>
    <style>
        :root {{
            --bg-color: #f8f9fa;
            --container-bg: #ffffff;
            --text-main: #333333;
            --text-muted: #6c757d;
            --border-color: #dee2e6;
            --accent-color: #2c3e50;
            --highlight-bg: #f8f9fa;
            --font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
        }}

        body {{
            font-family: var(--font-family);
            background-color: var(--bg-color);
            color: var(--text-main);
            line-height: 1.6;
            margin: 0;
            padding: 40px 20px;
        }}

        .container {{
            max-width: 1040px;
            margin: 0 auto;
            background: var(--container-bg);
            padding: 50px 70px;
            border-radius: 12px;
            box-shadow: 0 4px 24px rgba(0, 0, 0, 0.04);
            border: 1px solid var(--border-color);
        }}

        .header {{
            text-align: center;
            margin-bottom: 50px;
            padding-bottom: 30px;
            border-bottom: 2px solid #f0f0f0;
        }}

        .header h1 {{
            font-size: 32px;
            font-weight: 700;
            color: var(--accent-color);
            margin: 0 0 12px 0;
            letter-spacing: -0.5px;
        }}

        .header p {{
            font-size: 14px;
            color: var(--text-muted);
            margin: 0;
            text-transform: uppercase;
            letter-spacing: 1.5px;
            font-weight: 600;
        }}

        .section {{
            margin-bottom: 60px;
        }}

        .section-header {{
            display: flex;
            align-items: center;
            margin-bottom: 30px;
        }}

        .section h2 {{
            font-size: 20px;
            font-weight: 600;
            color: var(--accent-color);
            margin: 0;
            letter-spacing: -0.2px;
            white-space: nowrap;
        }}

        .section-header::after {{
            content: "";
            flex: 1;
            margin-left: 24px;
            height: 1px;
            background-color: #e9ecef;
        }}

        .table-wrap {{
            overflow-x: auto;
            border-radius: 8px;
            border: 1px solid var(--border-color);
            box-shadow: 0 2px 8px rgba(0,0,0,0.02);
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 15px;
            text-align: right;
        }}

        th, td {{
            padding: 16px 24px;
            border-bottom: 1px solid var(--border-color);
        }}

        th {{
            background-color: #f8f9fa;
            color: var(--text-muted);
            font-weight: 600;
            text-transform: uppercase;
            font-size: 12px;
            letter-spacing: 0.5px;
        }}

        th:first-child, td:first-child, td.model {{
            text-align: left;
        }}

        tr:last-child td {{
            border-bottom: none;
        }}

        tr.best {{
            background-color: #f8fbff;
            font-weight: 600;
            color: #0d6efd;
        }}
        
        tr:hover:not(.best) {{
            background-color: var(--highlight-bg);
        }}

        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 30px;
            align-items: start;
        }}

        .grid-3 {{
            grid-template-columns: repeat(3, 1fr);
        }}

        .figure {{
            background: #fff;
            padding: 24px;
            border-radius: 10px;
            border: 1px solid var(--border-color);
            box-shadow: 0 4px 12px rgba(0,0,0,0.02);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
            display: flex;
            flex-direction: column;
            align-items: center;
        }}
        
        .figure:hover {{
            transform: translateY(-2px);
            box-shadow: 0 8px 24px rgba(0,0,0,0.06);
        }}

        .figure img {{
            width: 100%;
            height: auto;
            display: block;
            border-radius: 4px;
        }}

        .caption {{
            font-size: 13px;
            color: var(--text-muted);
            text-align: center;
            margin-top: 16px;
            font-weight: 500;
        }}
        
        .full-width {{
            grid-column: 1 / -1;
            max-width: 700px;
            margin: 0 auto;
        }}

        @media (max-width: 900px) {{
            .grid-3 {{ grid-template-columns: 1fr; }}
        }}

        @media (max-width: 768px) {{
            .container {{ padding: 30px 20px; }}
            .grid {{ grid-template-columns: 1fr; }}
            .full-width {{ max-width: 100%; }}
        }}
    </style>
</head>

<body>
    <div class="container">
        <div class="header">
            <h1>BOLA Risk Prediction</h1>
            <p>Evaluation Report &bull; Silver-Train / Gold-Test</p>
        </div>

        <div class="section">
            <div class="section-header">
                <h2>Model Performance</h2>
            </div>
            <div class="table-wrap">
                <table>
                    <thead>
                        <tr>
                            <th>Rank</th>
                            <th>Model</th>
                            <th>F1-Score</th>
                            <th>Precision</th>
                            <th>Recall</th>
                        </tr>
                    </thead>
                    <tbody>
                        {rank_rows}
                    </tbody>
                </table>
            </div>
        </div>

        <div class="section">
            <div class="section-header">
                <h2>Metric Comparison</h2>
            </div>
            <div class="grid grid-3">
                <div class="figure">
                    <img src="data:image/png;base64,{chart_img['F1-Score']}" alt="F1-Score Comparison">
                    <div class="caption">F1-Score Comparison</div>
                </div>
                <div class="figure">
                    <img src="data:image/png;base64,{chart_img['Precision']}" alt="Precision Comparison">
                    <div class="caption">Precision Comparison</div>
                </div>
                <div class="figure">
                    <img src="data:image/png;base64,{chart_img['Recall']}" alt="Recall Comparison">
                    <div class="caption">Recall Comparison</div>
                </div>
            </div>
        </div>

        <div class="section">
            <div class="section-header">
                <h2>Precision-Recall Curve</h2>
            </div>
            <div class="figure full-width">
                <img src="data:image/png;base64,{pr_curve_img}" alt="Precision-Recall Curve">
                <div class="caption">Precision-Recall Curve across Thresholds (XGBoost)</div>
            </div>
        </div>

        <div class="section">
            <div class="section-header">
                <h2>Confusion Matrices</h2>
            </div>
            <div class="grid">
                {cm_html}
            </div>
        </div>

        <div class="section">
            <div class="section-header">
                <h2>SHAP Explainability (XGBoost)</h2>
            </div>
            <div class="grid">
                <div class="figure">
                    {"<img src='data:image/png;base64," + shap_imp_img + "' alt='Feature Importance'>" if shap_imp_img else ""}
                    <div class="caption">Feature Importance</div>
                </div>
                <div class="figure">
                    {"<img src='data:image/png;base64," + beeswarm_img + "' alt='SHAP Beeswarm'>" if beeswarm_img else ""}
                    <div class="caption">SHAP Beeswarm Distribution</div>
                </div>
            </div>
        </div>
    </div>
</body>
</html>"""

    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text(html, encoding="utf-8")
    print(f"Report saved to {REPORT_FILE}")
    webbrowser.open(REPORT_FILE.resolve().as_uri())
    print("Report opened in your browser.")

if __name__ == "__main__":
    build_html_report()