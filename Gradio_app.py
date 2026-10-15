import os
import time
import torch
import gradio as gr
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from dotenv import load_dotenv

load_dotenv()

# ── CONFIGURATION & MODEL LOADING ──────────────────────────────────────────
MODEL_REPO_ID = os.getenv("HF_REPO_ID", "Parthiv-Jethva/bankintent-ai")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Loading BankIntent AI model from Hugging Face: {MODEL_REPO_ID} on {DEVICE}...")
try:
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_REPO_ID)
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_REPO_ID)
    model.to(DEVICE)
    model.eval()
    id2label = {int(k): v for k, v in model.config.id2label.items()}
    print(f"[OK] Model successfully loaded with {len(id2label)} intent classes.")
except Exception as e:
    print(f"Error loading model from {MODEL_REPO_ID}: {e}")
    local_path = "best_model"
    if os.path.exists(local_path):
        print(f"Falling back to local checkpoint: {local_path}...")
        tokenizer = DistilBertTokenizer.from_pretrained(local_path)
        model = DistilBertForSequenceClassification.from_pretrained(local_path).to(DEVICE)
        model.eval()
        id2label = {int(k): v for k, v in model.config.id2label.items()}
        print(f"[OK] Loaded local checkpoint {local_path} with {len(id2label)} intent classes.")
    else:
        raise e

# ── DEPARTMENT ROUTING LOGIC ───────────────────────────────────────────────
def get_routing_info(intent: str):
    """Maps a raw Banking77 intent into an operational department, priority, and action."""
    intent_lower = intent.lower()

    if any(k in intent_lower for k in ["lost", "stolen", "compromised", "fraud", "blocked", "declined"]):
        return {
            "dept": "Fraud Prevention & Card Security",
            "priority": "High priority",
            "priority_color": "#b91c1c",
            "action": "Immediate card lockdown / manual security review required."
        }
    elif any(k in intent_lower for k in ["card", "pin", "contactless", "virtual"]):
        return {
            "dept": "Card Services & Logistics",
            "priority": "Standard priority",
            "priority_color": "#1d4ed8",
            "action": "Card dispatch, replacement status, or PIN reset service."
        }
    elif any(k in intent_lower for k in ["transfer", "standing", "beneficiary", "cancel_transfer", "direct_debit"]):
        return {
            "dept": "Domestic & International Payments",
            "priority": "Medium priority",
            "priority_color": "#b45309",
            "action": "Payment tracking, recall request, or SWIFT/SEPA clearance."
        }
    elif any(k in intent_lower for k in ["exchange", "rate", "usd", "eur", "currency", "fiat"]):
        return {
            "dept": "Foreign Exchange (Forex) & Cross-Border",
            "priority": "Standard priority",
            "priority_color": "#047857",
            "action": "Currency conversion rates, markup verification, or fee inquiry."
        }
    elif any(k in intent_lower for k in ["fee", "charge", "refund", "reverted"]):
        return {
            "dept": "Billing, Disputes & Fee Resolution",
            "priority": "Medium priority",
            "priority_color": "#b45309",
            "action": "Surcharge assessment and transaction dispute ticket."
        }
    elif any(k in intent_lower for k in ["verify", "identity", "source_of_funds", "passcode", "age"]):
        return {
            "dept": "KYC, Identity & Compliance",
            "priority": "Medium priority",
            "priority_color": "#6d28d9",
            "action": "Document verification, PEP/AML check, or account unlock."
        }
    else:
        return {
            "dept": "General Account Servicing",
            "priority": "Standard priority",
            "priority_color": "#475569",
            "action": "Standard customer inquiry triage and FAQ assistance."
        }

# ── UI HELPERS ─────────────────────────────────────────────────────────────
EMPTY_HTML = """
<div class="bi-empty">
    <div class="bi-empty-title">No query classified yet</div>
    <div class="bi-empty-text">Type a customer message or pick an example to see the detected intent and where it should be routed.</div>
</div>
"""

EMPTY_DIST_HTML = """
<div class="bi-dist-box">
    <div class="bi-dist-label">Top Predicted Intents</div>
    <div style="text-align: center; padding: 22px 16px;">
        <div class="bi-empty-title">No distribution to display</div>
        <div class="bi-empty-text">Top predicted intents with confidence percentages and progress bars will appear here.</div>
    </div>
</div>
"""

def build_distribution_html(top_names, top_scores):
    """Builds clean HTML distribution where progress bar is displayed after intent and percentage."""
    items_html = []
    for name, score in zip(top_names, top_scores):
        pretty = name.replace("_", " ").title()
        pct = float(score) * 100
        items_html.append(f"""
        <div class="bi-dist-item">
            <div class="bi-dist-header">
                <span class="bi-dist-title">{pretty}</span>
                <span class="bi-dist-pct">{pct:.1f}%</span>
            </div>
            <div class="bi-dist-bar-bg">
                <div class="bi-dist-bar-fill" style="width: {pct:.1f}%;"></div>
            </div>
        </div>
        """)
    return f'''<div class="bi-dist-box">
        <div class="bi-dist-label">Top Predicted Intents</div>
        {"".join(items_html)}
    </div>'''

# ── INFERENCE PIPELINE ─────────────────────────────────────────────────────
def predict_intent(query: str, top_k: int = 5):
    """Runs sequence classification on query and returns structured intent predictions."""
    if not query or not query.strip():
        return (
            "<div class='bi-empty'><div class='bi-empty-title'>Enter a banking query to classify</div></div>",
            EMPTY_DIST_HTML,
            "0.00 s"
        )

    start_time = time.perf_counter()
    k_val = min(int(top_k), len(id2label))

    inputs = tokenizer(
        query.strip(),
        return_tensors="pt",
        max_length=128,
        padding="max_length",
        truncation=True
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)[0]

    top_indices = torch.topk(probs, k=k_val)
    top_scores = [float(s) for s in top_indices.values.cpu().numpy()]
    top_names = [id2label[int(idx)] for idx in top_indices.indices.cpu().numpy()]
    best_id = int(top_indices.indices.cpu().numpy()[0])
    best_intent_raw = id2label[best_id]
    best_conf = float(top_scores[0]) * 100

    dist_html = build_distribution_html(top_names, top_scores)
    elapsed_sec = time.perf_counter() - start_time
    best_intent_pretty = best_intent_raw.replace("_", " ").title()
    routing = get_routing_info(best_intent_raw)

    # Confidence colour (readable on light backgrounds)
    conf_color = "#15803d" if best_conf >= 75 else ("#b45309" if best_conf >= 45 else "#b91c1c")
    conf_label = "High confidence" if best_conf >= 75 else ("Moderate confidence" if best_conf >= 45 else "Low confidence")
    pc = routing["priority_color"]

    # Result card HTML
    html_card = f"""
    <div class="bi-result">
        <div class="bi-result-top">
            <div>
                <div class="bi-label">Detected intent</div>
                <div class="bi-intent">{best_intent_pretty}</div>
                <code class="bi-code">{best_intent_raw}</code>
            </div>
            <span class="bi-pill" style="color:{pc}; background:{pc}14; border-color:{pc}40;">{routing['priority']}</span>
        </div>

        <div class="bi-conf">
            <div class="bi-conf-row">
                <span class="bi-conf-name">{conf_label}</span>
                <span class="bi-conf-val" style="color:{conf_color};">{best_conf:.1f}%</span>
            </div>
            <div class="bi-bar"><div class="bi-bar-fill" style="width:{best_conf:.1f}%; background:{conf_color};"></div></div>
        </div>

        <div class="bi-route" style="border-left-color:{pc};">
            <div class="bi-label">Recommended routing</div>
            <div class="bi-route-dept">{routing['dept']}</div>
            <div class="bi-route-action">{routing['action']}</div>
        </div>
    </div>
    """

    return html_card, dist_html, f"{elapsed_sec:.2f} s"

# ── CURATED SAMPLES ────────────────────────────────────────────────────────
SAMPLE_QUERIES = [
    ["I lost my credit card while travelling in Paris, please freeze it right away!", 5],
    ["Why was my card payment declined at the supermarket checkout?", 5],
    ["I want to exchange 500 USD into Euros inside the mobile app.", 5],
    ["How many days does an international wire transfer take to arrive?", 5],
    ["I see an unauthorized charge on my monthly account statement.", 5],
    ["How do I generate and activate a disposable virtual card for online shopping?", 5],
    ["I entered the wrong PIN three times and now my card is blocked.", 5],
    ["Can you help me set up an automatic recurring top-up for my balance?", 5],
    ["What documents do I need to submit to verify my proof of address?", 5]
]

# ── GRADIO APP INTERFACE ───────────────────────────────────────────────────
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {
    color-scheme: light;
    --bi-bg: #dde5ee;
    --bi-surface: #ffffff;
    --bi-surface-2: #eef2f7;
    --bi-border: #c3cedb;
    --bi-border-strong: #a9b8c9;
    --bi-ink: #0b2038;
    --bi-text: #1e3350;
    --bi-muted: #4b5f78;
    --bi-accent: #1e40af;
    --bi-accent-soft: #e0e8fb;
    --bi-radius: 10px;
}

body, .gradio-container {
    background: var(--bi-bg) !important;
    font-family: 'IBM Plex Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif !important;
    color: var(--bi-text);
}
.gradio-container {
    max-width: 1120px !important;
    margin: 0 auto !important;
}

/* Header */
.bi-header {
    background: linear-gradient(135deg, #0b2038 0%, #16365e 100%);
    border: 1px solid #0b2038;
    border-radius: var(--bi-radius);
    padding: 28px 32px 24px;
    margin-bottom: 18px;
    text-align: center;
}
.bi-header h1 {
    margin: 0 auto;
    font-size: 28px;
    font-weight: 700;
    letter-spacing: -0.02em;
    color: #ffffff !important;
    text-align: center;
}
.bi-header p {
    margin: 6px auto 0;
    font-size: 15px;
    color: #c7d6ea !important;
    max-width: 640px;
    line-height: 1.55;
    text-align: center;
}
.bi-stats {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin-top: 22px;
}
.bi-stat {
    background: rgba(255, 255, 255, 0.08);
    border: 1px solid rgba(255, 255, 255, 0.18);
    border-radius: 8px;
    padding: 12px 14px;
    min-width: 0;
    text-align: center;
    text-decoration: none;
    display: block;
    color: inherit;
    transition: background 0.2s ease, border-color 0.2s ease, transform 0.15s ease;
}
a.bi-stat:hover {
    background: rgba(255, 255, 255, 0.16);
    border-color: rgba(255, 255, 255, 0.38);
    transform: translateY(-2px);
}
.bi-stat-val {
    font-size: 19px;
    font-weight: 700;
    color: #ffffff !important;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.bi-stat-name {
    font-size: 12.5px;
    color: #b4c6de !important;
    margin-top: 2px;
}

/* Section headings */
.bi-section {
    font-size: 15px;
    font-weight: 600;
    color: var(--bi-ink);
    margin: 4px 0 10px;
}
.bi-block-heading {
    font-size: 17.5px;
    font-weight: 700;
    color: var(--bi-ink);
    margin: 22px 0 10px;
    display: flex;
    align-items: center;
    gap: 9px;
    letter-spacing: -0.01em;
    line-height: 1;
}
.bi-block-logo {
    font-size: 20px;
    line-height: 1;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    vertical-align: middle;
}
.bi-block-text {
    display: inline-flex;
    align-items: center;
    line-height: 1.25;
}

/* Result card */
.bi-result {
    background: var(--bi-surface);
    border: 1px solid var(--bi-border);
    border-radius: var(--bi-radius);
    padding: 22px;
    box-shadow: 0 1px 3px rgba(11, 32, 56, 0.10);
}
.bi-result-top {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    gap: 12px;
}
.bi-label {
    font-size: 12.5px;
    font-weight: 500;
    color: var(--bi-muted);
    margin-bottom: 4px;
}
.bi-intent {
    font-size: 24px;
    font-weight: 700;
    color: var(--bi-ink);
    letter-spacing: -0.015em;
    line-height: 1.2;
}
.bi-code {
    display: inline-block;
    margin-top: 8px;
    font-family: 'IBM Plex Mono', ui-monospace, monospace;
    font-size: 12.5px;
    color: var(--bi-accent);
    background: var(--bi-accent-soft);
    padding: 2px 8px;
    border-radius: 6px;
}
.bi-pill {
    border: 1px solid;
    padding: 4px 12px;
    border-radius: 9999px;
    font-size: 12.5px;
    font-weight: 600;
    white-space: nowrap;
}
.bi-conf { margin-top: 20px; }
.bi-conf-row {
    display: flex;
    justify-content: space-between;
    font-size: 13px;
    margin-bottom: 6px;
}
.bi-conf-name { color: var(--bi-muted); font-weight: 500; }
.bi-conf-val { font-weight: 700; }
.bi-bar {
    height: 8px;
    background: #d5deea;
    border-radius: 9999px;
    overflow: hidden;
}
.bi-bar-fill {
    height: 100%;
    border-radius: 9999px;
    transition: width 0.4s ease;
}
.bi-route {
    margin-top: 20px;
    background: var(--bi-surface-2);
    border: 1px solid var(--bi-border);
    border-left: 4px solid var(--bi-accent);
    border-radius: 8px;
    padding: 14px 16px;
}
.bi-route-dept {
    font-size: 15.5px;
    font-weight: 600;
    color: var(--bi-ink);
}
.bi-route-action {
    font-size: 13.5px;
    color: var(--bi-text);
    margin-top: 4px;
    line-height: 1.5;
}

/* Empty state */
.bi-empty {
    padding: 36px 24px;
    text-align: center;
    background: var(--bi-surface);
    border: 1px dashed var(--bi-border-strong);
    border-radius: var(--bi-radius);
}
.bi-empty-title { font-size: 15px; font-weight: 600; color: var(--bi-ink); }
.bi-empty-text { font-size: 13.5px; color: var(--bi-muted); margin-top: 6px; line-height: 1.5; }

/* Custom Confidence Distribution: Intent + Percentage, then Progress bar */
.bi-dist-box {
    background: var(--bi-surface);
    border: 1px solid var(--bi-border);
    border-radius: var(--bi-radius);
    padding: 16px 20px;
    box-shadow: 0 1px 3px rgba(11, 32, 56, 0.10);
    display: flex;
    flex-direction: column;
    gap: 13px;
}
.bi-dist-label {
    font-size: 13.5px;
    font-weight: 600;
    color: var(--bi-ink);
    margin-bottom: 2px;
}
.bi-dist-item {
    display: flex;
    flex-direction: column;
    gap: 5px;
}
.bi-dist-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.bi-dist-title {
    font-size: 14px;
    font-weight: 600;
    color: var(--bi-ink);
}
.bi-dist-pct {
    font-size: 13.5px;
    font-weight: 700;
    color: var(--bi-accent);
}
.bi-dist-bar-bg {
    height: 7px;
    background: #d5deea;
    border-radius: 9999px;
    overflow: hidden;
}
.bi-dist-bar-fill {
    height: 100%;
    background: var(--bi-accent);
    border-radius: 9999px;
    transition: width 0.3s ease;
}

/* Specs tab */
.bi-spec-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 14px;
    margin-bottom: 18px;
}
.bi-spec {
    background: var(--bi-surface);
    border: 1px solid var(--bi-border);
    border-radius: var(--bi-radius);
    padding: 18px;
}
.bi-spec-val { font-size: 26px; font-weight: 700; color: var(--bi-accent); }
.bi-spec-name { font-size: 13px; color: var(--bi-muted); margin-top: 2px; }
.bi-pipeline {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 14px;
    margin-top: 8px;
}
.bi-step {
    background: var(--bi-surface);
    border: 1px solid var(--bi-border);
    border-radius: var(--bi-radius);
    padding: 16px 18px;
}
.bi-step h4 { margin: 0 0 6px; font-size: 14.5px; font-weight: 600; color: var(--bi-ink); }
.bi-step p { margin: 0; font-size: 13.5px; line-height: 1.55; color: var(--bi-text); }

/* Intent chips */
.bi-chips { display: flex; flex-wrap: wrap; gap: 8px; }
.bi-chip {
    font-family: 'IBM Plex Mono', ui-monospace, monospace;
    font-size: 12.5px;
    color: var(--bi-text);
    background: var(--bi-surface);
    border: 1px solid var(--bi-border);
    border-radius: 6px;
    padding: 4px 10px;
}

/* Gradio component polish: force readable text on every surface */
.gradio-container .block, .gradio-container .form {
    background: var(--bi-surface) !important;
    border-color: var(--bi-border) !important;
    border-radius: var(--bi-radius) !important;
}
.gradio-container .block-title, .gradio-container .block-label,
.gradio-container label, .gradio-container label span,
.gradio-container span[data-testid="block-info"] {
    color: var(--bi-ink) !important;
    background: transparent !important;
    font-weight: 600;
}
.gradio-container textarea, .gradio-container input[type="text"], .gradio-container input[type="number"] {
    background: #ffffff !important;
    color: var(--bi-ink) !important;
    border: 1px solid var(--bi-border-strong) !important;
}
.gradio-container textarea::placeholder, .gradio-container input::placeholder {
    color: var(--bi-muted) !important;
    opacity: 1;
}
.gradio-container input[type="range"] { accent-color: var(--bi-accent); }

/* Examples table */
.gradio-container .examples,
.gradio-container .examples.block,
.gradio-container .examples .table-wrap,
.gradio-container .table-wrap,
.gradio-container .examples table,
.gradio-container .examples table th,
.gradio-container .examples table td {
    border: none !important;
    box-shadow: none !important;
}
.gradio-container table, .gradio-container .table-wrap, .gradio-container .examples {
    background: var(--bi-surface) !important;
    color: var(--bi-ink) !important;
}
.gradio-container table th {
    background: var(--bi-surface-2) !important;
    color: var(--bi-ink) !important;
}
.gradio-container table td, .gradio-container table td * {
    color: var(--bi-text) !important;
    background: transparent !important;
}
.gradio-container table tbody tr { background: var(--bi-surface) !important; }
.gradio-container table tbody tr:hover, .gradio-container table tbody tr:hover td {
    background: var(--bi-accent-soft) !important;
}
.gradio-container .gallery-item, .gradio-container .gallery-item * { color: var(--bi-text) !important; }

/* Label (confidence distribution) */
.gradio-container .output-class, .gradio-container .confidence-set,
.gradio-container .confidence-set *, .gradio-container .label-wrap, .gradio-container .label-wrap * {
    color: var(--bi-ink) !important;
}
.gradio-container .output-class { background: var(--bi-accent-soft) !important; }

/* Tabs */
.gradio-container .tab-container button, .gradio-container .tab-wrapper button, .gradio-container [role="tab"] {
    color: var(--bi-muted) !important;
    font-weight: 600;
}
.gradio-container [role="tab"][aria-selected="true"] {
    color: var(--bi-accent) !important;
}

/* Markdown (spec table) */
.gradio-container .prose, .gradio-container .prose p, .gradio-container .prose td,
.gradio-container .prose th, .gradio-container .prose li, .gradio-container .prose strong {
    color: var(--bi-text) !important;
}
.gradio-container .prose a { color: var(--bi-accent) !important; }
.gradio-container .prose code { background: var(--bi-accent-soft) !important; color: var(--bi-accent) !important; }

/* Buttons */
.gradio-container button.primary {
    background: var(--bi-accent) !important;
    border-color: var(--bi-accent) !important;
    color: #ffffff !important;
    font-weight: 600;
}
.gradio-container button.primary:hover { background: #172f86 !important; }
.gradio-container button.secondary {
    background: #ffffff !important;
    border: 1px solid var(--bi-border-strong) !important;
    color: var(--bi-ink) !important;
    font-weight: 600;
}
.gradio-container button.secondary:hover { background: var(--bi-surface-2) !important; }
.gradio-container button:focus-visible, .gradio-container textarea:focus-visible, .gradio-container input:focus-visible {
    outline: 2px solid var(--bi-accent) !important;
    outline-offset: 2px;
}

@media (max-width: 820px) {
    .bi-stats, .bi-spec-grid, .bi-pipeline { grid-template-columns: 1fr 1fr; }
    .bi-header { padding: 22px 18px; }
}
@media (max-width: 520px) {
    .bi-stats, .bi-spec-grid, .bi-pipeline { grid-template-columns: 1fr; }
}
"""

light_theme = gr.themes.Soft(
    primary_hue="blue",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("IBM Plex Sans"), "ui-sans-serif", "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("IBM Plex Mono"), "ui-monospace", "monospace"],
).set(
    body_background_fill="#dde5ee",
    body_background_fill_dark="#dde5ee",
    body_text_color="#1e3350",
    body_text_color_dark="#1e3350",
    body_text_color_subdued="#4b5f78",
    body_text_color_subdued_dark="#4b5f78",
    background_fill_primary="#ffffff",
    background_fill_primary_dark="#ffffff",
    background_fill_secondary="#eef2f7",
    background_fill_secondary_dark="#eef2f7",
    block_background_fill="#ffffff",
    block_background_fill_dark="#ffffff",
    block_border_color="#c3cedb",
    block_border_color_dark="#c3cedb",
    block_label_text_color="#0b2038",
    block_label_text_color_dark="#0b2038",
    block_title_text_color="#0b2038",
    block_title_text_color_dark="#0b2038",
    block_shadow="0 1px 3px rgba(11, 32, 56, 0.10)",
    input_background_fill="#ffffff",
    input_background_fill_dark="#ffffff",
    input_border_color="#a9b8c9",
    input_border_color_dark="#a9b8c9",
    table_even_background_fill="#ffffff",
    table_even_background_fill_dark="#ffffff",
    table_odd_background_fill="#f5f8fc",
    table_odd_background_fill_dark="#f5f8fc",
    button_primary_background_fill="#1e40af",
    button_primary_background_fill_dark="#1e40af",
    button_primary_background_fill_hover="#172f86",
    button_primary_background_fill_hover_dark="#172f86",
    button_primary_text_color="#ffffff",
    button_primary_text_color_dark="#ffffff",
    button_secondary_background_fill="#ffffff",
    button_secondary_background_fill_dark="#ffffff",
    button_secondary_text_color="#0b2038",
    button_secondary_text_color_dark="#0b2038",
)

EMPTY_STATE_HTML = EMPTY_HTML

with gr.Blocks(title="BankIntent AI - Intent Classification") as demo:
    gr.HTML(f"""
    <div class="bi-header">
        <h1 class="header-title">🏦 BankIntent AI</h1>
        <p class="header-tagline">DistilBERT-powered banking intent classification across 77 customer-support intents.</p>
        <div class="bi-stats">
            <div class="bi-stat"><div class="bi-stat-val">85.66%</div><div class="bi-stat-name">Test accuracy</div></div>
            <div class="bi-stat"><div class="bi-stat-val">84.94%</div><div class="bi-stat-name">Weighted F1</div></div>
            <div class="bi-stat"><div class="bi-stat-val">Banking77</div><div class="bi-stat-name">Dataset, 3,076 test samples</div></div>
            <a class="bi-stat" href="https://huggingface.co/{MODEL_REPO_ID}" target="_blank" rel="noopener noreferrer">
                <div class="bi-stat-val" title="https://huggingface.co/{MODEL_REPO_ID}">{MODEL_REPO_ID.split('/')[-1]} ↗</div>
                <div class="bi-stat-name">Model on Hugging Face</div>
            </a>
        </div>
    </div>
    """)

    with gr.Tabs():
        # ── TAB 1: INTERACTIVE PREDICTOR ──
        with gr.TabItem("Intent classifier"):
            gr.HTML("<div class='bi-block-heading' style='margin-top:12px;'><span class='bi-block-logo'>💬</span><span class='bi-block-text'>Customer Banking Query</span></div>")
            query_input = gr.Textbox(
                label="Customer Banking Query",
                show_label=False,
                placeholder="Type a customer question or issue, e.g. 'Why was my transaction declined?'",
                lines=4,
                autofocus=True
            )

            # 2. Controls on one line
            with gr.Row(equal_height=True):
                submit_btn = gr.Button("Classify intent", variant="primary", scale=3, min_width=140)
                clear_btn = gr.Button("Clear", variant="secondary", scale=1, min_width=90)

            # 3. Result and routing
            gr.HTML("<div class='bi-block-heading'><span class='bi-block-logo'>🎯</span><span class='bi-block-text'>Classification result and routing</span></div>")
            result_card = gr.HTML(EMPTY_STATE_HTML)

            # 4. Confidence distribution with Top-K & Latency controls
            gr.HTML("<div class='bi-block-heading'><span class='bi-block-logo'>📊</span><span class='bi-block-text'>Confidence distribution</span></div>")
            with gr.Row(equal_height=True):
                top_k_slider = gr.Slider(
                    minimum=1,
                    maximum=10,
                    value=5,
                    step=1,
                    label="Top-K candidates",
                    scale=4,
                    min_width=220
                )
                latency_box = gr.Textbox(
                    label="Inference latency",
                    value="-- s",
                    interactive=False,
                    scale=2,
                    min_width=140
                )

            label_output = gr.HTML(EMPTY_DIST_HTML)

            # 5. Examples
            gr.HTML("<div class='bi-block-heading'><span class='bi-block-logo'>💡</span><span class='bi-block-text'>Or select a common banking inquiry</span></div>")
            gr.Examples(
                examples=SAMPLE_QUERIES,
                inputs=[query_input, top_k_slider],
                label=None
            )

            # Event handlers
            submit_btn.click(
                fn=predict_intent,
                inputs=[query_input, top_k_slider],
                outputs=[result_card, label_output, latency_box]
            )
            query_input.submit(
                fn=predict_intent,
                inputs=[query_input, top_k_slider],
                outputs=[result_card, label_output, latency_box]
            )
            top_k_slider.change(
                fn=predict_intent,
                inputs=[query_input, top_k_slider],
                outputs=[result_card, label_output, latency_box]
            )
            clear_btn.click(
                fn=lambda: ("", EMPTY_STATE_HTML, EMPTY_DIST_HTML, "-- s"),
                inputs=None,
                outputs=[query_input, result_card, label_output, latency_box]
            )

        # ── TAB 2: MODEL & METRICS ──
        with gr.TabItem("Model and evaluation"):
            gr.HTML(f"""
            <div class="bi-section">Performance on the Banking77 test set</div>
            <div class="bi-spec-grid">
                <div class="bi-spec"><div class="bi-spec-val">85.66%</div><div class="bi-spec-name">Accuracy (3,076 queries)</div></div>
                <div class="bi-spec"><div class="bi-spec-val">84.94%</div><div class="bi-spec-name">Weighted F1 score</div></div>
                <div class="bi-spec"><div class="bi-spec-val">77</div><div class="bi-spec-name">Intent classes</div></div>
            </div>
            """)
            gr.Markdown(f"""
| Property | Detail |
| :--- | :--- |
| **Training data** | Banking77, 13,082 banking customer queries across 77 intents |
| **Architecture** | DistilBERT (`distilbert-base-uncased`), 66 million parameters |
| **Fine-tuning** | 3 epochs, AdamW (`lr=2e-5`, linear warmup) |
| **Hugging Face Hub** | [`{MODEL_REPO_ID}`](https://huggingface.co/{MODEL_REPO_ID}), model weights and tokenizer |
            """)
            gr.HTML("""
            <div class="bi-section" style="margin-top:18px;">How it works</div>
            <div class="bi-pipeline">
                <div class="bi-step"><h4>DistilBERT encoder</h4><p>A light, fast transformer that keeps about 97% of BERT's language understanding with 40% fewer parameters.</p></div>
                <div class="bi-step"><h4>Classification head</h4><p>A linear layer projects the 768-dimension pooled output to 77 intent logits.</p></div>
                <div class="bi-step"><h4>Softmax and routing</h4><p>Logits become a probability distribution, and the top intent maps to a support team: Cards, Payments, Forex, KYC or Disputes.</p></div>
            </div>
            """)

        # ── TAB 3: SUPPORTED INTENTS ──
        with gr.TabItem("Supported intents (77)"):
            chips_html = "".join(
                f"<span class='bi-chip'>{intent}</span>" for intent in sorted(id2label.values())
            )
            gr.HTML(f"""
            <div class="bi-section">All {len(id2label)} banking intents the model can detect</div>
            <div class="bi-chips">{chips_html}</div>
            """)

if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7861,
        share=False,
        inbrowser=True,
        theme=light_theme,
        css=custom_css
    )