from __future__ import annotations
import json, tempfile
from pathlib import Path
import pandas as pd
import gradio as gr

from src.config import SETTINGS
from src.data import validate_input, clean_input, merge_predictions, overview_metrics, reason_breakdown, product_summary
from src.demo_classifier import classify_dataframe
from src.costing import cost_summary

BASE = Path(__file__).parent
SAMPLE = BASE / "data" / "dhaga_returns_sample_300.csv"

CSS = """
:root { --dhaga-ink:#211f1c; --dhaga-muted:#726b63; --dhaga-line:#ded8d0; --dhaga-paper:#f7f4ef; --dhaga-card:#fffdfa; --dhaga-accent:#7f3f36; }
.gradio-container{max-width:1240px!important;margin:0 auto!important;background:var(--dhaga-paper)!important;color:var(--dhaga-ink)!important;font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif!important;}
footer{display:none!important}.prose h1,.prose h2,.prose h3{letter-spacing:-.02em}.wrap{box-shadow:none!important}
#brand{padding:14px 0 8px;border-bottom:1px solid var(--dhaga-line);margin-bottom:18px}.eyebrow{font-size:12px;letter-spacing:.14em;text-transform:uppercase;color:var(--dhaga-muted);font-weight:700}.title{font-size:30px;line-height:1.1;margin:7px 0 5px;font-weight:650}.sub{color:var(--dhaga-muted);font-size:14px}.metric-grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin:4px 0 18px}.metric{background:var(--dhaga-card);border:1px solid var(--dhaga-line);border-radius:12px;padding:16px}.metric .n{font-size:28px;font-weight:650;letter-spacing:-.03em}.metric .l{font-size:12px;color:var(--dhaga-muted);margin-top:5px}.section-card{background:var(--dhaga-card);border:1px solid var(--dhaga-line);border-radius:14px;padding:16px 18px}.insight{border-left:3px solid var(--dhaga-accent);padding-left:14px}.status-ok{color:#466347;font-weight:650}.status-warn{color:#8a5a21;font-weight:650}
button.primary{background:var(--dhaga-accent)!important;border-color:var(--dhaga-accent)!important}.tabs{border-top:0!important}
@media(max-width:800px){.metric-grid{grid-template-columns:repeat(2,minmax(0,1fr))}.title{font-size:24px}}
"""

def header_html():
    return '<div id="brand"><div class="eyebrow">DHAGA &amp; CO.</div><div class="title">Returns Intelligence</div><div class="sub">Turn return feedback into product-level actions.</div></div>'

def metric_html(m):
    return f'''<div class="metric-grid">
      <div class="metric"><div class="n">{m['total']}</div><div class="l">Returns analysed</div></div>
      <div class="metric"><div class="n">{m['originally_other']}</div><div class="l">Originally tagged “Other”</div></div>
      <div class="metric"><div class="n">{m['auto_pct']}%</div><div class="l">Automatically classified</div></div>
      <div class="metric"><div class="n">{m['needs_review']}</div><div class="l">Need review</div></div>
    </div>'''

def load_df(file_path):
    path = file_path if file_path else str(SAMPLE)
    df = pd.read_csv(path)
    ok, msg = validate_input(df)
    if not ok:
        raise gr.Error(msg)
    df = clean_input(df)
    summary = f"**{Path(path).name}**  ·  {len(df)} rows  ·  {df['sku'].nunique()} products  ·  {df['category'].nunique()} categories  ·  {(df['return_reason_selected']=='Other').sum()} tagged Other\n\n✓ {msg}"
    return df, summary

def on_upload(file_path):
    return load_df(file_path)

def use_sample():
    df, summary = load_df(str(SAMPLE))
    return str(SAMPLE), df, summary

def _top_attention(result):
    ps = product_summary(result)
    if ps.empty:
        return "<div class='section-card'>No product insight available.</div>"
    top = ps.iloc[0]
    return f"<div class='section-card insight'><b>{top['sku']} · {top['product_name']}</b><br><span class='sub'>{top['fit_pct']}% of confidently classified returns are fit-related. Most common issue: {top['top_issue']}.</span></div>"

def _demo_insight(result):
    ps = product_summary(result)
    if ps.empty:
        return {"headline":"No clear pattern yet","recommended_action":"Review the rows marked Needs Review.","evidence":"Insufficient confidently classified records."}
    top = ps.iloc[0]
    subset = result[(result['sku']==top['sku']) & (result['review_status']!='Needs Review')]
    issue = subset['ai_sub_reason'].value_counts().head(2)
    evidence = ", ".join([f"{k}: {v}" for k,v in issue.items()])
    return {"headline":f"{top['sku']} shows the strongest fit concentration in this run.","recommended_action":f"Review the size chart and vendor measurements for {top['sku']} before the next restock.","evidence":evidence or f"{len(subset)} classified returns."}

def run_analysis(df, progress=gr.Progress()):
    if df is None or len(df) == 0:
        raise gr.Error("Upload a valid returns CSV or load the sample dataset first.")
    ok, msg = validate_input(df)
    if not ok:
        raise gr.Error(msg)
    df = clean_input(df)
    progress(0.05, desc="Validating input")
    live = bool(SETTINGS.api_key)
    usage = None
    if live:
        try:
            from src.gemini_client import GeminiWorkflow
            wf = GeminiWorkflow()
            preds = wf.classify(df, progress=progress)
            usage = wf.usage
        except Exception as e:
            raise gr.Error(f"Gemini workflow failed: {type(e).__name__}: {e}")
    else:
        progress(0.35, desc="Previewing classifications in demo mode")
        preds = classify_dataframe(df)
    result = merge_predictions(df, preds)
    progress(0.82, desc="Aggregating product patterns")
    metrics = overview_metrics(result)
    reasons = reason_breakdown(result)
    review = result[result['review_status']=='Needs Review'][["return_id","sku","other_comment","ai_primary_reason","ai_sub_reason","ai_confidence","ai_explanation","model_used"]].copy()
    display = result[["return_id","sku","product_name","size_ordered","other_comment","ai_primary_reason","ai_sub_reason","ai_body_area","ai_confidence","review_status","model_used"]].copy()
    ps = product_summary(result)
    if live:
        summary = {"top_products":ps.head(5).to_dict('records'),"reason_breakdown":reasons.head(8).to_dict('records')}
        try:
            insight = wf.insight(summary)
        except Exception:
            insight = _demo_insight(result)
        costs = cost_summary(usage, len(df))
        tech = {"mode":"Live Gemini","fast_model":SETTINGS.model_fast,"strong_model":SETTINGS.model_strong,"confidence_threshold":SETTINGS.confidence_threshold,"usage_and_cost":costs}
    else:
        insight = _demo_insight(result)
        tech = {"mode":"Demo preview - heuristic classifier; set GEMINI_API_KEY for the required two-model workflow","fast_model":SETTINGS.model_fast,"strong_model":SETTINGS.model_strong,"confidence_threshold":SETTINGS.confidence_threshold,"usage_and_cost":"Unavailable in demo mode"}
    insight_md = f"### {insight['headline']}\n\n{insight['recommended_action']}\n\n**Evidence:** {insight['evidence']}"
    fd = tempfile.NamedTemporaryFile(prefix="dhaga_classified_", suffix=".csv", delete=False)
    result.to_csv(fd.name, index=False)
    fd.close()
    progress(1.0, desc="Analysis complete")
    return result, metric_html(metrics), reasons, _top_attention(result), display, review, ps, insight_md, json.dumps(tech, indent=2), fd.name

with gr.Blocks(title="Dhaga Returns Intelligence") as demo:
    gr.HTML(header_html())
    state_input = gr.State()
    state_result = gr.State()
    with gr.Row():
        with gr.Column(scale=3):
            upload = gr.File(label="Returns CSV", file_types=[".csv"], type="filepath")
        with gr.Column(scale=1):
            sample_btn = gr.Button("Use sample dataset")
            run_btn = gr.Button("Run analysis", variant="primary")
    file_summary = gr.Markdown("Upload a CSV or use the sample dataset.")
    with gr.Tabs():
        with gr.Tab("Overview"):
            metrics_html = gr.HTML(metric_html({"total":"—","originally_other":"—","auto_pct":"—","needs_review":"—"}))
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Top return issues")
                    reasons_table = gr.Dataframe(headers=["Reason","Returns","Percent"], interactive=False, wrap=True)
                with gr.Column():
                    gr.Markdown("### What needs attention")
                    attention = gr.HTML("<div class='section-card'>Run an analysis to surface product issues.</div>")
        with gr.Tab("Returns Analysis"):
            analysis_table = gr.Dataframe(interactive=False, wrap=True, max_height=520)
        with gr.Tab("Needs Review"):
            gr.Markdown("Comments that remain below the confidence threshold or could not be classified safely.")
            review_table = gr.Dataframe(interactive=False, wrap=True, max_height=500)
        with gr.Tab("Product Insights"):
            product_table = gr.Dataframe(interactive=False, wrap=True)
            insight_box = gr.Markdown("Run an analysis to generate a product-level recommendation.")
        with gr.Tab("Run Details"):
            gr.Markdown("Technical details are kept secondary for the category user, but remain visible for the CTO and project review.")
            tech = gr.Code(language="json", label="Workflow details")
            export = gr.File(label="Export classified returns")

    upload.upload(on_upload, inputs=upload, outputs=[state_input, file_summary])
    sample_btn.click(use_sample, outputs=[upload, state_input, file_summary])
    run_btn.click(run_analysis, inputs=state_input, outputs=[state_result, metrics_html, reasons_table, attention, analysis_table, review_table, product_table, insight_box, tech, export])

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=4).launch(server_name="0.0.0.0", server_port=7860, max_file_size="10mb", css=CSS)
