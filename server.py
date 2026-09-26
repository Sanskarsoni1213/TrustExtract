"""
TrustExtract Interactive Review Dashboard & HITL Studio
======================================================
FastAPI server providing live endpoints and a web UI to review:
  1. Processed documents & extraction outputs (Dual-path Path A & Path B).
  2. Confidence breakdowns, validation signals, and arithmetic checks.
  3. Forensic integrity analysis (ELA, PDF metadata, font consistency).
  4. Cross-document reconciliation conflict inspector.
  5. Live Human-in-the-Loop (HITL) review queue & active feedback calibration.
  6. Real-time Precision, Recall, F1, and Confidence Calibration charts.
"""

import os
import json
import base64
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

from trustextract.ingestion.pipeline import IngestionPipeline
from trustextract.extraction.pipeline import DualPathExtractor
from trustextract.extraction.path_b_vision import LocalVisionLayoutExtractor
from trustextract.evaluation.evaluate import evaluate_dataset, DocumentEvaluator
from trustextract.evaluation.human_feedback import (
    HumanCorrection,
    ReviewFeedbackStore,
    ActiveFeedbackCalibrator,
)
from trustextract.integrity.pipeline import IntegrityPipeline
from trustextract.reconciliation.reconciler import CrossDocumentReconciler

app = FastAPI(title="TrustExtract Review Studio", version="1.0.0")

FEEDBACK_STORE_PATH = "evaluation_results/dashboard_human_corrections.json"
feedback_store = ReviewFeedbackStore(storage_path=FEEDBACK_STORE_PATH)
calibrator = ActiveFeedbackCalibrator(feedback_store=feedback_store)
ingestion_pipeline = IngestionPipeline()
dual_extractor = DualPathExtractor(vision_model=LocalVisionLayoutExtractor(), n_consistency_passes=2)
integrity_pipeline = IntegrityPipeline()
reconciler = CrossDocumentReconciler()

DOCUMENTS = [
    {"id": "clean_invoice.pdf", "path": "sample_data/clean_invoice.pdf", "gt_path": "sample_data/ground_truth/clean_invoice.json", "type": "Invoice"},
    {"id": "clean_receipt.png", "path": "sample_data/clean_receipt.png", "gt_path": "sample_data/ground_truth/clean_receipt.json", "type": "Receipt"},
    {"id": "clean_po.pdf", "path": "sample_data/clean_po.pdf", "gt_path": "sample_data/ground_truth/clean_po.json", "type": "Purchase Order"},
    {"id": "clean_agreement.docx", "path": "sample_data/clean_agreement.docx", "gt_path": "sample_data/ground_truth/clean_agreement.json", "type": "Agreement"},
    {"id": "clean_payslip.png", "path": "sample_data/clean_payslip.png", "gt_path": "sample_data/ground_truth/clean_payslip.json", "type": "Payslip"},
    {"id": "degraded_payslip.jpg", "path": "sample_data/degraded_payslip.jpg", "gt_path": "sample_data/ground_truth/degraded_payslip.json", "type": "Degraded Payslip"},
]


class CorrectionPayload(BaseModel):
    document_id: str
    field_name: str
    original_value: Optional[Any] = None
    corrected_value: Any
    reviewer_id: str = "human_reviewer_01"
    notes: str = ""
    is_spurious_fp: bool = False


@app.get("/api/documents")
def list_documents():
    return DOCUMENTS


@app.get("/api/document/{doc_id}/process")
def process_doc(doc_id: str):
    doc_info = next((d for d in DOCUMENTS if d["id"] == doc_id), None)
    if not doc_info:
        raise HTTPException(status_code=404, detail="Document not found")

    file_path = doc_info["path"]
    res, norm_doc = ingestion_pipeline.process_document(file_path, document_id=doc_id)

    # Ingestion details
    ingest_result = {
        "status": res.document_status.value,
        "unprocessable_reason": res.unprocessable_reason,
        "page_count": len(norm_doc.pages) if norm_doc else 0,
        "format": norm_doc.original_format if norm_doc else "unknown"
    }

    if not norm_doc:
        return {
            "document_id": doc_id,
            "doc_type": doc_info["type"],
            "ingestion": ingest_result,
            "document_status": res.document_status.value,
            "escalation_reasons": [res.unprocessable_reason or "Document quality gate rejection"],
            "fields": {},
            "ground_truth": {},
            "evaluation": {"metrics": {"tp": 0, "fp": 0, "fn": 0}},
            "integrity": None
        }

    # Dual Path Extraction
    extract_res, path_a, path_b = dual_extractor.process_normalized_document(norm_doc)

    # Apply active feedback post-processing if any
    fields_dict = {}
    for k, fv in extract_res.fields.items():
        fields_dict[k] = {
            "value": fv.value,
            "confidence": round(fv.confidence, 4),
            "agreement": fv.extraction_agreement.value,
            "validation": fv.validation,
            "source": fv.source.dict() if fv.source else None,
            "breakdown": fv.confidence_breakdown.dict() if fv.confidence_breakdown else None
        }

    # Integrity Analysis
    integrity_report = integrity_pipeline.evaluate_document(norm_doc, file_path=file_path)
    integrity_dict = {
        "is_tampered": len(integrity_report.authenticity_flags) > 0,
        "overall_integrity_score": round(integrity_report.integrity_score, 3),
        "tamper_evidence": integrity_report.authenticity_flags,
        "ela_suspicious": integrity_report.ela_result.suspicious if integrity_report.ela_result else False,
        "font_suspicious": integrity_report.font_result.suspicious if integrity_report.font_result else False,
        "metadata_suspicious": integrity_report.pdf_result.suspicious if integrity_report.pdf_result else False,
    }

    # Ground truth comparison
    gt_data = {}
    if os.path.exists(doc_info["gt_path"]):
        with open(doc_info["gt_path"], "r", encoding="utf-8") as f:
            gt_data = json.load(f).get("fields", {})

    evaluator = DocumentEvaluator()
    eval_res = evaluator.evaluate_sample(extract_res.fields, gt_data, doc_id=doc_id)

    return {
        "document_id": doc_id,
        "doc_type": doc_info["type"],
        "ingestion": ingest_result,
        "document_status": extract_res.document_status.value,
        "escalation_reasons": extract_res.escalation_reasons,
        "fields": fields_dict,
        "ground_truth": gt_data,
        "evaluation": eval_res,
        "integrity": integrity_dict,
    }


@app.get("/api/evaluation/summary")
def get_evaluation_summary():
    manifest = [(d["path"], d["gt_path"]) for d in DOCUMENTS]
    report = evaluate_dataset(manifest, dual_extractor=dual_extractor)

    fields_summary = []
    for fname in sorted(report.field_metrics.keys()):
        m = report.field_metrics[fname]
        fields_summary.append({
            "field_name": fname,
            "type": m.field_type.value,
            "tp": m.true_positives,
            "fp": m.false_positives,
            "fn": m.false_negatives,
            "precision": m.precision,
            "recall": m.recall,
            "f1": m.f1_score,
            "total_gt": m.total_ground_truth,
            "total_extracted": m.total_extracted
        })

    calibration_summary = []
    for bname, b in report.calibration_buckets.items():
        calibration_summary.append({
            "bucket": bname,
            "count": b.total_count,
            "correct": b.correct_count,
            "accuracy": b.accuracy,
            "mean_confidence": b.mean_confidence,
            "gap": b.calibration_gap
        })

    return {
        "overall_precision": report.overall_precision,
        "overall_recall": report.overall_recall,
        "overall_f1": report.overall_f1,
        "sample_count": report.sample_count,
        "field_metrics": fields_summary,
        "calibration": calibration_summary,
        "per_document": report.per_document_details
    }


@app.get("/api/corrections")
def get_corrections():
    corrections = feedback_store.get_corrections()
    return [
        {
            "document_id": c.document_id,
            "field_name": c.field_name,
            "original_value": c.original_extracted_value,
            "corrected_value": c.corrected_value,
            "reviewer_id": c.reviewer_id,
            "timestamp": c.timestamp,
            "notes": c.notes,
            "is_spurious_fp": c.is_spurious_fp,
        }
        for c in corrections
    ]


@app.post("/api/corrections/submit")
def submit_correction(payload: CorrectionPayload):
    c = HumanCorrection(
        document_id=payload.document_id,
        field_name=payload.field_name,
        original_extracted_value=payload.original_value,
        corrected_value=payload.corrected_value,
        reviewer_id=payload.reviewer_id,
        notes=payload.notes,
        is_spurious_fp=payload.is_spurious_fp
    )
    feedback_store.record_correction(c)
    summary = calibrator.train_on_feedback()
    return {
        "status": "SUCCESS",
        "message": f"Recorded correction for {payload.field_name}",
        "calibrator_summary": summary,
        "aliases": calibrator.alias_dictionary
    }


@app.get("/", response_class=HTMLResponse)
def index_page():
    return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>TrustExtract &bull; Audit & Verification Studio</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg-base: #0a0d14;
      --bg-surface: #111622;
      --bg-elevated: #182032;
      --bg-hover: #212c44;
      --border-subtle: #1e293b;
      --border-bright: #334155;
      --primary: #38bdf8;
      --primary-glow: rgba(56, 189, 248, 0.15);
      --accent: #818cf8;
      --success: #34d399;
      --success-bg: rgba(52, 211, 153, 0.12);
      --warning: #fbbf24;
      --warning-bg: rgba(251, 191, 36, 0.12);
      --danger: #f87171;
      --danger-bg: rgba(248, 113, 113, 0.12);
      --text-main: #f1f5f9;
      --text-muted: #94a3b8;
      --text-dim: #64748b;
      --font-sans: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
      --font-mono: 'JetBrains Mono', monospace;
    }

    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background-color: var(--bg-base);
      color: var(--text-main);
      font-family: var(--font-sans);
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      overflow-x: hidden;
    }

    /* Top Navigation Bar */
    header {
      background: rgba(17, 22, 34, 0.85);
      backdrop-filter: blur(16px);
      border-bottom: 1px solid var(--border-subtle);
      position: sticky;
      top: 0;
      z-index: 100;
      padding: 0.85rem 2rem;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }

    .brand {
      display: flex;
      align-items: center;
      gap: 0.75rem;
    }

    .brand-badge {
      background: linear-gradient(135deg, #0284c7, #6366f1);
      width: 32px;
      height: 32px;
      border-radius: 8px;
      display: flex;
      align-items: center;
      justify-content: center;
      font-weight: 800;
      font-size: 0.95rem;
      color: #fff;
      box-shadow: 0 0 16px rgba(56, 189, 248, 0.35);
    }

    .brand-text h1 {
      font-size: 1.15rem;
      font-weight: 700;
      letter-spacing: -0.02em;
    }

    .brand-text span {
      font-size: 0.75rem;
      color: var(--text-muted);
      font-family: var(--font-mono);
    }

    nav {
      display: flex;
      gap: 0.5rem;
      background: var(--bg-surface);
      padding: 0.25rem;
      border-radius: 10px;
      border: 1px solid var(--border-subtle);
    }

    nav button {
      background: transparent;
      border: none;
      color: var(--text-muted);
      font-family: var(--font-sans);
      font-size: 0.85rem;
      font-weight: 600;
      padding: 0.5rem 1rem;
      border-radius: 7px;
      cursor: pointer;
      transition: all 0.2s ease;
    }

    nav button:hover {
      color: var(--text-main);
      background: var(--bg-hover);
    }

    nav button.active {
      color: #fff;
      background: linear-gradient(135deg, #0284c7, #4f46e5);
      box-shadow: 0 2px 10px rgba(2, 132, 199, 0.3);
    }

    /* Main Container */
    main {
      flex: 1;
      padding: 2rem;
      max-width: 1440px;
      margin: 0 auto;
      width: 100%;
    }

    /* Layout Sections */
    .tab-section { display: none; }
    .tab-section.active { display: block; animation: fadeIn 0.25s ease-out; }

    @keyframes fadeIn {
      from { opacity: 0; transform: translateY(6px); }
      to { opacity: 1; transform: translateY(0); }
    }

    /* Grid & Cards */
    .grid-2 {
      display: grid;
      grid-template-columns: 340px 1fr;
      gap: 1.5rem;
    }

    .card {
      background: var(--bg-surface);
      border: 1px solid var(--border-subtle);
      border-radius: 14px;
      padding: 1.5rem;
      box-shadow: 0 8px 24px rgba(0, 0, 0, 0.25);
    }

    .card-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 1.25rem;
      padding-bottom: 0.75rem;
      border-bottom: 1px solid var(--border-subtle);
    }

    .card-title {
      font-size: 1rem;
      font-weight: 700;
      color: var(--text-main);
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }

    /* Document List */
    .doc-item {
      padding: 0.85rem 1rem;
      border-radius: 10px;
      border: 1px solid var(--border-subtle);
      background: var(--bg-elevated);
      margin-bottom: 0.6rem;
      cursor: pointer;
      display: flex;
      justify-content: space-between;
      align-items: center;
      transition: all 0.2s ease;
    }

    .doc-item:hover {
      border-color: var(--primary);
      transform: translateX(4px);
    }

    .doc-item.selected {
      border-color: var(--primary);
      background: rgba(56, 189, 248, 0.08);
    }

    .doc-item-title {
      font-size: 0.88rem;
      font-weight: 600;
    }

    .doc-item-sub {
      font-size: 0.75rem;
      color: var(--text-muted);
      font-family: var(--font-mono);
    }

    /* Badges */
    .badge {
      font-size: 0.72rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      padding: 0.25rem 0.6rem;
      border-radius: 6px;
      display: inline-flex;
      align-items: center;
      gap: 0.35rem;
    }

    .badge-ok { background: var(--success-bg); color: var(--success); border: 1px solid rgba(52, 211, 153, 0.25); }
    .badge-review { background: var(--warning-bg); color: var(--warning); border: 1px solid rgba(251, 191, 36, 0.25); }
    .badge-unproc { background: var(--danger-bg); color: var(--danger); border: 1px solid rgba(248, 113, 113, 0.25); }
    .badge-primary { background: var(--primary-glow); color: var(--primary); border: 1px solid rgba(56, 189, 248, 0.25); }

    /* Tables */
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 0.85rem;
    }

    th {
      text-align: left;
      padding: 0.75rem 1rem;
      color: var(--text-dim);
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      font-size: 0.72rem;
      border-bottom: 1px solid var(--border-subtle);
    }

    td {
      padding: 0.85rem 1rem;
      border-bottom: 1px solid var(--border-subtle);
      vertical-align: middle;
    }

    tr:hover td {
      background: rgba(255, 255, 255, 0.02);
    }

    .mono {
      font-family: var(--font-mono);
    }

    /* Metric Tiles */
    .metric-row {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 1rem;
      margin-bottom: 1.5rem;
    }

    .metric-tile {
      background: var(--bg-surface);
      border: 1px solid var(--border-subtle);
      border-radius: 12px;
      padding: 1.25rem;
      display: flex;
      flex-direction: column;
      gap: 0.35rem;
    }

    .metric-tile span {
      font-size: 0.78rem;
      font-weight: 600;
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }

    .metric-tile strong {
      font-size: 1.8rem;
      font-weight: 800;
      color: var(--primary);
    }

    /* Input & Forms */
    .form-group {
      margin-bottom: 1rem;
    }

    .form-label {
      display: block;
      font-size: 0.8rem;
      font-weight: 600;
      color: var(--text-muted);
      margin-bottom: 0.4rem;
    }

    .form-control {
      width: 100%;
      background: var(--bg-elevated);
      border: 1px solid var(--border-bright);
      color: var(--text-main);
      padding: 0.65rem 0.85rem;
      border-radius: 8px;
      font-family: var(--font-sans);
      font-size: 0.85rem;
      outline: none;
      transition: border-color 0.2s;
    }

    .form-control:focus {
      border-color: var(--primary);
      box-shadow: 0 0 0 2px var(--primary-glow);
    }

    .btn {
      background: linear-gradient(135deg, #0284c7, #4f46e5);
      color: #fff;
      border: none;
      padding: 0.7rem 1.25rem;
      border-radius: 8px;
      font-weight: 600;
      font-size: 0.85rem;
      cursor: pointer;
      transition: all 0.2s ease;
      display: inline-flex;
      align-items: center;
      gap: 0.5rem;
    }

    .btn:hover {
      transform: translateY(-1px);
      box-shadow: 0 4px 14px rgba(2, 132, 199, 0.4);
    }

    .btn-secondary {
      background: var(--bg-elevated);
      border: 1px solid var(--border-bright);
      color: var(--text-main);
    }

    .btn-secondary:hover {
      background: var(--bg-hover);
      box-shadow: none;
    }
  </style>
</head>
<body>

  <header>
    <div class="brand">
      <div class="brand-badge">&Psi;</div>
      <div class="brand-text">
        <h1>TrustExtract Studio</h1>
        <span>Zero-Trust Document Verification &bull; Loops 1-9</span>
      </div>
    </div>
    <nav id="navbar">
      <button class="active" onclick="switchTab('inspector')">Document Inspector</button>
      <button onclick="switchTab('evaluation')">Evaluation & Calibration</button>
      <button onclick="switchTab('hitl')">HITL Review Queue</button>
      <button onclick="switchTab('integrity')">Forensics & Integrity</button>
    </nav>
  </header>

  <main>
    <!-- TAB 1: DOCUMENT INSPECTOR -->
    <section id="tab-inspector" class="tab-section active">
      <div class="grid-2">
        <div class="card">
          <div class="card-header">
            <span class="card-title">Test Corpus</span>
            <span class="badge badge-primary">6 Docs</span>
          </div>
          <div id="doc-list">
            <!-- Populated via JS -->
          </div>
        </div>

        <div class="card" id="doc-detail-card">
          <div class="card-header">
            <div>
              <h2 id="current-doc-name" style="font-size: 1.15rem; font-weight: 700;">Select a document</h2>
              <p id="current-doc-type" style="font-size: 0.8rem; color: var(--text-muted); font-family: var(--font-mono);"></p>
            </div>
            <div id="doc-status-badge"></div>
          </div>

          <div id="doc-content-body">
            <p style="color: var(--text-dim); text-align: center; padding: 3rem 0;">Loading document details...</p>
          </div>
        </div>
      </div>
    </section>

    <!-- TAB 2: EVALUATION & CALIBRATION -->
    <section id="tab-evaluation" class="tab-section">
      <div class="metric-row">
        <div class="metric-tile">
          <span>Overall Precision</span>
          <strong id="eval-precision">--</strong>
        </div>
        <div class="metric-tile">
          <span>Overall Recall</span>
          <strong id="eval-recall">--</strong>
        </div>
        <div class="metric-tile">
          <span>Overall F1-Score</span>
          <strong id="eval-f1">--</strong>
        </div>
        <div class="metric-tile">
          <span>Evaluated GT Fields</span>
          <strong id="eval-fields-count">--</strong>
        </div>
      </div>

      <div class="card" style="margin-bottom: 1.5rem;">
        <div class="card-header">
          <span class="card-title">Per-Field Precision & Recall (Ground Truth Evaluated)</span>
          <button class="btn btn-secondary" onclick="loadEvaluation()">Refresh Benchmarks</button>
        </div>
        <div style="overflow-x: auto;">
          <table>
            <thead>
              <tr>
                <th>Field Name</th>
                <th>Type</th>
                <th>TP</th>
                <th>FP</th>
                <th>FN</th>
                <th>Precision</th>
                <th>Recall</th>
                <th>F1-Score</th>
                <th>N (GT/Ext)</th>
              </tr>
            </thead>
            <tbody id="eval-fields-table"></tbody>
          </table>
        </div>
      </div>

      <div class="card">
        <div class="card-header">
          <span class="card-title">Confidence Calibration Analysis by Bucket</span>
        </div>
        <table>
          <thead>
            <tr>
              <th>Confidence Range</th>
              <th>Sample Count (N)</th>
              <th>Correct</th>
              <th>Empirical Accuracy</th>
              <th>Mean Confidence</th>
              <th>Calibration Gap</th>
            </tr>
          </thead>
          <tbody id="eval-calib-table"></tbody>
        </table>
      </div>
    </section>

    <!-- TAB 3: HITL REVIEW QUEUE -->
    <section id="tab-hitl" class="tab-section">
      <div class="grid-2">
        <div class="card">
          <div class="card-header">
            <span class="card-title">Submit Field Correction</span>
          </div>
          <form id="correction-form" onsubmit="handleCorrectionSubmit(event)">
            <div class="form-group">
              <label class="form-label">Document</label>
              <select id="corr-doc-id" class="form-control" required></select>
            </div>
            <div class="form-group">
              <label class="form-label">Field Name</label>
              <input type="text" id="corr-field-name" class="form-control" placeholder="e.g. EmployeeName" required />
            </div>
            <div class="form-group">
              <label class="form-label">Original Extracted Value (or null)</label>
              <input type="text" id="corr-orig-val" class="form-control" placeholder="e.g. None or raw string" />
            </div>
            <div class="form-group">
              <label class="form-label">Corrected Human Value</label>
              <input type="text" id="corr-new-val" class="form-control" placeholder="e.g. Sarah Chen" required />
            </div>
            <div class="form-group">
              <label class="form-label">Reviewer Notes</label>
              <input type="text" id="corr-notes" class="form-control" placeholder="Reason for correction" />
            </div>
            <button type="submit" class="btn" style="width: 100%;">Record & Train Active Feedback</button>
          </form>
        </div>

        <div class="card">
          <div class="card-header">
            <span class="card-title">Audited Human Corrections Log</span>
            <span id="corr-count-badge" class="badge badge-primary">0</span>
          </div>
          <div style="overflow-x: auto;">
            <table>
              <thead>
                <tr>
                  <th>Doc</th>
                  <th>Field</th>
                  <th>Original</th>
                  <th>Corrected</th>
                  <th>Reviewer</th>
                  <th>Timestamp</th>
                </tr>
              </thead>
              <tbody id="corrections-table-body"></tbody>
            </table>
          </div>
        </div>
      </div>
    </section>

    <!-- TAB 4: INTEGRITY & FORENSICS -->
    <section id="tab-integrity" class="tab-section">
      <div class="card">
        <div class="card-header">
          <span class="card-title">Forensic Tamper Detection Suite (Loop 4)</span>
        </div>
        <p style="color: var(--text-muted); font-size: 0.88rem; margin-bottom: 1.5rem;">
          TrustExtract runs multimodal integrity checks including Error Level Analysis (ELA) for image compression artifacts, PDF metadata modification tool sniffing, perceptual font discrepancy analysis, and perceptual deduplication.
        </p>
        <div id="integrity-results-container">
          <!-- Populated from current doc -->
        </div>
      </div>
    </section>
  </main>

  <script>
    let currentDocs = [];
    let selectedDocId = 'clean_invoice.pdf';

    function switchTab(tabId) {
      document.querySelectorAll('.tab-section').forEach(el => el.classList.remove('active'));
      document.querySelectorAll('nav button').forEach(el => el.classList.remove('active'));
      
      document.getElementById('tab-' + tabId).classList.add('active');
      event.target.classList.add('active');

      if (tabId === 'evaluation') loadEvaluation();
      if (tabId === 'hitl') loadCorrections();
      if (tabId === 'integrity') loadIntegrityForSelected();
    }

    async function loadDocuments() {
      const res = await fetch('/api/documents');
      currentDocs = await res.json();
      
      const listEl = document.getElementById('doc-list');
      const selectEl = document.getElementById('corr-doc-id');
      listEl.innerHTML = '';
      selectEl.innerHTML = '';

      currentDocs.forEach((doc, idx) => {
        const item = document.createElement('div');
        item.className = 'doc-item' + (doc.id === selectedDocId ? ' selected' : '');
        item.onclick = () => selectDocument(doc.id);
        item.innerHTML = `
          <div>
            <div class="doc-item-title">${doc.id}</div>
            <div class="doc-item-sub">${doc.type}</div>
          </div>
          <span class="badge ${doc.id.includes('degraded') ? 'badge-unproc' : 'badge-ok'}">${doc.id.includes('degraded') ? 'Degraded' : 'Clean'}</span>
        `;
        listEl.appendChild(item);

        const opt = document.createElement('option');
        opt.value = doc.id;
        opt.textContent = `${doc.id} (${doc.type})`;
        selectEl.appendChild(opt);
      });

      selectDocument(selectedDocId);
    }

    async function selectDocument(docId) {
      selectedDocId = docId;
      document.querySelectorAll('.doc-item').forEach(el => {
        el.classList.toggle('selected', el.innerText.includes(docId));
      });

      const body = document.getElementById('doc-content-body');
      body.innerHTML = '<p style="color: var(--text-dim); text-align: center; padding: 2rem 0;">Analyzing document with Dual-Path & Confidence Fuser...</p>';

      const res = await fetch(`/api/document/${docId}/process`);
      const data = await res.json();

      document.getElementById('current-doc-name').innerText = data.document_id;
      document.getElementById('current-doc-type').innerText = `Format: ${data.ingestion.format.toUpperCase()} | Pages: ${data.ingestion.page_count}`;

      const statusEl = document.getElementById('doc-status-badge');
      if (data.document_status === 'OK') {
        statusEl.innerHTML = '<span class="badge badge-ok">STATUS: OK</span>';
      } else if (data.document_status === 'NEEDS_REVIEW') {
        statusEl.innerHTML = '<span class="badge badge-review">STATUS: NEEDS_REVIEW</span>';
      } else {
        statusEl.innerHTML = '<span class="badge badge-unproc">STATUS: UNPROCESSABLE</span>';
      }

      let html = '';
      if (data.escalation_reasons && data.escalation_reasons.length > 0) {
        html += '<div style="background: var(--warning-bg); border: 1px solid rgba(251,191,36,0.3); border-radius: 8px; padding: 0.75rem 1rem; margin-bottom: 1.25rem;">';
        html += '<strong style="color: var(--warning); font-size: 0.82rem; text-transform: uppercase;">Escalation Reasons:</strong><ul style="margin-top: 0.35rem; padding-left: 1.2rem; font-size: 0.8rem; color: var(--text-main);">';
        data.escalation_reasons.forEach(r => html += `<li>${r}</li>`);
        html += '</ul></div>';
      }

      html += `
        <h3 style="font-size: 0.9rem; font-weight: 700; margin-bottom: 0.75rem;">Extracted Document Fields (Dual-Path Fused)</h3>
        <table>
          <thead>
            <tr>
              <th>Field Name</th>
              <th>Extracted Value</th>
              <th>Confidence</th>
              <th>Agreement</th>
              <th>Validation</th>
              <th>Ground Truth</th>
            </tr>
          </thead>
          <tbody>
      `;

      const allKeys = Object.keys(data.fields);
      if (allKeys.length === 0) {
        html += '<tr><td colspan="6" style="text-align: center; color: var(--text-dim);">Zero fields extracted (Unprocessable / Low Quality Gate).</td></tr>';
      } else {
        allKeys.sort().forEach(k => {
          const f = data.fields[k];
          const gt = data.ground_truth[k] !== undefined ? data.ground_truth[k] : '<em style="color:var(--text-dim)">None</em>';
          const confColor = f.confidence >= 0.90 ? 'var(--success)' : f.confidence >= 0.75 ? 'var(--warning)' : 'var(--danger)';
          
          html += `
            <tr>
              <td><strong>${k}</strong></td>
              <td class="mono" style="color: var(--primary);">${f.value}</td>
              <td><span class="mono" style="font-weight: 700; color: ${confColor};">${(f.confidence * 100).toFixed(1)}%</span></td>
              <td><span class="badge badge-primary">${f.agreement}</span></td>
              <td><span class="badge ${f.validation.includes('OK') ? 'badge-ok' : 'badge-review'}">${f.validation}</span></td>
              <td class="mono" style="color: var(--text-muted);">${gt}</td>
            </tr>
          `;
        });
      }

      html += '</tbody></table>';
      body.innerHTML = html;
    }

    async function loadEvaluation() {
      const res = await fetch('/api/evaluation/summary');
      const data = await res.json();

      document.getElementById('eval-precision').innerText = (data.overall_precision * 100).toFixed(2) + '%';
      document.getElementById('eval-recall').innerText = (data.overall_recall * 100).toFixed(2) + '%';
      document.getElementById('eval-f1').innerText = (data.overall_f1 * 100).toFixed(2) + '%';
      
      const totalGt = data.field_metrics.reduce((acc, m) => acc + m.total_gt, 0);
      document.getElementById('eval-fields-count').innerText = totalGt;

      const fTable = document.getElementById('eval-fields-table');
      fTable.innerHTML = '';
      data.field_metrics.forEach(m => {
        const row = document.createElement('tr');
        row.innerHTML = `
          <td><strong>${m.field_name}</strong></td>
          <td><span class="badge badge-primary">${m.type}</span></td>
          <td class="mono">${m.tp}</td>
          <td class="mono">${m.fp}</td>
          <td class="mono">${m.fn}</td>
          <td class="mono" style="font-weight:700; color: ${m.precision >= 0.99 ? 'var(--success)' : 'var(--warning)'}">${(m.precision * 100).toFixed(1)}%</td>
          <td class="mono" style="font-weight:700; color: ${m.recall >= 0.99 ? 'var(--success)' : 'var(--warning)'}">${(m.recall * 100).toFixed(1)}%</td>
          <td class="mono" style="font-weight:700;">${(m.f1 * 100).toFixed(1)}%</td>
          <td class="mono" style="color:var(--text-muted);">${m.total_gt}/${m.total_extracted}</td>
        `;
        fTable.appendChild(row);
      });

      const cTable = document.getElementById('eval-calib-table');
      cTable.innerHTML = '';
      data.calibration.forEach(b => {
        const row = document.createElement('tr');
        row.innerHTML = `
          <td><strong>${b.bucket}</strong></td>
          <td class="mono">${b.count}</td>
          <td class="mono">${b.correct}</td>
          <td class="mono">${(b.accuracy * 100).toFixed(2)}%</td>
          <td class="mono">${(b.mean_confidence * 100).toFixed(2)}%</td>
          <td class="mono" style="color: ${b.gap <= 0.05 ? 'var(--success)' : 'var(--warning)'}">${(b.gap * 100).toFixed(2)}%</td>
        `;
        cTable.appendChild(row);
      });
    }

    async function loadCorrections() {
      const res = await fetch('/api/corrections');
      const data = await res.json();
      
      document.getElementById('corr-count-badge').innerText = data.length;
      const body = document.getElementById('corrections-table-body');
      body.innerHTML = '';

      if (data.length === 0) {
        body.innerHTML = '<tr><td colspan="6" style="text-align: center; color: var(--text-dim);">No corrections recorded yet.</td></tr>';
        return;
      }

      data.forEach(c => {
        const row = document.createElement('tr');
        row.innerHTML = `
          <td><strong>${c.document_id}</strong></td>
          <td><span class="badge badge-primary">${c.field_name}</span></td>
          <td class="mono" style="color: var(--text-muted);">${c.original_value || 'None'}</td>
          <td class="mono" style="color: var(--success); font-weight:700;">${c.corrected_value}</td>
          <td>${c.reviewer_id}</td>
          <td class="mono" style="font-size: 0.75rem; color: var(--text-dim);">${c.timestamp.substring(0, 19).replace('T', ' ')}</td>
        `;
        body.appendChild(row);
      });
    }

    async function handleCorrectionSubmit(e) {
      e.preventDefault();
      const payload = {
        document_id: document.getElementById('corr-doc-id').value,
        field_name: document.getElementById('corr-field-name').value,
        original_value: document.getElementById('corr-orig-val').value || null,
        corrected_value: document.getElementById('corr-new-val').value,
        notes: document.getElementById('corr-notes').value
      };

      const res = await fetch('/api/corrections/submit', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      if (res.ok) {
        alert('Correction submitted and active feedback models updated!');
        document.getElementById('corr-field-name').value = '';
        document.getElementById('corr-orig-val').value = '';
        document.getElementById('corr-new-val').value = '';
        document.getElementById('corr-notes').value = '';
        loadCorrections();
      }
    }

    async function loadIntegrityForSelected() {
      const container = document.getElementById('integrity-results-container');
      container.innerHTML = '<p style="text-align: center; color: var(--text-dim);">Running ELA, font analysis, and PDF metadata inspection...</p>';

      const res = await fetch(`/api/document/${selectedDocId}/process`);
      const data = await res.json();

      if (!data.integrity) {
        container.innerHTML = '<p style="color: var(--warning);">Integrity suite skipped (document unprocessable).</p>';
        return;
      }

      const it = data.integrity;
      container.innerHTML = `
        <div class="metric-row">
          <div class="metric-tile">
            <span>Tamper Status</span>
            <strong style="color: ${it.is_tampered ? 'var(--danger)' : 'var(--success)'};">${it.is_tampered ? 'TAMPERED' : 'CLEAN'}</strong>
          </div>
          <div class="metric-tile">
            <span>Integrity Score</span>
            <strong>${(it.overall_integrity_score * 100).toFixed(1)}%</strong>
          </div>
          <div class="metric-tile">
            <span>ELA Artifacts</span>
            <strong style="color: ${it.ela_suspicious ? 'var(--danger)' : 'var(--success)'};">${it.ela_suspicious ? 'SUSPICIOUS' : 'PASS'}</strong>
          </div>
          <div class="metric-tile">
            <span>Font Consistency</span>
            <strong style="color: ${it.font_suspicious ? 'var(--danger)' : 'var(--success)'};">${it.font_suspicious ? 'SUSPICIOUS' : 'PASS'}</strong>
          </div>
        </div>
      `;
    }

    window.onload = loadDocuments;
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    print("Starting TrustExtract Studio at http://127.0.0.1:8000 ...")
    uvicorn.run(app, host="127.0.0.1", port=8000)
