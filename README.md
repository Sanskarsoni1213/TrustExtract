# TrustExtract

Document-level trust scoring, tamper detection, and cross-document reconciliation for scanned and photographed documents.

TrustExtract ingests messy, real-world documents — scans, photos, PDFs, DOCX — and returns structured JSON with a transparent, decomposed confidence score for every field, a document-level integrity verdict, and source provenance so a human reviewer can verify any value in one look. It is built around one core idea: **confidence should be evidence-based, not self-reported.**

## Why this exists

Off-the-shelf OCR has mostly solved *reading* documents. It has not solved the question that actually blocks automation: which extracted value can be trusted without a human looking at it, and which cannot. Most extraction systems answer this with a single, uncalibrated confidence number produced by the model itself. TrustExtract instead derives confidence from independent, checkable signals — agreement between two separate extraction paths, stability under perturbation, and business-logic validation — and fails loudly instead of guessing when a document cannot be processed reliably.

## What it does

- Ingests PDF, JPG, PNG, TIFF, HEIC, and DOCX, including low-quality inputs (skewed, blurry, compressed)
- Extracts structured fields — entities, amounts, dates, tables — via two independent extraction paths
- Fuses four orthogonal signals into one calibrated confidence **per field**, not per document
- Runs physical/forensic checks for tampering, duplication, and AI-generated origin
- Reconciles claims across a bundle of related documents and surfaces disagreements
- Attaches a source bounding box and encrypted crop reference to every extracted value
- Returns an explicit `UNPROCESSABLE` status rather than a confident guess when a document can't be read

## Architecture

The system is built as five architecturally independent stages, each answering a distinct question:

| Loop | Capability | Question Answered |
|------|-----------|-------------------|
| 1 | Ingestion & Quality Gates | Can we safely parse and normalize this file? |
| 2 | Dual-Path Extraction | What values does this document contain? |
| 3 | Confidence Fusion & Escalation | How much should we trust each extracted value? |
| 4 | Integrity & Tamper Detection | Was this document physically altered or fabricated? |
| 5 | Cross-Document Reconciliation | Do the documents in this bundle agree with each other? |

```
Untrusted File
      |
      v
[Loop 1] SafeParserBoundary -> DocumentNormalizer -> QualityChecker -> KMS Encryption at Rest
      |
      v
[Loop 2] Path A: RapidOCR  +  Path B: Vision Model  ->  Field Reconciler -> Table Extractor
      |                                                        |
      |                                          FieldValue + Provenance Crops
      v
[Loop 3] Business Validators -> Self-Consistency Sampler -> Agreement Signal
              -> ConfidenceFuser -> DocumentEscalationEngine
      |
      v
[Loop 4] Error Level Analysis -> PDF Metadata Forensics -> Font Consistency
              -> Duplicate Detection -> AI Generation Check -> IntegrityPipeline
      |
      v
[Loop 5] EntityGraph -> CrossDocumentReconciler
      |
      v
Final TrustExtractResult (JSON)
```

## Loop 1 — Ingestion & Quality Gates

`trustextract/ingestion/`

- **Sandbox** (`security/sandbox.py`): file-size limits, decompression-bomb guards, magic-byte format sniffing (prevents extension spoofing), zero-byte/corrupt/encrypted-PDF handling
- **Normalizer** (`normalizer.py`): converts every supported format into standardized page images with digital text layers where available
- **Deskew** (`deskew.py`): contour/Hough-based skew detection and bicubic rotation correction, verified accurate to <0.12° on 6° and 18° test samples
- **Quality gate** (`quality.py`): resolution, blur (Laplacian variance), contrast, and text-density checks — rejects unreadable pages before any extraction is attempted
- **Encryption at rest** (`security/crypto.py`): AES-256-GCM via a swappable `KMSEncryptionInterface`
- **Audit logging** (`security/audit.py`): every event logged with actor, timestamp, and hashed identifiers — no plaintext PII, ever

**Key decision:** quality gates run *before* extraction. A document that fails the gate returns `UNPROCESSABLE` with a specific reason rather than producing low-confidence fields downstream.

## Loop 2 — Dual-Path Extraction

`trustextract/extraction/`

Two extraction paths run independently over each page:

| | Path A | Path B |
|---|--------|--------|
| Engine | RapidOCR | Vision-capable model |
| Input | Raw page pixels | Raw page pixels |
| Independence | Character-level OCR | Semantic visual reading |

- Table structure is reconstructed separately with per-cell confidence
- Every field carries a bounding box and an encrypted provenance crop reference, enabling one-look source verification
- Field-to-region mapping bugs (e.g. regex over-matching "Subtotal" as "InvoiceTotal") were found and fixed during development — see `docs/` for the audit trail

> **Status note:** Path B currently runs against \[a real vision API / a documented reference simulator — update this line to match your current state\]. This is called out explicitly in `confidence.py`'s module docstring.

## Loop 3 — Confidence Fusion & Escalation

`trustextract/extraction/confidence.py`

Every field's final confidence is a weighted fusion of four independent signals:

```
final = 0.35 × native_ocr + 0.25 × agreement + 0.25 × consistency + 0.15 × validation
```

| Signal | Weight | Measures |
|--------|--------|----------|
| `native_ocr` | 0.35 | The winning path's own extraction confidence |
| `agreement` | 0.25 | Do Path A and Path B independently agree? |
| `consistency` | 0.25 | Is the value stable across 3 perturbed re-extractions? |
| `validation` | 0.15 | Does the value pass business-logic checks? |

**Field-type-aware agreement.** Comparison logic matches the semantics of the field:

| Field Type | Rule | Near-match allowed? |
|------------|------|---------------------|
| `IDENTIFIER` (InvoiceId, EmployeeId, PO#) | Strict exact match | No — a 1-character OCR ambiguity (`984I` vs `9841`) is a real conflict, not noise |
| `FREE_TEXT` (VendorName, EmployeeName) | Normalized, case/whitespace-insensitive | Yes — edit distance ≤2 or similarity ≥0.85 |
| `NUMERIC` (Subtotal, NetPay) | Parsed float comparison | N/A |
| `DATE` (InvoiceDate, PayDate) | Canonical-form comparison | No |

This distinction exists because early testing showed a flat/uniform comparison rule either flagged formatting noise as false disagreements, or — worse — let genuine identifier mismatches slip through as "near matches."

**Single-source reweighting.** When a field has no Path A coverage, the consistency term is excluded and the remaining weights are rescaled proportionally, so vision-only fields don't inherit an unearned perfect consistency score.

**Business validators** (`validators.py`): arithmetic checks (line items sum to subtotal, subtotal + tax = total), date ordering, identifier format/checksum. A failed validator caps the field's confidence regardless of agreement.

**Escalation** (`DocumentEscalationEngine`): `document_status` escalates to `NEEDS_REVIEW` via independent, individually-named rules (confidence below threshold, path disagreement, arithmetic failure) — every escalation has a specific, human-readable reason attached.

## Loop 4 — Integrity & Tamper Detection

`trustextract/integrity/`

Five forensic checks, kept architecturally separate from field-level confidence (they answer "was this altered," not "is this value right"):

| Check | Detects |
|-------|---------|
| Error Level Analysis (`ela.py`) | JPEG recompression artifacts from spliced/pasted regions |
| PDF Metadata Forensics (`pdf_forensics.py`) | Suspicious creation/modification timelines, editing-software signatures, incremental revision chains |
| Font Consistency (`font_analysis.py`) | Character pitch/height/baseline divergence from the document's own median |
| Duplicate Detection (`duplicate_detector.py`) | Exact (SHA-256) and near-duplicate (perceptual hash) resubmission |
| AI Generation Signals (`ai_detector.py`) | Missing/inconsistent capture metadata, generative-model fingerprints |

### Documented limitation: the tamper sensitivity boundary

Testing established a specific, verified hard boundary of single-document forensics:

| Tamper scenario | ELA | Font analysis | Arithmetic | Caught? |
|---|---|---|---|---|
| Large font-size splice | Detected | Detected | If sum breaks | Yes |
| Size-matched text edit, flat background | Missed | Missed | Only if it breaks a sum | Partial |
| Size-matched identifier edit, no arithmetic relationship | Missed | Missed | N/A | **No — single-document gap** |

This is a physical limit of 8×8 DCT recompression forensics, not a bug: a single-pixel stroke change occupying <5% of a DCT block does not produce enough energy divergence to separate from normal text noise. This boundary is locked in as a permanent regression test (`test_worst_case_subtle_tamper_boundary_finding`) so it can't silently regress or be assumed fixed.

**This is exactly what motivates Loop 5** — but see the caveat below.

## Loop 5 — Cross-Document Reconciliation

`trustextract/reconciliation/`

- **Entity graph** (`entity_graph.py`): documents contribute claims about shared real-world entities (name, ID number, amounts, dates) to a shared graph, with heterogeneous field names (`PurchaseOrderNumber` vs `InvoiceRef`) resolved to canonical entity keys via an alias registry
- **Reconciler** (`reconciler.py`): compares claims across documents using the same field-type-aware logic from Loop 3, surfaces conflicts into `cross_document_conflicts[]` with both sides' values, confidences, and provenance crops

**Verified end-to-end**: a bundle containing a tampered invoice (the exact worst-case identifier edit from the Loop 4 boundary finding) plus a genuine, corroborating purchase order was run through the full pipeline. Both documents entered Loops 1–4 individually as `OK` / `integrity_score: 1.0` / no flags. Loop 5 correctly surfaced the conflict and escalated both documents to `NEEDS_REVIEW`.

> **Caveat, stated plainly:** cross-document reconciliation is a *partial* mitigation, conditional on bundle composition — not a complete closure of the single-document tamper boundary. It only works when a second, independent, uncompromised document in the bundle asserts the same fact. A single tampered document submitted alone, with no corroborating document, is not caught by anything in this system. This is an open, documented gap.

## Output schema

Every document produces a `TrustExtractResult`:

```json
{
  "document_id": "doc_a1b2c3d4...",
  "document_status": "OK | NEEDS_REVIEW | UNPROCESSABLE",
  "unprocessable_reason": null,
  "integrity_score": 0.75,
  "authenticity_flags": ["ELA_ANOMALY_DETECTED", "FONT_INCONSISTENCY"],
  "escalation_reasons": [
    "Field 'InvoiceId': final confidence 0.842 < threshold 0.90",
    "Field 'InvoiceId': ocr!=vision disagreement"
  ],
  "fields": {
    "VendorName": {
      "value": "Contoso Logistics Inc.",
      "confidence": 0.963,
      "extraction_agreement": "ocr==vision",
      "validation": "PASSED",
      "confidence_breakdown": {
        "native_ocr": 0.95,
        "agreement_signal": 1.0,
        "consistency_signal": 1.0,
        "validation_signal": 1.0,
        "final": 0.963
      },
      "source": {
        "page": 1,
        "bbox": [0.05, 0.12, 0.35, 0.04],
        "crop_ref": "enc_ref_abc123..."
      }
    }
  },
  "tables": [],
  "cross_document_conflicts": []
}
```

Field names follow Azure AI Document Intelligence's naming conventions (`VendorName`, `InvoiceId`, `InvoiceTotal`, `DateOfBirth`, etc.) for interoperability.

## Repository structure

```
trustextract/
├── schema.py                  # Pydantic output schema
├── ingestion/
│   ├── pipeline.py            # Loop 1 orchestrator
│   ├── normalizer.py          # Multi-format normalization
│   ├── quality.py             # Quality gate checks
│   └── deskew.py              # Skew detection & correction
├── extraction/
│   ├── pipeline.py            # Loop 2+3 orchestrator
│   ├── path_a_ocr.py          # Path A: RapidOCR
│   ├── path_b_vision.py       # Path B: vision model
│   ├── confidence.py          # Signal fusion + escalation
│   ├── validators.py          # Arithmetic, date, format validators
│   ├── table.py                # Table structure extraction
│   └── crop.py                # Provenance crop manager
├── integrity/
│   ├── pipeline.py            # Loop 4 orchestrator
│   ├── ela.py                 # Error Level Analysis
│   ├── pdf_forensics.py       # PDF metadata forensics
│   ├── font_analysis.py       # Font consistency analysis
│   ├── duplicate_detector.py  # SHA-256 + perceptual hash dedup
│   └── ai_detector.py         # AI-generation detection
├── reconciliation/
│   ├── entity_graph.py        # Cross-document entity graph
│   └── reconciler.py          # Loop 5 reconciler
└── security/
    ├── audit.py                # Redacting audit logger
    ├── crypto.py                # KMS encryption interface
    └── sandbox.py               # Untrusted-input validation boundary

tests/                          # 61+ tests across all loops
evaluation_results/              # Precision/recall + calibration reports
sample_data/                     # Test documents (clean + degraded)
run_loop*_demo.py                # Per-loop demonstration scripts
```

## Running the demos

```bash
python -m pytest tests/ -v            # full test suite
python run_loop1_demo.py              # ingestion + quality gates
python run_loop2_demo.py              # dual-path extraction
python run_loop3_demo.py              # confidence fusion
python run_loop4_demo.py              # tamper/integrity detection
python run_loop5_demo.py              # cross-document reconciliation
python run_loop8_eval.py              # precision/recall + calibration report
```

## Known limitations

Stated here deliberately, not hidden — a system that names its own limits precisely is more trustworthy than one that claims to have none.

| Limitation | Impact | Status |
|---|---|---|
| Single-document tamper with no corroborating document | Nothing in the system catches this | Open, documented gap |
| Path B extraction backend | Confirm current state: real API vs. reference implementation | See `confidence.py` docstring |
| Fusion weights (0.35/0.25/0.25/0.15) | Manually tuned starting point, not empirically calibrated | Calibration pending labeled data at scale |
| ELA requires JPEG compression history | Pure PNG / lossless-origin documents get no ELA benefit | Inherent limitation of the technique |
| Confidence calibration | High-confidence bucket accuracy measured against ground truth — see `evaluation_results/` for current figures | Ongoing |

## Design principles

- **Architectural separation.** Each loop answers a different question and can be run, tested, and upgraded independently.
- **Evidence-based confidence, not self-reported confidence.** Every score decomposes into named, inspectable signals — never an opaque single number.
- **Field-type awareness.** Comparison logic respects what a field means: an edit distance of 1 on an identifier is a real conflict; the same distance on a name is formatting noise.
- **Explicit escalation.** `document_status` is never set to `NEEDS_REVIEW` without a corresponding, specific, human-readable reason.
- **Fail loudly.** A document that cannot be processed returns `UNPROCESSABLE` with a reason — never a confident guess.
- **State the gaps plainly.** Known limitations are documented in code, tests, and this README — not discovered later by a user.
