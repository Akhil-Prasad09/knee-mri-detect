# Build Plan — Automated Knee MRI Abnormality Detection for Clinical Decision Support

Team: Billakanti Ratan Sathwik (010), Akhil Prasad (014), Jakka Kartik Reddy (021)
Source docs: `docs/Major_Project_Proposal.docx`, `docs/major_abstract.docx`

## 1. What we are building (reconciled from proposal + abstract)

A web application where a clinician uploads a knee MRI exam, a deep learning model
predicts abnormalities with confidence scores, Grad-CAM heatmaps show *where* the
model looked, and a PDF report is generated. The model assists, it does not diagnose.

The abstract lists TensorFlow/Keras + Flask; the proposal lists PyTorch + FastAPI +
React/Streamlit + Grad-CAM + PDF reports. The proposal is the more recent and more
detailed document, so **the proposal stack wins**:

| Layer | Choice | Why |
|---|---|---|
| DL framework | PyTorch 2.x, `timm` EfficientNet-B3 | Proposal's algorithm table; timm gives pretrained weights in one line |
| Explainability | Grad-CAM (`pytorch-grad-cam`) | Proposal requirement |
| Backend | FastAPI + Uvicorn | Async, auto Swagger docs, easy file upload |
| Frontend | React (Vite) + Tailwind | Proposal's primary option; Streamlit kept as fallback if time runs short |
| DB | SQLite (dev) → PostgreSQL (demo) via SQLAlchemy | Stores uploads, predictions, reports |
| Reports | ReportLab (rule-based PDF) | Proposal requirement |
| Packaging | Docker Compose (api + web + db) | Proposal deliverable |
| Training compute | Google Colab / Kaggle GPU | Hardware section lists Colab as cloud option |

## 2. Dataset decision (the most important early call)

The proposal names 8 abnormalities (ACL, PCL, meniscus, cartilage, fracture, bone
marrow edema, effusion, osteoarthritis). **No public dataset labels all eight.**

Primary dataset: **Stanford MRNet** (1,370 exams, 3 planes — sagittal/coronal/axial,
labels: *abnormal*, *ACL tear*, *meniscal tear*). It is free for research, is the
standard benchmark, and is already multi-label — it matches the proposal's
methodology exactly (EfficientNet + sigmoid + BCE).

Secondary (stretch): **KneeMRI (Štajduhar et al.)** for ACL severity, and
**OAI / Kaggle knee X-ray OA** only if we want an osteoarthritis head.

Plan: implement the pipeline generically for *N* labels. Ship with 3 MRNet labels;
the UI, report and API already support adding more heads. The report explicitly
states which abnormalities the deployed model was trained to detect. Register for
MRNet access in week 1 (approval takes a few days).

## 3. Architecture

```
Browser (React)
   │  multipart upload (.npy / DICOM series / PNG stack)
   ▼
FastAPI  /api/v1
   ├─ /exams        POST upload → store → enqueue inference
   ├─ /exams/{id}   GET status, predictions, gradcam slice urls
   ├─ /exams/{id}/report  GET generated PDF
   └─ /health
   │
   ├─ services/preprocess.py   pydicom/np load → resize 224 → CLAHE → normalise
   ├─ services/inference.py    per-plane EfficientNet-B3 → slice features →
   │                           max-pool over slices → sigmoid per label
   ├─ services/explain.py      Grad-CAM on top-k slices → PNG overlays
   └─ services/report.py       ReportLab PDF (patient meta, table, heatmaps, disclaimer)
   │
SQLite/Postgres (exams, predictions, reports)   +   ./storage (files)
```

Model design (MRNet-style, fits an 8 GB laptop for inference):
- Input: one plane = stack of S slices, each 3×224×224 (grayscale repeated).
- Backbone: EfficientNet-B3 pretrained (ImageNet), shared across slices.
- Aggregation: global max-pool across slice dimension → 1536-d.
- Head: Linear → N logits → sigmoid. Loss: BCEWithLogits (pos_weight for imbalance).
- Train one model per plane; ensemble by averaging probabilities across planes
  (or a tiny logistic-regression fuser, as in the MRNet paper).
- Grad-CAM target layer: last conv block of the backbone, computed on the slice
  with the highest activation.

## 4. Milestones (12-week plan)

| Week | Milestone | Owner (suggested) | Done when |
|---|---|---|---|
| 1 | Repo, env, MRNet access request, literature table | All | `make setup` runs, dataset request sent |
| 2 | Data loader + preprocessing + EDA notebook | Akhil | `ml/data` tests pass, slice viewer notebook |
| 3 | Baseline training on sagittal plane (abnormal label) | Sathwik | AUC ≥ 0.85 on val |
| 4 | Multi-label, all 3 planes, augmentation, pos_weight | Sathwik | AUC per label logged to `ml/models/metrics.json` |
| 5 | Grad-CAM module + overlay export | Akhil | Heatmaps for any exam id |
| 6 | FastAPI: upload, inference, DB, status endpoints | Kartik | Swagger demo end-to-end on CPU |
| 7 | PDF report generator | Kartik | `/report` returns PDF with table + heatmaps |
| 8 | React dashboard: upload, progress, results, heatmap viewer | Akhil | Works against local API |
| 9 | Plane ensemble, threshold tuning, calibration, final eval | Sathwik | Test-set metrics table + confusion matrices |
| 10 | Docker Compose, Postgres, deployment doc | Kartik | `docker compose up` serves full app |
| 11 | Documentation, report, demo script, tests | All | Thesis chapters drafted |
| 12 | Buffer, presentation, demo rehearsal | All | Final PPT + video |

## 5. Repo layout

```
knee-mri-detect/
├── PLAN.md                 ← this file
├── README.md
├── docs/                   proposal, abstract, architecture notes
├── ml/                     training code (runs on Colab/Kaggle or local GPU)
│   ├── data/               dataset classes, transforms, MRNet loader
│   ├── training/           train.py, evaluate.py, config.yaml
│   ├── explain/            gradcam.py
│   └── models/             saved .pt weights + metrics.json (git-ignored weights)
├── backend/                FastAPI app
│   └── app/{api,core,ml,services}
├── frontend/               React + Vite dashboard
├── tests/                  pytest
├── notebooks/              EDA / experiments
├── data/                   raw + processed (git-ignored)
├── docker-compose.yml
└── Makefile
```

## 6. Evaluation plan

Per label: AUROC (primary, comparable to MRNet paper), sensitivity/specificity at
chosen threshold, F1. Report 5-fold or official MRNet split. Inference latency on
CPU (target < 10 s per exam). Qualitative Grad-CAM review with the internal guide.

## 7. Risks and mitigations

- **Dataset access delay** → start with the public MRNet sample / synthetic noise
  tensors so the pipeline is testable; request access on day 1.
- **8 GB RAM laptops** → train on Colab, run inference on CPU with batch = 1 plane.
- **Label scope gap vs proposal** → document clearly; N-label design lets us add
  heads if another dataset is obtained.
- **Overfitting (1,130 train exams)** → pretrained backbone, heavy augmentation,
  freeze early blocks, early stopping on val AUC.
- **Clinical disclaimer** → every result screen and PDF carries "for research /
  decision support only".

## 8. Definition of done

Trained weights + metrics, `docker compose up` launches dashboard, an exam can be
uploaded → predicted → explained → downloaded as PDF, tests pass in CI, and the
documentation matches what was built.
