# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

Primary (confirmed): project examiners and demo viewers of a B.Tech major project, watching a live walkthrough on a laptop/projector. Secondary: radiologists/clinicians as the notional end user of the decision-support tool.

## Product Purpose

Upload a knee MRI exam (sagittal / coronal / axial `.npy` slice stacks, MRNet layout), run a deep-learning model (EfficientNet-B3 per plane, ensembled), and return per-label probabilities for **abnormal**, **ACL tear**, **meniscus tear**, with Grad-CAM heatmaps and a downloadable PDF report. Success for the demo: the examiner understands the pipeline end to end in under two minutes and trusts that results are explainable.

## Positioning

Explainable (Grad-CAM) multi-plane ensemble with tuned per-label thresholds, wrapped in a complete clinical-style workflow (intake → inference → report), not just a notebook.

## Operating Context

Demo on synthetic MRNet data until Stanford AIMI grants real dataset access. Backend FastAPI at `/api/v1/exams`; inference runs as a background task, frontend polls every 2 s. Storage per exam under `storage/<id>/`.

## Capabilities and Constraints

- POST `/api/v1/exams` (multipart: `patient_ref`, `sagittal|coronal|axial` files) → `{id, status}`
- GET `/api/v1/exams/{id}` → `{id, status: pending|done|error, predictions, gradcam{label:url}, thresholds, created_at}`
- GET `/api/v1/exams/{id}/gradcam/{label}` → PNG; GET `/api/v1/exams/{id}/report` → PDF
- Planned (this build): GET `/api/v1/exams` list; GET `/api/v1/exams/{id}/slice/{plane}/{i}` PNG for in-browser slice scrubbing
- Labels and thresholds come from training config / `ml/models/eval.json`; never hardcode.
- Frontend: React 18 + Vite, no UI library. Keep it dependency-light (ponytail convention).
- Must carry a visible "research / decision support only, not a medical device" disclaimer.

## Brand Commitments

Name: **knee-mri-detect**. Visual direction pinned by user: Apple HIG-inspired, light.

## Evidence on Hand

No real patient data, clinical validation, or testimonials. Metrics in `ml/models/sagittal_metrics.json` / `eval.json` are from synthetic smoke training and must be labelled as such if shown.

## Product Principles

1. Explain every number: a probability is shown next to its threshold and its heatmap.
2. Status is always visible: pending / done / error never ambiguous.
3. Demo-legible: one screen tells the whole story without narration.
4. Honest: synthetic data and research-only status are stated, not hidden.
