---
title: Knee MRI Detect
emoji: 🦵
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Knee MRI abnormality detection with Grad-CAM (MRNet)
---

# Knee MRI Detect

EfficientNet-B3 ensemble over sagittal/coronal/axial planes, trained on Stanford's MRNet dataset
(ensemble AUC: abnormal 0.943 · ACL 0.960 · meniscus 0.847 on the MRNet validation split).
Upload MRNet-style `.npy` stacks, or click a bundled sample case. Grad-CAM heatmaps show which
slice and region drove each positive finding; a PDF report is generated per exam.

Inference runs on 2 CPU cores — an exam takes a couple of minutes. Exams are stored in ephemeral
`/tmp` and vanish on restart.

Source: https://github.com/Akhil-Prasad09/knee-mri-detect

**Research prototype for decision support. Not a medical device.** Sample images are from the
MRNet dataset (Stanford AIMI), included for demonstration.
