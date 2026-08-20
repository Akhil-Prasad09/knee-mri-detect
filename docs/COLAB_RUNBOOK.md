# Runbook: train the real model on Colab, driven from Claude Code + Claude in Chrome

Written 2026-08-20 for the next session. Read HANDOFF.md first for repo state.

## Known gotchas (hit on 2026-08-20 run)
- The repo was private, so the Colab VM could not clone it → made public with the user's explicit consent.
  Cell 1 now asserts the clone worked instead of failing silently.
- The Kaggle secret was pasted with a trailing newline → `Invalid header value`. Cell 3 `.strip()`s it.
- Colab's websocket drops repeatedly; execution on the VM continues but control commands do not land. Do NOT
  reload a GitHub-loaded notebook to fix it — the reload gives the notebook a new identity and orphans the
  running session (which keeps holding the only free GPU). **Use File > Save a copy in Drive BEFORE running**,
  so the notebook has a stable identity and reloads reattach.
- Interrupting cell 4 only kills the current `!python` subprocess; the Python `for` loop moves on to the next
  plane. To stop the whole run, interrupt twice or restart the session.
- A plane interrupted after epoch 0 still leaves a `.pt` on Drive, and the skip-if-exists check then treats it
  as finished. `FRESH = True` in cell 4 wipes the models dir for a clean run.

## Why this path
- Local training was tried: EfficientNet-B3 swaps on 16 GB unified memory (1 s → 35 s/step); B0 fits but
  pegs the laptop for ~6 h. Decision: train on Colab, keep the laptop for serving.
- Dataset: Kaggle mirror `cjinny/mrnet-v1` = Stanford MRNet v1.0 (1130 train / 120 valid × 3 planes).
  Already downloaded locally to `data/raw/MRNet-v1.0/` (git-ignored) — useful for local evaluate/tests.

## Hard rules for the browser session
- Never type, paste, or upload credentials (Google password, Kaggle token, `kaggle.json`). The notebook
  reads the Kaggle key from **Colab Secrets**, which the user sets by hand.
- Ask before: clicking "Run anyway" on the "notebook not authored by Google" dialog, granting the Drive
  OAuth prompt (user clicks it), downloading `models.zip`.
- If a Colab cell errors, read the traceback in the cell output; do not blindly re-run.

## Preconditions (user)
1. Logged into Google in Chrome.
2. Kaggle account with API token created (kaggle.com → Settings → API → Create New Token).
3. In Colab, Secrets panel: `KAGGLE_API_TOKEN` (the `KGAT_...` value), notebook access ON. Legacy alternative:
   `KAGGLE_USERNAME` + `KAGGLE_KEY`. **The user enters this. Claude never types, pastes, or reads back a token** —
   if the user offers one in chat, decline and ask them to put it in Secrets themselves (and rotate it).

## Steps

### Phase A — launch (≈10 min, interactive)
1. `mcp__claude-in-chrome__tabs_context_mcp`, create a tab, navigate to
   `https://colab.research.google.com/github/Akhil-Prasad09/knee-mri-detect/blob/main/notebooks/train_colab.ipynb`
2. Runtime → Change runtime type → T4 GPU → Save. Verify with screenshot.
3. Confirm secrets exist: open 🔑 Secrets panel, check the name(s) listed with toggle on (values stay hidden — do not
   reveal them). If missing → stop, ask user.
4. Run cell 1 (clone + pip + `nvidia-smi -L`). Expect a Tesla T4 line. pip takes ~2 min.
5. Run cell 2 (Drive mount). A Google OAuth popup appears → **user clicks through**. Expect `Mounted at /content/drive`.
6. Run cell 3 (Kaggle download, ~3–5 min at Colab bandwidth). Expect `1130`.
   - `401`/`403` → secret wrong; ask user to fix, re-run cell 3 only.
   - `ValueError: Invalid header value b'Bearer ...\r\n'` → the secret value has a trailing newline. The cell
     already `.strip()`s it; if an older notebook copy is loaded in the browser, edit the cell to add `.strip()`.

### Phase B — train (≈2.5–3 h, mostly waiting)
7. Run cell 4. Per plane: 20 epochs, each prints `[plane] epoch N train … val … auc {…}`.
   Poll with a Monitor/ScheduleWakeup every 15–20 min (not faster): screenshot the bottom of the cell output.
   Healthy: epoch lines advancing, ~2–3 min/epoch on T4. Report per-plane best AUC when each finishes.
   - Session disconnected / "Runtime disconnected": Reconnect, Runtime → Run all. Cell 4 skips planes whose
     `.pt` is already on Drive (best checkpoint saved every epoch), so at most one plane's progress is lost.
   - CUDA OOM (unlikely on T4 with batch 1): lower `img_size` to 192 in `colab.yaml` cell and re-run.
   - Keep the tab alive; Colab idles out on inactivity — a scroll/click every poll is enough.
8. Run cell 5 (evaluate). Expect `eval.json` printed with `sagittal/coronal/axial/ensemble`, each label with
   auc/threshold/f1/sensitivity/specificity. Sanity: ensemble AUCs should be roughly abnormal ≥0.90,
   acl ≥0.90, meniscus ≥0.80 (MRNet paper: 0.94/0.97/0.85). Far lower → something's off; report, don't ship.
9. Run cell 6 (zip). `models.zip` is now in Drive at `MyDrive/knee-mri-detect/models.zip`.

### Phase C — bring weights home (≈5 min)
10. Get `models.zip` onto the laptop. Preferred: open drive.google.com in the tab, locate
    `knee-mri-detect/models.zip`, **ask the user**, then download (lands in `~/Downloads`). Alternative: user
    downloads it themselves.
11. Terminal: `unzip -o ~/Downloads/models.zip -d ml/models/` → expect `sagittal.pt coronal.pt axial.pt
    *_metrics.json eval.json`.
12. Verify end to end (all with the real config, no `TRAIN_CONFIG` override now):
    - `pytest -q` → 4 passed (e2e will now run B3 inference on CPU: ~20–40 s, fine).
    - `python -m ml.training.evaluate --config ml/training/config.yaml` locally against `data/raw` — should
      reproduce eval.json numbers (same valid split).
    - `make api` + `cd frontend && npm run dev`; upload a real valid exam
      (`data/raw/MRNet-v1.0/valid/{sagittal,coronal,axial}/1130.npy`, label rows in `valid-*.csv`), check the
      three verdicts against the CSV labels and that Grad-CAM shows a heatmap on real anatomy. Screenshot.
13. Commit: `ml/models/*_metrics.json` (tracked), updated HANDOFF.md with real AUCs. `*.pt` and `eval.json`
    stay git-ignored — note in HANDOFF where the weights live (Drive path) so another machine can fetch them.
    Consider: copy `eval.json` thresholds into README "Results" table.

## Done looks like
- `ml/models/{sagittal,coronal,axial}.pt` + `eval.json` present locally, tests green on the real model,
  dashboard verified on a real valid exam, HANDOFF.md updated with AUCs and the Drive location of the weights.
