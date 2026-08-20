from pathlib import Path
import numpy as np
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, Response
import cv2
from ..core.db import SessionLocal, Exam
from ..core.config import STORAGE_DIR, PLANES, LABELS, THRESHOLDS
from ..services.inference import predict, get_model
from ..services.report import build_report

router = APIRouter(prefix="/api/v1/exams", tags=["exams"])


@router.post("")
async def create_exam(bg: BackgroundTasks, patient_ref: str = Form(""),
                      sagittal: UploadFile | None = File(None), coronal: UploadFile | None = File(None), axial: UploadFile | None = File(None)):
    files = {p: f for p, f in zip(PLANES, (sagittal, coronal, axial)) if f}
    if not files:
        raise HTTPException(400, "Upload at least one plane as a .npy stack")
    db = SessionLocal(); exam = Exam(patient_ref=patient_ref); db.add(exam); db.commit(); db.refresh(exam)
    d = Path(STORAGE_DIR) / str(exam.id); d.mkdir(parents=True, exist_ok=True)
    for p, f in files.items():
        (d / f"{p}.npy").write_bytes(await f.read())
    bg.add_task(run_inference, exam.id)
    return {"id": exam.id, "status": exam.status}


def run_inference(exam_id: int):
    db = SessionLocal(); exam = db.get(Exam, exam_id); d = Path(STORAGE_DIR) / str(exam_id)
    try:
        stacks = {p: np.load(d / f"{p}.npy") for p in PLANES if (d / f"{p}.npy").exists()}
        preds, tensors = predict(stacks)
        cams, meta = {}, {}
        try:
            from ml.explain.gradcam import gradcam_overlay
            import cv2
            plane, x = next(iter(tensors.items()))
            for i, label in enumerate(LABELS):
                if preds[label] >= THRESHOLDS[label]:
                    s, overlay = gradcam_overlay(get_model(plane), x, i)
                    out = d / f"gradcam_{label}.png"; cv2.imwrite(str(out), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)); cams[label] = str(out)
                    meta = {"plane": plane, "slices": {**meta.get("slices", {}), label: s}}
        except Exception as e:  # explainability is best-effort
            print("gradcam failed:", e)
        rp = d / "report.pdf"; build_report(str(rp), exam_id, exam.patient_ref, preds, cams)
        exam.predictions, exam.gradcam, exam.gradcam_meta, exam.report_path, exam.status = preds, cams, meta, str(rp), "done"
    except Exception as e:
        exam.status = "error"; exam.predictions = {"error": str(e)}
    db.commit()


def _planes(exam_id: int) -> dict[str, int]:
    """{plane: n_slices} for uploaded stacks; mmap so listing stays cheap."""
    d = Path(STORAGE_DIR) / str(exam_id)
    return {p: np.load(d / f"{p}.npy", mmap_mode="r").shape[0] for p in PLANES if (d / f"{p}.npy").exists()}


@router.get("")
def list_exams(limit: int = 50):
    rows = SessionLocal().query(Exam).order_by(Exam.id.desc()).limit(limit).all()
    return [{"id": e.id, "patient_ref": e.patient_ref, "status": e.status, "created_at": e.created_at, "planes": list(_planes(e.id)),
             "flagged": sum(v >= THRESHOLDS[k] for k, v in e.predictions.items() if k in THRESHOLDS) if e.status == "done" else None} for e in rows]


@router.get("/{exam_id}")
def get_exam(exam_id: int):
    exam = SessionLocal().get(Exam, exam_id)
    if not exam:
        raise HTTPException(404)
    return {"id": exam.id, "status": exam.status, "predictions": exam.predictions, "patient_ref": exam.patient_ref,
            "gradcam": {k: f"/api/v1/exams/{exam.id}/gradcam/{k}" for k in exam.gradcam}, "gradcam_meta": exam.gradcam_meta,
            "thresholds": THRESHOLDS, "created_at": exam.created_at, "planes": _planes(exam.id)}


@router.get("/{exam_id}/slice/{plane}/{i}")
def get_slice(exam_id: int, plane: str, i: int):
    f = Path(STORAGE_DIR) / str(exam_id) / f"{plane}.npy"
    if plane not in PLANES or not f.exists():
        raise HTTPException(404)
    stack = np.load(f, mmap_mode="r")
    if not 0 <= i < stack.shape[0]:
        raise HTTPException(404)
    img = cv2.normalize(np.asarray(stack[i], dtype=np.float32), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return Response(cv2.imencode(".png", img)[1].tobytes(), media_type="image/png",
                    headers={"Cache-Control": "public, max-age=31536000, immutable"})


@router.get("/{exam_id}/gradcam/{label}")
def get_gradcam(exam_id: int, label: str):
    exam = SessionLocal().get(Exam, exam_id)
    if not exam or label not in exam.gradcam:
        raise HTTPException(404)
    return FileResponse(exam.gradcam[label])


@router.get("/{exam_id}/report")
def get_report(exam_id: int):
    exam = SessionLocal().get(Exam, exam_id)
    if not exam or not exam.report_path:
        raise HTTPException(404)
    return FileResponse(exam.report_path, media_type="application/pdf", filename=f"exam_{exam_id}_report.pdf")
