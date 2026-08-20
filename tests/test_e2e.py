"""Upload -> predict -> gradcam -> PDF. Skips if no trained weights present."""
import io, os, time, numpy as np, pytest
from pathlib import Path
from fastapi.testclient import TestClient

pytestmark = pytest.mark.skipif(not list(Path(os.getenv("MODEL_DIR", "ml/models")).glob("*.pt")), reason="no weights")


def test_upload_to_report(tmp_path, monkeypatch):
    monkeypatch.setenv("STORAGE_DIR", str(tmp_path)); monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/t.db")
    import importlib, backend.app.core.config as cfg, backend.app.core.db as db, backend.app.api.exams as ex, backend.app.main as main
    for m in (cfg, db, ex, main): importlib.reload(m)
    buf = io.BytesIO(); np.save(buf, (np.random.rand(6, 256, 256) * 255).astype(np.uint8))
    with TestClient(main.app) as c:
        r = c.post("/api/v1/exams", data={"patient_ref": "P1"}, files={"sagittal": ("s.npy", buf.getvalue())}); assert r.status_code == 200
        eid = r.json()["id"]
        for _ in range(60):
            j = c.get(f"/api/v1/exams/{eid}").json()
            if j["status"] != "pending": break
            time.sleep(1)
        assert j["status"] == "done", j
        assert set(j["predictions"]) == set(cfg.LABELS)
        assert set(j["gradcam"]) == {k for k, v in j["predictions"].items() if v >= cfg.THRESHOLDS[k]}
        assert c.get(f"/api/v1/exams/{eid}/report").headers["content-type"] == "application/pdf"
        lst = c.get("/api/v1/exams").json(); assert lst[0]["id"] == eid and lst[0]["planes"] == ["sagittal"] and lst[0]["flagged"] == len(j["gradcam"])
        assert j["planes"] == {"sagittal": 6}
        if j["gradcam"]: assert j["gradcam_meta"]["plane"] == "sagittal" and set(j["gradcam_meta"]["slices"]) == set(j["gradcam"])
        assert c.get(f"/api/v1/exams/{eid}/slice/sagittal/3").headers["content-type"] == "image/png"
        assert c.get(f"/api/v1/exams/{eid}/slice/coronal/0").status_code == 404
