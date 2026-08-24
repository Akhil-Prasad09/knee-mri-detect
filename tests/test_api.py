from fastapi.testclient import TestClient
from backend.app.main import app


def test_health():
    with TestClient(app) as c:
        assert c.get("/health").json() == {"status": "ok"}


def test_samples(tmp_path, monkeypatch):
    import numpy as np
    (tmp_path / "acl_tear").mkdir()
    np.save(tmp_path / "acl_tear" / "sagittal.npy", (np.random.rand(4, 64, 64) * 255).astype(np.uint8))
    monkeypatch.setenv("SAMPLES_DIR", str(tmp_path)); monkeypatch.setenv("STORAGE_DIR", str(tmp_path / "st"))
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/t.db")
    import importlib, backend.app.core.config as cfg, backend.app.core.db as db, backend.app.api.exams as ex, backend.app.main as main
    for m in (cfg, db, ex, main): importlib.reload(m)
    from fastapi.testclient import TestClient
    with TestClient(main.app) as c:
        assert c.get("/api/v1/samples").json() == [{"id": "acl_tear", "planes": ["sagittal"]}]
        r = c.post("/api/v1/samples/acl_tear")
        assert r.status_code == 200 and r.json()["status"] == "pending"
        assert c.post("/api/v1/samples/nope").status_code == 404
