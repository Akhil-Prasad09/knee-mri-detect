import argparse, json, yaml, torch, numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from ml.data.mrnet import MRNetDataset
from ml.training.model import KneeMRINet


def run_epoch(model, loader, crit, opt, device, train):
    model.train(train); ys, ps, losses = [], [], []
    with torch.set_grad_enabled(train):
        for x, y in tqdm(loader, leave=False):
            x, y = x[0].to(device), y[0].to(device)
            logits = model(x); loss = crit(logits, y)
            if train:
                opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item()); ys.append(y.cpu().numpy()); ps.append(torch.sigmoid(logits).cpu().numpy())
    ys, ps = np.array(ys), np.array(ps)
    aucs = [roc_auc_score(ys[:, i], ps[:, i]) for i in range(ys.shape[1])]
    return float(np.mean(losses)), aucs


def main(cfg, plane):
    torch.manual_seed(cfg["seed"]); device = "cuda" if torch.cuda.is_available() else "cpu"
    tr = MRNetDataset(cfg["data_dir"], "train", plane, cfg["labels"], cfg["img_size"], train=True)
    va = MRNetDataset(cfg["data_dir"], "valid", plane, cfg["labels"], cfg["img_size"])
    pos = torch.tensor([(tr.df[l] == 0).sum() / max((tr.df[l] == 1).sum(), 1) for l in cfg["labels"]], dtype=torch.float32)
    model = KneeMRINet(len(cfg["labels"]), cfg["backbone"]).to(device)
    crit = torch.nn.BCEWithLogitsLoss(pos_weight=pos.to(device))
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    dl = lambda ds, sh: DataLoader(ds, batch_size=1, shuffle=sh, num_workers=cfg["num_workers"])
    best, out = -1, Path(cfg["out_dir"]); out.mkdir(parents=True, exist_ok=True)
    for ep in range(cfg["epochs"]):
        tl, _ = run_epoch(model, dl(tr, True), crit, opt, device, True)
        vl, aucs = run_epoch(model, dl(va, False), crit, opt, device, False)
        print(f"[{plane}] epoch {ep} train {tl:.3f} val {vl:.3f} auc {dict(zip(cfg['labels'], map(lambda a: round(a,3), aucs)))}")
        if np.mean(aucs) > best:
            best = np.mean(aucs); torch.save(model.state_dict(), out / f"{plane}.pt")
            json.dump({"plane": plane, "auc": dict(zip(cfg["labels"], aucs)), "epoch": ep}, open(out / f"{plane}_metrics.json", "w"), indent=2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default="ml/training/config.yaml"); ap.add_argument("--plane")
    a = ap.parse_args(); cfg = yaml.safe_load(open(a.config))
    for p in ([a.plane] if a.plane else cfg["planes"]):
        main(cfg, p)
