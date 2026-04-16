"""
Frozen ESM-2 + ChemBERTa probe for DTI prediction across temporal test subsets.

Requires:
  - phase3_temporal_split

Outputs:
  - results/esm2_probe_predictions.parquet
  - results/esm2_probe_metrics.json

Sentinels:
  - writes checkpoints/phase4_esm2_probe.done
"""

from __future__ import annotations

import json
import shutil
import traceback
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, EsmModel, EsmTokenizer

from utils.logging_utils import setup_logging
from utils.metrics import bootstrap_pearson_ci, concordance_index, pearson_r
from utils.sentinel import require_sentinel, write_sentinel
from utils.smiles import append_invalid_smiles, validate_smiles
from utils.status import StatusUpdater


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class PairDataset(Dataset):
    def __init__(
        self,
        protein_idx: np.ndarray,
        compound_idx: np.ndarray,
        y: np.ndarray,
        protein_emb: np.ndarray,
        compound_emb: np.ndarray,
    ):
        self.protein_idx = protein_idx.astype(np.int64)
        self.compound_idx = compound_idx.astype(np.int64)
        self.y = y.astype(np.float32)
        self.protein_emb = protein_emb.astype(np.float32)
        self.compound_emb = compound_emb.astype(np.float32)

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, i: int):
        p = self.protein_emb[int(self.protein_idx[i])]
        c = self.compound_emb[int(self.compound_idx[i])]
        x = np.concatenate([p, c], axis=0).astype(np.float32)
        return torch.from_numpy(x), torch.tensor(self.y[i], dtype=torch.float32)


def _select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_splits() -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    parts = []
    for ds in ["davis", "kiba"]:
        p = PROJECT_ROOT / "data" / "splits" / f"{ds}_temporal_splits.parquet"
        tmp = pd.read_parquet(p)
        tmp["dataset"] = ds
        parts.append(tmp)
    # Optional clean holdout arm (post-2021 ChEMBL34, Tanimoto < 0.4 vs ChEMBL27)
    holdout_path = PROJECT_ROOT / "data" / "splits" / "holdout_temporal_splits.parquet"
    if holdout_path.exists():
        tmp = pd.read_parquet(holdout_path)
        tmp["dataset"] = "holdout"
        parts.append(tmp)
    df = pd.concat(parts, ignore_index=True)
    df["split"] = df["split"].astype(str)
    df["subset"] = df["subset"].astype(str)
    df["dataset"] = df["dataset"].astype(str)
    df["smiles"] = df["smiles"].astype(str)
    df["sequence"] = df["sequence"].astype(str)
    df["affinity"] = pd.to_numeric(df["affinity"], errors="coerce")
    df = df[np.isfinite(df["affinity"].to_numpy())]

    # Per-dataset z-score normalization using train-split statistics.
    # Davis pKd (~2–12) and KIBA scores (~0–17) are on different scales; joint training
    # without normalization causes the model to anchor predictions to one dataset's range,
    # yielding near-zero Pearson r on the other. Normalization is affinity-scale-invariant
    # with respect to Pearson r, so test metrics are unaffected in interpretation.
    norm_stats: dict[str, dict[str, float]] = {}
    df = df.copy()
    df["affinity_raw"] = df["affinity"].copy()
    for ds in df["dataset"].unique():
        train_mask = (df["dataset"] == ds) & (df["split"] == "train")
        train_vals = df.loc[train_mask, "affinity"].to_numpy(dtype=float)
        if len(train_vals) == 0:
            # Holdout has no train split — compute stats from its own test pairs
            train_vals = df.loc[df["dataset"] == ds, "affinity"].to_numpy(dtype=float)
        mu = float(np.mean(train_vals))
        sigma = float(np.std(train_vals))
        if sigma < 1e-8:
            sigma = 1.0
        norm_stats[ds] = {"mean": mu, "std": sigma}
        ds_mask = df["dataset"] == ds
        df.loc[ds_mask, "affinity"] = (df.loc[ds_mask, "affinity"] - mu) / sigma

    return df, norm_stats


def _preflight_disk(logger) -> None:
    usage = shutil.disk_usage("/Users")
    free = usage.free
    logger.info("Disk free at /Users: %.2f GB", free / (1024**3))
    assert free > 20 * 1024**3, "Need > 20 GB free on /Users for model downloads and checkpoints."


def _embed_esm2(
    sequences: list[str],
    out_path: Path,
    device: torch.device,
    logger,
    status: StatusUpdater,
    phase: str,
) -> np.ndarray:
    if out_path.exists():
        logger.info("Loading cached ESM2 embeddings from %s.", out_path)
        return np.load(out_path)

    logger.info("Loading ESM-2 model/tokenizer...")
    tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
    model = EsmModel.from_pretrained("facebook/esm2_t33_650M_UR50D")
    model.eval()
    model.to(device)

    embs = []
    batch_size = 8
    for start in tqdm(range(0, len(sequences), batch_size), desc="ESM2 embeddings", unit="batch"):
        batch = sequences[start : start + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=1022)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            cls = outputs.last_hidden_state[:, 0, :].detach().float().cpu().numpy()
        embs.append(cls)
        if device.type == "cuda":
            torch.cuda.empty_cache()
        status.update(phase, "running", progress=min(0.49, 0.49 * (start + batch_size) / max(1, len(sequences))))

    arr = np.concatenate(embs, axis=0).astype(np.float32)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, arr)
    logger.info("Saved ESM2 embeddings to %s.", out_path)
    return arr


def _embed_chemberta(
    smiles_list: list[str],
    out_path: Path,
    device: torch.device,
    logger,
    status: StatusUpdater,
    phase: str,
) -> np.ndarray:
    if out_path.exists():
        logger.info("Loading cached ChemBERTa embeddings from %s.", out_path)
        return np.load(out_path)

    logger.info("Loading ChemBERTa model/tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
    model = AutoModel.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
    model.eval()
    model.to(device)

    embs = []
    batch_size = 32
    for start in tqdm(range(0, len(smiles_list), batch_size), desc="ChemBERTa embeddings", unit="batch"):
        batch = smiles_list[start : start + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=128)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            cls = outputs.last_hidden_state[:, 0, :].detach().float().cpu().numpy()
        embs.append(cls)
        if device.type == "cuda":
            torch.cuda.empty_cache()
        status.update(phase, "running", progress=min(0.79, 0.49 + 0.30 * (start + batch_size) / max(1, len(smiles_list))))

    arr = np.concatenate(embs, axis=0).astype(np.float32)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, arr)
    logger.info("Saved ChemBERTa embeddings to %s.", out_path)
    return arr


def _train_mlp(
    train_df: pd.DataFrame,
    protein_to_idx: dict[str, int],
    smiles_to_idx: dict[str, int],
    protein_emb: np.ndarray,
    compound_emb: np.ndarray,
    device: torch.device,
    logger,
    status: StatusUpdater,
    phase: str,
) -> nn.Module:
    protein_idx = train_df["sequence"].map(lambda s: protein_to_idx[s]).to_numpy()
    compound_idx = train_df["smiles"].map(lambda s: smiles_to_idx[s]).to_numpy()
    y = train_df["affinity"].to_numpy(dtype=np.float32)

    in_dim = int(protein_emb.shape[1] + compound_emb.shape[1])
    model = nn.Sequential(
        nn.Linear(in_dim, 512),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(256, 1),
    ).to(device)

    ds = PairDataset(protein_idx, compound_idx, y, protein_emb, compound_emb)
    dl = DataLoader(ds, batch_size=512, shuffle=True, num_workers=0)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    epochs = 50
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for x, yy in dl:
            x = x.to(device)
            yy = yy.to(device).view(-1, 1)
            pred = model(x)
            loss = loss_fn(pred, yy)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu().item()))
        avg = float(np.mean(losses)) if losses else float("nan")
        logger.info("Epoch %d/%d - train MSE: %.6f", epoch, epochs, avg)
        status.update(phase, "running", progress=min(0.95, 0.80 + 0.15 * epoch / epochs))

    model.eval()
    return model


def _predict(
    model: nn.Module,
    df: pd.DataFrame,
    protein_to_idx: dict[str, int],
    smiles_to_idx: dict[str, int],
    protein_emb: np.ndarray,
    compound_emb: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    protein_idx = df["sequence"].map(lambda s: protein_to_idx[s]).to_numpy(dtype=np.int64)
    compound_idx = df["smiles"].map(lambda s: smiles_to_idx[s]).to_numpy(dtype=np.int64)
    y = df["affinity"].to_numpy(dtype=np.float32)
    ds = PairDataset(protein_idx, compound_idx, y, protein_emb, compound_emb)
    dl = DataLoader(ds, batch_size=1024, shuffle=False, num_workers=0)
    preds = []
    model.eval()
    with torch.no_grad():
        for x, _ in dl:
            x = x.to(device)
            p = model(x).view(-1).detach().float().cpu().numpy()
            preds.append(p)
    return np.concatenate(preds, axis=0)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    r, ci = bootstrap_pearson_ci(y_true, y_pred, n_boot=1000, seed=0)
    mse = float(mean_squared_error(y_true, y_pred)) if y_true.size > 0 else float("nan")
    c = float(concordance_index(y_true, y_pred))
    out: dict[str, Any] = {"pearson_r": r, "mse": mse, "ci": c}
    if ci is not None:
        out["pearson_r_ci95"] = [ci[0], ci[1]]
    else:
        out["pearson_r_ci95"] = None
    return out


def main() -> int:
    phase = "phase4_esm2_probe"
    prereq = "phase3_temporal_split"
    status = StatusUpdater(PROJECT_ROOT)
    logger = setup_logging(PROJECT_ROOT, phase)
    status.update(phase, "running", progress=0.0)

    try:
        try:
            require_sentinel(prereq)
        except RuntimeError as e:
            logger.warning(str(e))
            status.update(phase, "blocked", error=str(e))
            return 0

        _preflight_disk(logger)

        device = _select_device()
        logger.info("Using device: %s", device)

        df, norm_stats = _load_splits()
        logger.info("Affinity normalization stats: %s", norm_stats)

        # Validate SMILES before embedding
        canon_smiles: list[Optional[str]] = []
        drop = np.zeros(len(df), dtype=bool)
        for i, smi in enumerate(df["smiles"].tolist()):
            ok, canon = validate_smiles(smi)
            if not ok or canon is None:
                append_invalid_smiles(PROJECT_ROOT, source=phase, key=f"row_{i}", smiles=smi, reason="rdkit_invalid")
                canon_smiles.append(None)
                drop[i] = True
            else:
                canon_smiles.append(canon)
        df = df.copy()
        df["smiles"] = canon_smiles
        df = df[~drop]

        sequences = df["sequence"].dropna().astype(str).unique().tolist()
        smiles_list = df["smiles"].dropna().astype(str).unique().tolist()

        protein_to_idx = {s: i for i, s in enumerate(sequences)}
        smiles_to_idx = {s: i for i, s in enumerate(smiles_list)}

        prot_emb_path = PROJECT_ROOT / "checkpoints" / "esm2_embeddings.npy"
        chem_emb_path = PROJECT_ROOT / "checkpoints" / "chembert_embeddings.npy"

        protein_emb = _embed_esm2(sequences, prot_emb_path, device=device, logger=logger, status=status, phase=phase)
        compound_emb = _embed_chemberta(smiles_list, chem_emb_path, device=device, logger=logger, status=status, phase=phase)

        # Train on all training pairs — contamination labels are for characterization, not filtering
        train_df = df[df["split"] == "train"].copy()
        if train_df.empty:
            raise RuntimeError("No training pairs found. Check Phase 3 split criteria.")
        logger.info("Training pairs: %d across datasets %s", len(train_df), sorted(train_df["dataset"].unique().tolist()))

        mlp = _train_mlp(
            train_df,
            protein_to_idx,
            smiles_to_idx,
            protein_emb,
            compound_emb,
            device=device,
            logger=logger,
            status=status,
            phase=phase,
        )

        # Evaluate per dataset (Davis, KIBA, holdout) + contamination class breakdown
        test_all = df[df["split"] == "test"].copy()

        metrics: dict[str, Any] = {}
        pred_frames: list[pd.DataFrame] = []

        for ds_name in sorted(test_all["dataset"].unique().tolist()):
            ds_test = test_all[test_all["dataset"] == ds_name].copy().reset_index(drop=True)
            if ds_test.empty:
                logger.warning("Empty test set for dataset=%s, skipping.", ds_name)
                continue
            preds = _predict(mlp, ds_test, protein_to_idx, smiles_to_idx, protein_emb, compound_emb, device=device)
            ds_test = ds_test.copy()
            ds_test["y_pred"] = preds.astype(float)
            pred_frames.append(ds_test)
            metrics[ds_name] = _metrics(ds_test["affinity"].to_numpy(dtype=float), preds.astype(float))
            logger.info("Dataset=%s n=%d Pearson=%.4f", ds_name, len(ds_test), metrics[ds_name]["pearson_r"])

            # Per contamination class within benchmarks (for DiD heterogeneity check)
            if ds_name != "holdout":
                for subset_name in ["contaminated", "clean"]:
                    sub = ds_test[ds_test["subset"] == subset_name]
                    if len(sub) >= 10:
                        key = f"{ds_name}_{subset_name}"
                        metrics[key] = _metrics(sub["affinity"].to_numpy(dtype=float), sub["y_pred"].to_numpy())
                        logger.info("  %s n=%d Pearson=%.4f", key, len(sub), metrics[key]["pearson_r"])

        results_dir = PROJECT_ROOT / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        metrics["_norm_stats"] = norm_stats
        metrics_path = results_dir / "esm2_probe_metrics.json"
        metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True))
        logger.info("Wrote %s.", metrics_path)

        pred_df = pd.concat(pred_frames, ignore_index=True) if pred_frames else pd.DataFrame()
        pred_path = results_dir / "esm2_probe_predictions.parquet"
        pred_df.to_parquet(pred_path, index=False)
        logger.info("Wrote %s (%d rows).", pred_path, len(pred_df))

        write_sentinel(phase)
        status.update(phase, "completed", progress=1.0)
        return 0

    except Exception as e:  # noqa: BLE001
        tb = traceback.format_exc()
        logger.error("Fatal error: %s", e)
        logger.error(tb)
        status.update(phase, "error", error=str(e))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
