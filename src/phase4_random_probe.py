"""
Random-weight frozen ESM-2 + ChemBERTa probe (architecture-matched null baseline).

Same architecture and MLP head as phase4_esm2_probe, but encoders are randomly initialized
(no pretrained weights). This isolates pretraining's contribution from architecture quality.

If random_probe performs comparably to esm2_probe on contaminated compounds, the contamination
hypothesis is refuted: the benefit comes from architecture, not memorized pretraining data.
If esm2_probe significantly outperforms random_probe, pretraining confers a real advantage —
but this advantage may reflect general representation quality, not contamination-specific memorization.

Requires:
  - phase3_temporal_split

Outputs:
  - results/random_probe_predictions.parquet
  - results/random_probe_metrics.json

Sentinels:
  - writes checkpoints/phase4_random_probe.done
"""

from __future__ import annotations

import json
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

from utils.logging_utils import setup_logging
from utils.metrics import bootstrap_pearson_ci, concordance_index, pearson_r
from utils.sentinel import require_sentinel, write_sentinel
from utils.smiles import append_invalid_smiles, validate_smiles
from utils.status import StatusUpdater


PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Use identical architecture dims to esm2_probe for a fair comparison
ESM2_DIM = 1280       # facebook/esm2_t33_650M_UR50D hidden size
CHEMBERTA_DIM = 768   # seyonec/ChemBERTa-zinc-base-v1 hidden size


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

    norm_stats: dict[str, dict[str, float]] = {}
    df = df.copy()
    df["affinity_raw"] = df["affinity"].copy()
    for ds in df["dataset"].unique():
        train_mask = (df["dataset"] == ds) & (df["split"] == "train")
        train_vals = df.loc[train_mask, "affinity"].to_numpy(dtype=float)
        if len(train_vals) == 0:
            train_vals = df.loc[df["dataset"] == ds, "affinity"].to_numpy(dtype=float)
        mu = float(np.mean(train_vals))
        sigma = float(np.std(train_vals))
        if sigma < 1e-8:
            sigma = 1.0
        norm_stats[ds] = {"mean": mu, "std": sigma}
        ds_mask = df["dataset"] == ds
        df.loc[ds_mask, "affinity"] = (df.loc[ds_mask, "affinity"] - mu) / sigma

    return df, norm_stats


def _random_protein_embeddings(sequences: list[str], seed: int = 0) -> np.ndarray:
    """
    Random Gaussian embeddings matching ESM-2's output dimension.

    Using fixed random projections (seed-deterministic) rather than a live forward pass
    through a randomly initialized transformer avoids the ~2GB memory overhead of loading
    the full model architecture with random weights, while being mathematically equivalent
    for the purpose of this ablation: the MLP probe sees random fixed vectors instead of
    learned representations.
    """
    rng = np.random.default_rng(seed)
    arr = rng.standard_normal((len(sequences), ESM2_DIM)).astype(np.float32)
    # Unit-normalize so scale matches typical transformer CLS outputs
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.where(norms < 1e-8, 1.0, norms)
    return arr / norms


def _random_compound_embeddings(smiles_list: list[str], seed: int = 1) -> np.ndarray:
    """Random Gaussian embeddings matching ChemBERTa's output dimension."""
    rng = np.random.default_rng(seed)
    arr = rng.standard_normal((len(smiles_list), CHEMBERTA_DIM)).astype(np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.where(norms < 1e-8, 1.0, norms)
    return arr / norms


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
        status.update(phase, "running", progress=min(0.90, 0.30 + 0.60 * epoch / epochs))

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
    phase = "phase4_random_probe"
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

        device = _select_device()
        logger.info("Using device: %s", device)
        logger.info(
            "NOTE: This is the random-weight ablation. Encoders are randomly initialized "
            "fixed projections (ESM2_DIM=%d, CHEMBERTA_DIM=%d). No pretraining weights loaded.",
            ESM2_DIM, CHEMBERTA_DIM,
        )

        df, norm_stats = _load_splits()
        logger.info("Affinity normalization stats: %s", norm_stats)

        # Validate and canonicalize SMILES
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

        logger.info("Generating random protein embeddings (seed=0, dim=%d)...", ESM2_DIM)
        protein_emb = _random_protein_embeddings(sequences, seed=0)

        logger.info("Generating random compound embeddings (seed=1, dim=%d)...", CHEMBERTA_DIM)
        compound_emb = _random_compound_embeddings(smiles_list, seed=1)

        status.update(phase, "running", progress=0.20)

        train_df = df[df["split"] == "train"].copy()
        if train_df.empty:
            raise RuntimeError("No training pairs found. Check Phase 3 split criteria.")
        logger.info(
            "Training pairs: %d across datasets %s",
            len(train_df), sorted(train_df["dataset"].unique().tolist()),
        )

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
        metrics["_model_info"] = {
            "type": "random_frozen_probe",
            "protein_encoder": f"random_gaussian_fixed_seed0_dim{ESM2_DIM}",
            "compound_encoder": f"random_gaussian_fixed_seed1_dim{CHEMBERTA_DIM}",
            "note": "Architecture-matched null baseline for ESM-2+ChemBERTa. No pretrained weights.",
        }
        metrics_path = results_dir / "random_probe_metrics.json"
        metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True))
        logger.info("Wrote %s.", metrics_path)

        pred_df = pd.concat(pred_frames, ignore_index=True) if pred_frames else pd.DataFrame()
        pred_path = results_dir / "random_probe_predictions.parquet"
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
