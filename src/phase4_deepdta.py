"""
DeepDTA reimplementation (no pretrained components) as negative control.

Requires:
  - phase3_temporal_split

Outputs:
  - results/deepdta_predictions.parquet
  - results/deepdta_metrics.json

Sentinels:
  - writes checkpoints/phase4_deepdta.done
"""

from __future__ import annotations

import json
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from utils.logging_utils import setup_logging
from utils.metrics import bootstrap_pearson_ci, concordance_index
from utils.sentinel import require_sentinel, write_sentinel
from utils.smiles import append_invalid_smiles, validate_smiles
from utils.status import StatusUpdater


PROJECT_ROOT = Path(__file__).resolve().parent.parent


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


def _build_smiles_vocab(smiles_list: list[str]) -> dict[str, int]:
    chars = sorted(set("".join(smiles_list)))
    return {ch: i + 1 for i, ch in enumerate(chars)}  # 0 reserved for pad/unknown


def _build_protein_vocab() -> dict[str, int]:
    aas = list("ACDEFGHIKLMNPQRSTVWY")
    vocab = {aa: i + 1 for i, aa in enumerate(aas)}
    vocab["X"] = len(vocab) + 1
    return vocab


def _encode_smiles(smiles: str, vocab: dict[str, int], max_len: int = 85) -> np.ndarray:
    arr = np.zeros(max_len, dtype=np.int64)
    for i, ch in enumerate(smiles[:max_len]):
        arr[i] = int(vocab.get(ch, 0))
    return arr


def _encode_protein(seq: str, vocab: dict[str, int], max_len: int = 1200) -> np.ndarray:
    arr = np.zeros(max_len, dtype=np.int64)
    x = int(vocab["X"])
    for i, ch in enumerate(seq[:max_len]):
        arr[i] = int(vocab.get(ch, x))
    return arr


class DeepDTADataset(Dataset):
    def __init__(
        self,
        smiles: list[str],
        sequences: list[str],
        y: np.ndarray,
        smiles_vocab: dict[str, int],
        protein_vocab: dict[str, int],
    ):
        self.smiles = smiles
        self.sequences = sequences
        self.y = y.astype(np.float32)
        self.smiles_vocab = smiles_vocab
        self.protein_vocab = protein_vocab

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, i: int):
        s = _encode_smiles(self.smiles[i], self.smiles_vocab)
        p = _encode_protein(self.sequences[i], self.protein_vocab)
        return torch.from_numpy(s), torch.from_numpy(p), torch.tensor(self.y[i], dtype=torch.float32)


class SmilesEncoder(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, 128, padding_idx=0)
        self.conv1 = nn.Conv1d(128, 32, 4)
        self.conv2 = nn.Conv1d(32, 64, 6)
        self.conv3 = nn.Conv1d(64, 96, 8)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.emb(x)  # (B, L, 128)
        x = x.transpose(1, 2)  # (B, 128, L)
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.act(self.conv3(x))
        x = self.pool(x).squeeze(-1)  # (B, 96)
        return x


class ProteinEncoder(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, 128, padding_idx=0)
        self.conv1 = nn.Conv1d(128, 32, 4)
        self.conv2 = nn.Conv1d(32, 64, 6)
        self.conv3 = nn.Conv1d(64, 96, 8)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.emb(x)  # (B, L, 128)
        x = x.transpose(1, 2)  # (B, 128, L)
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.act(self.conv3(x))
        x = self.pool(x).squeeze(-1)  # (B, 96)
        return x


class DeepDTA(nn.Module):
    def __init__(self, smiles_vocab_size: int, protein_vocab_size: int):
        super().__init__()
        self.smiles = SmilesEncoder(smiles_vocab_size)
        self.protein = ProteinEncoder(protein_vocab_size)
        self.fc = nn.Sequential(
            nn.Linear(192, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

    def forward(self, s: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        se = self.smiles(s)
        pe = self.protein(p)
        x = torch.cat([se, pe], dim=1)
        return self.fc(x).view(-1)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    r, ci = bootstrap_pearson_ci(y_true, y_pred, n_boot=1000, seed=0)
    mse = float(mean_squared_error(y_true, y_pred)) if y_true.size > 0 else float("nan")
    c = float(concordance_index(y_true, y_pred))
    out: dict[str, Any] = {"pearson_r": r, "mse": mse, "ci": c}
    out["pearson_r_ci95"] = [ci[0], ci[1]] if ci is not None else None
    return out


def _predict(model: nn.Module, dl: DataLoader, device: torch.device) -> np.ndarray:
    preds = []
    model.eval()
    with torch.no_grad():
        for s, p, _ in dl:
            s = s.to(device)
            p = p.to(device)
            yhat = model(s, p).detach().float().cpu().numpy()
            preds.append(yhat)
    return np.concatenate(preds, axis=0) if preds else np.array([], dtype=float)


def main() -> int:
    phase = "phase4_deepdta"
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

        df, norm_stats = _load_splits()
        logger.info("Affinity normalization stats: %s", norm_stats)

        # Validate SMILES and canonicalize
        canon_smiles = []
        drop = np.zeros(len(df), dtype=bool)
        for i, smi in enumerate(df["smiles"].tolist()):
            ok, canon = validate_smiles(smi)
            if not ok or canon is None:
                append_invalid_smiles(PROJECT_ROOT, source=phase, key=f"row_{i}", smiles=smi, reason="rdkit_invalid")
                drop[i] = True
                canon_smiles.append("")
            else:
                canon_smiles.append(canon)
        df = df.copy()
        df["smiles"] = canon_smiles
        df = df[~drop]

        # Train on all training pairs — contamination labels are for characterization, not filtering
        train_df = df[df["split"] == "train"].copy()
        if train_df.empty:
            raise RuntimeError("No training pairs found. Check Phase 3 split criteria.")
        logger.info("Training pairs: %d across datasets %s", len(train_df), sorted(train_df["dataset"].unique().tolist()))

        smiles_vocab = _build_smiles_vocab(train_df["smiles"].astype(str).tolist())
        protein_vocab = _build_protein_vocab()

        smiles_vocab_size = max(smiles_vocab.values(), default=0) + 1
        protein_vocab_size = max(protein_vocab.values(), default=0) + 1

        model = DeepDTA(smiles_vocab_size, protein_vocab_size).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=0.001)
        loss_fn = nn.MSELoss()

        train_ds = DeepDTADataset(
            train_df["smiles"].astype(str).tolist(),
            train_df["sequence"].astype(str).tolist(),
            train_df["affinity"].to_numpy(dtype=float),
            smiles_vocab,
            protein_vocab,
        )
        train_dl = DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=0)

        epochs = 100
        for epoch in range(1, epochs + 1):
            model.train()
            losses = []
            for s, p, y in train_dl:
                s = s.to(device)
                p = p.to(device)
                y = y.to(device)
                pred = model(s, p)
                loss = loss_fn(pred, y)
                opt.zero_grad()
                loss.backward()
                opt.step()
                losses.append(float(loss.detach().cpu().item()))
            logger.info("Epoch %d/%d - train MSE: %.6f", epoch, epochs, float(np.mean(losses)) if losses else float("nan"))
            if epoch % 5 == 0:
                status.update(phase, "running", progress=min(0.9, epoch / epochs))

        def make_dl(dd: pd.DataFrame) -> DataLoader:
            ds = DeepDTADataset(
                dd["smiles"].astype(str).tolist(),
                dd["sequence"].astype(str).tolist(),
                dd["affinity"].to_numpy(dtype=float),
                smiles_vocab,
                protein_vocab,
            )
            return DataLoader(ds, batch_size=512, shuffle=False, num_workers=0)

        # Evaluate per dataset (Davis, KIBA, holdout) + contamination class breakdown
        test_all = df[df["split"] == "test"].copy()

        metrics: dict[str, Any] = {}
        pred_frames: list[pd.DataFrame] = []

        for ds_name in sorted(test_all["dataset"].unique().tolist()):
            ds_test = test_all[test_all["dataset"] == ds_name].copy().reset_index(drop=True)
            if ds_test.empty:
                logger.warning("Empty test set for dataset=%s, skipping.", ds_name)
                continue
            preds = _predict(model, make_dl(ds_test), device=device)
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
        metrics_path = results_dir / "deepdta_metrics.json"
        metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True))
        logger.info("Wrote %s.", metrics_path)

        pred_df = pd.concat(pred_frames, ignore_index=True) if pred_frames else pd.DataFrame()
        pred_path = results_dir / "deepdta_predictions.parquet"
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

