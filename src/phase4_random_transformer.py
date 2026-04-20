"""
Random-transformer frozen probe for DTI prediction.

Same architecture as phase4_esm2_probe but with RANDOMLY INITIALISED (not pretrained)
ESM-2 and ChemBERTa transformers. This ablation properly tests whether transformer
inductive bias (positional encoding, self-attention structure) accounts for benchmark
performance, independent of any learned pretraining knowledge.

Contrast with phase4_random_probe, which uses fixed Gaussian projections. That ablation
only tests whether the MLP can extract signal from *any* fixed embedding of the right
dimensionality. This ablation tests whether the transformer's architectural structure
(without training) contributes over random projections.

The EsmConfig and RobertaConfig are loaded from cached HuggingFace configs (small JSON
files). No pretrained weights are downloaded.

Requires:
  - phase3_temporal_split

Outputs:
  - results/random_transformer_predictions.parquet
  - results/random_transformer_metrics.json

Sentinels:
  - writes checkpoints/phase4_random_transformer.done
"""

from __future__ import annotations

import argparse
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
from transformers import (
    AutoTokenizer,
    EsmConfig,
    EsmModel,
    EsmTokenizer,
    RobertaConfig,
    RobertaModel,
)

from utils.metrics import bootstrap_pearson_ci, concordance_index
from utils.sentinel import require_sentinel, write_sentinel
from utils.status import StatusUpdater
from utils.logging_utils import setup_logging
from utils.smiles import validate_smiles, append_invalid_smiles

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Default random seed — overridden by --seed CLI argument
_RANDOM_SEED = 7


def _select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
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

    # Convert Davis raw Kd (nM) → pKd = -log10(Kd_M) so that higher values denote
    # stronger binding, matching the sign convention of KIBA scores and holdout pChEMBL.
    df = df.copy()
    davis_mask = df["dataset"] == "davis"
    df.loc[davis_mask, "affinity"] = -np.log10(
        df.loc[davis_mask, "affinity"].to_numpy(dtype=float) * 1e-9
    )

    norm_stats: dict[str, dict[str, float]] = {}
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


def _embed_random_esm2(
    sequences: list[str],
    out_path: Path,
    device: torch.device,
    logger,
    status: StatusUpdater,
    phase: str,
) -> np.ndarray:
    if out_path.exists():
        logger.info("Loading cached random-ESM-2 embeddings from %s.", out_path)
        return np.load(out_path)

    logger.info(
        "Building randomly-initialised ESM-2 (same architecture, seed=%d)...", _RANDOM_SEED
    )
    torch.manual_seed(_RANDOM_SEED)
    tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
    config = EsmConfig.from_pretrained("facebook/esm2_t33_650M_UR50D")
    model = EsmModel(config)  # randomly initialised — no pretrained weights
    model.eval()
    model.to(device)

    embs = []
    batch_size = 8
    for start in tqdm(range(0, len(sequences), batch_size), desc="random-ESM2 embeddings", unit="batch"):
        batch = sequences[start : start + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=1022)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            cls = outputs.last_hidden_state[:, 0, :].detach().float().cpu().numpy()
        embs.append(cls)
        if device.type == "cuda":
            torch.cuda.empty_cache()
        status.update(phase, "running", progress=min(0.49, 0.10 + 0.39 * (start + batch_size) / max(1, len(sequences))))

    arr = np.concatenate(embs, axis=0).astype(np.float32)
    # Unit-normalise rows for numerical stability (same treatment as random_probe)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.where(norms < 1e-8, 1.0, norms)
    arr = arr / norms
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, arr)
    logger.info("Saved random-ESM-2 embeddings to %s.", out_path)
    return arr


def _embed_random_chemberta(
    smiles_list: list[str],
    out_path: Path,
    device: torch.device,
    logger,
    status: StatusUpdater,
    phase: str,
) -> np.ndarray:
    if out_path.exists():
        logger.info("Loading cached random-ChemBERTa embeddings from %s.", out_path)
        return np.load(out_path)

    logger.info(
        "Building randomly-initialised ChemBERTa (same architecture, seed=%d)...", _RANDOM_SEED
    )
    torch.manual_seed(_RANDOM_SEED + 1)
    tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
    config = RobertaConfig.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
    model = RobertaModel(config)  # randomly initialised — no pretrained weights
    model.eval()
    model.to(device)

    embs = []
    batch_size = 64
    for start in tqdm(range(0, len(smiles_list), batch_size), desc="random-ChemBERTa embeddings", unit="batch"):
        batch = smiles_list[start : start + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            cls = outputs.last_hidden_state[:, 0, :].detach().float().cpu().numpy()
        embs.append(cls)
        if device.type == "cuda":
            torch.cuda.empty_cache()
        status.update(phase, "running", progress=min(0.79, 0.49 + 0.30 * (start + batch_size) / max(1, len(smiles_list))))

    arr = np.concatenate(embs, axis=0).astype(np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.where(norms < 1e-8, 1.0, norms)
    arr = arr / norms
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, arr)
    logger.info("Saved random-ChemBERTa embeddings to %s.", out_path)
    return arr


class PairDataset(Dataset):
    def __init__(
        self,
        protein_idx: np.ndarray,
        compound_idx: np.ndarray,
        labels: np.ndarray,
        protein_emb: np.ndarray,
        compound_emb: np.ndarray,
    ):
        self.protein_idx = protein_idx.astype(np.int64)
        self.compound_idx = compound_idx.astype(np.int64)
        self.labels = labels.astype(np.float32)
        self.protein_emb = protein_emb.astype(np.float32)
        self.compound_emb = compound_emb.astype(np.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int):
        p = torch.from_numpy(self.protein_emb[self.protein_idx[i]])
        c = torch.from_numpy(self.compound_emb[self.compound_idx[i]])
        x = torch.cat([p, c], dim=0)
        return x, torch.tensor(self.labels[i])


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
    torch.manual_seed(_RANDOM_SEED)
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
    parser = argparse.ArgumentParser(description="Random-transformer frozen probe for DTI prediction.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for weight initialisation (default: 7)")
    args = parser.parse_args()

    # Override module-level seed with CLI arg
    global _RANDOM_SEED
    _RANDOM_SEED = args.seed

    phase = "phase4_random_transformer"
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

        logger.info("Random seed: %d", _RANDOM_SEED)
        device = _select_device()
        logger.info("Using device: %s", device)

        df, norm_stats = _load_splits()
        logger.info("Affinity normalization stats: %s", norm_stats)

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

        # Seed-specific checkpoint paths (seed=7 uses legacy names for backward compat)
        if _RANDOM_SEED == 7:
            prot_emb_path = PROJECT_ROOT / "checkpoints" / "random_transformer_esm2_embeddings.npy"
            chem_emb_path = PROJECT_ROOT / "checkpoints" / "random_transformer_chemberta_embeddings.npy"
        else:
            prot_emb_path = PROJECT_ROOT / "checkpoints" / f"random_transformer_esm2_embeddings_seed{_RANDOM_SEED}.npy"
            chem_emb_path = PROJECT_ROOT / "checkpoints" / f"random_transformer_chemberta_embeddings_seed{_RANDOM_SEED}.npy"

        protein_emb = _embed_random_esm2(
            sequences, prot_emb_path, device=device, logger=logger, status=status, phase=phase
        )
        compound_emb = _embed_random_chemberta(
            smiles_list, chem_emb_path, device=device, logger=logger, status=status, phase=phase
        )

        train_df = df[df["split"] == "train"].copy()
        if train_df.empty:
            raise RuntimeError("No training pairs found.")
        logger.info(
            "Training pairs: %d across datasets %s",
            len(train_df),
            sorted(train_df["dataset"].unique().tolist()),
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

        # Seed=7 keeps legacy filenames for backward compat with downstream scripts
        if _RANDOM_SEED == 7:
            metrics_path = results_dir / "random_transformer_metrics.json"
            pred_path = results_dir / "random_transformer_predictions.parquet"
        else:
            metrics_path = results_dir / f"random_transformer_seed{_RANDOM_SEED}_metrics.json"
            pred_path = results_dir / f"random_transformer_seed{_RANDOM_SEED}_predictions.parquet"

        metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True))
        logger.info("Wrote %s.", metrics_path)

        pred_df = pd.concat(pred_frames, ignore_index=True) if pred_frames else pd.DataFrame()
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
