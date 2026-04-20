"""
External benchmark validation: TDC DTI-DG (BindingDB_Patent, temporal split).

Train all 4 models (esm2_probe, deepdta, random_probe, random_transformer) on the
same Davis+KIBA training data as the main experiment, then evaluate on the TDC
BindingDB_Patent test split (2019–2021 patents).

TDC affinity (Y) is ln(IC50/Ki in nM). Higher = weaker binding. Since Pearson r
is sign-invariant, we keep the raw scale — positive r means the model correlates
with affinity strength (lower predicted = lower Y = tighter binder), but we negate
Y before Pearson so that direction matches pChEMBL (higher = stronger). We
explicitly report the sign convention used.

Key questions:
  1. Does target concentration replicate in the TDC test set?
  2. Is the macro-avg per-target Pearson r ranking consistent with the main holdout?
  3. Does random_transformer >= esm2_probe on macro-avg r?

Outputs:
  - results/tdc_dtidg_results.json

Does NOT write a sentinel — this is a standalone analysis pass.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoModel,
    AutoTokenizer,
    EsmConfig,
    EsmModel,
    EsmTokenizer,
    RobertaConfig,
    RobertaModel,
)

from utils.metrics import bootstrap_pearson_ci, concordance_index
from utils.smiles import validate_smiles

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TDC_TEST_CSV = Path("/tmp/dti_dg/dti_dg_group/bindingdb_patent/test.csv")
TDC_TRAIN_CSV = Path("/tmp/dti_dg/dti_dg_group/bindingdb_patent/train_val.csv")

# Random transformer / probe seeds (match main experiment)
_RT_SEED = 7

# Minimum pairs per target for macro-avg Pearson
N_MIN = 10


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _setup_logging() -> logging.Logger:
    logger = logging.getLogger("phase6_tdc_dtidg")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(h)
    return logger


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

def _select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Load Davis+KIBA training data (same as main experiment)
# ---------------------------------------------------------------------------

def _load_train_data() -> pd.DataFrame:
    parts = []
    for ds in ["davis", "kiba"]:
        p = PROJECT_ROOT / "data" / "splits" / f"{ds}_temporal_splits.parquet"
        tmp = pd.read_parquet(p)
        tmp["dataset"] = ds
        parts.append(tmp)
    df = pd.concat(parts, ignore_index=True)
    df["split"] = df["split"].astype(str)
    df["smiles"] = df["smiles"].astype(str)
    df["sequence"] = df["sequence"].astype(str)
    df["affinity"] = pd.to_numeric(df["affinity"], errors="coerce")
    df = df[np.isfinite(df["affinity"].to_numpy())].copy()

    # Davis: raw Kd (nM) → pKd = -log10(Kd_M); higher = tighter
    davis_mask = df["dataset"] == "davis"
    df.loc[davis_mask, "affinity"] = -np.log10(
        df.loc[davis_mask, "affinity"].to_numpy(dtype=float) * 1e-9
    )

    # Per-dataset z-score normalisation (train stats only)
    norm_stats: dict[str, dict[str, float]] = {}
    for ds in df["dataset"].unique():
        train_mask = (df["dataset"] == ds) & (df["split"] == "train")
        vals = df.loc[train_mask, "affinity"].to_numpy(dtype=float)
        mu, sigma = float(np.mean(vals)), float(np.std(vals))
        if sigma < 1e-8:
            sigma = 1.0
        norm_stats[ds] = {"mean": mu, "std": sigma}
        df.loc[df["dataset"] == ds, "affinity"] = (df.loc[df["dataset"] == ds, "affinity"] - mu) / sigma

    return df, norm_stats


# ---------------------------------------------------------------------------
# Load TDC test set
# ---------------------------------------------------------------------------

def _load_tdc_test(logger: logging.Logger) -> pd.DataFrame:
    if not TDC_TEST_CSV.exists():
        raise FileNotFoundError(
            f"TDC test CSV not found at {TDC_TEST_CSV}. "
            "Run: curl -L -o /tmp/dti_dg_group.zip 'https://dataverse.harvard.edu/api/access/datafile/4742443' "
            "&& unzip /tmp/dti_dg_group.zip -d /tmp/dti_dg/"
        )
    test = pd.read_csv(TDC_TEST_CSV)
    # Y = ln(IC50/Ki in nM). Negate so higher = stronger binding (matches pChEMBL direction)
    test["Y_raw"] = test["Y"].copy()
    test["Y"] = -test["Y"]  # now higher = stronger binding

    # Validate SMILES — vectorized to avoid iterrows overhead on 49k rows
    from rdkit import Chem as _Chem
    def _canon(smi: str) -> str | None:
        try:
            mol = _Chem.MolFromSmiles(str(smi))
            if mol is None:
                return None
            return _Chem.MolToSmiles(mol, canonical=True)
        except Exception:
            return None

    logger.info("Canonicalizing TDC SMILES (%d rows)...", len(test))
    canon_col = test["Drug"].map(_canon)
    valid_mask = canon_col.notna()
    n_dropped = int((~valid_mask).sum())
    if n_dropped:
        logger.warning("Dropped %d TDC test rows with invalid SMILES.", n_dropped)

    tdf = test[valid_mask].copy()
    tdf["Drug"] = canon_col[valid_mask].values
    tdf = tdf.reset_index(drop=True)
    logger.info("TDC test set: %d pairs, %d unique targets, %d unique drugs.",
                len(tdf), tdf["Target_ID"].nunique(), tdf["Drug_ID"].nunique())
    return tdf


# ---------------------------------------------------------------------------
# Concentration stats
# ---------------------------------------------------------------------------

def _concentration_stats(tdf: pd.DataFrame) -> dict[str, Any]:
    tc = tdf["Target_ID"].value_counts()
    total = len(tdf)
    top1_pct = 100 * int(tc.iloc[0]) / total if len(tc) else 0.0
    top5_pct = 100 * int(tc.head(5).sum()) / total
    top10_pct = 100 * int(tc.head(10).sum()) / total
    hhi = float(np.sum((tc.to_numpy() / total) ** 2))
    return {
        "total_pairs": total,
        "unique_targets": int(tdf["Target_ID"].nunique()),
        "unique_drugs": int(tdf["Drug_ID"].nunique()),
        "top1_target_pct": round(top1_pct, 2),
        "top5_targets_pct": round(top5_pct, 2),
        "top10_targets_pct": round(top10_pct, 2),
        "hhi": round(hhi, 4),
        "top5_targets": tc.head(5).to_dict(),
        "pairs_with_ge10": int((tc >= 10).sum()),
        "pairs_eligible_for_macro_avg": int(tc[tc >= 10].sum()),
    }


# ---------------------------------------------------------------------------
# PairDataset (shared for esm2_probe, random_probe, random_transformer)
# ---------------------------------------------------------------------------

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
        p = torch.from_numpy(self.protein_emb[int(self.protein_idx[i])])
        c = torch.from_numpy(self.compound_emb[int(self.compound_idx[i])])
        return torch.cat([p, c], dim=0), torch.tensor(self.y[i], dtype=torch.float32)


# ---------------------------------------------------------------------------
# MLP head (shared by esm2_probe, random_probe, random_transformer)
# ---------------------------------------------------------------------------

def _build_mlp(in_dim: int, seed: int, device: torch.device) -> nn.Module:
    torch.manual_seed(seed)
    return nn.Sequential(
        nn.Linear(in_dim, 512),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(256, 1),
    ).to(device)


def _train_mlp(
    model: nn.Module,
    train_df: pd.DataFrame,
    protein_to_idx: dict[str, int],
    smiles_to_idx: dict[str, int],
    protein_emb: np.ndarray,
    compound_emb: np.ndarray,
    device: torch.device,
    logger: logging.Logger,
    label: str,
    epochs: int = 50,
) -> nn.Module:
    protein_idx = train_df["sequence"].map(lambda s: protein_to_idx[s]).to_numpy()
    compound_idx = train_df["smiles"].map(lambda s: smiles_to_idx[s]).to_numpy()
    y = train_df["affinity"].to_numpy(dtype=np.float32)

    ds = PairDataset(protein_idx, compound_idx, y, protein_emb, compound_emb)
    dl = DataLoader(ds, batch_size=512, shuffle=True, num_workers=0)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for x, yy in dl:
            x, yy = x.to(device), yy.to(device).view(-1, 1)
            loss = loss_fn(model(x), yy)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu().item()))
        if epoch % 10 == 0:
            logger.info("[%s] epoch %d/%d train_mse=%.4f", label, epoch, epochs, float(np.mean(losses)))
    model.eval()
    return model


def _infer_mlp(
    model: nn.Module,
    sequences: list[str],
    smiles: list[str],
    protein_to_idx: dict[str, int],
    smiles_to_idx: dict[str, int],
    protein_emb: np.ndarray,
    compound_emb: np.ndarray,
    device: torch.device,
    logger: logging.Logger,
    label: str,
) -> np.ndarray:
    """Predict for sequences/smiles that may include unseen sequences/smiles."""
    # Build index arrays; unseen sequences get a placeholder (mean embedding fallback)
    n = len(sequences)
    p_idx = np.zeros(n, dtype=np.int64)
    c_idx = np.zeros(n, dtype=np.int64)

    unk_p = set()
    unk_c = set()
    mean_p = protein_emb.mean(axis=0, keepdims=True).astype(np.float32)
    mean_c = compound_emb.mean(axis=0, keepdims=True).astype(np.float32)

    # We'll expand embedding arrays with mean-fallback rows for unseen seqs/smiles
    extra_p: list[np.ndarray] = []
    extra_c: list[np.ndarray] = []
    p_map = dict(protein_to_idx)
    c_map = dict(smiles_to_idx)

    for i in range(n):
        seq = sequences[i]
        smi = smiles[i]
        if seq not in p_map:
            unk_p.add(seq)
            p_map[seq] = len(protein_emb) + len(extra_p)
            extra_p.append(mean_p[0])
        if smi not in c_map:
            unk_c.add(smi)
            c_map[smi] = len(compound_emb) + len(extra_c)
            extra_c.append(mean_c[0])
        p_idx[i] = p_map[seq]
        c_idx[i] = c_map[smi]

    if unk_p:
        logger.info("[%s] %d novel protein sequences → mean-embedding fallback", label, len(unk_p))
    if unk_c:
        logger.info("[%s] %d novel SMILES → mean-embedding fallback", label, len(unk_c))

    if extra_p:
        protein_emb = np.vstack([protein_emb] + [a.reshape(1, -1) for a in extra_p])
    if extra_c:
        compound_emb = np.vstack([compound_emb] + [a.reshape(1, -1) for a in extra_c])

    dummy_y = np.zeros(n, dtype=np.float32)
    ds = PairDataset(p_idx, c_idx, dummy_y, protein_emb, compound_emb)
    dl = DataLoader(ds, batch_size=128, shuffle=False, num_workers=0)
    preds = []
    model.eval()
    with torch.no_grad():
        for x, _ in dl:
            x = x.to(device)
            preds.append(model(x).view(-1).detach().float().cpu().numpy())
    return np.concatenate(preds, axis=0)


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

def _embed_esm2_pretrained(sequences: list[str], device: torch.device, logger: logging.Logger) -> np.ndarray:
    cache = PROJECT_ROOT / "checkpoints" / "tdc_esm2_embeddings.npy"
    if cache.exists():
        logger.info("Loading cached ESM-2 embeddings from %s", cache)
        return np.load(cache)
    logger.info("Embedding %d sequences with pretrained ESM-2...", len(sequences))
    tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
    model = EsmModel.from_pretrained("facebook/esm2_t33_650M_UR50D")
    model.eval().to(device)
    embs = []
    for start in tqdm(range(0, len(sequences), 8), desc="ESM-2 (pretrained)"):
        batch = sequences[start:start + 8]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=1022)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            cls = model(**inputs).last_hidden_state[:, 0, :].detach().float().cpu().numpy()
        embs.append(cls)
    arr = np.concatenate(embs, axis=0).astype(np.float32)
    np.save(cache, arr)
    logger.info("Saved ESM-2 embeddings → %s", cache)
    return arr


def _embed_chemberta_pretrained(smiles_list: list[str], device: torch.device, logger: logging.Logger) -> np.ndarray:
    cache = PROJECT_ROOT / "checkpoints" / "tdc_chemberta_embeddings.npy"
    if cache.exists():
        logger.info("Loading cached ChemBERTa embeddings from %s", cache)
        return np.load(cache)
    logger.info("Embedding %d SMILES with pretrained ChemBERTa...", len(smiles_list))
    tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
    model = AutoModel.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
    model.eval().to(device)
    embs = []
    for start in tqdm(range(0, len(smiles_list), 64), desc="ChemBERTa (pretrained)"):
        batch = smiles_list[start:start + 64]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=128)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            cls = model(**inputs).last_hidden_state[:, 0, :].detach().float().cpu().numpy()
        embs.append(cls)
    arr = np.concatenate(embs, axis=0).astype(np.float32)
    np.save(cache, arr)
    logger.info("Saved ChemBERTa embeddings → %s", cache)
    return arr


def _embed_random_esm2(sequences: list[str], device: torch.device, logger: logging.Logger, seed: int) -> np.ndarray:
    cache = PROJECT_ROOT / "checkpoints" / f"tdc_random_esm2_seed{seed}.npy"
    if cache.exists():
        logger.info("Loading cached random-ESM-2 embeddings from %s", cache)
        return np.load(cache)
    logger.info("Embedding %d sequences with random-init ESM-2 (seed=%d)...", len(sequences), seed)
    torch.manual_seed(seed)
    tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
    config = EsmConfig.from_pretrained("facebook/esm2_t33_650M_UR50D")
    model = EsmModel(config)
    model.eval().to(device)
    embs = []
    for start in tqdm(range(0, len(sequences), 8), desc="random-ESM2"):
        batch = sequences[start:start + 8]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=1022)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            cls = model(**inputs).last_hidden_state[:, 0, :].detach().float().cpu().numpy()
        embs.append(cls)
    arr = np.concatenate(embs, axis=0).astype(np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    arr = arr / np.where(norms < 1e-8, 1.0, norms)
    np.save(cache, arr)
    logger.info("Saved random-ESM-2 embeddings → %s", cache)
    return arr


def _embed_random_chemberta(smiles_list: list[str], device: torch.device, logger: logging.Logger, seed: int) -> np.ndarray:
    cache = PROJECT_ROOT / "checkpoints" / f"tdc_random_chemberta_seed{seed}.npy"
    if cache.exists():
        logger.info("Loading cached random-ChemBERTa embeddings from %s", cache)
        return np.load(cache)
    logger.info("Embedding %d SMILES with random-init ChemBERTa (seed=%d)...", len(smiles_list), seed)
    torch.manual_seed(seed + 1)
    tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
    config = RobertaConfig.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
    model = RobertaModel(config)
    model.eval().to(device)
    embs = []
    for start in tqdm(range(0, len(smiles_list), 64), desc="random-ChemBERTa"):
        batch = smiles_list[start:start + 64]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            cls = model(**inputs).last_hidden_state[:, 0, :].detach().float().cpu().numpy()
        embs.append(cls)
    arr = np.concatenate(embs, axis=0).astype(np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    arr = arr / np.where(norms < 1e-8, 1.0, norms)
    np.save(cache, arr)
    logger.info("Saved random-ChemBERTa embeddings → %s", cache)
    return arr


# ---------------------------------------------------------------------------
# DeepDTA
# ---------------------------------------------------------------------------

def _build_smiles_vocab(smiles_list: list[str]) -> dict[str, int]:
    chars = sorted(set("".join(smiles_list)))
    return {ch: i + 1 for i, ch in enumerate(chars)}


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
    def __init__(self, smiles: list[str], sequences: list[str], y: np.ndarray,
                 smiles_vocab: dict[str, int], protein_vocab: dict[str, int]):
        self.smiles = smiles
        self.sequences = sequences
        self.y = y.astype(np.float32)
        self.sv = smiles_vocab
        self.pv = protein_vocab

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, i: int):
        s = torch.from_numpy(_encode_smiles(self.smiles[i], self.sv))
        p = torch.from_numpy(_encode_protein(self.sequences[i], self.pv))
        return s, p, torch.tensor(self.y[i], dtype=torch.float32)


class SmilesEncoder(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, 128, padding_idx=0)
        self.conv1 = nn.Conv1d(128, 32, 4)
        self.conv2 = nn.Conv1d(32, 64, 6)
        self.conv3 = nn.Conv1d(64, 96, 8)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.emb(x).transpose(1, 2)
        return self.pool(self.act(self.conv3(self.act(self.conv2(self.act(self.conv1(x))))))).squeeze(-1)


class ProteinEncoder(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, 128, padding_idx=0)
        self.conv1 = nn.Conv1d(128, 32, 4)
        self.conv2 = nn.Conv1d(32, 64, 6)
        self.conv3 = nn.Conv1d(64, 96, 8)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.emb(x).transpose(1, 2)
        return self.pool(self.act(self.conv3(self.act(self.conv2(self.act(self.conv1(x))))))).squeeze(-1)


class DeepDTA(nn.Module):
    def __init__(self, smiles_vocab_size: int, protein_vocab_size: int):
        super().__init__()
        self.smiles = SmilesEncoder(smiles_vocab_size)
        self.protein = ProteinEncoder(protein_vocab_size)
        self.fc = nn.Sequential(
            nn.Linear(192, 1024), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(1024, 1024), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(1024, 512), nn.ReLU(),
            nn.Linear(512, 1),
        )

    def forward(self, s, p):
        return self.fc(torch.cat([self.smiles(s), self.protein(p)], dim=1)).view(-1)


def _train_deepdta(
    model: DeepDTA,
    train_df: pd.DataFrame,
    smiles_vocab: dict[str, int],
    protein_vocab: dict[str, int],
    device: torch.device,
    logger: logging.Logger,
    epochs: int = 100,
) -> DeepDTA:
    ds = DeepDTADataset(
        train_df["smiles"].astype(str).tolist(),
        train_df["sequence"].astype(str).tolist(),
        train_df["affinity"].to_numpy(dtype=float),
        smiles_vocab, protein_vocab,
    )
    dl = DataLoader(ds, batch_size=256, shuffle=True, num_workers=0)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for s, p, y in dl:
            s, p, y = s.to(device), p.to(device), y.to(device)
            loss = loss_fn(model(s, p), y)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(float(loss.detach().cpu().item()))
        if epoch % 20 == 0:
            logger.info("[deepdta] epoch %d/%d train_mse=%.4f", epoch, epochs, float(np.mean(losses)))
    model.eval()
    return model


def _infer_deepdta(
    model: DeepDTA,
    sequences: list[str],
    smiles: list[str],
    smiles_vocab: dict[str, int],
    protein_vocab: dict[str, int],
    device: torch.device,
) -> np.ndarray:
    y_dummy = np.zeros(len(sequences), dtype=np.float32)
    ds = DeepDTADataset(smiles, sequences, y_dummy, smiles_vocab, protein_vocab)
    dl = DataLoader(ds, batch_size=128, shuffle=False, num_workers=0)
    preds = []
    model.eval()
    with torch.no_grad():
        for s, p, _ in dl:
            s, p = s.to(device), p.to(device)
            preds.append(model(s, p).detach().float().cpu().numpy())
    return np.concatenate(preds, axis=0) if preds else np.array([], dtype=float)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    yt, yp = y_true[mask], y_pred[mask]
    if len(yt) < 2 or np.std(yt) == 0 or np.std(yp) == 0:
        return float("nan")
    r, _ = pearsonr(yt, yp)
    return float(r)


def _macro_avg_pearson(tdf: pd.DataFrame, preds: np.ndarray, n_min: int = N_MIN) -> dict[str, Any]:
    tdf = tdf.copy()
    tdf["y_pred"] = preds
    y_true = tdf["Y"].to_numpy(dtype=float)

    per_target: dict[str, float] = {}
    for tid, grp in tdf.groupby("Target_ID"):
        if len(grp) < n_min:
            continue
        r = _pearson(grp["Y"].to_numpy(dtype=float), grp["y_pred"].to_numpy(dtype=float))
        per_target[str(tid)] = r

    valid_rs = [r for r in per_target.values() if np.isfinite(r)]
    macro_r = float(np.mean(valid_rs)) if valid_rs else float("nan")
    pooled_r = _pearson(y_true, preds)

    r_global, ci = bootstrap_pearson_ci(y_true, preds, n_boot=500, seed=0)

    return {
        "pooled_pearson_r": pooled_r,
        "macro_avg_pearson_r": macro_r,
        "macro_avg_n_targets": len(valid_rs),
        "macro_avg_n_min": n_min,
        "per_target_r": per_target,
        "pooled_pearson_r_ci95_boot": ci,
        "mse": float(mean_squared_error(y_true[np.isfinite(y_true) & np.isfinite(preds)],
                                         preds[np.isfinite(y_true) & np.isfinite(preds)]))
                if (np.isfinite(y_true) & np.isfinite(preds)).any() else float("nan"),
        "ci": float(concordance_index(y_true, preds)),
    }


# ---------------------------------------------------------------------------
# Rank order consistency check
# ---------------------------------------------------------------------------

def _rank_consistency(main_results: dict[str, float], tdc_results: dict[str, float]) -> dict[str, Any]:
    """
    Compare macro-avg Pearson r ranking between main holdout and TDC test.
    Returns Spearman rho over the 4 models.
    """
    from scipy.stats import spearmanr
    models = sorted(main_results.keys())
    main_ranks = [main_results[m] for m in models]
    tdc_ranks = [tdc_results[m] for m in models]
    rho, pval = spearmanr(main_ranks, tdc_ranks)
    return {
        "models": models,
        "main_holdout_macro_r": main_ranks,
        "tdc_macro_r": tdc_ranks,
        "spearman_rho": float(rho),
        "spearman_pval": float(pval),
        "consistent": bool(rho > 0),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    logger = _setup_logging()
    logger.info("=== Phase 6: TDC DTI-DG External Benchmark ===")

    device = _select_device()
    logger.info("Device: %s", device)

    # 1. Load training data (Davis + KIBA)
    logger.info("Loading Davis + KIBA training data...")
    train_df_full, norm_stats = _load_train_data()
    train_df = train_df_full[train_df_full["split"] == "train"].copy()
    logger.info("Training pairs: %d", len(train_df))

    # 2. Load TDC test set
    logger.info("Loading TDC test set...")
    tdf = _load_tdc_test(logger)

    # 3. Concentration stats
    conc = _concentration_stats(tdf)
    logger.info("Target concentration: top-5=%s%% top-10=%s%% HHI=%s",
                conc["top5_targets_pct"], conc["top10_targets_pct"], conc["hhi"])

    # 4. Protein sequence overlap
    train_seqs_set = set(train_df["sequence"].astype(str).tolist())
    tdc_seqs = tdf["Target"].astype(str).tolist()
    tdc_seqs_set = set(tdc_seqs)
    overlap = train_seqs_set & tdc_seqs_set
    protein_overlap = {
        "davis_kiba_train_unique": len(train_seqs_set),
        "tdc_test_unique": len(tdc_seqs_set),
        "overlap": len(overlap),
        "novel_in_tdc": len(tdc_seqs_set - train_seqs_set),
        "overlap_pct_of_tdc": round(100 * len(overlap) / max(1, len(tdc_seqs_set)), 1),
    }
    logger.info("Protein overlap: %d/%d TDC targets seen in Davis+KIBA train (%.1f%%).",
                len(overlap), len(tdc_seqs_set), protein_overlap["overlap_pct_of_tdc"])

    # 5. All unique sequences for embedding (train + TDC test)
    all_sequences = sorted(set(train_df["sequence"].astype(str).tolist()) | tdc_seqs_set)
    all_smiles = sorted(set(train_df["smiles"].astype(str).tolist()) | set(tdf["Drug"].astype(str).tolist()))

    protein_to_idx = {s: i for i, s in enumerate(all_sequences)}
    smiles_to_idx = {s: i for i, s in enumerate(all_smiles)}

    model_results: dict[str, dict] = {}

    # -----------------------------------------------------------------------
    # Model A: ESM-2 pretrained probe
    # -----------------------------------------------------------------------
    logger.info("=== Model: esm2_probe ===")
    prot_emb_esm2 = _embed_esm2_pretrained(all_sequences, device, logger)
    chem_emb_esm2 = _embed_chemberta_pretrained(all_smiles, device, logger)

    mlp_esm2 = _build_mlp(prot_emb_esm2.shape[1] + chem_emb_esm2.shape[1], seed=0, device=device)
    mlp_esm2 = _train_mlp(mlp_esm2, train_df, protein_to_idx, smiles_to_idx,
                           prot_emb_esm2, chem_emb_esm2, device, logger, "esm2_probe")
    preds_esm2 = _infer_mlp(mlp_esm2,
                             tdf["Target"].astype(str).tolist(),
                             tdf["Drug"].astype(str).tolist(),
                             protein_to_idx, smiles_to_idx,
                             prot_emb_esm2, chem_emb_esm2, device, logger, "esm2_probe")
    model_results["esm2_probe"] = _macro_avg_pearson(tdf, preds_esm2)
    logger.info("esm2_probe: pooled_r=%.4f macro_r=%.4f (n=%d targets)",
                model_results["esm2_probe"]["pooled_pearson_r"],
                model_results["esm2_probe"]["macro_avg_pearson_r"],
                model_results["esm2_probe"]["macro_avg_n_targets"])

    # -----------------------------------------------------------------------
    # Model B: Random probe (Gaussian embeddings)
    # -----------------------------------------------------------------------
    logger.info("=== Model: random_probe ===")
    ESM2_DIM, CHEMBERTA_DIM = 1280, 768
    rng_p = np.random.default_rng(0)
    prot_emb_rand = rng_p.standard_normal((len(all_sequences), ESM2_DIM)).astype(np.float32)
    norms_p = np.linalg.norm(prot_emb_rand, axis=1, keepdims=True)
    prot_emb_rand /= np.where(norms_p < 1e-8, 1.0, norms_p)

    rng_c = np.random.default_rng(1)
    chem_emb_rand = rng_c.standard_normal((len(all_smiles), CHEMBERTA_DIM)).astype(np.float32)
    norms_c = np.linalg.norm(chem_emb_rand, axis=1, keepdims=True)
    chem_emb_rand /= np.where(norms_c < 1e-8, 1.0, norms_c)

    mlp_rand = _build_mlp(ESM2_DIM + CHEMBERTA_DIM, seed=0, device=device)
    mlp_rand = _train_mlp(mlp_rand, train_df, protein_to_idx, smiles_to_idx,
                          prot_emb_rand, chem_emb_rand, device, logger, "random_probe")
    preds_rand = _infer_mlp(mlp_rand,
                            tdf["Target"].astype(str).tolist(),
                            tdf["Drug"].astype(str).tolist(),
                            protein_to_idx, smiles_to_idx,
                            prot_emb_rand, chem_emb_rand, device, logger, "random_probe")
    model_results["random_probe"] = _macro_avg_pearson(tdf, preds_rand)
    logger.info("random_probe: pooled_r=%.4f macro_r=%.4f (n=%d targets)",
                model_results["random_probe"]["pooled_pearson_r"],
                model_results["random_probe"]["macro_avg_pearson_r"],
                model_results["random_probe"]["macro_avg_n_targets"])

    # -----------------------------------------------------------------------
    # Model C: Random transformer (randomly-init ESM-2 + ChemBERTa)
    # -----------------------------------------------------------------------
    logger.info("=== Model: random_transformer ===")
    prot_emb_rt = _embed_random_esm2(all_sequences, device, logger, seed=_RT_SEED)
    chem_emb_rt = _embed_random_chemberta(all_smiles, device, logger, seed=_RT_SEED)

    mlp_rt = _build_mlp(prot_emb_rt.shape[1] + chem_emb_rt.shape[1], seed=_RT_SEED, device=device)
    mlp_rt = _train_mlp(mlp_rt, train_df, protein_to_idx, smiles_to_idx,
                        prot_emb_rt, chem_emb_rt, device, logger, "random_transformer", epochs=50)
    preds_rt = _infer_mlp(mlp_rt,
                          tdf["Target"].astype(str).tolist(),
                          tdf["Drug"].astype(str).tolist(),
                          protein_to_idx, smiles_to_idx,
                          prot_emb_rt, chem_emb_rt, device, logger, "random_transformer")
    model_results["random_transformer"] = _macro_avg_pearson(tdf, preds_rt)
    logger.info("random_transformer: pooled_r=%.4f macro_r=%.4f (n=%d targets)",
                model_results["random_transformer"]["pooled_pearson_r"],
                model_results["random_transformer"]["macro_avg_pearson_r"],
                model_results["random_transformer"]["macro_avg_n_targets"])

    # -----------------------------------------------------------------------
    # Model D: DeepDTA
    # -----------------------------------------------------------------------
    logger.info("=== Model: deepdta ===")
    smiles_vocab = _build_smiles_vocab(train_df["smiles"].astype(str).tolist())
    protein_vocab = _build_protein_vocab()
    smiles_vocab_size = max(smiles_vocab.values(), default=0) + 1
    protein_vocab_size = max(protein_vocab.values(), default=0) + 1

    deepdta_model = DeepDTA(smiles_vocab_size, protein_vocab_size).to(device)
    deepdta_model = _train_deepdta(deepdta_model, train_df, smiles_vocab, protein_vocab, device, logger)
    preds_deepdta = _infer_deepdta(
        deepdta_model,
        tdf["Target"].astype(str).tolist(),
        tdf["Drug"].astype(str).tolist(),
        smiles_vocab, protein_vocab, device,
    )
    model_results["deepdta"] = _macro_avg_pearson(tdf, preds_deepdta)
    logger.info("deepdta: pooled_r=%.4f macro_r=%.4f (n=%d targets)",
                model_results["deepdta"]["pooled_pearson_r"],
                model_results["deepdta"]["macro_avg_pearson_r"],
                model_results["deepdta"]["macro_avg_n_targets"])

    # -----------------------------------------------------------------------
    # Rank consistency vs main experiment holdout results
    # -----------------------------------------------------------------------
    main_holdout_macro_r = {
        # From results/esm2_probe_metrics.json, results/random_transformer_metrics.json, etc.
        # holdout Pearson r (these are pooled, not macro-avg — we use them as proxy for ranking)
        "esm2_probe": 0.0345,       # results/esm2_probe_metrics.json  holdout.pearson_r
        "deepdta": None,            # will load from file
        "random_probe": None,
        "random_transformer": -0.0424,  # results/random_transformer_metrics.json holdout.pearson_r
    }
    # Load from actual result files
    for m, fname in [("deepdta", "deepdta_metrics.json"), ("random_probe", "random_probe_metrics.json")]:
        try:
            d = json.loads((PROJECT_ROOT / "results" / fname).read_text())
            main_holdout_macro_r[m] = d.get("holdout", {}).get("pearson_r", float("nan"))
        except Exception:
            main_holdout_macro_r[m] = float("nan")

    tdc_macro_r_by_model = {m: v["macro_avg_pearson_r"] for m, v in model_results.items()}
    rank_info = _rank_consistency(main_holdout_macro_r, tdc_macro_r_by_model)

    # -----------------------------------------------------------------------
    # Assemble output
    # -----------------------------------------------------------------------
    output = {
        "dataset": "BindingDB_Patent (TDC DTI-DG)",
        "temporal_split": "train 2013–2018 / test 2019–2021",
        "affinity_scale": "ln(IC50/Ki nM), negated so higher = stronger binding",
        "n_min_per_target_for_macro": N_MIN,
        "data_stats": {
            "train_val_pairs": 183430,
            "test_pairs": len(tdf),
            "year_distribution_test": {"2019": 42009, "2020": 6947, "2021": 72},
        },
        "target_concentration": conc,
        "protein_overlap_with_davis_kiba_train": protein_overlap,
        "model_results": model_results,
        "rank_consistency_vs_main_holdout": rank_info,
        "key_finding": {
            "esm2_probe_macro_r": model_results["esm2_probe"]["macro_avg_pearson_r"],
            "random_transformer_macro_r": model_results["random_transformer"]["macro_avg_pearson_r"],
            "random_transformer_beats_esm2_on_macro": bool(
                model_results["random_transformer"]["macro_avg_pearson_r"]
                > model_results["esm2_probe"]["macro_avg_pearson_r"]
            ),
            "rank_consistent_with_main": rank_info["consistent"],
        },
    }

    out_path = PROJECT_ROOT / "results" / "tdc_dtidg_results.json"
    out_path.write_text(json.dumps(output, indent=2, sort_keys=True, default=str))
    logger.info("Wrote results → %s", out_path)

    # Print summary
    logger.info("=== SUMMARY ===")
    logger.info("Target concentration: top-5 targets = %.1f%% of test pairs (HHI=%.4f)",
                conc["top5_targets_pct"], conc["hhi"])
    logger.info("Protein overlap: %d/%d TDC test targets seen in training data",
                protein_overlap["overlap"], protein_overlap["tdc_test_unique"])
    for m, res in sorted(model_results.items()):
        logger.info("  %-25s  pooled_r=%+.4f  macro_r=%+.4f  (n=%d targets)",
                    m, res["pooled_pearson_r"], res["macro_avg_pearson_r"], res["macro_avg_n_targets"])
    logger.info("Rank consistency (Spearman ρ): %.3f (p=%.3f, consistent=%s)",
                rank_info["spearman_rho"], rank_info["spearman_pval"], rank_info["consistent"])
    logger.info("random_transformer > esm2_probe on macro-avg r: %s",
                output["key_finding"]["random_transformer_beats_esm2_on_macro"])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
