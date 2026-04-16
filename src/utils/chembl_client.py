from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import Optional
from urllib.parse import quote

import requests

from .http_utils import RateLimiter, request_with_backoff


class ChEMBLClient:
    """
    Minimal ChEMBL REST client with throttling + backoff.

    - Enforces >=100ms between requests.
    - Exponential backoff on 429/503 (2s base, 120s max).
    """

    def __init__(self, min_interval_s: float = 0.1, cache_dir: Optional[Path] = None):
        self._rate = RateLimiter(min_interval_s=min_interval_s)
        self._session = requests.Session()
        self._base = "https://www.ebi.ac.uk/chembl/api/data"
        self._cache_dir = cache_dir
        self._cache_dir.mkdir(parents=True, exist_ok=True) if self._cache_dir else None

    def _get_json(self, url: str) -> dict:
        self._rate.wait()
        resp = request_with_backoff(
            "GET",
            url,
            session=self._session,
            timeout=60,
            backoff_base_s=2.0,
            backoff_max_s=120.0,
            max_retries=8,
            headers={"Accept": "application/json"},
        )
        resp.raise_for_status()
        return resp.json()

    def search_by_smiles(self, smiles: str) -> list[str]:
        """
        Returns a list of ChEMBL molecule IDs matching the SMILES.

        Strategy:
        1) canonical_smiles exact match
        2) canonical_smiles flexmatch (if supported by backend)
        3) similarity endpoint at 100%
        """
        q = quote(smiles, safe="")

        # 1) Exact match
        url1 = f"{self._base}/molecule.json?molecule_structures__canonical_smiles={q}&limit=1000"
        data = self._get_json(url1)
        mols = data.get("molecules") or data.get("molecule") or []
        ids = [m.get("molecule_chembl_id") for m in mols if isinstance(m, dict)]
        ids = [i for i in ids if i]
        if ids:
            return ids

        # 2) Flexmatch (may not be enabled everywhere)
        url2 = f"{self._base}/molecule.json?molecule_structures__canonical_smiles__flexmatch={q}&limit=1000"
        try:
            data = self._get_json(url2)
            mols = data.get("molecules") or []
            ids = [m.get("molecule_chembl_id") for m in mols if isinstance(m, dict)]
            ids = [i for i in ids if i]
            if ids:
                return ids
        except Exception:  # noqa: BLE001
            pass

        # 3) Similarity at 100 (exact by fingerprint)
        url3 = f"{self._base}/similarity/{q}/100.json?limit=1000"
        try:
            data = self._get_json(url3)
            mols = data.get("molecules") or []
            ids = [m.get("molecule_chembl_id") for m in mols if isinstance(m, dict)]
            ids = [i for i in ids if i]
            return ids
        except Exception:  # noqa: BLE001
            return []

    def get_first_known_date(self, chembl_id: str) -> Optional[str]:
        """
        Returns an ISO date string (YYYY-MM-DD) approximating the earliest known date.

        Preference order:
        1) molecule.first_approval (year) -> YYYY-01-01
        2) earliest activity document_year -> YYYY-01-01
        """
        cid = quote(chembl_id, safe="")

        try:
            mol = self._get_json(f"{self._base}/molecule/{cid}.json")
            if isinstance(mol, dict):
                year = mol.get("first_approval")
                if isinstance(year, int) and year > 0:
                    return f"{year:04d}-01-01"
        except Exception:  # noqa: BLE001
            pass

        # earliest associated publication year from activities
        # ChEMBL supports ordering by document_year on the activity endpoint.
        try:
            url = (
                f"{self._base}/activity.json?"
                f"molecule_chembl_id={cid}&"
                f"order_by=document_year&"
                f"limit=1&"
                f"only=document_year"
            )
            data = self._get_json(url)
            acts = data.get("activities") or []
            if acts and isinstance(acts[0], dict):
                year = acts[0].get("document_year")
                if isinstance(year, int) and year > 0:
                    return f"{year:04d}-01-01"
        except Exception:  # noqa: BLE001
            pass

        return None

