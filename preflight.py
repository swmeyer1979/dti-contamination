#!/usr/bin/env python3
from __future__ import annotations

import importlib
import json
import platform
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import requests


PROJECT_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class CheckResult:
    package: str
    version: str
    status: str
    note: str = ""


def _get_pkg_version(mod) -> str:
    version_attrs = ["__version__", "version"]
    for attr in version_attrs:
        if hasattr(mod, attr):
            v = getattr(mod, attr)
            if isinstance(v, str):
                return v
    return "unknown"


def check_python() -> CheckResult:
    ok = sys.version_info >= (3, 9)
    status = "OK" if ok else "MISSING"
    version = platform.python_version()
    note = "" if ok else "Python >= 3.9 required"
    return CheckResult("python", version, status, note)


def check_import(package: str, import_name: Optional[str] = None) -> CheckResult:
    name = import_name or package
    try:
        mod = importlib.import_module(name)
        return CheckResult(package, _get_pkg_version(mod), "OK")
    except Exception as e:  # noqa: BLE001
        return CheckResult(package, "-", "MISSING", str(e))


def check_disk() -> CheckResult:
    usage = shutil.disk_usage(str(PROJECT_ROOT))
    free_gb = usage.free / (1024**3)
    if free_gb < 100:
        status = "MISSING"
        note = f"ERROR: free disk {free_gb:.1f} GB (< 100 GB)"
    elif free_gb < 150:
        status = "WARN"
        note = f"WARN: free disk {free_gb:.1f} GB (< 150 GB)"
    else:
        status = "OK"
        note = f"free {free_gb:.1f} GB"
    return CheckResult("disk_free", f"{free_gb:.1f} GB", status, note)


def check_mmseqs2() -> CheckResult:
    path = shutil.which("mmseqs")
    if path is None:
        return CheckResult("mmseqs2", "-", "WARN", "Not on PATH (brew install mmseqs2)")
    return CheckResult("mmseqs2", path, "OK")


def check_internet() -> CheckResult:
    try:
        r = requests.get("https://www.ebi.ac.uk", timeout=5)
        if r.status_code >= 200 and r.status_code < 400:
            return CheckResult("internet", str(r.status_code), "OK")
        return CheckResult("internet", str(r.status_code), "WARN", "Non-2xx/3xx response")
    except Exception as e:  # noqa: BLE001
        return CheckResult("internet", "-", "WARN", str(e))


def _ascii_table(rows: list[list[str]]) -> str:
    widths = [0] * len(rows[0])
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    lines = []
    border = "+".join([""] + ["-" * (w + 2) for w in widths] + [""])
    lines.append(border)
    for idx, row in enumerate(rows):
        line = "|".join([""] + [f" {cell.ljust(widths[i])} " for i, cell in enumerate(row)] + [""])
        lines.append(line)
        if idx == 0:
            lines.append(border)
    lines.append(border)
    return "\n".join(lines)


def main() -> int:
    checks: list[CheckResult] = []
    checks.append(check_python())

    pkgs = [
        ("rdkit", "rdkit"),
        ("chembl_webresource_client", "chembl_webresource_client"),
        ("pubchempy", "pubchempy"),
        ("FPSim2", "FPSim2"),
        ("torch", "torch"),
        ("transformers", "transformers"),
        ("scipy", "scipy"),
        ("pandas", "pandas"),
        ("numpy", "numpy"),
        ("seaborn", "seaborn"),
        ("statsmodels", "statsmodels"),
        ("tqdm", "tqdm"),
        ("pyarrow", "pyarrow"),
        ("pingouin", "pingouin"),
        ("sklearn", "sklearn"),
        ("requests", "requests"),
    ]
    for pkg, mod in pkgs:
        checks.append(check_import(pkg, mod))

    checks.append(check_disk())
    checks.append(check_mmseqs2())
    checks.append(check_internet())

    rows = [["package", "version", "status", "note"]]
    for c in checks:
        rows.append([c.package, c.version, c.status, c.note])
    print(_ascii_table(rows))

    status_summary = {"OK": 0, "WARN": 0, "MISSING": 0}
    for c in checks:
        status_summary[c.status] = status_summary.get(c.status, 0) + 1
    (PROJECT_ROOT / "logs").mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "logs" / "preflight.json").write_text(json.dumps([c.__dict__ for c in checks], indent=2))

    if any(c.package == "python" and c.status != "OK" for c in checks):
        return 1
    if any(c.package == "disk_free" and c.status == "MISSING" for c in checks):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

