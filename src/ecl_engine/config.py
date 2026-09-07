from __future__ import annotations

from pathlib import Path
import yaml


ROOT = Path(__file__).resolve().parents[2]


def load_config(path: Path | None = None) -> dict:
    config_path = path or ROOT / "config" / "project.yaml"
    with config_path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    for key, value in cfg.get("data", {}).items():
        if isinstance(value, str):
            candidate = Path(value).expanduser()
            cfg["data"][key] = str(candidate if candidate.is_absolute() else ROOT / candidate)
    cfg["paths"] = {
        "root": ROOT,
        "database": ROOT / "database" / "ifrs9_ecl.sqlite3",
        "models": ROOT / "outputs" / "models",
        "tables": ROOT / "outputs" / "tables",
        "report": ROOT / "outputs" / "IFRS_9_Mortgage_ECL_Model.xlsx",
    }
    _validate_config(cfg)
    return cfg


def _validate_config(cfg: dict) -> None:
    weights = [float(values["weight"]) for values in cfg["scenarios"].values()]
    if any(weight < 0 for weight in weights) or abs(sum(weights) - 1.0) > 1e-12:
        raise ValueError("Scenario weights must be non-negative and sum to 100%")
    if int(cfg["default"]["dpd_threshold"]) < 90:
        raise ValueError("The configured project default threshold must be at least 90 DPD")
    if not 0 < float(cfg["pd"]["maximum_12m_pd"]) < 1:
        raise ValueError("maximum_12m_pd must be between zero and one")
    if pd_timestamp(cfg["pd"]["calibration_start_date"]) >= pd_timestamp(cfg["pd"]["test_start_date"]):
        raise ValueError("PD calibration must begin before the out-of-time test")
    for key in ("unemployment_publication_lag_months", "gdp_publication_lag_months", "hpi_publication_lag_months"):
        if int(cfg["macro"][key]) < 0:
            raise ValueError(f"{key} must be non-negative")
    if cfg["lgd"].get("freddie_cashflow_sign_convention") not in {
        "auto_detect", "legacy_recoveries_positive", "current_recoveries_negative"
    }:
        raise ValueError("Unsupported Freddie Mac cash-flow sign convention")
    missing = [key for key, value in cfg["data"].items() if isinstance(value, str) and not Path(value).exists()]
    if missing:
        raise FileNotFoundError(f"Configured input files are missing: {missing}")


def pd_timestamp(value):
    from datetime import date
    return date.fromisoformat(str(value)[:10])


def ensure_directories(cfg: dict) -> None:
    for key in ("database", "models", "tables", "report"):
        path = Path(cfg["paths"][key])
        (path.parent if path.suffix else path).mkdir(parents=True, exist_ok=True)
