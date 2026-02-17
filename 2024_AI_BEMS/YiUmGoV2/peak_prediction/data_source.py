"""
data_source.py  --  Peak Prediction Data Access Layer

Abstraction for data access in both CSV (dev) and PostgreSQL DB (production) modes.

Usage:
    import json, data_source as DS
    with open('_config.json') as f:
        config = json.load(f)
    source = DS.create_data_source(config)
    devices = DS.read_enabled_devices(source, config)
    df = DS.read_sensor_data(source, config, bldg_id='B0019', dev_id=2001)
    DS.write_peak_result(source, config, 'B0019', '207.33@13:15')
"""

import os
import logging
from datetime import datetime, timedelta

import pandas as pd

logger = logging.getLogger(__name__)

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def _resolve_path(rel_path: str) -> str:
    """Resolve a path relative to the peak_prediction/ directory."""
    return os.path.normpath(os.path.join(_SCRIPT_DIR, rel_path))


# ===================================================================
# 1. create_data_source
# ===================================================================
def create_data_source(config: dict):
    """Return a data-access handle (SQLAlchemy engine for DB, config dict for CSV)."""
    mode = config.get("data_source", "csv")

    if mode == "db":
        try:
            from sqlalchemy import create_engine
        except ImportError as exc:
            raise ImportError(
                "sqlalchemy is required for DB mode. "
                "Install with: pip install sqlalchemy psycopg2-binary"
            ) from exc

        db_cfg = config["db"]
        url = (
            f"postgresql://{db_cfg['user']}:{db_cfg['password']}"
            f"@{db_cfg['host']}:{db_cfg['port']}/{db_cfg['database']}"
        )
        engine = create_engine(url)
        logger.info("DB engine created: %s:%s/%s",
                     db_cfg["host"], db_cfg["port"], db_cfg["database"])
        return engine

    logger.info("CSV mode: data_path=%s", config["csv"].get("data_path"))
    return config


# ===================================================================
# 2. read_enabled_devices
# ===================================================================
def read_enabled_devices(source, config: dict) -> list[dict]:
    """Return list of devices enabled for peak prediction.

    Each element: {'bldg_id': str, 'dev_id': int}.
    """
    mode = config.get("data_source", "csv")

    if mode == "db":
        table = config["peak"]["config_table"]
        query = (
            f'SELECT "BLDG_ID" AS bldg_id, "DEV_ID" AS dev_id '
            f'FROM "{table}" '
            f"WHERE \"FALT_PRCV_YN\" = 'Y'"
        )
        df = pd.read_sql(query, source)
        # Filter to only dev_id matching peak config
        peak_dev_id = str(config["peak"]["dev_id"])
        df = df[df["dev_id"].astype(str) == peak_dev_id]
        devices = df.to_dict(orient="records")
        logger.info("DB: %d peak devices found", len(devices))
        return devices

    # CSV mode
    csv_path = config["csv"].get("config_peak_devices_path")
    if csv_path:
        abs_path = _resolve_path(csv_path)
        if os.path.isfile(abs_path):
            df = pd.read_csv(abs_path, dtype={"BLDG_ID": str, "DEV_ID": int})
            # Support both PEAK_PRCV_YN and FALT_PRCV_YN column names
            yn_col = "PEAK_PRCV_YN" if "PEAK_PRCV_YN" in df.columns else "FALT_PRCV_YN"
            df = df[df[yn_col] == "Y"]
            devices = [{"bldg_id": row["BLDG_ID"], "dev_id": row["DEV_ID"]}
                       for _, row in df.iterrows()]
            logger.info("CSV: read %d enabled devices from %s", len(devices), abs_path)
            return devices
        logger.warning("CSV: device config not found: %s", abs_path)

    # Fallback
    devices = [{"bldg_id": "B0019", "dev_id": 2001}]
    logger.info("CSV: returning fallback device list")
    return devices


# ===================================================================
# 3. read_sensor_data  (unchanged from anomaly_detection)
# ===================================================================
def read_sensor_data(
    source,
    config: dict,
    bldg_id: str,
    dev_id: int,
    fetch_hours: int | None = None,
) -> pd.DataFrame:
    """Read sensor data for a device. Returns DataFrame with [colec_dt, colec_val].

    Parameters:
        fetch_hours: None=use config default (336h), 0=all history, N=last N hours.
    """
    mode = config.get("data_source", "csv")
    tag_cd = config["data"]["tag_cd"]
    if fetch_hours is None:
        fetch_hours = config["data"]["fetch_window_hours"]

    if mode == "db":
        table = config["data"]["collection_table"]
        if fetch_hours > 0:
            cutoff = datetime.now() - timedelta(hours=fetch_hours)
            query = (
                f'SELECT "COLEC_DT" AS colec_dt, "COLEC_VAL" AS colec_val '
                f'FROM "{table}" '
                f"WHERE \"BLDG_ID\" = %(bldg_id)s "
                f"  AND \"DEV_ID\" = %(dev_id)s "
                f"  AND \"TAG_CD\" = %(tag_cd)s "
                f"  AND \"COLEC_DT\" >= %(cutoff)s "
                f'ORDER BY "COLEC_DT" ASC'
            )
            params = {"bldg_id": bldg_id, "dev_id": str(dev_id),
                      "tag_cd": str(tag_cd), "cutoff": cutoff}
        else:
            query = (
                f'SELECT "COLEC_DT" AS colec_dt, "COLEC_VAL" AS colec_val '
                f'FROM "{table}" '
                f"WHERE \"BLDG_ID\" = %(bldg_id)s "
                f"  AND \"DEV_ID\" = %(dev_id)s "
                f"  AND \"TAG_CD\" = %(tag_cd)s "
                f'ORDER BY "COLEC_DT" ASC'
            )
            params = {"bldg_id": bldg_id, "dev_id": str(dev_id),
                      "tag_cd": str(tag_cd)}
        df = pd.read_sql(query, source, params=params)
        df["colec_dt"] = pd.to_datetime(df["colec_dt"])
        df["colec_val"] = df["colec_val"].astype(float)
        label = f"{fetch_hours}h" if fetch_hours > 0 else "all"
        logger.info("DB: %d rows for dev_id=%s (%s)", len(df), dev_id, label)
        return df

    # CSV mode
    csv_path = _resolve_path(config["csv"]["data_path"])
    logger.info("CSV: reading %s (dev_id=%s, tag_cd=%s)", csv_path, dev_id, tag_cd)

    CHUNK_SIZE = 500_000
    usecols = ["dev_id", "tag_cd", "colec_dt", "colec_val"]
    chunks: list[pd.DataFrame] = []

    for chunk in pd.read_csv(
        csv_path,
        usecols=usecols,
        dtype={"dev_id": str, "tag_cd": str, "colec_val": float},
        parse_dates=["colec_dt"],
        chunksize=CHUNK_SIZE,
    ):
        mask = (chunk["dev_id"] == str(dev_id)) & (chunk["tag_cd"] == str(tag_cd))
        filtered = chunk.loc[mask]
        if not filtered.empty:
            chunks.append(filtered)

    if not chunks:
        logger.warning("CSV: no data for dev_id=%s tag_cd=%s", dev_id, tag_cd)
        return pd.DataFrame(columns=["colec_dt", "colec_val"])

    df = pd.concat(chunks, ignore_index=True)
    df["colec_dt"] = pd.to_datetime(df["colec_dt"]).dt.floor("min")
    df = df.sort_values("colec_dt").reset_index(drop=True)

    if fetch_hours > 0:
        cutoff = df["colec_dt"].iloc[-1] - timedelta(hours=fetch_hours)
        df = df[df["colec_dt"] >= cutoff].reset_index(drop=True)

    df = df[["colec_dt", "colec_val"]]
    label = f"{fetch_hours}h" if fetch_hours > 0 else "all"
    logger.info("CSV: %d rows (%s) for dev_id=%s", len(df), label, dev_id)
    return df


# ===================================================================
# 4. write_peak_result  (replaces write_anomaly_result)
# ===================================================================
def write_peak_result(
    source,
    config: dict,
    bldg_id: str,
    peak_info: str,
) -> None:
    """Persist a peak prediction result.

    Args:
        source: Data source handle.
        config: Configuration dict.
        bldg_id: Building ID string (e.g., "B0019").
        peak_info: Formatted string "207.33@13:15" (peak_power@peak_time).

    DB mode:  INSERT INTO MAX_DMAND_FCST_H (USE_DT, BLDG_ID, DLY_MAX_DMAND_FCST_INF).
    CSV mode: print + append to output/peak_results.csv.
    """
    mode = config.get("data_source", "csv")
    now = datetime.now()

    if mode == "db":
        table = config["peak"]["result_table"]  # MAX_DMAND_FCST_H
        row = pd.DataFrame([{
            "USE_DT": now,
            "BLDG_ID": bldg_id,
            "DLY_MAX_DMAND_FCST_INF": peak_info,
        }])
        row.to_sql(table, source, if_exists="append", index=False)
        logger.info("DB: inserted peak result for %s: %s", bldg_id, peak_info)
        return

    # CSV mode
    msg = (
        f"[{now:%Y-%m-%d %H:%M:%S}] "
        f"bldg_id={bldg_id}, "
        f"DLY_MAX_DMAND_FCST_INF={peak_info}"
    )
    print(msg)
    logger.info("CSV (write_peak_result): %s", msg)

    result_path = config["csv"].get("peak_results_path")
    if result_path:
        abs_path = _resolve_path(result_path)
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)
        row = pd.DataFrame([{
            "USE_DT": now.strftime("%Y-%m-%d %H:%M:%S"),
            "BLDG_ID": bldg_id,
            "DLY_MAX_DMAND_FCST_INF": peak_info,
        }])
        write_header = not os.path.isfile(abs_path)
        row.to_csv(abs_path, mode="a", header=write_header, index=False)
        logger.info("CSV: result appended to %s", abs_path)
