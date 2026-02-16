"""
db_connection.py
Abstraction layer for data access in both CSV (dev) and PostgreSQL DB (production) modes.

Usage:
    import json
    with open('_config.json') as f:
        config = json.load(f)
    import db_connection as DB
    source = DB.create_data_source(config)
    devices = DB.read_enabled_devices(source, config)
    df = DB.read_sensor_data(source, config, bldg_id='B0019', dev_id=2001)
    DB.write_anomaly_result(source, config, 'B0019', 2001, 85.3, 'Normal operation')
"""

import os
import logging
from datetime import datetime, timedelta

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Resolve paths relative to this script's directory (YiUmGoV2/)
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def _resolve_path(rel_path: str) -> str:
    """Resolve a path that is relative to the YiUmGoV2/ directory."""
    return os.path.normpath(os.path.join(_SCRIPT_DIR, rel_path))


# ===================================================================
# 1. create_data_source
# ===================================================================
def create_data_source(config: dict):
    """
    Factory that returns a data-access handle.

    - CSV mode  (config['data_source'] == 'csv'):  returns config itself (no
      connection needed; individual functions open the CSV on demand).
    - DB  mode  (config['data_source'] == 'db'):   returns a SQLAlchemy engine
      connected to the PostgreSQL database.
    """
    mode = config.get("data_source", "csv")

    if mode == "db":
        try:
            from sqlalchemy import create_engine
        except ImportError as exc:
            raise ImportError(
                "sqlalchemy is required for DB mode. "
                "Install it with: pip install sqlalchemy psycopg2-binary"
            ) from exc

        db_cfg = config["db"]
        url = (
            f"postgresql://{db_cfg['user']}:{db_cfg['password']}"
            f"@{db_cfg['host']}:{db_cfg['port']}/{db_cfg['database']}"
        )
        engine = create_engine(url)
        logger.info("DB engine created: %s:%s/%s", db_cfg["host"], db_cfg["port"], db_cfg["database"])
        return engine

    # CSV mode -- just hand back the config; CSV readers will open files lazily.
    logger.info("CSV mode: data_path=%s", config["csv"].get("data_path"))
    return config


# ===================================================================
# 2. read_enabled_devices
# ===================================================================
def read_enabled_devices(source, config: dict) -> list[dict]:
    """
    Return a list of devices that have anomaly detection enabled.

    Each element is a dict with keys 'bldg_id' and 'dev_id'.

    - DB mode:  SELECT BLDG_ID, DEV_ID FROM DEV_USE_PURP_REL_R
                WHERE FALT_PRCV_YN = 'Y'
    - CSV mode: returns a hardcoded test list (the CSV files do not contain
                the DEV_USE_PURP_REL_R table).
    """
    mode = config.get("data_source", "csv")

    if mode == "db":
        table = config["anomaly"]["config_table"]  # DEV_USE_PURP_REL_R
        query = (
            f'SELECT "BLDG_ID" AS bldg_id, "DEV_ID" AS dev_id '
            f'FROM "{table}" '
            f"WHERE \"FALT_PRCV_YN\" = 'Y'"
        )
        df = pd.read_sql(query, source)
        devices = df.to_dict(orient="records")
        logger.info("DB: %d devices with FALT_PRCV_YN='Y'", len(devices))
        return devices

    # CSV mode -- read from enabled_devices CSV file
    csv_path = config["csv"].get("enabled_devices_path")
    if csv_path:
        abs_path = _resolve_path(csv_path)
        if os.path.isfile(abs_path):
            df = pd.read_csv(abs_path, dtype={"BLDG_ID": str, "DEV_ID": int, "FALT_PRCV_YN": str})
            df = df[df["FALT_PRCV_YN"] == "Y"]
            devices = [{"bldg_id": row["BLDG_ID"], "dev_id": row["DEV_ID"]} for _, row in df.iterrows()]
            logger.info("CSV mode: read %d enabled devices from %s", len(devices), abs_path)
            return devices
        logger.warning("CSV mode: enabled_devices file not found: %s, using fallback", abs_path)

    # Fallback: hardcoded test devices (backwards compat)
    devices = [{"bldg_id": "B0019", "dev_id": 2001}]
    logger.info("CSV mode: returning hardcoded device list (%d devices)", len(devices))
    return devices


# ===================================================================
# 3. read_sensor_data
# ===================================================================
def read_sensor_data(
    source,
    config: dict,
    bldg_id: str,
    dev_id: int,
) -> pd.DataFrame:
    """
    Read sensor collection data for a single device and return a DataFrame
    with columns ['colec_dt', 'colec_val'].

    - DB mode:  queries DATA_COLEC_H for the last N hours (N from config).
    - CSV mode: reads the CSV file with minimal columns, filters by dev_id
                and tag_cd, then takes the last fetch_window_hours of data
                by time (consistent with DB mode).

    Returns
    -------
    pd.DataFrame
        Columns: colec_dt (datetime64), colec_val (float64).
        Sorted by colec_dt ascending.
    """
    mode = config.get("data_source", "csv")
    tag_cd = config["data"]["tag_cd"]  # 30001
    fetch_hours = config["data"]["fetch_window_hours"]

    if mode == "db":
        table = config["data"]["collection_table"]  # DATA_COLEC_H
        # Fetch exactly fetch_window_hours of data from DB.
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
        df = pd.read_sql(
            query,
            source,
            params={"bldg_id": bldg_id, "dev_id": str(dev_id), "tag_cd": str(tag_cd), "cutoff": cutoff},
        )
        df["colec_dt"] = pd.to_datetime(df["colec_dt"])
        df["colec_val"] = df["colec_val"].astype(float)
        logger.info("DB: read %d rows for dev_id=%s tag_cd=%s (lookback=%dh)", len(df), dev_id, tag_cd, fetch_hours)
        return df

    # ------------------------------------------------------------------
    # CSV mode
    # ------------------------------------------------------------------
    csv_path = _resolve_path(config["csv"]["data_path"])
    logger.info("CSV mode: reading %s (dev_id=%s, tag_cd=%s)", csv_path, dev_id, tag_cd)

    # The CSV is ~4.3 GB.  We read only the columns we need and filter
    # in chunks to avoid loading the entire file into memory.
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
        # dev_id and tag_cd are quoted strings in the CSV (e.g. "2001").
        mask = (chunk["dev_id"] == str(dev_id)) & (chunk["tag_cd"] == str(tag_cd))
        filtered = chunk.loc[mask]
        if not filtered.empty:
            chunks.append(filtered)

    if not chunks:
        logger.warning("CSV: no data found for dev_id=%s tag_cd=%s", dev_id, tag_cd)
        return pd.DataFrame(columns=["colec_dt", "colec_val"])

    df = pd.concat(chunks, ignore_index=True)
    df["colec_dt"] = pd.to_datetime(df["colec_dt"]).dt.floor("min")
    df = df.sort_values("colec_dt").reset_index(drop=True)

    # Keep only the last fetch_window_hours by TIME (consistent with DB mode cutoff)
    cutoff = df["colec_dt"].iloc[-1] - timedelta(hours=fetch_hours)
    df = df[df["colec_dt"] >= cutoff].reset_index(drop=True)
    df = df[["colec_dt", "colec_val"]]

    logger.info("CSV: returning %d rows (%dh window) for dev_id=%s", len(df), fetch_hours, dev_id)
    return df


# ===================================================================
# 4. write_anomaly_result
# ===================================================================
def write_anomaly_result(
    source,
    config: dict,
    bldg_id: str,
    dev_id: int,
    ad_score: float,
    ad_desc: str,
) -> None:
    """
    Persist an anomaly-detection result.

    - DB mode:  INSERT INTO FALT_PRCV_FCST (USE_DT, BLDG_ID, DEV_ID,
                AD_SCORE, AD_DESC).
    - CSV mode: log/print the result (no file write).
    """
    mode = config.get("data_source", "csv")
    now = datetime.now()

    if mode == "db":
        table = config["anomaly"]["result_table"]  # FALT_PRCV_FCST
        row = pd.DataFrame(
            [
                {
                    "USE_DT": now,
                    "BLDG_ID": bldg_id,
                    "DEV_ID": str(dev_id),
                    "AD_SCORE": ad_score,
                    "AD_DESC": ad_desc,
                }
            ]
        )
        row.to_sql(table, source, if_exists="append", index=False)
        logger.info(
            "DB: inserted anomaly result for %s/%s score=%.2f",
            bldg_id, dev_id, ad_score,
        )
        return

    # CSV mode -- write to result CSV file and also log
    status = "ANOMALY" if ad_score <= config["anomaly"]["score_threshold"] else "NORMAL"
    msg = (
        f"[{now:%Y-%m-%d %H:%M:%S}] "
        f"bldg_id={bldg_id}, dev_id={dev_id}, "
        f"ad_score={ad_score:.2f} ({status}), "
        f"ad_desc={ad_desc}"
    )
    print(msg)
    logger.info("CSV mode (write_anomaly_result): %s", msg)

    # Append to result CSV if path is configured
    result_path = config["csv"].get("result_path")
    if result_path:
        abs_path = _resolve_path(result_path)
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)
        row = pd.DataFrame([{
            "USE_DT": now.strftime("%Y-%m-%d %H:%M:%S"),
            "BLDG_ID": bldg_id,
            "DEV_ID": str(dev_id),
            "AD_SCORE": round(ad_score, 2),
            "AD_DESC": ad_desc,
        }])
        write_header = not os.path.isfile(abs_path)
        row.to_csv(abs_path, mode="a", header=write_header, index=False)
        logger.info("CSV mode: result appended to %s", abs_path)
