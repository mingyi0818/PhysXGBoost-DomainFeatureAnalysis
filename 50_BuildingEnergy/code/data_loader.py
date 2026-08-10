"""Data loading and building-physics feature engineering for the UCI
Appliances Energy Prediction dataset.

Raw features (27 columns):
    lights, T1..T9, RH_1..RH_9, T_out, Press_mm_hg, RH_out, Windspeed,
    Visibility, Tdewpoint, rv1, rv2

Domain features (14 derived columns, defined in paper_draft.md Section 2.2):
    THI_out, T_dew_indoor, dT_indoor_outdoor, enthalpy_out, stack_effect,
    wind_chill, spatial_T_range, spatial_RH_range, T_indoor_mean,
    RH_indoor_mean, THI_indoor, enthalpy_indoor, hour_sin, hour_cos
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import DATA_PATH, INDOOR_RH_COLS, INDOOR_TEMP_COLS

TARGET_COL = "y"  # the 'Appliances' energy consumption (Wh)


def load_raw_frame() -> pd.DataFrame:
    """Read the CSV and parse the date column into a datetime index.

    The dataset stores the timestamp as 'date' without a separator
    between the date and the time (e.g. '2016-01-1117:00:00'); we split
    it into the standard 'YYYY-MM-DD HH:MM:SS' form first.
    """
    df = pd.read_csv(DATA_PATH)
    raw = df["date"].astype(str)
    fixed = raw.str.replace(
        r"^(\d{4}-\d{2}-\d{2})(\d{2}:\d{2}:\d{2})$", r"\1 \2", regex=True
    )
    df["date"] = pd.to_datetime(fixed, format="%Y-%m-%d %H:%M:%S",
                               errors="coerce")
    return df


# --- Magnus dew-point ----------------------------------------------------
def _magnus_dewpoint(temp_c: np.ndarray, rh_percent: np.ndarray) -> np.ndarray:
    """Dew-point temperature from the Magnus-Tetens approximation.

    gamma = ln(RH/100) + (17.27 * T) / (237.7 + T)
    Td   = (237.7 * gamma) / (17.27 - gamma)
    """
    rh = np.clip(rh_percent, 1.0, None)  # avoid log(0)
    gamma = np.log(rh / 100.0) + (17.27 * temp_c) / (237.7 + temp_c)
    return (237.7 * gamma) / (17.27 - gamma)


def _wind_chill(t_out: np.ndarray, wind: np.ndarray) -> np.ndarray:
    """Osczevski-Bluestein (2005) wind chill equivalent temperature.

    Only physically meaningful when T_out <= 10 C and wind > 1.3 m/s, but
    we compute it everywhere for consistency with the paper's feature
    definition; tree models can learn to ignore out-of-range values.
    """
    w = np.clip(wind, 0.1, None)  # avoid 0**0.16 issues
    w16 = np.power(w, 0.16)
    return 13.12 + 0.6215 * t_out - 11.37 * w16 + 0.3965 * t_out * w16


def build_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with the 14 building-physics derived features
    appended as new columns.  Does not modify the input frame.
    """
    out = df.copy()

    t_indoor = df[INDOOR_TEMP_COLS].to_numpy()
    rh_indoor = df[INDOOR_RH_COLS].to_numpy()
    t_out = df["T_out"].to_numpy()
    rh_out = df["RH_out"].to_numpy()
    press = df["Press_mm_hg"].to_numpy()
    wind = df["Windspeed"].to_numpy()

    t_indoor_mean = t_indoor.mean(axis=1)
    rh_indoor_mean = rh_indoor.mean(axis=1)

    # 1. Temperature-Humidity Index (outdoor) -- Steadman (1979)
    out["THI_out"] = t_out - 0.55 * (1.0 - rh_out / 100.0) * (t_out - 14.5)

    # 2. Indoor dew-point via Magnus formula (outdoor Tdewpoint is already
    #    a raw column, so we derive the indoor counterpart).
    out["T_dew_indoor"] = _magnus_dewpoint(t_indoor_mean, rh_indoor_mean)

    # 3. Indoor-outdoor temperature difference
    out["dT_indoor_outdoor"] = t_indoor_mean - t_out

    # 4. Air enthalpy of outdoor air (kJ/kg)
    out["enthalpy_out"] = t_out * (1.01 + 1.88 * rh_out / 100.0)

    # 5. Stack effect = dT * (P / 760)
    out["stack_effect"] = out["dT_indoor_outdoor"] * (press / 760.0)

    # 6. Wind chill equivalent temperature
    out["wind_chill"] = _wind_chill(t_out, wind)

    # 7. Spatial temperature range across the nine rooms
    out["spatial_T_range"] = t_indoor.max(axis=1) - t_indoor.min(axis=1)

    # 8. Spatial humidity range across the nine rooms
    out["spatial_RH_range"] = rh_indoor.max(axis=1) - rh_indoor.min(axis=1)

    # 9. Mean indoor temperature
    out["T_indoor_mean"] = t_indoor_mean

    # 10. Mean indoor humidity
    out["RH_indoor_mean"] = rh_indoor_mean

    # 11. Indoor THI (same formula, indoor mean values)
    out["THI_indoor"] = (
        t_indoor_mean
        - 0.55 * (1.0 - rh_indoor_mean / 100.0) * (t_indoor_mean - 14.5)
    )

    # 12. Indoor air enthalpy
    out["enthalpy_indoor"] = t_indoor_mean * (1.01 + 1.88 * rh_indoor_mean / 100.0)

    # 13-14. Circular hour encoding from the timestamp
    hour = df["date"].dt.hour.to_numpy()
    out["hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
    out["hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)

    return out


# Column lists ---------------------------------------------------------------
RAW_FEATURE_COLS = [
    "lights",
    *INDOOR_TEMP_COLS,
    *INDOOR_RH_COLS,
    "T_out",
    "Press_mm_hg",
    "RH_out",
    "Windspeed",
    "Visibility",
    "Tdewpoint",
    "rv1",
    "rv2",
]

DOMAIN_FEATURE_COLS = [
    "THI_out",
    "T_dew_indoor",
    "dT_indoor_outdoor",
    "enthalpy_out",
    "stack_effect",
    "wind_chill",
    "spatial_T_range",
    "spatial_RH_range",
    "T_indoor_mean",
    "RH_indoor_mean",
    "THI_indoor",
    "enthalpy_indoor",
    "hour_sin",
    "hour_cos",
]


def get_feature_matrix(feature_set: str) -> tuple[pd.DataFrame, pd.Series]:
    """Return (X, y) for the requested feature set.

    feature_set in {"Raw", "Domain"}.
    For "Domain" the returned matrix contains the raw columns PLUS the
    14 derived columns (augmentation, not replacement), matching the
    paper's "physics-augmented feature set" wording.
    """
    df = load_raw_frame()
    df = build_domain_features(df)
    y = df[TARGET_COL]

    if feature_set == "Raw":
        X = df[RAW_FEATURE_COLS].copy()
    elif feature_set == "Domain":
        X = df[RAW_FEATURE_COLS + DOMAIN_FEATURE_COLS].copy()
    else:
        raise ValueError(f"Unknown feature_set: {feature_set!r}")
    return X, y
