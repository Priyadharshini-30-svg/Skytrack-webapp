# pipeline_logic.py
import pandas as pd
import numpy as np
from pykalman import KalmanFilter
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from pyproj import Transformer
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


def apply_kalman_filter(series, noise=0.25):
    kf = KalmanFilter(initial_state_mean=series.iloc[0], n_dim_obs=1,
                      observation_covariance=noise,
                      transition_covariance=noise)
    state_means, _ = kf.smooth(series)
    return pd.Series(state_means.flatten())


def kalman_filter_columns(df):
    for col in df.columns:
        if any(k in col for k in ["CNO", "DOP", "SATX", "SATY", "SATZ"]):
            df[col] = apply_kalman_filter(df[col].fillna(method='ffill'))
    return df


def get_top4_satellites(df, sat_indices):
    scores = []
    for i in sat_indices:
        try:
            los_conditions = (
                (df[f"CNO[{i}]"] > 50) &
                (df[f"Elev[{i}]"] > 40) &
                (df["PDOP"] < 2.0) &
                (df["HDOP"] < 1.8) &
                (df["VDOP"] < 2.5)
            )
            scores.append((i, los_conditions.sum()))
        except:
            continue
    top4 = sorted(scores, key=lambda x: -x[1])[:4]
    return [i for i, _ in top4]


def extract_satellite_xyz(df, top_ids):
    x, y, z, cno, ids = [], [], [], [], []
    for i in top_ids:
        try:
            x.append(df[f"SATX[{i}]"]) 
            y.append(df[f"SATY[{i}]"])
            z.append(df[f"SATZ[{i}]"])
            cno.append(df[f"CNO[{i}]"])
            ids.append(i)
        except:
            continue
    if not x or not y or not z:
        return None
    x = np.mean(np.array(x), axis=0)
    y = np.mean(np.array(y), axis=0)
    z = np.mean(np.array(z), axis=0)
    return x[0], y[0], z[0], np.mean(cno), ids


def ecef_to_geodetic(x, y, z):
    transformer = Transformer.from_crs(4978, 4326, always_xy=True)
    lon, lat, alt = transformer.transform(x, y, z)
    return lat, lon, alt


def run_pipeline(gps_csv, navic_csv, true_coords):
    gps_raw = pd.read_csv(gps_csv)
    navic_raw = pd.read_csv(navic_csv)

    gps_raw = kalman_filter_columns(gps_raw)
    navic_raw = kalman_filter_columns(navic_raw)

    gps_ids = get_top4_satellites(gps_raw, range(9, 34))
    navic_ids = get_top4_satellites(navic_raw, range(0, 9))

    gps_data = extract_satellite_xyz(gps_raw, gps_ids)
    navic_data = extract_satellite_xyz(navic_raw, navic_ids)

    if not gps_data or not navic_data:
        return None

    # Combine GPS and NavIC XYZ
    x = np.mean([gps_data[0], navic_data[0]])
    y = np.mean([gps_data[1], navic_data[1]])
    z = np.mean([gps_data[2], navic_data[2]])

    lat, lon, alt = ecef_to_geodetic(x, y, z)

    timestamp = gps_raw["System Time"].iloc[0] if "System Time" in gps_raw.columns else "Not available"

    avg_cno = (gps_data[3] + navic_data[3]) / 2
    sat_ids = gps_data[4] + navic_data[4]

    pdop = gps_raw["PDOP"].mean()
    hdop = gps_raw["HDOP"].mean()
    vdop = gps_raw["VDOP"].mean()

    return {
        "latitude": round(lat, 7),
        "longitude": round(lon, 7),
        "altitude": round(alt, 2),
        "timestamp": str(timestamp),
        "avg_cno": round(avg_cno, 2),
        "pdop": round(pdop, 2),
        "hdop": round(hdop, 2),
        "vdop": round(vdop, 2),
        "sat_ids": sat_ids,
        "mode": "Hybrid LOS"
    }
