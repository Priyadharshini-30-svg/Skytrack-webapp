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

# ---------- Kalman Filter ----------
def apply_kalman_filter(series, noise=0.25):
    kf = KalmanFilter(initial_state_mean=series.iloc[0], n_dim_obs=1,
                      observation_covariance=noise,
                      transition_covariance=noise)
    state_means, _ = kf.smooth(series)
    return pd.Series(state_means.flatten())

def kalman_filter_columns(df):
    # ✅ Parse System Time
    df["System Time"] = pd.to_datetime(df["System Time"], errors='coerce')
    df.dropna(subset=["System Time"], inplace=True)

    # ✅ Downsample every 60 seconds
    df = df.set_index("System Time").resample("60S").first().dropna(how="all").reset_index()

    # ✅ Apply Kalman filter to selected columns
    for col in df.columns:
        if any(k in col for k in ["CNO", "DOP", "SATX", "SATY", "SATZ"]):
            df[col] = apply_kalman_filter(df[col].ffill())
    return df

# ---------- Labeling & Satellite Selection ----------
def evaluate_los_score(row, i):
    score = 0
    try:
        if row.get(f"CNO[{i}]") > 50: score += 1
        if row.get(f"Elev[{i}]") > 40: score += 1
        if row.get("PDOP") < 2.0: score += 1
        if row.get("HDOP") < 1.8: score += 1
        if row.get("VDOP") < 2.5: score += 1
    except: pass
    return score

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
        except: continue
    top4 = sorted(scores, key=lambda x: -x[1])[:4]
    return [i for i, _ in top4]

def expand_and_label(df, indices):
    data = []
    for _, row in df.iterrows():
        for i in indices:
            try:
                label = 1 if row.get(f"Label[{i}]") == "LOS" else 0
                data.append({
                    "Elev": row[f"Elev[{i}]"],
                    "Azim": row[f"Azim[{i}]"],
                    "CNO": row[f"CNO[{i}]"],
                    "SATX": row[f"SATX[{i}]"],
                    "SATY": row[f"SATY[{i}]"],
                    "SATZ": row[f"SATZ[{i}]"],
                    "PDOP": row["PDOP"],
                    "HDOP": row["HDOP"],
                    "VDOP": row["VDOP"],
                    "TrueLabel": label
                })
            except: continue
    return pd.DataFrame(data)

def relabel_rule_based(df):
    relabeled = []
    for _, row in df.iterrows():
        los, nlos = 0, 0
        if row["CNO"] > 50: los += 1
        elif row["CNO"] < 41: nlos += 1
        if row["Elev"] > 40: los += 1
        elif row["Elev"] < 30: nlos += 1
        if row["PDOP"] < 2.0: los += 1
        elif row["PDOP"] > 2.6: nlos += 1
        if row["HDOP"] < 1.8: los += 1
        elif row["HDOP"] > 2.3: nlos += 1
        if row["VDOP"] < 2.5: los += 1
        elif row["VDOP"] > 2.9: nlos += 1
        if nlos >= 3:
            relabeled.append(0)
        elif los >= 4:
            relabeled.append(1)
        else:
            relabeled.append(row["TrueLabel"])
    df["TrueLabel"] = relabeled
    return df

# ---------- Training ----------
def train_hybrid_model(df):
    X = df.drop(columns=["TrueLabel"])
    y = df["TrueLabel"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)

    rf.fit(X_train, y_train)
    mlp.fit(X_train, y_train)

    y_pred = np.logical_or(rf.predict(X_test), mlp.predict(X_test)).astype(int)
    return X_test.assign(Predicted=y_pred, TrueLabel=y_test.values), y_test, y_pred

# ---------- Confusion Matrix Plotting ----------
def plot_conf_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["NLOS", "LOS"], yticklabels=["NLOS", "LOS"])
    plt.title(f"{title} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(f"{title.lower().replace(' ', '_')}_confusion_matrix.png")
    plt.close()

# ---------- Main Pipeline ----------
def process_all(gps_File, navic_File):
    gps_raw = pd.read_csv(gps_File)
    navic_raw = pd.read_csv(navic_File)

    gps_raw = kalman_filter_columns(gps_raw)
    navic_raw = kalman_filter_columns(navic_raw)

    gps_ids = get_top4_satellites(gps_raw, range(9, 34))
    navic_ids = get_top4_satellites(navic_raw, range(0, 9))

    gps_df = expand_and_label(gps_raw, gps_ids)
    navic_df = expand_and_label(navic_raw, navic_ids)

    gps_df = relabel_rule_based(gps_df)
    navic_df = relabel_rule_based(navic_df)

    gps_result, y_gps, p_gps = train_hybrid_model(gps_df)
    navic_result, y_nav, p_nav = train_hybrid_model(navic_df)

    plot_conf_matrix(y_gps, p_gps, "GPS")
    plot_conf_matrix(y_nav, p_nav, "NavIC")

    return gps_result, navic_result
