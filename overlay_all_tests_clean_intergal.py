import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.signal import find_peaks
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge

BASE = os.path.dirname(os.path.abspath(__file__))
angle_dir = os.path.join(BASE, "Joint Angles")
moment_dir = os.path.join(BASE, "Moment")



# Detect valid angle and moment column
def get_valid_moment_column(df):
    """Automatically pick a working hip flexion moment column."""
    candidates = [
        "hip_flexion_l_moment",
        "hip_flexion_r_moment",
        "hip_flexion_moment",
        "hip_moment",
    ]

    # 1) Exact match
    for col in candidates:
        if col in df.columns and df[col].notna().sum() > 20:
            return col

    # 2) Fuzzy search for any usable hip moment
    for col in df.columns:
        if ("hip" in col.lower()) and ("moment" in col.lower()):
            if df[col].notna().sum() > 20:
                return col

    return None


def get_valid_angle_column(df):
    """Select hip flexion angle column (used for cycle detection)."""
    candidates = [
        "hip_flexion_l_angle",
        "hip_flexion_r_angle",
        "hip_flexion_angle",
        "hip_flexion",
        "hip_angle",
    ]

    for col in candidates:
        if col in df.columns and df[col].notna().sum() > 20:
            return col

    # fuzzy fallback
    for col in df.columns:
        if "hip" in col.lower() and "flex" in col.lower():
            return col

    return df.columns[1]   # fallback: 2nd column


# Cycle Detection
def extract_cycles(time, angle, torque):
    peaks, _ = find_peaks(angle, prominence=2, distance=20)

    cycles = []
    for i in range(len(peaks)-1):
        i0 = peaks[i]
        i1 = peaks[i+1]

        if i1 <= i0:
            continue

        t = time[i0:i1]
        tq = torque[i0:i1]

        if np.all(np.isnan(tq)):
            continue

        phi = (t - t[0]) / (t[-1] - t[0])
        cycles.append((phi, tq))

    return cycles


# Resample cycle to fixed 0-1 phase
def resample_cycle(phi, tq, N=200):
    phi_u = np.linspace(0, 1, N)
    tq_u = np.interp(phi_u, phi, tq)
    return tq_u


# Main Processing
angle_dir = "Joint Angles"
moment_dir = "Moment"

# I add this function because I work this file in both windows and mac
# This leads to useless file that needs to be cleaned up
tests = sorted({
    f.lower().split("_moment")[0]
    for f in os.listdir(moment_dir)
    if f.endswith(".csv") and not f.startswith("._")
})

print("Found tests:", len(tests))

all_cycles = []

for test_id in tests:
    moment_file = None
    angle_file = None

    # match files
    for f in os.listdir(moment_dir):
      if f.startswith("._"):
        continue
      if test_id in f.lower():
        moment_file = os.path.join(moment_dir, f)

    for f in os.listdir(angle_dir):
      if f.startswith("._"):
        continue
      if test_id in f.lower():
        angle_file = os.path.join(angle_dir, f)

    if moment_file is None or angle_file is None:
        print(f"Missing: {test_id}")
        continue

    ang = pd.read_csv(angle_file)
    mom = pd.read_csv(moment_file)

    angle_col = get_valid_angle_column(ang)
    moment_col = get_valid_moment_column(mom)

    if moment_col is None:
        print(f"No valid hip moment column: {test_id}")
        continue

    time = ang.iloc[:,0].values
    hip_angle = ang[angle_col].values
    hip_moment = mom[moment_col].values

    cycles = extract_cycles(time, hip_angle, hip_moment)

    for phi, tq in cycles:
        tq_u = resample_cycle(phi, tq)
        all_cycles.append(tq_u)

    print(f"OK: {test_id}, cycles = {len(cycles)}")


from sklearn.neighbors import LocalOutlierFactor

curves = np.array(all_cycles)
N, M = curves.shape

# 1) LOF Outlier Detection
curves = np.array(all_cycles)   # shape = (N, M)

mask = ~np.isnan(curves).any(axis=1)
curves = curves[mask]

print("Valid curves after removing NaN:", curves.shape[0])


from sklearn.neighbors import LocalOutlierFactor

lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
labels = lof.fit_predict(curves)

good_lof = labels == 1
curves_lof = curves[good_lof]

lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
labels = lof.fit_predict(curves)

good_lof = labels == 1


# 2) MAD Filtering
median_curve = np.nanmedian(curves, axis=0)
mad = np.nanmedian(np.abs(curves - median_curve), axis=1)

mad_thresh = np.nanmedian(mad) + 2.0 * np.nanstd(mad)
good_mad = mad < mad_thresh


# 3) Envelope Filtering
upper_env = np.nanpercentile(curves, 90, axis=0)
lower_env = np.nanpercentile(curves, 10, axis=0)

good_env = []
for c in curves:
    above = c > (upper_env + 0.2 * np.abs(upper_env))
    below = c < (lower_env - 0.2 * np.abs(lower_env))
    if np.any(above) or np.any(below):
        good_env.append(False)
    else:
        good_env.append(True)

good_env = np.array(good_env)


# Final selection
good = good_lof & good_mad & good_env
clean_curves = curves[good]

print(f"Clean: {len(clean_curves)} / {len(curves)} (removed {len(curves)-len(clean_curves)})")

#   Plotting
plt.figure(figsize=(12,6))

phi_u = np.linspace(0, 1, clean_curves.shape[1])

X = np.tile(phi_u, clean_curves.shape[0]).reshape(-1, 1)
y = clean_curves.reshape(-1)

mask = np.isfinite(y)
X = X[mask]
y = y[mask]

deg = 20
model = make_pipeline(PolynomialFeatures(deg, include_bias=False), Ridge(alpha=1e-3))
model.fit(X, y)

phi_fit = np.linspace(0, 1, 200).reshape(-1, 1)
tau_fit = model.predict(phi_fit)

plt.figure(figsize=(12,6))
plt.plot(phi_fit[:,0], tau_fit, linewidth=3)
plt.title(f"Regression Torque Map: tau = f(phi), deg={deg}")
plt.xlabel("Phase (0–1)")
plt.ylabel("Hip Torque (Nm)")
plt.grid(True)
plt.show()

print("Polynomial coefficients (highest order last):")
print(model.named_steps["ridge"].coef_)
print("Intercept:", model.named_steps["ridge"].intercept_)
