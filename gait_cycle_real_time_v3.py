import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

angle_paths = [
    "BT10_stairs_2_5_up_off_angle_filt.csv",
    "BT10_stairs_2_7_up_off_angle_filt.csv",
    "BT10_stairs_2_11_up_off_angle_filt.csv",
]

colors = ["blue", "green", "red"]

def causal_lowpass(x, alpha=0.1):
    y = np.zeros_like(x, dtype=float)
    y[0] = x[0]
    for k in range(1, len(x)):
        y[k] = alpha * x[k] + (1 - alpha) * y[k - 1]
    return y

def dirty_derivative(x, dt, tau=0.05):
    dx = np.zeros_like(x, dtype=float)
    alpha = tau / (tau + dt)
    for k in range(1, len(x)):
        raw_diff = (x[k] - x[k - 1]) / dt
        dx[k] = alpha * dx[k - 1] + (1 - alpha) * raw_diff
    return dx

all_loops = []

plt.figure(figsize=(10,4))

for path, color in zip(angle_paths, colors):

    df = pd.read_csv(path)

    angle_col = [c for c in df.columns if "hip" in c.lower() and "flex" in c.lower()][0]

    time = df.iloc[:,0].astype(float).values
    theta = df[angle_col].astype(float).values

    mask = np.isfinite(time) & np.isfinite(theta)
    time = time[mask]
    theta = theta[mask]

    dt = np.mean(np.diff(time))

    theta_f = causal_lowpass(theta, alpha=0.2)
    velocity_f = dirty_derivative(theta_f, dt, tau=0.02)

    phi = np.arctan2(velocity_f, theta_f) / np.pi

    t0 = time[0]
    time_aligned = time - t0

    plt.plot(time_aligned, phi, color=color, label=path.split("_")[2])

    peaks, _ = find_peaks(theta_f, prominence=2, distance=20)

    print(f"{path}: detected cycles =", len(peaks))

    trim = 3

    for i in range(len(peaks)-1):

        idx0, idx1 = peaks[i], peaks[i+1]

        if idx1 - idx0 <= 2*trim + 5:
            continue

        th = theta_f[idx0+trim : idx1-trim]
        vel = velocity_f[idx0+trim : idx1-trim]

        th_centered = th - np.mean(th)
        vel_centered = vel - np.mean(vel)

        all_loops.append((th_centered, vel_centered, color))

plt.xlabel("Time (s)")
plt.ylabel("Phase / π")
plt.title("Phase vs Time (2_5, 2_7, 2_11)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


plt.figure(figsize=(7,7))

for th, vel, color in all_loops:
    plt.plot(th, vel, color=color, linewidth=2, alpha=0.7)

plt.xlabel("Centered Hip Angle")
plt.ylabel("Centered Hip Angular Velocity")
plt.title("Phase-Space Loops (Centers Aligned)")
plt.grid(True)
plt.tight_layout()
plt.show()