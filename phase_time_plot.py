import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path = "BT10_stairs_2_11_up_off_angle_filt.csv"
df = pd.read_csv(path)

t = df.iloc[:, 0].astype(float).to_numpy()
cols = [c for c in df.columns if ("hip" in c.lower() and "flex" in c.lower())]
hip_col = cols[0] if cols else df.columns[1]
theta = df[hip_col].astype(float).to_numpy()

m = np.isfinite(t) & np.isfinite(theta)
t = t[m]
theta = theta[m]

v = np.gradient(theta, t)
zc = np.where((v[:-1] < 0) & (v[1:] >= 0))[0]

phi = np.full_like(theta, np.nan, dtype=float)
for k in range(len(zc) - 1):
    i0, i1 = zc[k], zc[k + 1]
    if i1 <= i0:
        continue
    denom = t[i1] - t[i0]
    if denom <= 0:
        continue
    phi[i0:i1] = (t[i0:i1] - t[i0]) / denom

mm = np.isfinite(phi)
plt.figure(figsize=(12, 4))
plt.plot(t[mm], phi[mm], linewidth=2)
plt.xlabel("Time (s)")
plt.ylabel("Phase  (0-1)")
plt.title("Phase vs Time")
plt.grid(True)
plt.tight_layout()
plt.show()


