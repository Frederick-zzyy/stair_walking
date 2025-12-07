import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

#Load hip flexion data
path = "BT10_stairs_2_5_up_off_angle_filt.csv"
df = pd.read_csv(path)

# auto-detect hip flexion column
possible_cols = [c for c in df.columns if "hip" in c.lower() and "flex" in c.lower()]
hip_col = possible_cols[0] if possible_cols else df.columns[1]

theta = df[hip_col].astype(float).values
time = df[df.columns[0]].astype(float).values



#Detect gait cycles using hip flexion peaks
peaks, _ = find_peaks(theta, prominence=2, distance=20)

print("Detected cycles:", len(peaks))
print("Peak indices:", peaks)



#Plot hip flexion signal + peaks
plt.figure(figsize=(10,4))
plt.plot(time, theta, label="Hip flexion")
plt.scatter(time[peaks], theta[peaks], color='red', s=50, label="Cycle peaks")
plt.xlabel("Time (s)")
plt.ylabel("Hip Flexion (deg)")
plt.title(f"Hip Flexion with Detected Cycles = {len(peaks)}")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()



#Compute Phase-Space Loops
loops = []

for i in range(len(peaks)-1):
    idx0, idx1 = peaks[i], peaks[i+1]
    
    th = theta[idx0:idx1]
    tt = time[idx0:idx1]
    dt = np.mean(np.diff(tt))
    
    th_mean = np.mean(th)
    y = np.cumsum((th - th_mean) * dt)
    
    loops.append((th, y))



#Plot phase-space closed loops
plt.figure(figsize=(6,6))

for th, y in loops:
    plt.plot(th, y, linewidth=2)

plt.xlabel("Hip Flexion (deg)")
plt.ylabel("Integral of (θ - mean θ)")
plt.title("Phase-Space Loops (Each Loop = 1 Gait Cycle)")
plt.grid(True)
plt.tight_layout()
plt.show()
