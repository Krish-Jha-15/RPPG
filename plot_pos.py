import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("pos_signal_windowed.csv")
pos = df["POS"].to_numpy(dtype=np.float64)
FPS = 28.671786
time = np.arange(len(pos)) / FPS

plt.figure(figsize=(12, 5))
plt.plot(time, pos)

plt.xlabel("Time (seconds)")
plt.ylabel("POS amplitude")
plt.title("Windowed POS Signal")

plt.grid(True)
plt.tight_layout()
plt.show()


plt.figure(figsize=(12, 5))
samples_to_show = int(10 * FPS)

plt.plot(
    time[:samples_to_show],
    pos[:samples_to_show]
)

plt.xlabel("Time (seconds)")
plt.ylabel("POS amplitude")
plt.title("First 10 Seconds of POS Signal")

plt.grid(True)
plt.tight_layout()
plt.show()