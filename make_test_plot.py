import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

T = np.linspace(140, 185, 900)                     # temperature, °C
signal = -0.8 * np.exp(-((T - 156.5) / 1.2) ** 2)  # one endothermic peak
noise = np.random.default_rng(0).normal(0, 0.01, T.size)

plt.plot(T, signal + noise, lw=1)
plt.xlabel("Temperature (°C)")
plt.ylabel("Heat flow (W/g), exo up")
plt.title("Test curve")
plt.savefig("test_plot.png", dpi=150)
print("saved test_plot.png")
