import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler

np.random.seed(42)
X = 10 * np.random.rand(100, 2) + 5

# Zero-center the data
scaler_zero_centered = StandardScaler(with_std=False)
X_zero_centered = scaler_zero_centered.fit_transform(X)

# Normalize the data to [0, 1]
scaler_normalized = MinMaxScaler()
X_normalized = scaler_normalized.fit_transform(X)

# Visualize original, centered, and normalized data
fig, axs = plt.subplots(1, 3, figsize=(18, 6))

axs[0].scatter(X[:, 0], X[:, 1], color="blue")
axs[0].set_title("Original Data")
axs[0].set_xlim(-15, 15)
axs[0].set_ylim(-15, 15)

axs[1].scatter(X_zero_centered[:, 0], X_zero_centered[:, 1], color="blue")
axs[1].set_title("Zero-Centered Data")
axs[1].set_xlim(-15, 15)
axs[1].set_ylim(-15, 15)

axs[2].scatter(X_normalized[:, 0], X_normalized[:, 1], color="blue")
axs[2].set_title("Normalized Data")
axs[2].set_xlim(-0.1, 1.1)
axs[2].set_ylim(-0.1, 1.1)

plt.tight_layout()
plt.show()
