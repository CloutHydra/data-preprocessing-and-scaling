# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.datasets import load_wine
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt

# 1. Data Loading
wine = load_wine()
X = wine.data
y = wine.target
feature_names = wine.feature_names
target_names = wine.target_names

# Convert to DataFrame
X_df = pd.DataFrame(X, columns=feature_names)
y_df = pd.DataFrame(y, columns=["Cultivar"])


def equilateral_encode(n_classes):
    """Generate equilateral encoding vectors for n_classes."""
    if n_classes < 2:
        raise ValueError("Number of classes must be at least 2.")

    eq_matrix = np.zeros((n_classes, n_classes - 1))
    for i in range(1, n_classes):
        eq_matrix[:i, i - 1] = -1 / i
        eq_matrix[i, i - 1] = i
        eq_matrix[:i, :i] *= np.sqrt((i**2 + i) / (i**2 + i + 1))
    return eq_matrix


# 2. Data Preprocessing
onehot_encoder = OneHotEncoder(sparse_output=False)
y_onehot = onehot_encoder.fit_transform(y_df[["Cultivar"]])
y_onehot_df = pd.DataFrame(
    y_onehot, columns=[f"Cultivar_{i}" for i in range(len(target_names))]
)

# 3. Normalization
minmax_scaler = MinMaxScaler()
X_minmax = minmax_scaler.fit_transform(X_df)
X_minmax_df = pd.DataFrame(X_minmax, columns=feature_names)

standard_scaler = StandardScaler()
X_standard = standard_scaler.fit_transform(X_df)
X_standard_df = pd.DataFrame(X_standard, columns=feature_names)

alcohol_reciprocal = 1 / X_df["alcohol"]
alcohol_reciprocal_df = pd.DataFrame(
    {"alcohol_reciprocal": alcohol_reciprocal}
)

# 4. Visualization
plt.style.use("default")
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

axes[0, 0].hist(X_df["malic_acid"], bins=20, alpha=0.7, color="blue", edgecolor="black")
axes[0, 0].set_title("Malic Acid - Before Min-Max Scaling")
axes[0, 0].set_xlabel("Malic Acid")
axes[0, 0].set_ylabel("Frequency")

axes[0, 1].hist(X_minmax_df["malic_acid"], bins=20, alpha=0.7, color="red", edgecolor="black")
axes[0, 1].set_title("Malic Acid - After Min-Max Scaling")
axes[0, 1].set_xlabel("Malic Acid (Scaled)")
axes[0, 1].set_ylabel("Frequency")

axes[0, 2].hist(X_df["ash"], bins=20, alpha=0.7, color="blue", edgecolor="black")
axes[0, 2].set_title("Ash - Before Standard Scaling")
axes[0, 2].set_xlabel("Ash")
axes[0, 2].set_ylabel("Frequency")

axes[1, 0].hist(X_standard_df["ash"], bins=20, alpha=0.7, color="red", edgecolor="black")
axes[1, 0].set_title("Ash - After Standard Scaling")
axes[1, 0].set_xlabel("Ash (Standardized)")
axes[1, 0].set_ylabel("Frequency")

axes[1, 1].hist(X_df["alcohol"], bins=20, alpha=0.7, color="blue", edgecolor="black")
axes[1, 1].set_title("Alcohol - Before Reciprocal")
axes[1, 1].set_xlabel("Alcohol")
axes[1, 1].set_ylabel("Frequency")

axes[1, 2].hist(alcohol_reciprocal, bins=20, alpha=0.7, color="red", edgecolor="black")
axes[1, 2].set_title("Alcohol - After Reciprocal")
axes[1, 2].set_xlabel("1 / Alcohol")
axes[1, 2].set_ylabel("Frequency")

plt.tight_layout()
plt.show()

# 5. Equilateral Encoding
n_classes = len(target_names)
eq_encoding = equilateral_encode(n_classes)
y_equilateral = np.array([eq_encoding[i] for i in y])
y_equilateral_df = pd.DataFrame(
    y_equilateral, columns=[f"EQ_{i}" for i in range(n_classes - 1)]
)
