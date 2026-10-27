import pandas as pd
import numpy as np

input_file = "normalized_rgb.csv"

df = pd.read_csv(input_file)

print("Normalized RGB data:")
print(df.head())


r = df["r"].to_numpy(dtype=np.float64)
g = df["g"].to_numpy(dtype=np.float64)
b = df["b"].to_numpy(dtype=np.float64)

print("\nNumber of samples:", len(r))


X1 = g - b
X2 = -2 * r + g + b


std_X1 = np.std(X1)
std_X2 = np.std(X2)

if std_X2 == 0:
    raise ValueError("Standard deviation of X2 is zero.")

alpha = std_X1 / std_X2

print("\nPOS parameters")
print("std(X1):", std_X1)
print("std(X2):", std_X2)
print("alpha:", alpha)


pos_signal = X1 + alpha * X2

print("\nPOS signal:")
print(pos_signal[:10])


pos_df = pd.DataFrame({
    "Frame": df["Frame"],
    "POS": pos_signal
})

output_file = "pos_signal.csv"

pos_df.to_csv(
    output_file,
    index=False
)

print("\nSuccessfully saved:", output_file)