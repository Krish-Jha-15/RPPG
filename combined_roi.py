import pandas as pd
import numpy as np

input_file = "all_roi_rgb.csv"
df = pd.read_csv(input_file)

print("Original Data:")
print(df.head())

# Forehead
R_forehead = df["Forehead_R"]
G_forehead = df["Forehead_G"]
B_forehead = df["Forehead_B"]


# Left Cheek
R_left_cheek = df["LeftCheek_R"]
G_left_cheek = df["LeftCheek_G"]
B_left_cheek = df["LeftCheek_B"]


# Right Cheek
R_right_cheek = df["RightCheek_R"]
G_right_cheek = df["RightCheek_G"]
B_right_cheek = df["RightCheek_B"]


R_final = (
    R_forehead +
    R_left_cheek +
    R_right_cheek
) / 3


G_final = (
    G_forehead +
    G_left_cheek +
    G_right_cheek
) / 3


B_final = (
    B_forehead +
    B_left_cheek +
    B_right_cheek
) / 3


final_df = pd.DataFrame({

    "Frame": df["Frame"],

    "R": R_final,

    "G": G_final,

    "B": B_final

})

print("\nAveraged RGB Data:")
print(final_df.head())


print("\nShape of Final Data:")
print(final_df.shape)


output_file = "average_rgb.csv"

final_df.to_csv(
    output_file,
    index=False
)


print(f"\nSuccessfully saved: {output_file}")

# Calculate mean of each RGB channel
R_mean = final_df["R"].mean()
G_mean = final_df["G"].mean()
B_mean = final_df["B"].mean()

# Normalize each channel
r = (final_df["R"] - R_mean) / R_mean
g = (final_df["G"] - G_mean) / G_mean
b = (final_df["B"] - B_mean) / B_mean

normalized_df = pd.DataFrame({
    "Frame": final_df["Frame"],
    "r": r,
    "g": g,
    "b": b
})

print("\nNormalized RGB Data:")
print(normalized_df.head())

print("\nShape of Normalized Data:")
print(normalized_df.shape)

output_file = "normalized_rgb.csv"

normalized_df.to_csv(
    output_file,
    index=False
)

print(f"\nSuccessfully saved: {output_file}")