import pandas as pd
import numpy as np


def average_rgb(input_file):

    # Read CSV file
    df = pd.read_csv(input_file)

    print("Original Data:")
    print(df.head())

    # Average R channel
    R_final = (
        df["Forehead_R"] +
        df["LeftCheek_R"] +
        df["RightCheek_R"]
    ) / 3

    # Average G channel
    G_final = (
        df["Forehead_G"] +
        df["LeftCheek_G"] +
        df["RightCheek_G"]
    ) / 3

    # Average B channel
    B_final = (
        df["Forehead_B"] +
        df["LeftCheek_B"] +
        df["RightCheek_B"]
    ) / 3

    # Create final DataFrame
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

    # Save output CSV
    output_file = "average_rgb.csv"

    final_df.to_csv(output_file, index=False)

    print(f"\nSuccessfully saved: {output_file}")

    return final_df


# Function call
# final_df = average_rgb("all_roi_rgb.csv")
