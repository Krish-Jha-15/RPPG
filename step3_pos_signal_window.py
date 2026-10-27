import pandas as pd
import numpy as np


def calculate_pos(input_file, FPS, WINDOW_SECONDS=1.6):

    # Read averaged RGB data
    df = pd.read_csv(input_file)

    R = df["R"].to_numpy(dtype=np.float64)
    G = df["G"].to_numpy(dtype=np.float64)
    B = df["B"].to_numpy(dtype=np.float64)

    frames = df["Frame"].to_numpy()

    # Calculate window length
    WINDOW_LENGTH = round(FPS * WINDOW_SECONDS)

    print("FPS:", FPS)
    print("Window length:", WINDOW_LENGTH)

    N = len(df)
    number_of_windows = N - WINDOW_LENGTH + 1

    print("Total samples:", N)
    print("Number of POS windows:", max(0, number_of_windows))

    if WINDOW_LENGTH < 1 or WINDOW_LENGTH > N:
        raise ValueError(
            "Window length must be between 1 and the number of samples."
        )

    # Initialize POS signal
    H = np.zeros(N, dtype=np.float64)

    # Sliding window POS calculation
    for start in range(number_of_windows):

        end = start + WINDOW_LENGTH

        R_window = R[start:end]
        G_window = G[start:end]
        B_window = B[start:end]

        # Normalize RGB channels
        R_mean = np.mean(R_window)
        G_mean = np.mean(G_window)
        B_mean = np.mean(B_window)

        if R_mean == 0 or G_mean == 0 or B_mean == 0:
            continue

        Rn = R_window / R_mean
        Gn = G_window / G_mean
        Bn = B_window / B_mean

        # POS projections
        S1 = Gn - Bn
        S2 = -2 * Rn + Gn + Bn

        std_S1 = np.std(S1)
        std_S2 = np.std(S2)

        if std_S2 == 0:
            continue

        alpha = std_S1 / std_S2

        h = S1 + alpha * S2
        h = h - np.mean(h)

        # Overlap-add
        H[start:end] += h

    # Create output DataFrame
    pos_df = pd.DataFrame({
        "Frame": frames,
        "POS": H
    })

    # Save POS signal
    output_file = "pos_signal_windowed.csv"
    pos_df.to_csv(output_file, index=False)

    print("\n========== WINDOWED POS RESULT ==========")
    print("Output shape:", pos_df.shape)
    print("\nFirst 10 POS values:")
    print(pos_df.head(10))
    print("\nPOS signal mean:", np.mean(H))
    print("POS signal std:", np.std(H))
    print("\nSuccessfully saved:", output_file)
    print("=========================================")

    return pos_df


# Function call
# pos_df = calculate_pos("average_rgb.csv",FPS=28.671786)
