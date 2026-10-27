from pathlib import Path
import re
import cv2
import numpy as np

# Folder containing the annotated diagnostic frames
BASE_DIR = Path(__file__).resolve().parent

INPUT_DIR = BASE_DIR / "vid4_roi_diagnostic_frames"
OUTPUT_PATH = INPUT_DIR / "roi_contact_sheet.jpg"

# Inspect the beginning, middle, and later problem intervals
SELECTED_TIMES = [0, 10, 20, 25, 30, 35, 55, 60, 65]

TILE_WIDTH = 400
TILE_HEIGHT = 330
COLUMNS = 3

images = []

for path in sorted(INPUT_DIR.glob("*.jpg")):

    match = re.search(r"_(\d+)s\.jpg$", path.name)

    if not match:
        continue

    seconds = int(match.group(1))

    if seconds not in SELECTED_TIMES:
        continue

    image = cv2.imread(str(path))

    if image is None:
        print("Could not read:", path.name)
        continue

    # Resize to fit a common tile size.
    image = cv2.resize(
        image,
        (TILE_WIDTH, 280)
    )

    # Add a label beneath each image.
    tile = np.full(
        (TILE_HEIGHT, TILE_WIDTH, 3),
        255,
        dtype=np.uint8
    )

    tile[:280] = image

    cv2.putText(
        tile,
        f"Time: {seconds} seconds",
        (10, 310),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 0),
        2
    )

    images.append((seconds, tile))

images.sort(key=lambda item: item[0])

if not images:
    raise RuntimeError(
        f"No selected images found in {INPUT_DIR}"
    )

rows = (len(images) + COLUMNS - 1) // COLUMNS

sheet = np.full(
    (rows * TILE_HEIGHT, COLUMNS * TILE_WIDTH, 3),
    230,
    dtype=np.uint8
)

for i, (_, tile) in enumerate(images):
    row = i // COLUMNS
    col = i % COLUMNS

    y = row * TILE_HEIGHT
    x = col * TILE_WIDTH

    sheet[
        y:y + TILE_HEIGHT,
        x:x + TILE_WIDTH
    ] = tile

if not cv2.imwrite(str(OUTPUT_PATH), sheet):
    raise RuntimeError("Failed to save contact sheet.")

print("Images included:", len(images))
print("Saved contact sheet:", OUTPUT_PATH)