from pathlib import Path
import cv2
import mediapipe as mp
import numpy as np


def extract_roi_rgb(vid):
    video_path = str(vid)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    base_dir = Path(__file__).resolve().parent
    model_path = base_dir / "blaze_face_short_range.tflite"
    if not model_path.is_file():
        cap.release()
        raise FileNotFoundError(f"MediaPipe face model not found: {model_path}")

    BaseOptions = mp.tasks.BaseOptions
    FaceDetector = mp.tasks.vision.FaceDetector
    FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
    RunningMode = mp.tasks.vision.RunningMode
    options = FaceDetectorOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=RunningMode.IMAGE,
        min_detection_confidence=0.5,
    )
    face_detector = FaceDetector.create_from_options(options)

    forehead_values = []
    left_cheek_values = []
    right_cheek_values = []
    frame_numbers = []

    frame_count = 0
    forehead_replaced_count = 0
    left_cheek_replaced_count = 0
    right_cheek_replaced_count = 0
    low_skin_forehead_count = 0
    low_skin_left_count = 0
    low_skin_right_count = 0
    cheek_color_mismatch_count = 0
    forehead_coverage_sum = 0.0
    left_coverage_sum = 0.0
    right_coverage_sum = 0.0

    def clip_coordinates(x1, y1, x2, y2, width, height):
        x1 = max(0, min(width, int(round(x1))))
        y1 = max(0, min(height, int(round(y1))))
        x2 = max(0, min(width, int(round(x2))))
        y2 = max(0, min(height, int(round(y2))))
        return x1, y1, x2, y2

    def normalized_keypoint_xy(keypoint, width, height):
        px = float(keypoint.x)
        py = float(keypoint.y)
        if -0.01 <= px <= 1.01 and -0.01 <= py <= 1.01:
            px *= width
            py *= height
        return px, py

    def get_roi_boxes(detection, width, height):
        bbox = detection.bounding_box
        bx1 = float(bbox.origin_x)
        by1 = float(bbox.origin_y)
        bx2 = bx1 + float(bbox.width)
        by2 = by1 + float(bbox.height)
        bw = max(1.0, bx2 - bx1)
        bh = max(1.0, by2 - by1)

        keypoints = getattr(detection, "keypoints", None) or []
        if len(keypoints) >= 4:
            pts = [normalized_keypoint_xy(kp, width, height) for kp in keypoints[:4]]
            eyes = sorted(pts[:2], key=lambda p: p[0])
            (eye_lx, eye_ly), (eye_rx, eye_ry) = eyes
            nose_x, nose_y = pts[2]
            mouth_x, mouth_y = pts[3]
            eye_distance = eye_rx - eye_lx
            eye_mid_x = 0.5 * (eye_lx + eye_rx)
            eye_mid_y = 0.5 * (eye_ly + eye_ry)
            vertical_gap = mouth_y - eye_mid_y

            plausible = (
                eye_distance >= 0.10 * bw
                and vertical_gap >= 0.08 * bh
                and bx1 - 0.10 * bw <= eye_lx <= bx2 + 0.10 * bw
                and bx1 - 0.10 * bw <= eye_rx <= bx2 + 0.10 * bw
                and eye_mid_y < nose_y < mouth_y + 0.15 * bh
                and 0 <= eye_mid_x < width
                and 0 <= eye_mid_y < height
            )
            if plausible:
                forehead_w = 1.38 * eye_distance
                fx1 = eye_mid_x - 0.5 * forehead_w
                fx2 = eye_mid_x + 0.5 * forehead_w
                fy1 = eye_mid_y - 0.82 * vertical_gap
                fy2 = eye_mid_y - 0.26 * vertical_gap

                cheek_w = 0.53 * eye_distance
                cheek_y1 = eye_mid_y + 0.20 * vertical_gap
                cheek_y2 = eye_mid_y + 0.62 * vertical_gap
                left_cx = eye_lx + 0.07 * eye_distance
                right_cx = eye_rx - 0.07 * eye_distance
                lx1, lx2 = left_cx - 0.5 * cheek_w, left_cx + 0.5 * cheek_w
                rx1, rx2 = right_cx - 0.5 * cheek_w, right_cx + 0.5 * cheek_w
                ly1 = ry1 = cheek_y1
                ly2 = ry2 = cheek_y2

                margin_x = 0.035 * bw
                margin_y = 0.015 * bh
                fx1 = max(fx1, bx1 + margin_x)
                fx2 = min(fx2, bx2 - margin_x)
                lx1 = max(lx1, bx1 + margin_x)
                lx2 = min(lx2, bx2 - margin_x)
                rx1 = max(rx1, bx1 + margin_x)
                rx2 = min(rx2, bx2 - margin_x)
                return {
                    "forehead": clip_coordinates(fx1, fy1, fx2, fy2, width, height),
                    "left": clip_coordinates(lx1, ly1, lx2, ly2, width, height),
                    "right": clip_coordinates(rx1, ry1, rx2, ry2, width, height),
                    "method": "landmarks",
                }

        face_x1 = bx1 - 0.15 * bw
        face_x2 = bx2 + 0.15 * bw
        face_y1 = by1 - 0.35 * bh
        face_y2 = by2 + 0.10 * bh
        face_x1, face_y1, face_x2, face_y2 = clip_coordinates(
            face_x1, face_y1, face_x2, face_y2, width, height
        )
        fw = max(1, face_x2 - face_x1)
        fh = max(1, face_y2 - face_y1)
        fx1 = face_x1 + 0.25 * fw
        fx2 = face_x1 + 0.75 * fw
        fy1 = face_y1 + 0.20 * fh
        fy2 = face_y1 + 0.43 * fh
        lx1 = face_x1 + 0.27 * fw
        lx2 = face_x1 + 0.44 * fw
        rx1 = face_x1 + 0.56 * fw
        rx2 = face_x1 + 0.73 * fw
        ly1 = ry1 = face_y1 + 0.46 * fh
        ly2 = ry2 = face_y1 + 0.64 * fh
        return {
            "forehead": clip_coordinates(fx1, fy1, fx2, fy2, width, height),
            "left": clip_coordinates(lx1, ly1, lx2, ly2, width, height),
            "right": clip_coordinates(rx1, ry1, rx2, ry2, width, height),
            "method": "bbox-fallback",
        }

    def skin_mask(roi_bgr):
        if roi_bgr.size == 0:
            return np.zeros(roi_bgr.shape[:2], dtype=bool)
        ycrcb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2YCrCb)
        y_channel, cr_channel, cb_channel = cv2.split(ycrcb)
        b_channel, g_channel, r_channel = cv2.split(roi_bgr)
        mask = (
            (y_channel >= 35)
            & (cr_channel >= 130)
            & (cr_channel <= 182)
            & (cb_channel >= 72)
            & (cb_channel <= 138)
            & (r_channel.astype(np.int16) >= 0.90 * g_channel.astype(np.int16))
            & (g_channel.astype(np.int16) >= 0.72 * b_channel.astype(np.int16))
        ).astype(np.uint8) * 255
        if mask.shape[0] >= 7 and mask.shape[1] >= 7:
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        return mask > 0

    def measure_skin_rgb(roi_bgr):
        h, w = roi_bgr.shape[:2]
        if h < 1 or w < 1:
            return np.zeros(3, dtype=np.float64), 0.0, False
        roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
        mask = skin_mask(roi_bgr)
        skin_pixels = int(np.count_nonzero(mask))
        coverage = skin_pixels / float(mask.size)
        min_pixels = max(18, int(0.04 * mask.size))
        if skin_pixels >= min_pixels:
            return np.mean(roi_rgb[mask], axis=0).astype(np.float64), coverage, True

        cy1, cy2 = int(0.15 * h), max(int(0.85 * h), 1)
        cx1, cx2 = int(0.15 * w), max(int(0.85 * w), 1)
        center_bgr = roi_bgr[cy1:cy2, cx1:cx2]
        center_rgb = cv2.cvtColor(center_bgr, cv2.COLOR_BGR2RGB)
        center_mask = skin_mask(center_bgr)
        if np.count_nonzero(center_mask) >= max(12, int(0.04 * center_mask.size)):
            return np.mean(center_rgb[center_mask], axis=0).astype(np.float64), coverage, False

        return np.mean(center_rgb, axis=(0, 1)).astype(np.float64), coverage, False

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            height, width = frame.shape[:2]
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            results = face_detector.detect(mp_image)

            if results.detections:
                detection = max(
                    results.detections,
                    key=lambda d: d.bounding_box.width * d.bounding_box.height,
                )
                boxes = get_roi_boxes(detection, width, height)
                fx1, fy1, fx2, fy2 = boxes["forehead"]
                lx1, ly1, lx2, ly2 = boxes["left"]
                rx1, ry1, rx2, ry2 = boxes["right"]
                forehead_roi = frame[fy1:fy2, fx1:fx2]
                left_roi = frame[ly1:ly2, lx1:lx2]
                right_roi = frame[ry1:ry2, rx1:rx2]

                rois_ok = all(
                    roi.size > 0 and roi.shape[0] >= 5 and roi.shape[1] >= 5
                    for roi in (forehead_roi, left_roi, right_roi)
                )
                if rois_ok:
                    forehead_mean, forehead_coverage, forehead_reliable = measure_skin_rgb(forehead_roi)
                    left_mean, left_coverage, left_reliable = measure_skin_rgb(left_roi)
                    right_mean, right_coverage, right_reliable = measure_skin_rgb(right_roi)

                    forehead_coverage_sum += forehead_coverage
                    left_coverage_sum += left_coverage
                    right_coverage_sum += right_coverage

                    if forehead_coverage < 0.35 or not forehead_reliable:
                        low_skin_forehead_count += 1
                    if left_coverage < 0.30 or not left_reliable:
                        low_skin_left_count += 1
                    if right_coverage < 0.30 or not right_reliable:
                        low_skin_right_count += 1

                    if not left_reliable and right_reliable:
                        left_mean = right_mean.copy()
                        left_cheek_replaced_count += 1
                    elif not right_reliable and left_reliable:
                        right_mean = left_mean.copy()
                        right_cheek_replaced_count += 1
                    elif not left_reliable and not right_reliable:
                        cheek_color_mismatch_count += 1

                    def chromaticity(rgb_mean):
                        total = float(np.sum(rgb_mean))
                        if total <= 1e-8:
                            return np.zeros(3, dtype=np.float64)
                        return rgb_mean / total

                    cheek_chroma_distance = float(
                        np.linalg.norm(chromaticity(left_mean) - chromaticity(right_mean))
                    )
                    if cheek_chroma_distance > 0.17:
                        cheek_color_mismatch_count += 1

                    cheek_luma = 0.5 * (
                        0.2126 * left_mean[0] + 0.7152 * left_mean[1] + 0.0722 * left_mean[2]
                        + 0.2126 * right_mean[0] + 0.7152 * right_mean[1] + 0.0722 * right_mean[2]
                    )
                    forehead_luma = (
                        0.2126 * forehead_mean[0]
                        + 0.7152 * forehead_mean[1]
                        + 0.0722 * forehead_mean[2]
                    )
                    forehead_obstructed = (
                        forehead_coverage < 0.35
                        or not forehead_reliable
                        or (cheek_luma > 15.0 and forehead_luma < 0.62 * cheek_luma)
                    )
                    if forehead_obstructed:
                        forehead_mean = 0.5 * (left_mean + right_mean)
                        forehead_replaced_count += 1

                    forehead_values.append(forehead_mean)
                    left_cheek_values.append(left_mean)
                    right_cheek_values.append(right_mean)
                    frame_numbers.append(frame_count)

                    cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), (255, 0, 0), 2)
                    cv2.putText(
                        frame, f"Forehead {100 * forehead_coverage:.0f}%",
                        (fx1, max(20, fy1 - 5)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.48, (255, 0, 0), 2,
                    )
                    cv2.rectangle(frame, (lx1, ly1), (lx2, ly2), (0, 255, 255), 2)
                    cv2.putText(
                        frame, f"Left {100 * left_coverage:.0f}%",
                        (lx1, max(20, ly1 - 5)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.48, (0, 255, 255), 2,
                    )
                    cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (0, 255, 255), 2)
                    cv2.putText(
                        frame, f"Right {100 * right_coverage:.0f}%",
                        (rx1, max(20, ry1 - 5)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.48, (0, 255, 255), 2,
                    )
                    if forehead_obstructed:
                        cv2.putText(
                            frame, "Forehead excluded", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2,
                        )
                    if not left_reliable or not right_reliable:
                        cv2.putText(
                            frame, "Check cheek skin mask", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2,
                        )

            cv2.imshow("ROI Extraction - press q to stop", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        face_detector.close()
        cv2.destroyAllWindows()

    forehead_values = np.asarray(forehead_values, dtype=np.float64).reshape(-1, 3)
    left_cheek_values = np.asarray(left_cheek_values, dtype=np.float64).reshape(-1, 3)
    right_cheek_values = np.asarray(right_cheek_values, dtype=np.float64).reshape(-1, 3)

    print("Total frames processed:", frame_count)
    print("Frames with valid ROIs:", len(frame_numbers))
    print("Forehead replaced by cheek average:", forehead_replaced_count)
    print("Left cheek replaced by right cheek:", left_cheek_replaced_count)
    print("Right cheek replaced by left cheek:", right_cheek_replaced_count)
    print("Frames with low forehead skin coverage:", low_skin_forehead_count)
    print("Frames with low left-cheek skin coverage:", low_skin_left_count)
    print("Frames with low right-cheek skin coverage:", low_skin_right_count)
    print("Frames with cheek color mismatch warning:", cheek_color_mismatch_count)
    if len(frame_numbers):
        print("Mean forehead skin coverage:", f"{100 * forehead_coverage_sum / len(frame_numbers):.1f}%")
        print("Mean left-cheek skin coverage:", f"{100 * left_coverage_sum / len(frame_numbers):.1f}%")
        print("Mean right-cheek skin coverage:", f"{100 * right_coverage_sum / len(frame_numbers):.1f}%")
    print("Forehead RGB shape:", forehead_values.shape)
    print("Left cheek RGB shape:", left_cheek_values.shape)
    print("Right cheek RGB shape:", right_cheek_values.shape)

    output_path = base_dir / "all_roi_rgb.csv"
    if len(frame_numbers) > 0:
        all_roi_data = np.column_stack(
            (frame_numbers, forehead_values, left_cheek_values, right_cheek_values)
        )
        np.savetxt(
            output_path,
            all_roi_data,
            delimiter=",",
            header=(
                "Frame,"
                "Forehead_R,Forehead_G,Forehead_B,"
                "LeftCheek_R,LeftCheek_G,LeftCheek_B,"
                "RightCheek_R,RightCheek_G,RightCheek_B"
            ),
            comments="",
            fmt="%.6f",
        )
        print("CSV saved:", output_path)
    else:
        print("No valid face ROIs detected. CSV not saved.")

    return forehead_values, left_cheek_values, right_cheek_values


if __name__ == "__main__":
    pass
