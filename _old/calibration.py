from pathlib import Path
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox

# -------- Settings to adjust --------
IMAGE_DIR = Path("images")

# print(f"Looking in: {IMAGE_DIR.resolve()}")
# print(f"Found {len(image_paths)} image(s)")

# Number of INNER corners: columns, rows.
# Example: a board with 10 squares across and 7 down has (9, 6) inner corners.
BOARD_SIZE = (8, 5)

# Physical distance between adjacent inner corners, in your preferred unit
# (e.g. millimeters). The calibration scale will use that same unit.
SQUARE_SIZE = 29.0

OUTPUT_FILE = Path("camera_calibration.npz")

# Set to None to disable automatic view rejection.
# Views with higher per-image reprojection RMSE are excluded, if enough views remain.
MAX_VIEW_RMSE_PX = 2.0
MIN_VIEWS = 10
# ------------------------------------


def detect_corners(gray, board_size):
    # Try several contrast treatments. These preserve the image dimensions.
    variants = [gray]

    # Local contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    variants.append(clahe.apply(gray))

    # Global histogram equalization
    variants.append(cv2.equalizeHist(gray))

    # Stretch the useful grayscale range, ignoring a few extreme pixels
    low, high = np.percentile(gray, (1, 99))
    if high > low:
        stretched = np.clip(
            (gray.astype(np.float32) - low) * (255.0 / (high - low)),
            0,
            255,
        ).astype(np.uint8)
        variants.append(stretched)

    # Gamma variants: one brightens darker regions, one darkens bright regions
    for gamma in (0.7, 1.4):
        lut = np.array(
            [((i / 255.0) ** gamma) * 255 for i in range(256)],
            dtype=np.uint8,
        )
        variants.append(cv2.LUT(gray, lut))

    # SB detector flags; use whichever are available in this OpenCV build
    sb_flags = getattr(cv2, "CALIB_CB_NORMALIZE_IMAGE", 0)
    sb_flags |= getattr(cv2, "CALIB_CB_EXHAUSTIVE", 0)
    sb_flags |= getattr(cv2, "CALIB_CB_ACCURACY", 0)

    classic_flags = (
        cv2.CALIB_CB_ADAPTIVE_THRESH
        | cv2.CALIB_CB_NORMALIZE_IMAGE
    )
    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER,
        40,
        1e-3,
    )

    for candidate in variants:
        # Try the SB detector first
        if hasattr(cv2, "findChessboardCornersSB"):
            try:
                found, corners = cv2.findChessboardCornersSB(
                    candidate, board_size, flags=sb_flags
                )
                if found:
                    return True, corners.astype(np.float32)
            except cv2.error:
                pass

        # Fall back to the classic detector
        found, corners = cv2.findChessboardCorners(
            candidate, board_size, classic_flags
        )
        if found:
            corners = cv2.cornerSubPix(
                candidate,
                corners,
                winSize=(11, 11),
                zeroZone=(-1, -1),
                criteria=criteria,
            )
            return True, corners.astype(np.float32)

    return False, None


def fit_camera(object_points, image_points, image_size):
    return cv2.calibrateCamera(
        object_points,
        image_points,
        image_size,  # (width, height)
        None,
        None,
    )


def per_view_rmse(object_points, image_points, rvecs, tvecs, camera_matrix, dist_coeffs):
    errors = []
    for obj, observed, rvec, tvec in zip(
        object_points, image_points, rvecs, tvecs
    ):
        projected, _ = cv2.projectPoints(
            obj, rvec, tvec, camera_matrix, dist_coeffs
        )
        delta = observed.reshape(-1, 2) - projected.reshape(-1, 2)
        errors.append(float(np.sqrt(np.mean(np.sum(delta * delta, axis=1)))))
    return np.asarray(errors)


# GUI window
def prompt_for_settings(default_folder, default_board_size, default_square_size):
    root = tk.Tk()
    root.title("Camera Calibration Settings")
    root.option_add("*Font", "TkDefaultFont 11 bold")
    root.resizable(False, False)

    bg_color = "#353535"
    input_color = "#949494"
    btn_color = "#FC7121"
    root.configure(bg=bg_color)

    folder_var = tk.StringVar(value=str(default_folder.resolve()))
    columns_var = tk.StringVar(value=str(default_board_size[0]))
    rows_var = tk.StringVar(value=str(default_board_size[1]))
    square_size_var = tk.StringVar(value=str(default_square_size))
    result = {}

    def browse_for_folder():
        folder = filedialog.askdirectory(
            title="Select the folder containing calibration images",
            initialdir=folder_var.get(),
        )
        if folder:
            folder_var.set(folder)

    def start_calibration():
        from pathlib import Path

        folder = Path(folder_var.get()).expanduser()
        if not folder.is_dir():
            messagebox.showerror("Invalid folder", "Select an existing image folder.")
            return

        try:
            columns = int(columns_var.get())
            rows = int(rows_var.get())
            square_size = float(square_size_var.get())

            if columns < 2 or rows < 2:
                raise ValueError
            if not np.isfinite(square_size) or square_size <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror(
                "Invalid settings",
                "Enter at least 2 inner corners for each board dimension "
                "and a positive square size.",
            )
            return

        result["settings"] = (folder, (columns, rows), square_size)

        root.destroy()

    tk.Label(root, text="Image folder:", bg=bg_color, fg="white").grid(
        row=0, column=0, padx=8, pady=8, sticky="w"
    )
    tk.Entry(root, textvariable=folder_var, width=45).grid(
        row=0, column=1, padx=8, pady=8
    )
    tk.Button(root, text="Browse", command=browse_for_folder).grid(
        row=0, column=2, padx=8, pady=8
    )

    tk.Label(root, text="Columns (squares in col - 1):", bg=bg_color, fg="white").grid(
        row=1, column=0, padx=8, pady=8, sticky="w"
    )
    tk.Entry(root, textvariable=columns_var, width=8).grid(
        row=1, column=1, padx=8, pady=8, sticky="w"
    )

    tk.Label(root, text="Rows (squares in rows - 1):", bg=bg_color, fg="white").grid(
        row=2, column=0, padx=8, pady=8, sticky="w"
    )
    tk.Entry(root, textvariable=rows_var, width=8).grid(
        row=2, column=1, padx=8, pady=8, sticky="w"
    )
    tk.Label(
        root, text="Square size (e.g. mm):", bg=bg_color, fg="white"
    ).grid(row=3, column=0, padx=8, pady=8, sticky="w")
    tk.Entry(root, textvariable=square_size_var, width=8).grid(
        row=3, column=1, padx=8, pady=8, sticky="w"
    )

    tk.Button(root, text="Start calibration", bg=btn_color, fg="white", command=start_calibration).grid(
        row=4, column=0, columnspan=3, pady=12
    )

    root.mainloop()
    return result.get("settings")


def main(image_dir, board_size, square_size):
    print("Starting calibration...")

    if len(board_size) != 2 or min(board_size) < 2:
        raise ValueError("board_size must be (columns, rows) of inner corners.")
    if square_size <= 0:
        raise ValueError("square_size must be greater than zero.")

    image_paths = sorted(
        path
        for pattern in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
        for path in image_dir.glob(pattern)
    )
    print(f"Looking in: {image_dir.resolve()}")
    print(f"Found {len(image_paths)} image(s)")

    if not image_paths:
        raise FileNotFoundError(
            f"No images found in {image_dir.resolve()}"
        )

    # 3D checkerboard points, with the board lying on Z=0.
    board_points = np.zeros(
        (board_size[0] * board_size[1], 3), dtype=np.float32
    )
    board_points[:, :2] = (
        np.mgrid[0:board_size[0], 0:board_size[1]]
        .T.reshape(-1, 2)
        .astype(np.float32)
        * square_size
    )

    object_points = []
    image_points = []
    image_size = None
    accepted_paths = []
    rejected_paths = []

    for path in image_paths:
        image = cv2.imread(str(path))
        if image is None:
            print(f"Skipping unreadable image: {path}")
            rejected_paths.append(path)
            continue

        height, width = image.shape[:2]
        current_size = (width, height)

        if image_size is None:
            image_size = current_size
        elif current_size != image_size:
            print(
                f"Skipping {path}: size {current_size} differs from "
                f"{image_size}"
            )
            rejected_paths.append(path)
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        found, corners = detect_corners(gray, board_size)

        if not found:
            print(f"No complete checkerboard detected: {path}")
            rejected_paths.append(path)
            continue

        object_points.append(board_points.copy())
        image_points.append(corners)
        accepted_paths.append(path)
        print(f"Detected board: {path}")

    if len(object_points) < MIN_VIEWS:
        raise RuntimeError(
            f"Only {len(object_points)} usable views; need at least "
            f"{MIN_VIEWS}. Check board_size and add more images."
        )

    rms, camera_matrix, dist_coeffs, rvecs, tvecs = fit_camera(
        object_points, image_points, image_size
    )
    errors = per_view_rmse(
        object_points, image_points, rvecs, tvecs,
        camera_matrix, dist_coeffs
    )

    if MAX_VIEW_RMSE_PX is not None:
        keep = errors <= MAX_VIEW_RMSE_PX
        if keep.sum() >= MIN_VIEWS and not keep.all():
            for path, error, use_view in zip(accepted_paths, errors, keep):
                if not use_view:
                    print(f"Excluding high-error view ({error:.3f}px): {path}")

            object_points = [
                points for points, use_view in zip(object_points, keep)
                if use_view
            ]
            image_points = [
                points for points, use_view in zip(image_points, keep)
                if use_view
            ]
            accepted_paths = [
                path for path, use_view in zip(accepted_paths, keep)
                if use_view
            ]

            rms, camera_matrix, dist_coeffs, rvecs, tvecs = fit_camera(
                object_points, image_points, image_size
            )
            errors = per_view_rmse(
                object_points, image_points, rvecs, tvecs,
                camera_matrix, dist_coeffs
            )

    print(f"\nCalibration RMS: {rms:.4f} px")
    print(f"Mean per-view RMSE: {errors.mean():.4f} px")
    print(f"Worst per-view RMSE: {errors.max():.4f} px")
    print(f"Used views: {len(accepted_paths)}")
    print("Camera matrix:\n", camera_matrix)
    print("Distortion coefficients:\n", dist_coeffs.ravel())

    np.savez(
        OUTPUT_FILE,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        image_size=np.asarray(image_size, dtype=np.int32),
        rms=np.asarray(rms),
        per_view_rmse=errors,
        used_images=np.asarray([str(p) for p in accepted_paths]),
    )
    print(f"Saved calibration to: {OUTPUT_FILE}")


if __name__ == "__main__":
    settings = prompt_for_settings(IMAGE_DIR, BOARD_SIZE, SQUARE_SIZE)
    if settings is not None:
        main(*settings)
