import contextlib
import glob
import io
import json
import os
import re
import socket
import struct
import subprocess
import threading
import time
import tkinter as tk
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, ttk


import cv2
import numpy as np
from PIL import Image, ImageTk

# import the proojector calibration file when trying to use the external file
# but with with newly conda environment (opencv 4.14 NOT 5.0.0 originally) it seems to work fine with
# the inbuilt function "start_projector_calibration_from_sidebar", further info there...
# import projectorCalibrate

import exportUnityCalibration

SONY_API_URL = "http://10.0.0.1:10000/camera"

IMAGE_DIR = Path("images")
BOARD_SIZE = (8, 5)
SQUARE_SIZE = 29.0
OUTPUT_FILE = Path("results/result_camera_calibration.json")
MAX_VIEW_RMSE_PX = 2.0
MIN_VIEWS = 10
COMMON_RESOLUTION_OPTIONS = [
    "3840x2160", "2560x1440", "1920x1080", "1600x900",
    "1280x960", "1280x720", "1024x768", "960x720",
    "800x600", "720x576", "720x480", "640x480",
    "640x360", "480x360", "320x240",
]
CALIBRATION_LOG_COLOR_RULES = [
    ("Rejected", "#ff6b6b"),
    ("no chessboard corners found", "#ff6b6b"),
    ("Error", "#ff6b6b"),
    ("Warning", "#ffb86c"),
    ("Accepted", "#7ae582"),
    ("Saved", "#7ae582"),
    ("Calibration complete", "#7ae582"),
    ("Finished calibration successfully", "#7ae582"),
    ("Starting calibration", "#7cc8ff"),
]


def detect_corners(gray, board_size):
    variants = [gray]
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    variants.append(clahe.apply(gray))
    variants.append(cv2.equalizeHist(gray))

    low, high = np.percentile(gray, (1, 99))
    if high > low:
        stretched = np.clip(
            (gray.astype(np.float32) - low) * (255.0 / (high - low)),
            0,
            255,
        ).astype(np.uint8)
        variants.append(stretched)

    for gamma in (0.7, 1.4):
        lut = np.array(
            [((i / 255.0) ** gamma) * 255 for i in range(256)],
            dtype=np.uint8,
        )
        variants.append(cv2.LUT(gray, lut))

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
        if hasattr(cv2, "findChessboardCornersSB"):
            try:
                found, corners = cv2.findChessboardCornersSB(
                    candidate, board_size, flags=sb_flags
                )
                if found:
                    return True, corners.astype(np.float32)
            except cv2.error:
                pass

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
        image_size,
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


def run_camera_calibration(image_dir, board_size, square_size, output_file=OUTPUT_FILE,
                           max_view_rmse_px=MAX_VIEW_RMSE_PX, min_views=MIN_VIEWS,
                           expected_size=None):
    image_dir = Path(image_dir).expanduser()
    if len(board_size) != 2 or min(board_size) < 2:
        raise ValueError(
            "board_size must be (columns, rows) of inner corners.")
    if square_size <= 0:
        raise ValueError("square_size must be greater than zero.")
    if expected_size is not None:
        try:
            expected_size = tuple(int(value) for value in expected_size)
        except (TypeError, ValueError):
            raise ValueError("expected_size must be a (width, height) tuple.")
        if len(expected_size) != 2 or min(expected_size) <= 0:
            raise ValueError(
                "expected_size must contain positive width and height values.")
        print(
            f"[calibration] Restricting calibration images to {expected_size[0]}x{expected_size[1]} pixels."
        )

    print(f"[calibration] Scanning folder: {image_dir.resolve()}")
    image_paths = sorted(
        path
        for pattern in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
        for path in image_dir.glob(pattern)
    )
    print(
        f"[calibration] Found {len(image_paths)} candidate image(s) in folder.")

    if not image_paths:
        raise FileNotFoundError(f"No images found in {image_dir.resolve()}")

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
        print(f"[calibration] Processing {path.name}...")
        image = cv2.imread(str(path))
        if image is None:
            print(
                f"[calibration] Rejected {path.name}: could not decode image.")
            rejected_paths.append(path)
            continue

        height, width = image.shape[:2]
        current_size = (width, height)

        if expected_size is not None and current_size != expected_size:
            print(
                f"[calibration] Rejected {path.name}: size mismatch {current_size} != {expected_size}."
            )
            rejected_paths.append(path)
            continue

        if image_size is None:
            image_size = current_size
            print(
                f"[calibration] Image size detected: {current_size[0]}x{current_size[1]}.")
        elif current_size != image_size:
            print(
                f"[calibration] Rejected {path.name}: size mismatch {current_size} != {image_size}.")
            rejected_paths.append(path)
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        found, corners = detect_corners(gray, board_size)

        if not found:
            print(
                f"[calibration] Rejected {path.name}: no chessboard corners found.")
            rejected_paths.append(path)
            continue

        object_points.append(board_points.copy())
        image_points.append(corners)
        accepted_paths.append(path)
        print(
            f"[calibration] Accepted {path.name} ({len(accepted_paths)} usable so far).")

    if len(object_points) < min_views:
        raise RuntimeError(
            f"Only {len(object_points)} usable views; need at least {min_views}. Check board_size and add more images."
        )

    print(
        f"[calibration] Ready to calibrate with {len(object_points)} usable image(s).")
    rms, camera_matrix, dist_coeffs, rvecs, tvecs = fit_camera(
        object_points, image_points, image_size
    )
    errors = per_view_rmse(
        object_points, image_points, rvecs, tvecs,
        camera_matrix, dist_coeffs
    )

    if max_view_rmse_px is not None:
        keep = errors <= max_view_rmse_px
        if keep.sum() >= min_views and not keep.all():
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

    print(f"[calibration] RMS reprojection error: {rms:.4f}px")
    print(f"[calibration] Saving calibration output to {output_file}")
    # np.savez(
    #     output_file,
    #     camera_matrix=camera_matrix,
    #     dist_coeffs=dist_coeffs,
    #     image_size=np.asarray(image_size, dtype=np.int32),
    #     rms=np.asarray(rms),
    #     per_view_rmse=errors,
    #     used_images=np.asarray([str(p) for p in accepted_paths]),
    # )

    # instead output a json file for use ingraypattern calibration process
    json_file = Path(output_file).with_suffix(".json")

    camCalibConfig = {
        "camera": {
            "P": camera_matrix.tolist(),
            "distortion": dist_coeffs.reshape(-1).tolist(),
        },
        "image_size": np.asarray(image_size, dtype=np.int32).tolist(),
        "rms": float(rms),
        "per_view_rmse": np.asarray(errors).tolist(),
        "used_images": [str(p) for p in accepted_paths],
    }

    with json_file.open("w", encoding="utf-8") as f:
        json.dump(camCalibConfig, f, indent=2)

    print(f"Saved calibration to: {json_file}")
    ##

    print(f"[calibration] Finished calibration successfully.")

    return {
        "rms": float(rms),
        "camera_matrix": camera_matrix,
        "dist_coeffs": dist_coeffs,
        "image_size": image_size,
        "per_view_rmse": errors,
        "used_images": accepted_paths,
        "output_file": str(output_file),
        "rejected_paths": rejected_paths,
    }


def find_capture_directories(capture_root):
    root = Path(capture_root).expanduser()
    if not root.exists():
        raise FileNotFoundError(f"Capture directory does not exist: {root}")

    candidate_dirs = sorted(
        path for path in root.iterdir() if path.is_dir() and path.name.startswith("capture_")
    )
    valid_dirs = [
        path for path in candidate_dirs if any(path.glob("graycode_*"))
    ]
    if valid_dirs:
        return valid_dirs

    if any(root.glob("graycode_*")):
        return [root]

    return []


def loadCameraParam(json_file):
    with open(json_file, 'r') as f:
        param_data = json.load(f)
        P = param_data['camera']['P']
        d = param_data['camera']['distortion']
        return np.array(P).reshape([3, 3]), np.array(d)


def printNumpyWithIndent(tar, indentchar):
    print(indentchar + str(tar).replace('\n', '\n' + indentchar))


def calibrate(dirnames, gc_fname_lists, proj_shape, chess_shape, chess_block_size, gc_step, black_thr, white_thr, camP, camD):
    objps = np.zeros((chess_shape[0] * chess_shape[1], 3), np.float32)
    objps[:, :2] = chess_block_size * \
        np.mgrid[0:chess_shape[0], 0:chess_shape[1]].T.reshape(-1, 2)

    print('Calibrating ...')
    gc_height = int((proj_shape[0] - 1) / gc_step) + 1
    gc_width = int((proj_shape[1] - 1) / gc_step) + 1
    structured_light = getattr(cv2, "structured_light_GrayCodePattern", None)
    if structured_light is not None and hasattr(structured_light, "create"):
        graycode = structured_light.create(gc_width, gc_height)
    else:
        graycode = GrayCodePatternFallback(gc_width, gc_height)
    graycode.setBlackThreshold(black_thr)
    graycode.setWhiteThreshold(white_thr)

    cam_shape = cv2.imread(gc_fname_lists[0][0], cv2.IMREAD_GRAYSCALE).shape
    patch_size_half = int(np.ceil(cam_shape[1] / 180))
    print('  patch size :', patch_size_half * 2 + 1)

    cam_corners_list = []
    cam_objps_list = []
    cam_corners_list2 = []
    proj_objps_list = []
    proj_corners_list = []

    for dname, gc_filenames in zip(dirnames, gc_fname_lists):
        print('  checking \'' + str(dname) + '\'')
        if len(gc_filenames) != graycode.getNumberOfPatternImages() + 2:
            print('Error : invalid number of images in \'' + str(dname) + '\'')
            return None

        imgs = []
        for fname in gc_filenames:
            img = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
            if cam_shape != img.shape:
                print('Error : image size of \'' + fname + '\' is mismatch')
                return None
            imgs.append(img)
        black_img = imgs.pop()
        white_img = imgs.pop()

        res, cam_corners = cv2.findChessboardCorners(white_img, chess_shape)
        if not res:
            print('Error : chessboard was not found in \'' +
                  gc_filenames[-2] + '\'')
            return None
        cam_objps_list.append(objps)
        cam_corners_list.append(cam_corners)

        proj_objps = []
        proj_corners = []
        cam_corners2 = []
        for corner, objp in zip(cam_corners, objps):
            c_x = int(round(corner[0][0]))
            c_y = int(round(corner[0][1]))
            src_points = []
            dst_points = []
            for dx in range(-patch_size_half, patch_size_half + 1):
                for dy in range(-patch_size_half, patch_size_half + 1):
                    x = c_x + dx
                    y = c_y + dy
                    if int(white_img[y, x]) - int(black_img[y, x]) <= black_thr:
                        continue
                    err, proj_pix = graycode.getProjPixel(imgs, x, y)
                    if not err:
                        src_points.append((x, y))
                        dst_points.append(gc_step * np.array(proj_pix))
            if len(src_points) < patch_size_half ** 2:
                print(
                    '    Warning : corner', c_x, c_y,
                    'was skiped because decoded pixels were too few (check your images and threasholds)')
                continue
            h_mat, _ = cv2.findHomography(
                np.array(src_points), np.array(dst_points))
            point = h_mat @ np.array([corner[0][0],
                                     corner[0][1], 1]).transpose()
            point_pix = point[0:2] / point[2]
            proj_objps.append(objp)
            proj_corners.append([point_pix])
            cam_corners2.append(corner)
        if len(proj_corners) < 3:
            print('Error : too few corners were found in \'' +
                  str(dname) + '\' (less than 3)')
            return None
        proj_objps_list.append(np.float32(proj_objps))
        proj_corners_list.append(np.float32(proj_corners))
        cam_corners_list2.append(np.float32(cam_corners2))

    print('Initial solution of camera\'s intrinsic parameters')
    cam_rvecs = []
    cam_tvecs = []
    if camP is None:
        ret, cam_int, cam_dist, cam_rvecs, cam_tvecs = cv2.calibrateCamera(
            cam_objps_list, cam_corners_list, cam_shape, None, None, None, None)
        print('  RMS :', ret)
    else:
        for objp, corners in zip(cam_objps_list, cam_corners_list):
            ret, cam_rvec, cam_tvec = cv2.solvePnP(objp, corners, camP, camD)
            cam_rvecs.append(cam_rvec)
            cam_tvecs.append(cam_tvec)
            print('  RMS :', ret)
        cam_int = camP
        cam_dist = camD
    print('  Intrinsic parameters :')
    printNumpyWithIndent(cam_int, '    ')
    print('  Distortion parameters :')
    printNumpyWithIndent(cam_dist, '    ')
    print()

    print('Initial solution of projector\'s parameters')
    ret, proj_int, proj_dist, proj_rvecs, proj_tvecs = cv2.calibrateCamera(
        proj_objps_list, proj_corners_list, proj_shape, None, None, None, None)
    print('  RMS :', ret)
    print('  Intrinsic parameters :')
    printNumpyWithIndent(proj_int, '    ')
    print('  Distortion parameters :')
    printNumpyWithIndent(proj_dist, '    ')
    print()

    print('=== Result ===')
    ret, cam_int, cam_dist, proj_int, proj_dist, cam_proj_rmat, cam_proj_tvec, E, F = cv2.stereoCalibrate(
        proj_objps_list, cam_corners_list2, proj_corners_list, cam_int, cam_dist, proj_int, proj_dist, None)
    print('  RMS :', ret)
    print('  Camera intrinsic parameters :')
    printNumpyWithIndent(cam_int, '    ')
    print('  Camera distortion parameters :')
    printNumpyWithIndent(cam_dist, '    ')
    print('  Projector intrinsic parameters :')
    printNumpyWithIndent(proj_int, '    ')
    print('  Projector distortion parameters :')
    printNumpyWithIndent(proj_dist, '    ')
    print('  Rotation matrix / translation vector from camera to projector')
    print('  (they translate points from camera coord to projector coord) :')
    printNumpyWithIndent(cam_proj_rmat, '    ')
    printNumpyWithIndent(cam_proj_tvec, '    ')
    print()

    fs = cv2.FileStorage('results/result_projector_calibration.xml',
                         cv2.FILE_STORAGE_WRITE)
    fs.write('img_shape', cam_shape)
    fs.write('rms', ret)
    fs.write('cam_int', cam_int)
    fs.write('cam_dist', cam_dist)
    fs.write('proj_int', proj_int)
    fs.write('proj_dist', proj_dist)
    fs.write('rotation', cam_proj_rmat)
    fs.write('translation', cam_proj_tvec)
    fs.release()
    return {
        'rms': ret,
        'camera_intrinsic': cam_int,
        'camera_distortion': cam_dist,
        'projector_intrinsic': proj_int,
        'projector_distortion': proj_dist,
        'rotation': cam_proj_rmat,
        'translation': cam_proj_tvec,
        'output_file': 'result_projector_calibration.xml',
    }


def run_structured_light_calibration(capture_root, proj_shape, chess_shape, chess_block_size,
                                     gc_step, black_thr=40, white_thr=5, camera_json_path=''):
    capture_root = Path(capture_root).expanduser()
    dirnames = find_capture_directories(capture_root)
    if not dirnames:
        raise FileNotFoundError(
            f"No capture_* directories with graycode_* images were found in {capture_root}"
        )

    gc_fname_lists = []
    for dname in dirnames:
        gc_fnames = sorted(dname.glob('graycode_*'))
        if len(gc_fnames) == 0:
            continue
        gc_fname_lists.append([str(path) for path in gc_fnames])
        print(f" '{dname}' was found")

    if not gc_fname_lists:
        raise FileNotFoundError(
            f"No graycode images were found in the capture folders under {capture_root}"
        )

    camP = None
    cam_dist = None
    if camera_json_path and Path(camera_json_path).expanduser().suffix.lower() == '.json':
        camP, cam_dist = loadCameraParam(Path(camera_json_path).expanduser())
        print('JSON FILE FOUND, loading camera parameters')
        print(camP)
        print(cam_dist)

    return calibrate(
        dirnames,
        gc_fname_lists,
        proj_shape,
        chess_shape,
        chess_block_size,
        gc_step,
        black_thr,
        white_thr,
        camP,
        cam_dist,
    )


def format_color(value):
    if value is None:
        return "--"
    return f"{value:.4f}"


def enumerate_camera_indexes(limit=10):
    candidates = []

    try:
        base_dir = Path("/sys/class/video4linux")
        for device_dir in sorted(base_dir.glob("video*"), key=lambda p: p.name):
            match = re.search(r"(\d+)$", device_dir.name)
            if not match:
                continue
            idx = int(match.group(1))
            if idx < 0 or idx >= limit:
                continue
            device_path = (device_dir / "device")
            try:
                resolved = str(device_path.resolve())
            except OSError:
                continue
            if "virtual" in resolved:
                continue
            candidates.append(idx)
    except OSError:
        pass

    if not candidates:
        try:
            for device_path in sorted(Path("/dev").glob("video*"), key=lambda p: p.name):
                match = re.search(r"(\d+)$", device_path.name)
                if not match:
                    continue
                idx = int(match.group(1))
                if 0 <= idx < limit:
                    candidates.append(idx)
        except OSError:
            pass

    deduped = []
    seen = set()
    for idx in sorted(set(candidates)):
        try:
            sys_path = Path(f"/sys/class/video4linux/video{idx}/device")
            resolved = str(sys_path.resolve())
        except OSError:
            resolved = str(idx)
        if "virtual" in resolved:
            continue
        if resolved in seen:
            continue
        seen.add(resolved)
        deduped.append(idx)

    return deduped or list(range(limit))


def sony_request(method, params):
    payload = json.dumps({"method": method, "params": params,
                         "id": 1, "version": "1.0"}).encode("utf-8")
    request = urllib.request.Request(SONY_API_URL, data=payload, headers={
                                     "Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=5) as response:
        body = response.read()
    return json.loads(body.decode("utf-8", errors="replace"))


def sony_start_liveview():
    result = sony_request("startLiveview", [""])
    return result.get("result", [None])[0]


def build_capture_path(folder, camera_name, prefix="capture"):
    os.makedirs(folder, exist_ok=True)
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_",
                       str(camera_name)).strip("_") or "camera"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"{prefix}_{safe_name}_{timestamp}.jpg"
    return os.path.join(folder, filename)


def generate_graycode_pattern_images(output_dir, proj_height, proj_width, graycode_step=1):
    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    height = int(proj_height)
    width = int(proj_width)
    step = max(1, int(graycode_step))
    if height <= 0 or width <= 0:
        raise ValueError("Projector height and width must both be positive.")

    gc_height = int((height - 1) / step) + 1
    gc_width = int((width - 1) / step) + 1

    structured_light = getattr(cv2, "structured_light_GrayCodePattern", None)
    if structured_light is not None and hasattr(structured_light, "create"):
        graycode = structured_light.create(gc_width, gc_height)
        patterns = graycode.generate()[1]
    else:
        bit_count_x = max(1, int(np.ceil(np.log2(gc_width)))
                          ) if gc_width > 1 else 1
        bit_count_y = max(1, int(np.ceil(np.log2(gc_height)))
                          ) if gc_height > 1 else 1
        patterns = []

        for bit in range(bit_count_x):
            pattern = np.zeros((gc_height, gc_width), dtype=np.uint8)
            for y in range(gc_height):
                for x in range(gc_width):
                    code = x ^ (x >> 1)
                    bit_value = (code >> (bit_count_x - 1 - bit)) & 1
                    pattern[y, x] = 255 if bit_value else 0
            patterns.append(pattern)

        for bit in range(bit_count_y):
            pattern = np.zeros((gc_height, gc_width), dtype=np.uint8)
            for y in range(gc_height):
                for x in range(gc_width):
                    code = y ^ (y >> 1)
                    bit_value = (code >> (bit_count_y - 1 - bit)) & 1
                    pattern[y, x] = 255 if bit_value else 0
            patterns.append(pattern)

    expanded_patterns = []
    for pattern in patterns:
        expanded = np.zeros((height, width), dtype=np.uint8)
        for y in range(height):
            source_y = min(gc_height - 1, int(y / step))
            for x in range(width):
                source_x = min(gc_width - 1, int(x / step))
                expanded[y, x] = pattern[source_y, source_x]
        expanded_patterns.append(expanded)

    expanded_patterns.append(np.ones((height, width), dtype=np.uint8) * 255)
    expanded_patterns.append(np.zeros((height, width), dtype=np.uint8))

    generated_paths = []
    for index, pattern in enumerate(expanded_patterns):
        path = output_dir / f"pattern_{index:02d}.png"
        cv2.imwrite(str(path), pattern)
        generated_paths.append(str(path))

    return generated_paths


class GrayCodePatternFallback:
    def __init__(self, width, height):
        self.width = int(max(width, 1))
        self.height = int(max(height, 1))
        self.black_threshold = 40
        self.white_threshold = 5

    def setBlackThreshold(self, value):
        self.black_threshold = int(value)

    def setWhiteThreshold(self, value):
        self.white_threshold = int(value)

    def getNumberOfPatternImages(self):
        x_bits = max(1, int(np.ceil(np.log2(self.width)))
                     ) if self.width > 1 else 1
        y_bits = max(1, int(np.ceil(np.log2(self.height)))
                     ) if self.height > 1 else 1
        return x_bits + y_bits

    @staticmethod
    def _gray_to_binary(bits):
        if not bits:
            return 0
        decoded = [bits[0]]
        for bit in bits[1:]:
            decoded.append(decoded[-1] ^ int(bit))
        value = 0
        for bit in decoded:
            value = (value << 1) | int(bit)
        return value

    def getProjPixel(self, imgs, x, y):
        if not imgs:
            return False, (0, 0)

        x_bits = max(1, int(np.ceil(np.log2(self.width)))
                     ) if self.width > 1 else 1
        y_bits = max(1, int(np.ceil(np.log2(self.height)))
                     ) if self.height > 1 else 1

        x_values = []
        for image in imgs[:x_bits]:
            value = int(image[y, x])
            x_values.append(1 if value > max(1, self.white_threshold) else 0)

        y_values = []
        for image in imgs[x_bits:x_bits + y_bits]:
            value = int(image[y, x])
            y_values.append(1 if value > max(1, self.white_threshold) else 0)

        if len(x_values) != x_bits or len(y_values) != y_bits:
            return False, (0, 0)

        proj_x = self._gray_to_binary(x_values)
        proj_y = self._gray_to_binary(y_values)
        if proj_x >= self.width or proj_y >= self.height:
            return False, (float(proj_x), float(proj_y))
        return True, (float(proj_x), float(proj_y))


class CameraCaptureApp:
    @staticmethod
    def _calculate_sidebar_columns(available_width, column_count=4, min_column_width=180, gap=8):
        if available_width <= 0:
            available_width = min_column_width * \
                column_count + gap * (column_count - 1)

        usable_width = max(available_width - gap * (column_count - 1), 0)
        column_width = max(min_column_width, usable_width // column_count)
        total_width = column_width * column_count + gap * (column_count - 1)
        return {
            "column_width": column_width,
            "gap": gap,
            "total_width": total_width,
        }

    def _refresh_sidebar_layout(self):
        if not hasattr(self, "sidebar"):
            return

        available_width = max(600, self.root.winfo_width() - 420)
        layout = self._calculate_sidebar_columns(available_width)

        self.sidebar.configure(width=layout["total_width"])
        for index, pane in enumerate((
            self.basic_sidebar,
            self.advanced_sidebar,
            self.calibration_sidebar,
            self.projector_sidebar,
        )):
            pane.configure(width=layout["column_width"])
            self.sidebar.grid_columnconfigure(
                index, minsize=layout["column_width"], weight=1)
            pane.grid_configure(padx=(0, layout["gap"] if index < 3 else 0))

        for canvas in (
            self.basic_canvas,
            self.advanced_canvas,
            self.calibration_canvas,
            self.projector_canvas,
        ):
            canvas.configure(width=layout["column_width"])

    def __init__(self, root):
        self.root = root
        self.root.title("Camera & Projector Calibration App")
        self.root.geometry("1550x720")
        self.root.minsize(1200, 680)
        self.root.configure(bg="#0f1115")
        self._button_style = {
            "relief": "raised",
            "bd": 1,
            "highlightthickness": 0,
            "activebackground": "#4d535a",
            "activeforeground": "white",
            "cursor": "hand2",
            "borderwidth": 1,
        }
        self.camera_indexes = enumerate_camera_indexes(limit=8)
        self.camera_index = self.camera_indexes[0] if self.camera_indexes else 0
        self.zoom_factor = 1.0
        self.latest_frame = None
        self.photo_image = None
        self.cap = None
        self.after_id = None
        self._loop_generation = 0
        self.sony_stream = None
        self.sony_liveview_url = None
        self.sony_buffer = b""

        self.folder_var = tk.StringVar(
            value=os.path.expanduser("dataCollect/01_camPics"))
        self.graycode_pattern_dir_var = tk.StringVar(
            value=os.path.expanduser("dataCollect/graycode_pattern"))
        self.projector_pattern_dir_var = self.graycode_pattern_dir_var
        self.graycode_capture_dir_var = tk.StringVar(
            value=os.path.expanduser("dataCollect/capture_0"))
        # was "camera_param_path_var" once
        self.camera_json_dir_var = tk.StringVar(
            value=os.path.expanduser("results/result_camera_calibration.json"))
        self.projector_capture_dir_var = self.graycode_capture_dir_var
        self.projector_slideshow_speed_var = tk.StringVar(value="3.0")
        self.capture_interval_var = tk.StringVar(value="2.0")
        self.status_var = tk.StringVar(value="Ready")
        self.camera_names = ["Sony HTTP API"] + \
            [f"Camera {idx}" for idx in self.camera_indexes]
        self.projector_monitors = self.get_attached_monitors()
        self.projector_monitor_var = tk.StringVar(
            value=(list(self.projector_monitors.keys())[0] if self.projector_monitors else "Primary"))
        self.projector_slideshow_window = None
        self.projector_slideshow_label = None
        self.projector_slideshow_running = False
        self.projector_slideshow_images = []
        self.projector_slideshow_index = 0
        self.projector_slideshow_after_id = None
        self.camera_var = tk.StringVar(value=self.camera_names[0])
        self.camera_resolution_var = tk.StringVar(value="")

        self.interval_after_id = None
        self.interval_progress_after_id = None
        self.interval_active = False
        self.interval_start_time = None
        self.interval_duration = 0.0
        self.capture_lock = threading.Lock()
        self.osc_port = int(os.getenv("CAMERA_OSC_PORT", "10000"))
        self.osc_socket = None
        self.osc_server_thread = None
        self.osc_server_running = False

        self.exposure_auto_var = tk.BooleanVar(value=True)
        self.focus_auto_var = tk.BooleanVar(value=True)
        self.white_balance_auto_var = tk.BooleanVar(value=True)

        self.exposure_value_var = tk.StringVar(value="0")
        self.focus_value_var = tk.StringVar(value="0")
        self.white_balance_value_var = tk.StringVar(value="4000")

        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        self.calibration_columns_var = tk.StringVar(value=str(BOARD_SIZE[0]))
        self.calibration_rows_var = tk.StringVar(value=str(BOARD_SIZE[1]))
        self.calibration_square_size_var = tk.StringVar(value=str(SQUARE_SIZE))

        self.calibration_width_var = tk.StringVar(value="1920")
        self.calibration_height_var = tk.StringVar(value="1080")

        self.foto_width_var = tk.StringVar(value="1616")
        self.foto_height_var = tk.StringVar(value="1080")

        self.graycode_step_var = tk.StringVar(value="1")
        self.black_threshold_var = tk.StringVar(value="40")
        self.white_threshold_var = tk.StringVar(value="5")
        # self.camera_param_path_var = tk.StringVar(value="")

        self.unity_scale_var = tk.StringVar(value="0.001")

        main = tk.Frame(root, bg="#0f1115")
        main.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)

        main.grid_columnconfigure(0, weight=3)
        main.grid_columnconfigure(1, weight=1)
        main.grid_columnconfigure(2, weight=1)
        main.grid_columnconfigure(3, weight=1)
        main.grid_rowconfigure(0, weight=1)

        preview_panel = tk.Frame(main, bg="#1a1a1a", bd=2, relief=tk.SUNKEN)
        preview_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        preview_panel.grid_rowconfigure(0, weight=3)
        preview_panel.grid_rowconfigure(1, weight=1)
        preview_panel.grid_columnconfigure(0, weight=1)

        self.preview_label = tk.Label(preview_panel, bg="black")
        self.preview_label.grid(row=0, column=0, sticky="nsew")

        self.calibration_output = tk.Text(
            preview_panel,
            height=5,
            bg="#101418",
            fg="#dfe7f3",
            insertbackground="white",
            wrap=tk.WORD,
            state="normal",
        )
        self.calibration_output.bind("<Key>", lambda event: "break")
        self._configure_calibration_log_tags()
        self.calibration_output.grid(
            row=1, column=0, sticky="nsew", pady=(6, 0))

        sidebar_layout = self._calculate_sidebar_columns(1200)
        self.sidebar = tk.Frame(
            main, bg="#1a1a1a", width=sidebar_layout["total_width"])
        self.sidebar.grid(row=0, column=1, sticky="nsew")
        self.sidebar.grid_propagate(False)
        self.sidebar.grid_rowconfigure(0, weight=1)
        self.sidebar.grid_columnconfigure(0, weight=1)
        self.sidebar.grid_columnconfigure(1, weight=1)
        self.sidebar.grid_columnconfigure(2, weight=1)
        self.sidebar.grid_columnconfigure(3, weight=1)

        self.basic_sidebar = tk.Frame(
            self.sidebar, bg="#1a1a1a", width=sidebar_layout["column_width"])
        self.basic_sidebar.grid(
            row=0, column=0, sticky="nsew", padx=(0, sidebar_layout["gap"]))
        self.basic_sidebar.grid_propagate(False)
        self.basic_sidebar.grid_rowconfigure(0, weight=1)
        self.basic_sidebar.grid_columnconfigure(0, weight=1)

        self.advanced_sidebar = tk.Frame(
            self.sidebar, bg="#1a1a1a", width=sidebar_layout["column_width"])
        self.advanced_sidebar.grid(
            row=0, column=1, sticky="nsew", padx=(0, sidebar_layout["gap"]))
        self.advanced_sidebar.grid_propagate(False)
        self.advanced_sidebar.grid_rowconfigure(0, weight=1)
        self.advanced_sidebar.grid_columnconfigure(0, weight=1)

        self.calibration_sidebar = tk.Frame(
            self.sidebar, bg="#1a1a1a", width=sidebar_layout["column_width"])
        self.calibration_sidebar.grid(
            row=0, column=2, sticky="nsew", padx=(0, sidebar_layout["gap"]))
        self.calibration_sidebar.grid_propagate(False)
        self.calibration_sidebar.grid_rowconfigure(0, weight=1)
        self.calibration_sidebar.grid_columnconfigure(0, weight=1)

        self.projector_sidebar = tk.Frame(
            self.sidebar, bg="#1a1a1a", width=sidebar_layout["column_width"])
        self.projector_sidebar.grid(row=0, column=3, sticky="nsew")
        self.projector_sidebar.grid_propagate(False)
        self.projector_sidebar.grid_rowconfigure(0, weight=1)
        self.projector_sidebar.grid_columnconfigure(0, weight=1)

        self.basic_canvas = tk.Canvas(
            self.basic_sidebar, bg="#1a1a1a", highlightthickness=0, width=sidebar_layout["column_width"])
        # self.basic_sidebar, bg="#1a1a1a", highlightthickness=0)
        self.basic_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.basic_scrollbar = ttk.Scrollbar(
            self.basic_sidebar, orient=tk.VERTICAL, command=self.basic_canvas.yview)
        self.basic_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.basic_canvas.configure(yscrollcommand=self.basic_scrollbar.set)

        self.advanced_canvas = tk.Canvas(
            self.advanced_sidebar, bg="#1a1a1a", highlightthickness=0, width=sidebar_layout["column_width"])
        # self.advanced_sidebar, bg="#1a1a1a", highlightthickness=0)
        self.advanced_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.advanced_scrollbar = ttk.Scrollbar(
            self.advanced_sidebar, orient=tk.VERTICAL, command=self.advanced_canvas.yview)
        self.advanced_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.advanced_canvas.configure(
            yscrollcommand=self.advanced_scrollbar.set)

        self.calibration_canvas = tk.Canvas(
            self.calibration_sidebar, bg="#1a1a1a", highlightthickness=0, width=sidebar_layout["column_width"])
        # self.calibration_sidebar, bg="#1a1a1a", highlightthickness=0)
        self.calibration_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.calibration_scrollbar = ttk.Scrollbar(
            self.calibration_sidebar, orient=tk.VERTICAL, command=self.calibration_canvas.yview)
        self.calibration_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.calibration_canvas.configure(
            yscrollcommand=self.calibration_scrollbar.set)

        self.projector_canvas = tk.Canvas(
            self.projector_sidebar, bg="#1a1a1a", highlightthickness=0, width=sidebar_layout["column_width"])
        self.projector_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.projector_scrollbar = ttk.Scrollbar(
            self.projector_sidebar, orient=tk.VERTICAL, command=self.projector_canvas.yview)
        self.projector_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.projector_canvas.configure(
            yscrollcommand=self.projector_scrollbar.set)

        self.root.bind("<Configure>", self._refresh_sidebar_layout)
        self._refresh_sidebar_layout()

        basic_area = tk.Frame(self.basic_canvas,
                              bg="#1a1a1a", padx=12, pady=12)
        self.basic_canvas.create_window(
            (0, 0), window=basic_area, anchor="nw", width=180)
        basic_area.bind("<Configure>", lambda event: self.basic_canvas.configure(
            scrollregion=self.basic_canvas.bbox("all")))

        advanced_area = tk.Frame(self.advanced_canvas,
                                 bg="#1a1a1a", padx=12, pady=12)
        self.advanced_canvas.create_window(
            (0, 0), window=advanced_area, anchor="nw", width=180)
        advanced_area.bind("<Configure>", lambda event: self.advanced_canvas.configure(
            scrollregion=self.advanced_canvas.bbox("all")))

        camera_settings_label = tk.Label(
            advanced_area,
            text="Camera Settings",
            font=("Helvetica", 12, "bold"),
            bg="#1a1a1a",
            fg="yellow",
            anchor="w",
        )
        camera_settings_label.pack(fill=tk.X, pady=(0, 8))

        calibration_area = tk.Frame(self.calibration_canvas,
                                    bg="#1a1a1a", padx=12, pady=12)
        self.calibration_canvas.create_window(
            (0, 0), window=calibration_area, anchor="nw", width=180)
        calibration_area.bind("<Configure>", lambda event: self.calibration_canvas.configure(
            scrollregion=self.calibration_canvas.bbox("all")))

        projector_area = tk.Frame(self.projector_canvas,
                                  bg="#1a1a1a", padx=12, pady=12)
        self.projector_canvas.create_window(
            (0, 0), window=projector_area, anchor="nw", width=180)
        projector_area.bind("<Configure>", lambda event: self.projector_canvas.configure(
            scrollregion=self.projector_canvas.bbox("all")))

        self.root.bind_all("<MouseWheel>", self.on_sidebar_scroll)
        self.root.bind_all("<Button-4>", self.on_sidebar_scroll)
        self.root.bind_all("<Button-5>", self.on_sidebar_scroll)

        camera_label = tk.Label(
            basic_area, text="Camera", font=("Helvetica", 12, "bold"), bg="#1a1a1a", fg="yellow", anchor="w")
        camera_label.pack(fill=tk.X, pady=(0, 4))

        self.camera_menu = ttk.Combobox(
            basic_area, textvariable=self.camera_var, values=self.camera_names, state="readonly")
        self.camera_menu.pack(fill=tk.X, pady=(0, 10))
        self.camera_menu.bind("<<ComboboxSelected>>", self.on_camera_selected)

        resolution_label = tk.Label(
            basic_area, text="Resolution", bg="#1a1a1a", fg="white", anchor="w")
        resolution_label.pack(fill=tk.X, pady=(0, 4))

        self.camera_resolution_menu = ttk.Combobox(
            basic_area,
            textvariable=self.camera_resolution_var,
            values=[],
            state="disabled",
        )
        self.camera_resolution_menu.pack(fill=tk.X, pady=(0, 10))
        self.camera_resolution_menu.bind(
            "<<ComboboxSelected>>", self.on_camera_resolution_selected)
        self.camera_resolution_var.set("")
        self.camera_resolution_menu.configure(values=[], state="disabled")

        self.zoom_in_button = tk.Button(
            basic_area, text="Zoom in", font=("Helvetica", 10), command=self.zoom_in, bg="#3a3d42", fg="white", relief="flat", padx=10, pady=8)
        self.zoom_in_button.pack(fill=tk.X, pady=(0, 8))

        self.zoom_out_button = tk.Button(
            basic_area, text="Zoom out", font=("Helvetica", 10), command=self.zoom_out, bg="#3a3d42", fg="white", relief="flat", padx=10, pady=8)
        self.zoom_out_button.pack(fill=tk.X, pady=(0, 8))

        self.status_label = tk.Label(basic_area, textvariable=self.status_var,
                                     bg="#1a1a1a", fg="#dfe9ff", anchor="w", justify="left", wraplength=260)
        self.status_label.pack(fill=tk.X, pady=(0, 12))

        checker_label = tk.Label(
            basic_area,
            text="Checkerboard\nSeries Capture",
            font=("Helvetica", 12, "bold"),
            bg="#1a1a1a",
            fg="yellow",
            anchor="w",
        )
        checker_label.pack(fill=tk.X, pady=(0, 8))

        folder_label = tk.Label(
            basic_area, text="Save Folder", bg="#1a1a1a", fg="white", anchor="w")
        folder_label.pack(fill=tk.X, pady=(18, 4))

        folder_entry = tk.Entry(basic_area, textvariable=self.folder_var,
                                bg="#202429", fg="white", insertbackground="white")
        folder_entry.pack(fill=tk.X, pady=(0, 8))

        self.browse_button = tk.Button(basic_area, text="Browse", command=self.browse_folder,
                                       bg="#3a3d42", fg="white", relief="flat", padx=10, pady=6)
        self.browse_button.pack(fill=tk.X, pady=(0, 12))

        self.interval_progress = ttk.Progressbar(
            basic_area,
            orient=tk.HORIZONTAL,
            length=220,
            mode="determinate",
            maximum=100,
            value=0,
            style="IntervalCountdown.Horizontal.TProgressbar",
        )
        self.interval_status_var = tk.StringVar(value="")
        self.interval_label = tk.Label(
            basic_area,
            textvariable=self.interval_status_var,
            bg="#1a1a1a",
            fg="#ffe9a8",
            anchor="w",
        )

        self.interval_progress.pack(fill=tk.X, pady=(0, 4))
        self.interval_label.pack(fill=tk.X, pady=(0, 4))

        interval_label = tk.Label(
            basic_area, text="Interval (s)", bg="#1a1a1a", fg="white", anchor="w")
        interval_label.pack(fill=tk.X, pady=(16, 4))

        self.interval_entry = tk.Entry(
            basic_area,
            textvariable=self.capture_interval_var,
            bg="#202429",
            fg="white",
            insertbackground="white",
        )
        self.interval_entry.pack(fill=tk.X, pady=(0, 8))
        self.interval_entry.bind(
            "<Return>", lambda _event: self.validate_interval())
        self.interval_entry.bind(
            "<KP_Enter>", lambda _event: self.validate_interval())

        self.interval_button = tk.Button(
            basic_area,
            text="Interval Shots",
            font=("Helvetica", 10),
            command=self.toggle_interval_capture,
            bg="#ffa55b",
            fg="white",
            relief="flat",
            padx=10,
            pady=8,
        )
        self.interval_button.pack(fill=tk.X, pady=(0, 8))

        self.take_button = tk.Button(basic_area, text="Take Picture", command=self.take_picture,
                                     bg="#5b9dff", fg="white", font=("Helvetica", 10), relief="flat", padx=10, pady=8)
        self.take_button.pack(fill=tk.X, pady=(0, 8))

        self.gallery_expanded = True
        self.gallery_toggle_button = tk.Button(
            basic_area,
            text="Gallery (0) ▾",
            command=self.toggle_gallery,
            bg="#3a3d42",
            fg="white",
            relief="flat",
            padx=10,
            pady=6,
        )
        self.gallery_toggle_button.pack(fill=tk.X, pady=(16, 4))

        self.gallery_content = tk.Frame(basic_area, bg="#1a1a1a")
        self.gallery_content.pack(fill=tk.BOTH, expand=True)
        self.gallery_frame = tk.Frame(self.gallery_content, bg="#1a1a1a")
        self.gallery_frame.pack(fill=tk.BOTH, expand=True)
        self.refresh_gallery_from_folder()

        self.osc_expanded = False
        self.osc_toggle_button = tk.Button(
            basic_area,
            text="OSC ▸",
            command=self.toggle_osc,
            bg="#3a3d42",
            fg="white",
            relief="flat",
            padx=10,
            pady=6,
        )
        self.osc_toggle_button.pack(fill=tk.X, pady=(12, 6))

        self.osc_content = tk.Frame(basic_area, bg="#1a1a1a")

        self.osc_bind_ip_var = tk.StringVar(value="127.0.0.1")
        self.osc_port_var = tk.StringVar(value=str(self.osc_port))
        self.osc_take_picture_cmd_var = tk.StringVar(
            value="/camera/take_picture")
        self.osc_interval_start_cmd_var = tk.StringVar(
            value="/camera/interval/start")
        self.osc_interval_stop_cmd_var = tk.StringVar(
            value="/camera/interval/stop")
        self.osc_interval_time_cmd_var = tk.StringVar(
            value="/camera/interval/time")
        self.osc_projector_slideshow_start_cmd_var = tk.StringVar(
            value="/projector/slideshow/start")
        self.osc_projector_slideshow_stop_cmd_var = tk.StringVar(
            value="/projector/slideshow/stop")

        osc_ip_label = tk.Label(
            self.osc_content, text="OSC IP", bg="#1a1a1a", fg="white", anchor="w")
        osc_ip_label.pack(fill=tk.X, pady=(0, 4))
        self.osc_ip_entry = tk.Entry(
            self.osc_content,
            textvariable=self.osc_bind_ip_var,
            bg="#202429",
            fg="white",
            insertbackground="white",
        )
        self.osc_ip_entry.pack(fill=tk.X, pady=(0, 8))

        osc_port_label = tk.Label(
            self.osc_content, text="OSC Port", bg="#1a1a1a", fg="white", anchor="w")
        osc_port_label.pack(fill=tk.X, pady=(0, 4))
        self.osc_port_entry = tk.Entry(
            self.osc_content,
            textvariable=self.osc_port_var,
            bg="#202429",
            fg="white",
            insertbackground="white",
        )
        self.osc_port_entry.pack(fill=tk.X, pady=(0, 8))

        osc_commands = [
            ("Take picture", self.osc_take_picture_cmd_var),
            ("Interval start", self.osc_interval_start_cmd_var),
            ("Interval stop", self.osc_interval_stop_cmd_var),
            ("Interval time", self.osc_interval_time_cmd_var),
            ("Projector slideshow start", self.osc_projector_slideshow_start_cmd_var),
            ("Projector slideshow stop", self.osc_projector_slideshow_stop_cmd_var),
        ]
        for label_text, value_var in osc_commands:
            osc_cmd_label = tk.Label(
                self.osc_content, text=label_text, bg="#1a1a1a", fg="white", anchor="w")
            osc_cmd_label.pack(fill=tk.X, pady=(8, 4))
            entry = tk.Entry(
                self.osc_content,
                textvariable=value_var,
                bg="#202429",
                fg="white",
                insertbackground="white",
            )
            entry.pack(fill=tk.X, pady=(0, 8))

        self.osc_apply_button = tk.Button(
            self.osc_content,
            text="Apply OSC",
            command=self.apply_osc_settings,
            bg="#3a3d42",
            fg="white",
            relief="flat",
            padx=10,
            pady=6,
        )
        self.osc_apply_button.pack(fill=tk.X, pady=(0, 12))

        self.cv2_controls_frame = tk.Frame(advanced_area, bg="#1a1a1a")
        self.cv2_controls_frame.pack(fill=tk.X, pady=(0, 8))
        self.control_bg_cycle = ["#3b3a1a", "#1d2a3a", "#3b3a1a", "#1d2a3a"]

        self.exposure_frame = tk.Frame(
            self.cv2_controls_frame, bg=self.control_bg_cycle[0], padx=10, pady=10)
        self.exposure_frame.pack(fill=tk.X, pady=(0, 8), ipady=4)
        self.exposure_auto_btn = tk.Button(self.exposure_frame, text="Auto exposure: ON",
                                           command=self.toggle_exposure_auto, bg="#3a3d42", fg="white", relief="flat")
        self.exposure_auto_btn.pack(fill=tk.X, pady=(0, 4))
        self.exposure_slider = tk.Scale(self.exposure_frame, from_=0, to=100, orient=tk.HORIZONTAL,
                                        bg=self.control_bg_cycle[0], fg="white", highlightthickness=0, command=self.on_exposure_slider_change)
        self.exposure_slider.pack(fill=tk.X, pady=(0, 4))
        self.exposure_entry = tk.Entry(self.exposure_frame, textvariable=self.exposure_value_var,
                                       width=8, bg="#202429", fg="white", insertbackground="white")
        self.exposure_entry.pack(fill=tk.X)
        self.exposure_entry.bind(
            "<Return>", lambda _event: self.sync_entry_value("exposure"))
        self.exposure_entry.bind(
            "<KP_Enter>", lambda _event: self.sync_entry_value("exposure"))
        self.exposure_entry.bind(
            "<FocusOut>", lambda _event: self.sync_entry_value("exposure"))

        self.focus_frame = tk.Frame(
            self.cv2_controls_frame, bg=self.control_bg_cycle[1], padx=10, pady=10)
        self.focus_frame.pack(fill=tk.X, pady=(0, 8), ipady=4)
        self.focus_auto_btn = tk.Button(self.focus_frame, text="Auto focus: ON",
                                        command=self.toggle_focus_auto, bg="#3a3d42", fg="white", relief="flat")
        self.focus_auto_btn.pack(fill=tk.X, pady=(0, 4))
        self.focus_slider = tk.Scale(self.focus_frame, from_=0, to=100, orient=tk.HORIZONTAL,
                                     bg=self.control_bg_cycle[1], fg="white", highlightthickness=0, command=self.on_focus_slider_change)
        self.focus_slider.pack(fill=tk.X, pady=(0, 4))
        self.focus_entry = tk.Entry(self.focus_frame, textvariable=self.focus_value_var,
                                    width=8, bg="#202429", fg="white", insertbackground="white")
        self.focus_entry.pack(fill=tk.X)
        self.focus_entry.bind(
            "<Return>", lambda _event: self.sync_entry_value("focus"))
        self.focus_entry.bind(
            "<KP_Enter>", lambda _event: self.sync_entry_value("focus"))
        self.focus_entry.bind(
            "<FocusOut>", lambda _event: self.sync_entry_value("focus"))

        self.white_balance_frame = tk.Frame(
            self.cv2_controls_frame, bg=self.control_bg_cycle[0], padx=10, pady=10)
        self.white_balance_frame.pack(fill=tk.X, pady=(0, 8), ipady=4)
        self.white_balance_auto_btn = tk.Button(self.white_balance_frame, text="Auto white balance: ON",
                                                command=self.toggle_white_balance_auto, bg="#3a3d42", fg="white", relief="flat")
        self.white_balance_auto_btn.pack(fill=tk.X, pady=(0, 4))
        self.white_balance_slider = tk.Scale(self.white_balance_frame, from_=2000, to=15000, orient=tk.HORIZONTAL,
                                             bg=self.control_bg_cycle[0], fg="white", highlightthickness=0, command=self.on_white_balance_slider_change)
        self.white_balance_slider.pack(fill=tk.X, pady=(0, 4))
        self.white_balance_entry = tk.Entry(
            self.white_balance_frame, textvariable=self.white_balance_value_var, width=8, bg="#202429", fg="white", insertbackground="white")
        self.white_balance_entry.pack(fill=tk.X)
        self.white_balance_entry.bind(
            "<Return>", lambda _event: self.sync_entry_value("white_balance"))
        self.white_balance_entry.bind(
            "<KP_Enter>", lambda _event: self.sync_entry_value("white_balance"))
        self.white_balance_entry.bind(
            "<FocusOut>", lambda _event: self.sync_entry_value("white_balance"))

        self.cv2_extra_controls = {}
        for idx, (name, label, auto_prop, manual_prop, min_value, max_value, default_value) in enumerate([
            ("brightness", "Brightness", None, getattr(
                cv2, "CAP_PROP_BRIGHTNESS", None), -100, 100, 0),
            ("contrast", "Contrast", None, getattr(
                cv2, "CAP_PROP_CONTRAST", None), 0, 100, 50),
            ("saturation", "Saturation", None, getattr(
                cv2, "CAP_PROP_SATURATION", None), 0, 100, 50),
            ("hue", "Hue", None, getattr(cv2, "CAP_PROP_HUE", None), -180, 180, 0),
            ("gain", "Gain", getattr(cv2, "CAP_PROP_AUTO_GAIN", None) if hasattr(
                cv2, "CAP_PROP_AUTO_GAIN") else None, getattr(cv2, "CAP_PROP_GAIN", None), 0, 100, 0),
            ("gamma", "Gamma", None, getattr(
                cv2, "CAP_PROP_GAMMA", None), 0, 100, 50),
            ("sharpness", "Sharpness", None, getattr(
                cv2, "CAP_PROP_SHARPNESS", None), 0, 100, 0),
        ]):
            if manual_prop is None:
                continue
            frame_bg = self.control_bg_cycle[idx % len(self.control_bg_cycle)]
            frame = tk.Frame(self.cv2_controls_frame,
                             bg=frame_bg, padx=10, pady=10)
            frame.pack(fill=tk.X, pady=(0, 8), ipady=4)

            value_var = tk.StringVar(value=str(default_value))
            auto_var = tk.BooleanVar(
                value=True) if auto_prop is not None else None

            if auto_prop is not None:
                auto_btn = tk.Button(frame, text=f"Auto {label}: ON", command=lambda key=name: self.toggle_cv2_auto_control(
                    key), bg="#3a3d42", fg="white", relief="flat")
                auto_btn.pack(fill=tk.X, pady=(0, 4))
                setattr(self, f"{name}_auto_btn", auto_btn)
                self.cv2_extra_controls[name] = {
                    "frame": frame,
                    "label": label,
                    "auto_prop": auto_prop,
                    "manual_prop": manual_prop,
                    "auto_var": auto_var,
                    "value_var": value_var,
                    "slider": None,
                    "entry": None,
                    "default_value": default_value,
                    "min_value": min_value,
                    "max_value": max_value,
                }
            else:
                tk.Label(frame, text=f"{label}", bg="#1a1a1a", fg="white", anchor="w").pack(
                    fill=tk.X, pady=(0, 4))
                self.cv2_extra_controls[name] = {
                    "frame": frame,
                    "label": label,
                    "auto_prop": None,
                    "manual_prop": manual_prop,
                    "auto_var": None,
                    "value_var": value_var,
                    "slider": None,
                    "entry": None,
                    "default_value": default_value,
                    "min_value": min_value,
                    "max_value": max_value,
                }

            slider = tk.Scale(frame, from_=min_value, to=max_value, orient=tk.HORIZONTAL, bg=frame_bg, fg="white",
                              highlightthickness=0, command=lambda value, key=name: self.on_cv2_manual_value_changed(key, value))
            slider.pack(fill=tk.X, pady=(0, 4))
            self.cv2_extra_controls[name]["slider"] = slider

            entry = tk.Entry(frame, textvariable=value_var, width=8,
                             bg="#202429", fg="white", insertbackground="white")
            entry.pack(fill=tk.X)
            entry.bind("<Return>", lambda _event,
                       key=name: self.sync_cv2_entry_value(key))
            entry.bind("<KP_Enter>", lambda _event,
                       key=name: self.sync_cv2_entry_value(key))
            entry.bind("<FocusOut>", lambda _event,
                       key=name: self.sync_cv2_entry_value(key))
            self.cv2_extra_controls[name]["entry"] = entry

        self.cv2_controls_frame.pack_forget()

        calibration_label = tk.Label(
            calibration_area, text="Camera Calibration", font=("Helvetica", 12, "bold"), bg="#1a1a1a", fg="yellow", anchor="w")
        calibration_label.pack(fill=tk.X, pady=(0, 8))

        width_label = tk.Label(
            calibration_area, text="Foto Pixel Width (px)", bg="#1a1a1a", fg="white", anchor="w")
        width_label.pack(fill=tk.X, pady=(8, 4))
        self.foto_width_entry = tk.Entry(
            calibration_area, textvariable=self.foto_width_var,
            bg="#202429", fg="white", insertbackground="white"
        )
        self.foto_width_entry.pack(fill=tk.X, pady=(0, 8))

        height_label = tk.Label(
            calibration_area, text="Foto Pixel Height (px)", bg="#1a1a1a", fg="white", anchor="w")
        height_label.pack(fill=tk.X, pady=(0, 4))
        self.foto_height_entry = tk.Entry(
            calibration_area, textvariable=self.foto_height_var,
            bg="#202429", fg="white", insertbackground="white"
        )
        self.foto_height_entry.pack(fill=tk.X, pady=(0, 8))

        columns_label = tk.Label(
            calibration_area, text="Checker Board Columns", bg="#1a1a1a", fg="white", anchor="w")
        columns_label.pack(fill=tk.X, pady=(0, 4))
        self.calibration_columns_entry = tk.Entry(
            calibration_area, textvariable=self.calibration_columns_var,
            bg="#202429", fg="white", insertbackground="white"
        )
        self.calibration_columns_entry.pack(fill=tk.X, pady=(0, 8))

        rows_label = tk.Label(
            calibration_area, text="Checker Board Rows", bg="#1a1a1a", fg="white", anchor="w")
        rows_label.pack(fill=tk.X, pady=(0, 4))
        self.calibration_rows_entry = tk.Entry(
            calibration_area, textvariable=self.calibration_rows_var,
            bg="#202429", fg="white", insertbackground="white"
        )
        self.calibration_rows_entry.pack(fill=tk.X, pady=(0, 8))

        square_label = tk.Label(
            calibration_area, text="Single Square Size", bg="#1a1a1a", fg="white", anchor="w")
        square_label.pack(fill=tk.X, pady=(0, 4))
        self.calibration_square_size_entry = tk.Entry(
            calibration_area, textvariable=self.calibration_square_size_var,
            bg="#202429", fg="white", insertbackground="white"
        )
        self.calibration_square_size_entry.pack(fill=tk.X, pady=(0, 8))

        self.calibration_start_button = tk.Button(
            calibration_area,
            text="Run\nCamera Calibration",
            font=("Helvetica", 10),
            command=self.start_calibration_from_sidebar,
            bg="#ffa55b",
            fg="white",
            relief="flat",
            padx=10,
            pady=8,
        )
        self.calibration_start_button.pack(fill=tk.X, pady=(0, 8))

        projCalibration_label = tk.Label(
            projector_area, text="Projector Calibration", font=("Helvetica", 12, "bold"), bg="#1a1a1a", fg="yellow", anchor="w")
        projCalibration_label.pack(fill=tk.X, pady=(0, 8))

        width_label = tk.Label(
            projector_area, text="Projector Pixel Width (px)", bg="#1a1a1a", fg="white", anchor="w")
        width_label.pack(fill=tk.X, pady=(8, 4))
        self.calibration_width_entry = tk.Entry(
            projector_area, textvariable=self.calibration_width_var,
            bg="#202429", fg="white", insertbackground="white"
        )
        self.calibration_width_entry.pack(fill=tk.X, pady=(0, 8))

        height_label = tk.Label(
            projector_area, text="Projector Pixel Height (px)", bg="#1a1a1a", fg="white", anchor="w")
        height_label.pack(fill=tk.X, pady=(0, 4))
        self.calibration_height_entry = tk.Entry(
            projector_area, textvariable=self.calibration_height_var,
            bg="#202429", fg="white", insertbackground="white"
        )
        self.calibration_height_entry.pack(fill=tk.X, pady=(0, 8))

        graycode_step_label = tk.Label(
            projector_area, text="Graycode Step", bg="#1a1a1a", fg="white", anchor="w")
        graycode_step_label.pack(fill=tk.X, pady=(0, 4))
        self.graycode_step_entry = tk.Entry(
            projector_area, textvariable=self.graycode_step_var,
            bg="#202429", fg="white", insertbackground="white"
        )
        self.graycode_step_entry.pack(fill=tk.X, pady=(0, 8))

        pattern_folder_label = tk.Label(
            projector_area, text="Graycode Pattern Folder", bg="#1a1a1a", fg="white", anchor="w")
        pattern_folder_label.pack(fill=tk.X, pady=(0, 4))
        self.pattern_folder_entry = tk.Entry(
            projector_area, textvariable=self.graycode_pattern_dir_var,
            bg="#202429", fg="white", insertbackground="white"
        )
        self.pattern_folder_entry.pack(fill=tk.X, pady=(0, 8))
        self.pattern_folder_browse_button = tk.Button(
            projector_area, text="Browse Folder", command=self.browse_graycode_pattern_folder,
            bg="#3a3d42", fg="white", relief="flat", padx=10, pady=6)
        self.pattern_folder_browse_button.pack(fill=tk.X, pady=(0, 8))
        self.generate_pattern_button = tk.Button(
            projector_area,
            text="Generate\nGraycode Pattern",
            font=("Helvetica", 10),
            command=self.generate_graycode_pattern_from_sidebar,
            bg="#5aa9ff",
            fg="white",
            relief="flat",
            padx=10,
            pady=8,
        )
        self.generate_pattern_button.pack(fill=tk.X, pady=(0, 12))

        black_threshold_label = tk.Label(
            projector_area, text="Black Threshold", bg="#1a1a1a", fg="white", anchor="w")
        black_threshold_label.pack(fill=tk.X, pady=(0, 4))
        self.black_threshold_entry = tk.Entry(
            projector_area, textvariable=self.black_threshold_var,
            bg="#202429", fg="white", insertbackground="white"
        )
        self.black_threshold_entry.pack(fill=tk.X, pady=(0, 8))

        white_threshold_label = tk.Label(
            projector_area, text="White Threshold", bg="#1a1a1a", fg="white", anchor="w")
        white_threshold_label.pack(fill=tk.X, pady=(0, 4))
        self.white_threshold_entry = tk.Entry(
            projector_area, textvariable=self.white_threshold_var,
            bg="#202429", fg="white", insertbackground="white"
        )
        self.white_threshold_entry.pack(fill=tk.X, pady=(0, 8))

        camera_param_label = tk.Label(
            projector_area, text="Camera Parameter JSON", bg="#1a1a1a", fg="white", anchor="w")
        camera_param_label.pack(fill=tk.X, pady=(0, 4))
        self.camera_param_entry = tk.Entry(
            projector_area, textvariable=self.camera_json_dir_var,
            bg="#202429", fg="white", insertbackground="white"
        )
        self.camera_param_entry.pack(fill=tk.X, pady=(0, 8))
        self.camera_param_browse_button = tk.Button(
            projector_area, text="Browse JSON", command=self.browse_camera_parameter_json,
            bg="#3a3d42", fg="white", relief="flat", padx=10, pady=6)
        self.camera_param_browse_button.pack(fill=tk.X, pady=(0, 12))

        self.structured_light_calibration_button = tk.Button(
            projector_area,
            text="Run\nProjector Calibration",
            font=("Helvetica", 10),
            command=self.start_projector_calibration_from_sidebar,
            bg="#5aa9ff",
            fg="white",
            relief="flat",
            padx=10,
            pady=8,
        )
        self.structured_light_calibration_button.pack(fill=tk.X, pady=(0, 12))

        self.unity_scale_label = tk.Label(
            projector_area, text="Unity Scale set to milimeter", bg="#1a1a1a", fg="white", anchor="w")
        self.unity_scale_label.pack(fill=tk.X, pady=(0, 4))

        # self.unity_scale_entry = tk.Entry(
        #     projector_area, textvariable=self.unity_scale_var,
        #     bg="#202429", fg="white", insertbackground="white"
        # )
        # self.unity_scale_entry.pack(fill=tk.X, pady=(0, 8))

        def update_unityScale_value(value):
            value = float(value)
            if value < 0.009:
                self.unity_scale_slider.set(0.001)
                self.unity_scale_label.config(
                    text="Unity Scale set to milimeter")
            elif value < 0.05:
                self.unity_scale_slider.set(0.01)
                self.unity_scale_label.config(
                    text="Unity Scale set to centimeter")
            else:
                self.unity_scale_slider.set(0.1)
                self.unity_scale_label.config(text="Unity Scale set to meter")

        self.unity_scale_slider = tk.Scale(projector_area, from_=0, to=0.1, resolution=0.001,
                                           orient='horizontal', command=update_unityScale_value)
        self.unity_scale_slider.pack(fill=tk.X, pady=(0, 8))
        self.unity_scale_slider.set(0.001)

        self.export_unity_button = tk.Button(
            projector_area,
            text="Unity XML export",
            font=("Helvetica", 10),
            command=self.export_unity_xml_from_sidebar,
            bg="#ff5d13",
            fg="white",
            relief="flat",
            padx=10,
            pady=8,
        )
        self.export_unity_button.pack(fill=tk.X, pady=(0, 12))

        slideCalibration_label = tk.Label(
            calibration_area, text="Graypattern\nSeries Capture", font=("Helvetica", 12, "bold"), bg="#1a1a1a", fg="yellow", anchor="w")
        slideCalibration_label.pack(fill=tk.X, pady=(0, 8))

        projector_capture_label = tk.Label(
            calibration_area, text="Capture Folder", bg="#1a1a1a", fg="white", anchor="w")
        projector_capture_label.pack(fill=tk.X, pady=(0, 4))
        projector_capture_entry = tk.Entry(
            calibration_area, textvariable=self.graycode_capture_dir_var,
            bg="#202429", fg="white", insertbackground="white")
        projector_capture_entry.pack(fill=tk.X, pady=(0, 8))
        self.projector_capture_browse_button = tk.Button(
            calibration_area, text="Browse", command=self.browse_graycode_capture_folder,
            bg="#3a3d42", fg="white", relief="flat", padx=10, pady=6)
        self.projector_capture_browse_button.pack(fill=tk.X, pady=(0, 12))

        projector_monitor_label = tk.Label(
            calibration_area, text="Projector Monitor", bg="#1a1a1a", fg="white", anchor="w")
        projector_monitor_label.pack(fill=tk.X, pady=(0, 4))
        self.projector_monitor_menu = ttk.Combobox(
            calibration_area,
            textvariable=self.projector_monitor_var,
            values=list(self.projector_monitors.keys()
                        ) if self.projector_monitors else ["Primary"],
            state="readonly",
        )
        self.projector_monitor_menu.pack(fill=tk.X, pady=(0, 8))
        self.projector_monitor_menu.bind(
            "<<ComboboxSelected>>", self.on_projector_monitor_selected)

        projector_speed_label = tk.Label(
            calibration_area, text="Slideshow Speed (s)", bg="#1a1a1a", fg="white", anchor="w")
        projector_speed_label.pack(fill=tk.X, pady=(8, 4))
        self.projector_slideshow_speed_entry = tk.Entry(
            calibration_area, textvariable=self.projector_slideshow_speed_var,
            bg="#202429", fg="white", insertbackground="white")
        self.projector_slideshow_speed_entry.pack(fill=tk.X, pady=(0, 8))

        projector_button_row = tk.Frame(calibration_area, bg="#1a1a1a")
        projector_button_row.pack(fill=tk.X, pady=(0, 12))

        self.projector_test_button = tk.Button(
            projector_button_row,
            text="Test\nOutput",
            command=self.show_projector_test_output,
            bg="#ffa55b",
            fg="white",
            relief="flat",
            padx=8,
            pady=8,
        )
        self.projector_test_button.pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6))

        self.projector_slideshow_button = tk.Button(
            projector_button_row,
            text="Start\nSlideshow",
            command=self.toggle_projector_slideshow,
            bg="#5b9dff",
            fg="white",
            relief="flat",
            padx=8,
            pady=8,
        )
        self.projector_slideshow_button.pack(
            side=tk.LEFT, fill=tk.X, expand=True)

        self._style_buttons_and_sliders()
        self.start_osc_server()
        self.open_camera()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def _osc_read_string(self, data, offset):
        end = data.find(b"\x00", offset)
        if end == -1:
            end = len(data)
        value = data[offset:end].decode("utf-8", errors="replace")
        padded = (len(value.encode("utf-8")) + 1)
        padded = (padded + 3) & ~3
        return value, offset + padded

    def _parse_osc_message(self, packet):
        if len(packet) < 4:
            return None

        address, offset = self._osc_read_string(packet, 0)
        if offset >= len(packet):
            return {"address": address, "args": []}

        if packet[offset:offset + 1] != b",":
            return {"address": address, "args": []}

        tag_end = packet.find(b"\x00", offset)
        if tag_end == -1:
            tag_end = len(packet)
        type_string = packet[offset:tag_end].decode("utf-8", errors="replace")
        if not type_string.startswith(","):
            return {"address": address, "args": []}
        type_string = type_string[1:]

        offset = (tag_end + 4) & ~3
        args = []
        for tag in type_string:
            if offset + 4 > len(packet):
                break
            if tag == "i":
                value = struct.unpack(">i", packet[offset:offset + 4])[0]
                args.append(value)
                offset += 4
            elif tag == "f":
                value = struct.unpack(">f", packet[offset:offset + 4])[0]
                args.append(value)
                offset += 4
            elif tag == "s":
                value, offset = self._osc_read_string(packet, offset)
                args.append(value)
            elif tag == "b":
                blob_size = struct.unpack(">i", packet[offset:offset + 4])[0]
                offset += 4
                args.append(packet[offset:offset + blob_size])
                offset = (offset + blob_size + 3) & ~3
            else:
                break
        return {"address": address, "args": args}

    def _queue_on_ui(self, func, *args, **kwargs):
        if self.root.winfo_exists():
            self.root.after(0, lambda: func(*args, **kwargs))

    def _osc_pad_string(self, value):
        encoded = value.encode("utf-8") + b"\x00"
        padding = (-len(encoded)) % 4
        return encoded + (b"\x00" * padding)

    def _send_osc_float(self, address, value, port=7001, host=None):
        if host is None:
            host = self.osc_bind_ip_var.get().strip() if hasattr(
                self, "osc_bind_ip_var") else "127.0.0.1"
        if not host or host == "0.0.0.0":
            host = "127.0.0.1"

        packet = self._osc_pad_string(address)
        packet += self._osc_pad_string(",f")
        packet += struct.pack(">f", float(value))

        sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sender.sendto(packet, (host, int(port)))
        finally:
            sender.close()

    def get_osc_command_map(self):
        return {
            "take_picture": self.osc_take_picture_cmd_var.get().strip() or "/camera/take_picture",
            "interval_start": self.osc_interval_start_cmd_var.get().strip() or "/camera/interval/start",
            "interval_stop": self.osc_interval_stop_cmd_var.get().strip() or "/camera/interval/stop",
            "interval_time": self.osc_interval_time_cmd_var.get().strip() or "/camera/interval/time",
            "projector_slideshow_start": self.osc_projector_slideshow_start_cmd_var.get().strip() or "/projector/slideshow/start",
            "projector_slideshow_stop": self.osc_projector_slideshow_stop_cmd_var.get().strip() or "/projector/slideshow/stop",
        }

    def apply_osc_settings(self):
        try:
            port = int(self.osc_port_var.get().strip())
            if port <= 0 or port > 65535:
                raise ValueError
            self.osc_port = port
            self.osc_bind_ip_var.set(
                self.osc_bind_ip_var.get().strip() or "0.0.0.0")
            self.status_var.set(
                f"OSC configured on {self.osc_bind_ip_var.get()}:{self.osc_port}")
            self.stop_osc_server()
            self.start_osc_server()
        except ValueError:
            self.status_var.set(
                "OSC port must be an integer between 1 and 65535.")

    def _handle_osc_message(self, message, _address):
        address = message.get("address", "").strip().lower()
        args = message.get("args", [])
        commands = self.get_osc_command_map()
        command_set = {cmd.lower().strip() for cmd in commands.values()}

        if address in command_set:
            if address == commands["take_picture"].strip().lower():
                self._queue_on_ui(self.take_picture)
                return
            if address == commands["interval_start"].strip().lower():
                value = args[0] if args else None
                if value is not None:
                    self._queue_on_ui(self.set_interval_from_osc, float(value))
                self._queue_on_ui(self.toggle_interval_capture_if_needed)
                return
            if address == commands["interval_stop"].strip().lower():
                self._queue_on_ui(self.stop_interval_capture)
                return
            if address == commands["interval_time"].strip().lower():
                if args:
                    self._queue_on_ui(
                        self.set_interval_from_osc, float(args[0]))
                return
            if address == commands["projector_slideshow_start"].strip().lower():
                self._queue_on_ui(self.start_projector_slideshow)
                return
            if address == commands["projector_slideshow_stop"].strip().lower():
                self._queue_on_ui(self.stop_projector_slideshow)
                return

        legacy = {
            "/camera/take_picture",
            "/take_picture",
            "/capture",
            "/camera/capture",
            "/camera/interval/start",
            "/interval/start",
            "/camera/interval/stop",
            "/interval/stop",
            "/camera/interval/time",
            "/interval/time",
            "/camera/interval/set",
            "/interval/set",
            "/projector/slideshow/start",
            "/projector/slideshow/stop",
        }
        if address in legacy:
            if address in {"/camera/take_picture", "/take_picture", "/capture", "/camera/capture"}:
                self._queue_on_ui(self.take_picture)
                return
            if address in {"/camera/interval/start", "/interval/start"}:
                value = args[0] if args else None
                if value is not None:
                    self._queue_on_ui(self.set_interval_from_osc, float(value))
                self._queue_on_ui(self.toggle_interval_capture_if_needed)
                return
            if address in {"/camera/interval/stop", "/interval/stop"}:
                self._queue_on_ui(self.stop_interval_capture)
                return
            if address in {"/camera/interval/time", "/interval/time", "/camera/interval/set", "/interval/set"}:
                if args:
                    self._queue_on_ui(
                        self.set_interval_from_osc, float(args[0]))
                return
            if address == "/projector/slideshow/start":
                self._queue_on_ui(self.start_projector_slideshow)
                return
            if address == "/projector/slideshow/stop":
                self._queue_on_ui(self.stop_projector_slideshow)
                return

    def _handle_osc_error(self, exc):
        if hasattr(self, "status_var"):
            self.root.after(
                0, lambda: self.status_var.set(f"OSC error: {exc}"))

    def stop_osc_server(self):
        self.osc_server_running = False
        if self.osc_socket is not None:
            try:
                self.osc_socket.close()
            except Exception:
                pass
            self.osc_socket = None

    def start_osc_server(self):
        if self.osc_server_running:
            return
        try:
            host = self.osc_bind_ip_var.get().strip() or "0.0.0.0"
            port = int(self.osc_port_var.get().strip() or self.osc_port)
            self.osc_port = port
            self.osc_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.osc_socket.setsockopt(
                socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.osc_socket.bind((host, port))
            self.osc_socket.settimeout(0.25)
            self.osc_server_running = True
            self.osc_server_thread = threading.Thread(
                target=self._osc_server_loop, daemon=True)
            self.osc_server_thread.start()
            self.status_var.set(f"OSC listening on {host}:{port}")
        except OSError as exc:
            self.status_var.set(f"OSC unavailable: {exc}")

    def _osc_server_loop(self):
        while self.osc_server_running and self.osc_socket is not None:
            try:
                data, _address = self.osc_socket.recvfrom(4096)
            except socket.timeout:
                continue
            except OSError:
                break
            try:
                message = self._parse_osc_message(data)
                if message is not None:
                    self._handle_osc_message(message, _address)
            except Exception as exc:
                self._handle_osc_error(exc)

    def set_interval_from_osc(self, seconds):
        try:
            interval = float(seconds)
            if interval <= 0:
                raise ValueError("Interval must be > 0")
            self.capture_interval_var.set(str(interval))

            if self.interval_active:
                if self.interval_after_id is not None:
                    try:
                        self.root.after_cancel(self.interval_after_id)
                    except Exception:
                        pass
                    self.interval_after_id = None
                if self.interval_progress_after_id is not None:
                    try:
                        self.root.after_cancel(self.interval_progress_after_id)
                    except Exception:
                        pass
                    self.interval_progress_after_id = None
                self.interval_duration = interval
                self.interval_start_time = time.monotonic()
                self._show_interval_progress()
                self._update_interval_progress()
                self._schedule_interval_capture(interval)

            self.status_var.set(f"Interval set to {interval:.2f}s")
        except (TypeError, ValueError) as exc:
            self.status_var.set(f"Invalid interval OSC value: {exc}")

    def toggle_interval_capture_if_needed(self):
        if self.interval_active:
            return
        interval = self.validate_interval()
        if interval is None:
            return
        self.toggle_interval_capture()

    def _style_buttons_and_sliders(self):
        self.root.style = ttk.Style()
        self.root.style.theme_use("clam")
        self.root.style.configure(
            "IntervalCountdown.Horizontal.TProgressbar",
            background="#ffb000",
            troughcolor="#2d2f35",
            lightcolor="#ffb000",
            darkcolor="#ffb000",
        )
        self.root.style.map(
            "IntervalCountdown.Horizontal.TProgressbar",
            background=[("selected", "#ffb000"), ("active", "#ff9300")],
            troughcolor=[("active", "#2d2f35")],
        )

        for button in [
            self.take_button,
            self.zoom_in_button,
            self.zoom_out_button,
            self.exposure_auto_btn,
            self.focus_auto_btn,
            self.white_balance_auto_btn,
            self.gallery_toggle_button,
            *[getattr(self, f"{name}_auto_btn", None)
              for name in self.cv2_extra_controls.keys() if hasattr(self, f"{name}_auto_btn")],
            self.browse_button,
        ]:
            if button is not None:
                button.configure(
                    relief="raised",
                    bd=1,
                    highlightthickness=0,
                    activebackground="#4d535a",
                    activeforeground="white",
                    cursor="hand2",
                    borderwidth=1,
                )

        for slider in [
            self.exposure_slider,
            self.focus_slider,
            self.white_balance_slider,
            *[control["slider"] for control in self.cv2_extra_controls.values()
              if control.get("slider") is not None],
        ]:
            slider.configure(
                length=220,
                sliderlength=18,
                showvalue=0,
                troughcolor="#1b1f24",
                highlightthickness=0,
                bd=0,
                relief="flat",
                activebackground="#d9eaff",
            )

        for entry in [
            self.exposure_entry,
            self.focus_entry,
            self.white_balance_entry,
            *[control["entry"] for control in self.cv2_extra_controls.values()
              if control.get("entry") is not None],
        ]:
            entry.configure(
                highlightthickness=1,
                highlightbackground="#383d44",
                highlightcolor="#5fa8ff",
                relief="sunken",
                bd=1,
            )

    def _cancel_update_loop(self):
        self._loop_generation += 1
        if self.after_id is not None:
            try:
                self.root.after_cancel(self.after_id)
            except Exception:
                pass
            self.after_id = None

    def _schedule_update(self, generation=None):
        if generation is None:
            generation = self._loop_generation
        if self.after_id is not None:
            try:
                self.root.after_cancel(self.after_id)
            except Exception:
                pass
        self.after_id = self.root.after(
            30, lambda: self.update_frame(generation))

    def close_local_source(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def close_sony_source(self):
        if self.sony_stream is not None:
            try:
                self.sony_stream.close()
            except Exception:
                pass
            self.sony_stream = None
        self.sony_liveview_url = None
        self.sony_buffer = b""

    def open_camera(self):
        self._cancel_update_loop()
        selected = self.camera_var.get()
        self.close_local_source()
        self.close_sony_source()
        self.latest_frame = None

        if selected == "Sony HTTP API":
            self.hide_cv2_controls()
            self.camera_resolution_var.set("")
            self.camera_resolution_menu.configure(values=[], state="disabled")
            self.open_sony_camera()
            return

        match = re.search(r"(\d+)", selected)
        if not match:
            self.status_var.set("No valid camera selected.")
            return

        self.camera_index = int(match.group(1))
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            self.status_var.set(
                f"Camera {self.camera_index} could not be opened.")
            return

        self.status_var.set(f"Live preview from camera {self.camera_index}")
        self.show_cv2_controls()
        self.sync_cv2_controls_from_camera()
        self.update_camera_resolution_choices()
        self._schedule_update()

    def open_sony_camera(self):
        try:
            self.sony_liveview_url = sony_start_liveview()
            if not self.sony_liveview_url:
                raise RuntimeError("Sony API returned no live view URL.")
            self.status_var.set("Live preview from Sony HTTP API")
            self.sony_stream = urllib.request.urlopen(
                self.sony_liveview_url, timeout=3)
            self.sony_buffer = b""
            self._schedule_update()
        except Exception as exc:
            self.status_var.set(f"Sony API unavailable: {exc}")

    def get_camera_resolution_options(self, camera_index=None):
        if camera_index is None:
            selected = self.camera_var.get()
            if selected == "Sony HTTP API":
                return []
            match = re.search(r"(\d+)", selected)
            if not match:
                return []
            camera_index = int(match.group(1))

        if hasattr(self, "camera_indexes") and camera_index not in self.camera_indexes:
            return []

        cap = getattr(self, "cap", None)
        if cap is not None and cap.isOpened() and getattr(self, "camera_index", -1) == camera_index:
            options = []
            seen = set()
            for resolution in COMMON_RESOLUTION_OPTIONS:
                width, height = [int(value)
                                 for value in resolution.split("x", 1)]
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
                actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                if actual_width <= 0 or actual_height <= 0:
                    continue
                resolved = f"{actual_width}x{actual_height}"
                if resolved not in seen:
                    seen.add(resolved)
                    options.append(resolved)

            if not options:
                actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                if actual_width > 0 and actual_height > 0:
                    options.append(f"{actual_width}x{actual_height}")

            if options:
                return sorted(
                    options,
                    key=lambda item: (-int(item.split("x")
                                      [0]), -int(item.split("x")[1])),
                )

        return list(COMMON_RESOLUTION_OPTIONS)

    def update_camera_resolution_choices(self):
        selected = self.camera_var.get()
        if selected == "Sony HTTP API":
            self.camera_resolution_var.set("")
            self.camera_resolution_menu.configure(values=[], state="disabled")
            return

        match = re.search(r"(\d+)", selected)
        if not match:
            self.camera_resolution_var.set("")
            self.camera_resolution_menu.configure(values=[], state="disabled")
            return

        options = self.get_camera_resolution_options()
        if not options:
            self.camera_resolution_var.set("")
            self.camera_resolution_menu.configure(values=[], state="disabled")
            return

        self.camera_resolution_menu.configure(values=options, state="readonly")
        if self.cap is not None and self.cap.isOpened():
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            current = f"{actual_width}x{actual_height}"
            if current in options:
                self.camera_resolution_var.set(current)
                return
        self.camera_resolution_var.set(options[0])

    def apply_camera_resolution(self, resolution):
        if self.camera_var.get() == "Sony HTTP API":
            return
        match = re.search(r"(\d+)", self.camera_var.get())
        if not match:
            return

        try:
            width, height = [int(part) for part in resolution.split("x", 1)]
        except (TypeError, ValueError):
            return

        if self.cap is not None and self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.camera_resolution_var.set(f"{actual_width}x{actual_height}")
            self.status_var.set(
                f"Camera resolution set to {actual_width}x{actual_height}")
            self.sync_cv2_controls_from_camera()
            self._schedule_update()

    def on_camera_resolution_selected(self, event=None):
        resolution = self.camera_resolution_var.get().strip()
        if resolution:
            self.apply_camera_resolution(resolution)

    def on_camera_selected(self, event):
        selected = self.camera_var.get()
        if selected == "Sony HTTP API":
            self.camera_resolution_var.set("")
            self.camera_resolution_menu.configure(values=[], state="disabled")
            self.open_camera()
            return

        self.open_camera()
        self.update_camera_resolution_choices()

    def on_sidebar_scroll(self, event):
        if not self.sidebar.winfo_exists():
            return

        widget_under_cursor = self.sidebar.winfo_containing(
            event.x_root, event.y_root)
        if widget_under_cursor is None:
            return

        if hasattr(event, "delta"):
            delta = event.delta
        elif event.num == 4:
            delta = -120
        elif event.num == 5:
            delta = 120
        else:
            delta = 0

        self.sidebar_canvas.yview_scroll(int(-delta / 120), "units")
        return "break"

    def hide_cv2_controls(self):
        self.cv2_controls_frame.pack_forget()

    def show_cv2_controls(self):
        self.cv2_controls_frame.pack(fill=tk.X, pady=(0, 8))

    def _set_slider_and_entry(self, slider, value_var, value):
        try:
            numeric_value = float(value)
            slider.set(int(round(numeric_value)))
            value_var.set(str(int(round(numeric_value))))
        except (TypeError, ValueError):
            pass

    def sync_entry_value(self, control_name):
        mapping = {
            "exposure": (self.exposure_slider, self.exposure_value_var, self.apply_exposure_value),
            "focus": (self.focus_slider, self.focus_value_var, self.apply_focus_value),
            "white_balance": (self.white_balance_slider, self.white_balance_value_var, self.apply_white_balance_value),
        }
        slider, value_var, apply_callback = mapping.get(
            control_name, (None, None, None))
        if slider is None or value_var is None:
            return

        try:
            numeric_value = float(value_var.get())
        except (TypeError, ValueError):
            return

        rounded_value = int(round(numeric_value))
        slider.set(rounded_value)
        value_var.set(str(rounded_value))
        if apply_callback is not None:
            apply_callback()

    def update_cv2_manual_visibility(self):
        for auto_var, slider, entry in [
            (self.exposure_auto_var, self.exposure_slider, self.exposure_entry),
            (self.focus_auto_var, self.focus_slider, self.focus_entry),
            (self.white_balance_auto_var,
             self.white_balance_slider, self.white_balance_entry),
        ]:
            if auto_var.get():
                slider.pack_forget()
                entry.pack_forget()
                slider.configure(state=tk.DISABLED)
                entry.configure(state=tk.DISABLED)
            else:
                slider.pack(fill=tk.X, pady=(0, 4))
                entry.pack(fill=tk.X)
                slider.configure(state=tk.NORMAL)
                entry.configure(state=tk.NORMAL)

        for control in self.cv2_extra_controls.values():
            slider = control["slider"]
            entry = control["entry"]
            auto_var = control.get("auto_var")
            if auto_var is not None and auto_var.get():
                if slider is not None:
                    slider.pack_forget()
                    slider.configure(state=tk.DISABLED)
                if entry is not None:
                    entry.pack_forget()
                    entry.configure(state=tk.DISABLED)
            else:
                if slider is not None:
                    slider.pack(fill=tk.X, pady=(0, 4))
                    slider.configure(state=tk.NORMAL)
                if entry is not None:
                    entry.pack(fill=tk.X)
                    entry.configure(state=tk.NORMAL)

    def update_cv2_control_buttons(self):
        self.exposure_auto_btn.config(
            text=f"Auto exposure: {'ON' if self.exposure_auto_var.get() else 'OFF'}")
        self.focus_auto_btn.config(
            text=f"Auto focus: {'ON' if self.focus_auto_var.get() else 'OFF'}")
        self.white_balance_auto_btn.config(
            text=f"Auto white balance: {'ON' if self.white_balance_auto_var.get() else 'OFF'}")
        for name, control in self.cv2_extra_controls.items():
            auto_var = control.get("auto_var")
            if auto_var is None:
                continue
            button = getattr(self, f"{name}_auto_btn", None)
            if button is not None:
                button.config(
                    text=f"Auto {control['label']}: {'ON' if auto_var.get() else 'OFF'}")
        self.update_cv2_manual_visibility()

    def sync_cv2_controls_from_camera(self):
        if self.cap is None or not self.cap.isOpened():
            return

        for property_name, auto_var, manual_prop, value_var, slider, default_value in [
            ("exposure", self.exposure_auto_var, getattr(cv2, "CAP_PROP_EXPOSURE",
             None), self.exposure_value_var, self.exposure_slider, 0),
            ("focus", self.focus_auto_var, getattr(cv2, "CAP_PROP_FOCUS",
             None), self.focus_value_var, self.focus_slider, 0),
            ("white_balance", self.white_balance_auto_var, getattr(cv2, "CAP_PROP_WB_TEMPERATURE",
             None), self.white_balance_value_var, self.white_balance_slider, 4000),
        ]:
            auto_prop = getattr(cv2, f"CAP_PROP_AUTO_{property_name.upper().replace('_', '')}",
                                None) if property_name != "white_balance" else getattr(cv2, "CAP_PROP_AUTO_WB", None)
            if auto_prop is not None:
                auto_var.set(bool(self.cap.get(auto_prop)))
            if manual_prop is not None:
                manual_value = self.cap.get(manual_prop)
                if manual_value is None or (manual_value == 0 and default_value != 0):
                    manual_value = default_value
                self._set_slider_and_entry(slider, value_var, manual_value)

        for name, control in self.cv2_extra_controls.items():
            auto_prop = control.get("auto_prop")
            auto_var = control.get("auto_var")
            manual_prop = control.get("manual_prop")
            value_var = control.get("value_var")
            slider = control.get("slider")
            if auto_prop is not None and auto_var is not None:
                try:
                    auto_var.set(bool(self.cap.get(auto_prop)))
                except Exception:
                    pass
            if manual_prop is not None and value_var is not None and slider is not None:
                try:
                    manual_value = self.cap.get(manual_prop)
                    if manual_value is None:
                        manual_value = control.get("default_value", 0)
                    self._set_slider_and_entry(slider, value_var, manual_value)
                except Exception:
                    pass

        self.update_cv2_control_buttons()

    def toggle_exposure_auto(self):
        self.exposure_auto_var.set(not self.exposure_auto_var.get())
        self.apply_cv2_toggle_state("exposure")
        self.update_cv2_control_buttons()

    def toggle_focus_auto(self):
        self.focus_auto_var.set(not self.focus_auto_var.get())
        self.apply_cv2_toggle_state("focus")
        self.update_cv2_control_buttons()

    def toggle_white_balance_auto(self):
        self.white_balance_auto_var.set(not self.white_balance_auto_var.get())
        self.apply_cv2_toggle_state("white_balance")
        self.update_cv2_control_buttons()

    def toggle_cv2_auto_control(self, control_name):
        control = self.cv2_extra_controls.get(control_name)
        if control is None or control.get("auto_prop") is None or control.get("auto_var") is None:
            return
        control["auto_var"].set(not control["auto_var"].get())
        self.apply_cv2_auto_state(control_name)
        self.update_cv2_control_buttons()

    def apply_cv2_toggle_state(self, control_name):
        if self.cap is None or not self.cap.isOpened():
            return

        mapping = {
            "exposure": (getattr(cv2, "CAP_PROP_AUTO_EXPOSURE", None), getattr(cv2, "CAP_PROP_EXPOSURE", None)),
            "focus": (getattr(cv2, "CAP_PROP_AUTOFOCUS", None), getattr(cv2, "CAP_PROP_FOCUS", None)),
            "white_balance": (getattr(cv2, "CAP_PROP_AUTO_WB", None), getattr(cv2, "CAP_PROP_WB_TEMPERATURE", None)),
        }
        auto_prop, manual_prop = mapping.get(control_name, (None, None))
        if auto_prop is None:
            return

        auto_enabled = getattr(self, f"{control_name}_auto_var").get()
        self.cap.set(auto_prop, 1.0 if auto_enabled else 0.0)

        if manual_prop is not None:
            value = getattr(self, f"{control_name}_value_var").get()
            try:
                self.cap.set(manual_prop, float(value))
            except Exception:
                pass

    def apply_cv2_auto_state(self, control_name):
        if self.cap is None or not self.cap.isOpened():
            return
        control = self.cv2_extra_controls.get(control_name)
        if control is None:
            return
        auto_prop = control.get("auto_prop")
        if auto_prop is None:
            return
        auto_var = control.get("auto_var")
        if auto_var is None:
            return
        self.cap.set(auto_prop, 1.0 if auto_var.get() else 0.0)
        manual_prop = control.get("manual_prop")
        value_var = control.get("value_var")
        if manual_prop is not None and value_var is not None:
            try:
                self.cap.set(manual_prop, float(value_var.get()))
            except Exception:
                pass

    def on_exposure_slider_change(self, value):
        self.exposure_value_var.set(str(int(float(value))))
        self.apply_exposure_value()

    def apply_exposure_value(self):
        if self.cap is None or not self.cap.isOpened() or self.exposure_auto_var.get():
            return
        try:
            value = float(self.exposure_value_var.get())
            self.cap.set(getattr(cv2, "CAP_PROP_EXPOSURE", 0), value)
        except Exception:
            pass

    def on_focus_slider_change(self, value):
        self.focus_value_var.set(str(int(float(value))))
        self.apply_focus_value()

    def apply_focus_value(self):
        if self.cap is None or not self.cap.isOpened() or self.focus_auto_var.get():
            return
        try:
            value = float(self.focus_value_var.get())
            self.cap.set(getattr(cv2, "CAP_PROP_FOCUS", 0), value)
        except Exception:
            pass

    def on_white_balance_slider_change(self, value):
        self.white_balance_value_var.set(str(int(float(value))))
        self.apply_white_balance_value()

    def apply_white_balance_value(self):
        if self.cap is None or not self.cap.isOpened() or self.white_balance_auto_var.get():
            return
        try:
            value = float(self.white_balance_value_var.get())
            self.cap.set(getattr(cv2, "CAP_PROP_WB_TEMPERATURE", 0), value)
        except Exception:
            pass

    def on_cv2_manual_value_changed(self, name, value):
        control = self.cv2_extra_controls.get(name)
        if control is None:
            return
        control["value_var"].set(str(int(float(value))))
        self.apply_cv2_manual_value(name)

    def sync_cv2_entry_value(self, control_name):
        control = self.cv2_extra_controls.get(control_name)
        if control is None:
            return

        slider = control.get("slider")
        value_var = control.get("value_var")
        if slider is None or value_var is None:
            return

        try:
            numeric_value = float(value_var.get())
        except (TypeError, ValueError):
            return

        rounded_value = int(round(numeric_value))
        slider.set(rounded_value)
        value_var.set(str(rounded_value))
        self.apply_cv2_manual_value(control_name)

    def apply_cv2_manual_value(self, control_name):
        if self.cap is None or not self.cap.isOpened():
            return
        control = self.cv2_extra_controls.get(control_name)
        if control is None:
            return
        if control.get("auto_var") is not None and control["auto_var"].get():
            return
        manual_prop = control.get("manual_prop")
        value_var = control.get("value_var")
        if manual_prop is None or value_var is None:
            return
        try:
            self.cap.set(manual_prop, float(value_var.get()))
        except Exception:
            pass

    def _decode_sony_jpeg(self):
        if len(self.sony_buffer) < 136:
            return None

        if self.sony_buffer[0] != 255:
            self.sony_buffer = self.sony_buffer[1:]
            return None

        payload_header = self.sony_buffer[8:136]
        if len(payload_header) < 128:
            return None

        jpeg_size = payload_header[4] * 65536 + \
            payload_header[5] * 256 + payload_header[6]
        padding_size = payload_header[7]
        total_size = 8 + 128 + jpeg_size + padding_size
        if len(self.sony_buffer) < total_size:
            return None

        jpeg_bytes = self.sony_buffer[136: 136 + jpeg_size]
        self.sony_buffer = self.sony_buffer[total_size:]

        if len(jpeg_bytes) == 0:
            return None

        frame = cv2.imdecode(np.frombuffer(
            jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        return frame

    def apply_zoom(self, frame):
        if self.zoom_factor <= 1.0 or frame is None:
            return frame

        height, width = frame.shape[:2]
        crop_w = max(1, int(width / self.zoom_factor))
        crop_h = max(1, int(height / self.zoom_factor))
        x = (width - crop_w) // 2
        y = (height - crop_h) // 2
        cropped = frame[y: y + crop_h, x: x + crop_w]
        return cv2.resize(cropped, (width, height), interpolation=cv2.INTER_AREA)

    def update_frame(self, generation=None):
        if generation is not None and generation != self._loop_generation:
            return

        selected = self.camera_var.get()

        if selected == "Sony HTTP API":
            self._update_sony_frame(generation)
            return

        if self.cap is None or not self.cap.isOpened():
            return

        ret, frame = self.cap.read()
        if not ret:
            self.status_var.set("Could not read the next camera frame.")
            return

        preview_frame = self.apply_zoom(frame)
        self._show_frame(preview_frame)
        self.latest_frame = preview_frame
        self._schedule_update(generation)

    def _update_sony_frame(self, generation=None):
        if self.sony_stream is None:
            return

        try:
            chunk = self.sony_stream.read(4096)
            if not chunk:
                self.status_var.set("Sony stream ended.")
                self.close_sony_source()
                return

            self.sony_buffer += chunk
            frame = self._decode_sony_jpeg()
            if frame is not None:
                preview_frame = self.apply_zoom(frame)
                self._show_frame(preview_frame)
                self.latest_frame = preview_frame

        except (urllib.error.URLError, OSError, ValueError) as exc:
            self.status_var.set(f"Sony stream error: {exc}")
            self.close_sony_source()
            return

        self._schedule_update(generation)

    def _show_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        image.thumbnail((900, 700))
        self.photo_image = ImageTk.PhotoImage(image)
        self.preview_label.configure(image=self.photo_image)
        self.preview_label.image = self.photo_image

    def get_attached_monitors(self):
        monitors = {}
        root = getattr(self, "root", None)
        width = getattr(root, "winfo_screenwidth",
                        lambda: 1920)() if root is not None else 1920
        height = getattr(root, "winfo_screenheight",
                         lambda: 1080)() if root is not None else 1080

        try:
            result = subprocess.run(
                ["xrandr", "--listmonitors"],
                capture_output=True,
                text=True,
                check=False,
            )
        except FileNotFoundError:
            result = None

        if result is not None and result.returncode == 0:
            lines = result.stdout.strip().splitlines()[1:]
            for line in lines:
                if not line.strip():
                    continue
                parts = line.split()
                if len(parts) < 3:
                    continue
                name = re.sub(r"^[^A-Za-z0-9._-]*", "", parts[1])
                geometry = parts[2]
                match = re.search(
                    r"(\d+)\/\d+x(\d+)\/\d+\+(-?\d+)\+(-?\d+)", geometry
                )
                if match is None:
                    continue
                monitor_width = int(match.group(1))
                monitor_height = int(match.group(2))
                offset_x = int(match.group(3))
                offset_y = int(match.group(4))
                monitors[name] = (offset_x, offset_y,
                                  monitor_width, monitor_height)

        if not monitors:
            monitors["Primary"] = (0, 0, width, height)
        return monitors

    def on_projector_monitor_selected(self, event=None):
        monitor_name = self.projector_monitor_var.get().strip()
        if monitor_name and self.projector_slideshow_window is not None and self.projector_slideshow_window.winfo_exists():
            self._configure_projector_window_geometry()

    def ensure_projector_pattern_folder(self):
        folder = self.graycode_pattern_dir_var.get().strip(
        ) or os.path.expanduser("graycode_pattern")
        self.graycode_pattern_dir_var.set(folder)
        os.makedirs(folder, exist_ok=True)
        return folder

    def ensure_projector_capture_folder(self):
        capture_var = getattr(self, "projector_capture_dir_var", getattr(
            self, "graycode_capture_dir_var", None))
        if capture_var is None:
            capture_var = self.graycode_capture_dir_var
        folder = capture_var.get().strip() or os.path.expanduser("capture_0")
        capture_var.set(folder)
        if hasattr(self, "graycode_capture_dir_var"):
            self.graycode_capture_dir_var.set(folder)
        os.makedirs(folder, exist_ok=True)
        return folder

    def browse_graycode_pattern_folder(self):
        pattern_var = getattr(
            self, "projector_pattern_dir_var", self.graycode_pattern_dir_var)
        directory = filedialog.askdirectory(initialdir=pattern_var.get())
        if directory:
            pattern_var.set(directory)
            if hasattr(self, "graycode_pattern_dir_var"):
                self.graycode_pattern_dir_var.set(directory)

    def browse_graycode_capture_folder(self):
        capture_var = getattr(
            self, "projector_capture_dir_var", self.graycode_capture_dir_var)
        directory = filedialog.askdirectory(initialdir=capture_var.get())
        if directory:
            capture_var.set(directory)
            if hasattr(self, "graycode_capture_dir_var"):
                self.graycode_capture_dir_var.set(directory)

    def generate_graycode_pattern_from_sidebar(self):
        try:
            proj_height = int(self.calibration_height_var.get().strip())
            proj_width = int(self.calibration_width_var.get().strip())
            graycode_step = int(self.graycode_step_var.get().strip() or "1")
            if proj_height <= 0 or proj_width <= 0:
                raise ValueError(
                    "Projector width and height must be positive.")
            if graycode_step <= 0:
                raise ValueError("Graycode step must be greater than zero.")

            folder = self.ensure_projector_pattern_folder()
            self._append_calibration_output(
                f"Generating Gray-code pattern images in {folder} for {proj_height}x{proj_width} ...")
            self.status_var.set(
                f"Generating Gray-code patterns in {folder}...")

            files = generate_graycode_pattern_images(
                folder,
                proj_height,
                proj_width,
                graycode_step,
            )
            self._append_calibration_output(
                f"Generated {len(files)} Gray-code images in {folder}.")
            self.status_var.set(
                f"Generated {len(files)} Gray-code images in {folder}.")
            return files
        except Exception as exc:
            self._append_calibration_output(
                f"Graycode generation failed: {exc}")
            self.status_var.set(f"Graycode generation failed: {exc}")
            return []

    def get_graycode_pattern_files(self):
        folder = self.ensure_projector_pattern_folder()
        if not os.path.isdir(folder):
            return []
        extensions = {".png", ".jpg", ".jpeg",
                      ".bmp", ".gif", ".tif", ".tiff", ".webp"}
        files = []
        for entry in os.listdir(folder):
            full_path = os.path.join(folder, entry)
            if os.path.isfile(full_path) and os.path.splitext(entry)[1].lower() in extensions:
                files.append(full_path)
        return sorted(files, key=lambda item: os.path.basename(item).lower())

    def get_next_graycode_capture_index(self):
        folder = self.ensure_projector_capture_folder()
        max_index = -1
        for entry in os.listdir(folder):
            match = re.fullmatch(r"graycode_(\d+)\.png",
                                 entry, flags=re.IGNORECASE)
            if match:
                max_index = max(max_index, int(match.group(1)))
        return max_index + 1 if max_index >= 0 else 0

    def _configure_projector_window_geometry(self):
        if self.projector_slideshow_window is None or not self.projector_slideshow_window.winfo_exists():
            return
        screen_name = self.projector_monitor_var.get().strip() or "Primary"
        geometry = self.projector_monitors.get(screen_name, (0, 0, 1920, 1080))
        x_pos, y_pos, width, height = geometry
        self.projector_slideshow_window.geometry(
            f"{width}x{height}+{x_pos}+{y_pos}")
        self.projector_slideshow_window.attributes("-topmost", True)

    def _display_projector_image(self, image_path):
        if self.projector_slideshow_window is None or not self.projector_slideshow_window.winfo_exists():
            return
        if self.projector_slideshow_label is None:
            self.projector_slideshow_label = tk.Label(
                self.projector_slideshow_window,
                bg="black",
                compound="center",
                anchor="center",
            )
            self.projector_slideshow_label.pack(fill=tk.BOTH, expand=True)

        try:
            image = Image.open(image_path).convert("RGB")
            width = self.projector_slideshow_window.winfo_width() or 1920
            height = self.projector_slideshow_window.winfo_height() or 1080
            image = image.resize(
                (max(1, width), max(1, height)), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(image)
            self.projector_slideshow_label.configure(image=photo)
            self.projector_slideshow_label.image = photo
        except Exception:
            self.projector_slideshow_label.configure(
                text=os.path.basename(image_path), fg="white", bg="black")

    def _capture_projector_photo(self):
        if self.latest_frame is None:
            self.status_var.set(
                "No live frame available for projector capture.")
            return None

        folder = self.ensure_projector_capture_folder()
        index = self.get_next_graycode_capture_index()
        path = os.path.join(folder, f"graycode_{index:02d}.png")
        success = cv2.imwrite(path, self.latest_frame)
        if not success:
            raise RuntimeError(f"Could not save projector capture to {path}.")
        self.status_var.set(f"Saved projector capture: {path}")
        return path

    def _find_projector_test_image(self):
        folder = self.ensure_projector_pattern_folder()
        candidates = [
            os.path.join(folder, "pattern_09.png"),
            os.path.join(folder, "pattern_9.png"),
            os.path.join(folder, "pattern_09.jpg"),
            os.path.join(folder, "pattern_9.jpg"),
        ]
        for candidate in candidates:
            if os.path.isfile(candidate):
                return candidate

        patterns = [
            "pattern_*.png",
            "pattern_*.jpg",
            "*.png",
            "*.jpg",
        ]
        for pattern in patterns:
            matches = sorted(
                glob.glob(os.path.join(folder, pattern)),
                key=lambda item: os.path.basename(item).lower(),
            )
            if matches:
                return matches[0]

        images = self.get_graycode_pattern_files()
        if images:
            return images[0]
        return None

    def _show_projector_window(self, image_path, auto_close_seconds=0):
        if self.projector_slideshow_window is not None and self.projector_slideshow_window.winfo_exists():
            self.projector_slideshow_window.destroy()
        self.projector_slideshow_window = tk.Toplevel(self.root)
        self.projector_slideshow_window.configure(bg="black")
        self.projector_slideshow_window.overrideredirect(True)
        self.projector_slideshow_window.attributes("-topmost", True)
        self._configure_projector_window_geometry()
        self.projector_slideshow_window.update_idletasks()

        label = tk.Label(
            self.projector_slideshow_window,
            bg="black",
            compound="center",
            anchor="center",
        )
        label.pack(fill=tk.BOTH, expand=True)
        self.projector_slideshow_label = label

        try:
            image = Image.open(image_path).convert("RGB")
            width = max(1, self.projector_slideshow_window.winfo_width())
            height = max(1, self.projector_slideshow_window.winfo_height())
            image = image.resize((width, height), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(image)
            label.configure(image=photo)
            label.image = photo
        except Exception:
            label.configure(text=os.path.basename(
                image_path), fg="white", bg="black")

        if auto_close_seconds > 0:
            self.root.after(auto_close_seconds * 1000,
                            self._close_projector_test_window)

    def _close_projector_test_window(self):
        if self.projector_slideshow_window is not None and self.projector_slideshow_window.winfo_exists():
            self.projector_slideshow_window.destroy()
        self.projector_slideshow_window = None
        self.projector_slideshow_label = None
        self.status_var.set("Projector test output closed.")

    def show_projector_test_output(self):
        image_path = self._find_projector_test_image()
        if not image_path:
            self.status_var.set(
                "No projector test image found in graycode_pattern.")
            return

        self.status_var.set(
            f"Testing projector output on {self.projector_monitor_var.get()}.")
        self._show_projector_window(image_path, auto_close_seconds=3)

    def _get_projector_slideshow_interval(self):
        try:
            interval_seconds = float(
                self.projector_slideshow_speed_var.get().strip())
        except (TypeError, ValueError):
            interval_seconds = 3.0
        if interval_seconds < 3.0:
            interval_seconds = 3.0
            self.projector_slideshow_speed_var.set(str(interval_seconds))
        return interval_seconds

    def _capture_projector_photo_delayed(self):
        if not self.projector_slideshow_running:
            return

        current_index = self.projector_slideshow_index
        last_index = len(self.projector_slideshow_images) - 1

        if current_index == 0:
            self.projector_slideshow_index = 1
            interval_seconds = self._get_projector_slideshow_interval()
            self.projector_slideshow_after_id = self.projector_slideshow_window.after(
                int(interval_seconds * 1000),
                self._advance_projector_slideshow,
            )
            return

        try:
            self._capture_projector_photo()
        except Exception as exc:
            self.status_var.set(f"Projector capture error: {exc}")
            self.stop_projector_slideshow()
            return

        if current_index >= last_index:
            self.stop_projector_slideshow()
            return

        self.projector_slideshow_index = current_index + 1
        interval_seconds = self._get_projector_slideshow_interval()
        self.projector_slideshow_after_id = self.projector_slideshow_window.after(
            int(interval_seconds * 1000),
            self._advance_projector_slideshow,
        )

    def _advance_projector_slideshow(self):
        if not self.projector_slideshow_running:
            return
        if not self.projector_slideshow_images:
            self.stop_projector_slideshow()
            return

        image_path = self.projector_slideshow_images[self.projector_slideshow_index]
        self._display_projector_image(image_path)
        self.projector_slideshow_after_id = self.projector_slideshow_window.after(
            1000,
            self._capture_projector_photo_delayed,
        )

    def start_projector_slideshow(self):
        images = self.get_graycode_pattern_files()
        if not images:
            self.status_var.set("No graycode_pattern images found.")
            return

        self.projector_slideshow_images = images
        self.projector_slideshow_index = 0
        self.projector_slideshow_running = True

        if self.projector_slideshow_window is None or not self.projector_slideshow_window.winfo_exists():
            self.projector_slideshow_window = tk.Toplevel(self.root)
            self.projector_slideshow_window.configure(bg="black")
            self.projector_slideshow_window.overrideredirect(True)
            self.projector_slideshow_window.attributes("-topmost", True)
            self.projector_slideshow_label = None
            self._configure_projector_window_geometry()

        self.projector_slideshow_window.deiconify()
        self.projector_slideshow_window.lift()
        self.projector_slideshow_button.config(text="Stop Slideshow")
        self.status_var.set("Projector slideshow started.")
        self._advance_projector_slideshow()

    def stop_projector_slideshow(self):
        self.projector_slideshow_running = False
        if self.projector_slideshow_after_id is not None:
            try:
                if self.projector_slideshow_window is not None and self.projector_slideshow_window.winfo_exists():
                    self.projector_slideshow_window.after_cancel(
                        self.projector_slideshow_after_id)
            except Exception:
                pass
            self.projector_slideshow_after_id = None
        if self.projector_slideshow_window is not None and self.projector_slideshow_window.winfo_exists():
            self.projector_slideshow_window.destroy()
        self.projector_slideshow_window = None
        self.projector_slideshow_label = None
        self.projector_slideshow_button.config(text="Start Slideshow")
        self.status_var.set("Projector slideshow stopped.")

    def toggle_projector_slideshow(self):
        if self.projector_slideshow_running:
            self.stop_projector_slideshow()
        else:
            self.start_projector_slideshow()

    def toggle_osc(self):
        self.osc_expanded = not self.osc_expanded
        if self.osc_expanded:
            self.osc_content.pack(fill=tk.X, pady=(0, 8))
            self.osc_toggle_button.config(text="OSC ▾")
        else:
            self.osc_content.pack_forget()
            self.osc_toggle_button.config(text="OSC ▸")

    def toggle_gallery(self):
        self.gallery_expanded = not self.gallery_expanded
        if self.gallery_expanded:
            self.gallery_content.pack(
                fill=tk.BOTH, expand=True, before=self.osc_toggle_button)
        else:
            self.gallery_content.pack_forget()
        self.update_gallery_button_label()

    def browse_folder(self):
        directory = filedialog.askdirectory(initialdir=self.folder_var.get())
        if directory:
            self.folder_var.set(directory)
            self.refresh_gallery_from_folder()

    def ensure_folder(self):
        folder = self.folder_var.get().strip() or os.path.expanduser(
            "~/Pictures/CameraCapture")
        self.folder_var.set(folder)
        os.makedirs(folder, exist_ok=True)
        return folder

    def get_gallery_image_count(self):
        try:
            folder = self.ensure_folder()
        except AttributeError:
            if not hasattr(self, "folder_var"):
                return 0
            folder = self.folder_var.get().strip() or os.path.expanduser(
                "~/Pictures/CameraCapture")

        if not os.path.isdir(folder):
            return 0

        image_extensions = {".jpg", ".jpeg", ".png",
                            ".bmp", ".gif", ".tif", ".tiff", ".webp"}
        count = 0
        for entry in os.listdir(folder):
            full_path = os.path.join(folder, entry)
            if os.path.isfile(full_path) and os.path.splitext(entry)[1].lower() in image_extensions:
                count += 1
        return count

    def update_gallery_button_label(self):
        if not hasattr(self, "gallery_toggle_button"):
            return
        count = self.get_gallery_image_count()
        indicator = "▾" if self.gallery_expanded else "▸"
        self.gallery_toggle_button.config(
            text=f"Gallery ({count}) {indicator}")

    def refresh_gallery_from_folder(self):
        try:
            folder = self.ensure_folder()
        except AttributeError:
            if not hasattr(self, "folder_var"):
                return
            folder = self.folder_var.get().strip() or os.path.expanduser(
                "~/Pictures/CameraCapture")
        if not hasattr(self, "gallery_frame") or self.gallery_frame is None:
            self.update_gallery_button_label()
            return

        try:
            for child in self.gallery_frame.winfo_children():
                child.destroy()
        except Exception:
            pass

        if not os.path.isdir(folder):
            self.update_gallery_button_label()
            return

        image_extensions = {".jpg", ".jpeg", ".png",
                            ".bmp", ".gif", ".tif", ".tiff", ".webp"}
        existing_files = []
        for entry in os.listdir(folder):
            full_path = os.path.join(folder, entry)
            if os.path.isfile(full_path) and os.path.splitext(entry)[1].lower() in image_extensions:
                existing_files.append(full_path)

        for path in sorted(existing_files, key=os.path.getmtime, reverse=True):
            try:
                self.add_gallery_thumbnail(path)
            except Exception:
                continue

        self.update_gallery_button_label()

    def _configure_calibration_log_tags(self):
        if not hasattr(self, "calibration_output") or self.calibration_output is None:
            return
        for index, (_, color) in enumerate(CALIBRATION_LOG_COLOR_RULES):
            tag_name = f"calib_color_{index}"
            self.calibration_output.tag_configure(tag_name, foreground=color)

    def _append_calibration_output(self, message):
        if not hasattr(self, "calibration_output") or self.calibration_output is None:
            return
        text = str(message).rstrip("\n")
        if not text:
            return
        self.calibration_output.insert(tk.END, text + "\n")

        line_start = self.calibration_output.index("end-1c linestart")
        for index, (pattern, _) in enumerate(CALIBRATION_LOG_COLOR_RULES):
            tag_name = f"calib_color_{index}"
            lower_pattern = pattern.lower()
            lower_text = text.lower()
            search_start = 0
            while True:
                match_index = lower_text.find(lower_pattern, search_start)
                if match_index == -1:
                    break
                start_index = self.calibration_output.index(
                    f"{line_start} + {match_index} chars"
                )
                end_index = self.calibration_output.index(
                    f"{start_index} + {len(pattern)} chars"
                )
                self.calibration_output.tag_add(
                    tag_name, start_index, end_index)
                search_start = match_index + len(pattern)

        self.calibration_output.see(tk.END)

    def _clear_calibration_output(self):
        if not hasattr(self, "calibration_output") or self.calibration_output is None:
            return
        self.calibration_output.delete("1.0", tk.END)

    def browse_camera_parameter_json(self):
        file_path = filedialog.askopenfilename(
            title="Select camera parameter JSON",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialdir=os.path.expanduser("~"),
        )
        if file_path:
            self.camera_json_dir_var.set(file_path)

    def export_unity_xml_from_sidebar(self):
        # call the external script and pass the parameters
        # python exportUnityCalibration.py results/result_projector_calibration.xml results/UnityProjectorCalibration.xml --width 1920 --height 1080 --unit-to-unity 0.001

        proj_width = int(self.calibration_width_var.get().strip())
        proj_height = int(self.calibration_height_var.get().strip())
        # unity_scale = int(self.unity_scale_var.get().strip())
        unity_scale = float(self.unity_scale_slider.get())

        subprocess.run(['python', 'exportUnityCalibration.py', 'results/result_projector_calibration.xml',
                       'results/UnityProjectorCalibration.xml', '--width', str(proj_width), '--height', str(proj_height), '--unit-to-unity', str(unity_scale)])

        self.status_var.set(
            f"Running external script exportUnityCalibration.py {proj_width} || {proj_height} || {unity_scale}")

# changed this so I can call the external script via tkinter button
    # def start_projector_calibration_from_sidebar(self):
    #     # call the external script and pass the parameters
    #     # projectorCalibrate.main(h, w, vert, horiz, size, step, black_thr, white_thr)

    #     proj_width = int(self.calibration_width_var.get().strip())
    #     proj_height = int(self.calibration_height_var.get().strip())
    #     columns = int(self.calibration_columns_var.get())
    #     rows = int(self.calibration_rows_var.get())
    #     square_size = float(self.calibration_square_size_var.get())
    #     graycode_step = int(self.graycode_step_var.get().strip() or "1")
    #     black_thr = int(self.black_threshold_var.get().strip() or "40")
    #     white_thr = int(self.white_threshold_var.get().strip() or "5")
    #     camera_json = self.camera_json_dir_var.get().strip()

    #     # subprocess.run(['python', 'projectorCalibrate.py',
    #     #                '1080', '1920', '8', '5', '29.0', '1'])
    #     # subprocess.run(['python', 'projectorCalibrate.py', str(h), str(w), str(
    #     #     vert), str(horiz), str(size), str(step)])
    #     subprocess.run(['python', 'projectorCalibrate.py',
    #                    str(proj_width), str(proj_height), str(columns), str(rows), str(square_size), str(graycode_step)])

    def start_projector_calibration_from_sidebar(self):
        try:
            capture_dir = Path(
                self.graycode_capture_dir_var.get()).expanduser()
            if not capture_dir.exists():
                raise FileNotFoundError(
                    f"Capture folder does not exist: {capture_dir}")

            proj_width = int(self.calibration_width_var.get().strip())
            proj_height = int(self.calibration_height_var.get().strip())
            columns = int(self.calibration_columns_var.get())
            rows = int(self.calibration_rows_var.get())
            square_size = float(self.calibration_square_size_var.get())
            graycode_step = int(self.graycode_step_var.get().strip() or "1")
            black_thr = int(self.black_threshold_var.get().strip() or "40")
            white_thr = int(self.white_threshold_var.get().strip() or "5")
            camera_json = self.camera_json_dir_var.get().strip()

            if proj_width <= 0 or proj_height <= 0:
                raise ValueError(
                    "Projector pixel width and height must be positive.")
            if columns < 2 or rows < 2:
                raise ValueError(
                    "Board dimensions must be at least 2x2 inner corners.")
            if square_size <= 0:
                raise ValueError("Square size must be greater than zero.")
            if graycode_step <= 0:
                raise ValueError("Graycode step must be greater than zero.")

            self._clear_calibration_output()
            self._append_calibration_output(
                f"Starting structured-light calibration for {capture_dir}...")
            self.status_var.set(
                f"Running projector calibration on {capture_dir}...")

            class _CalibrationLogStream(io.TextIOBase):
                def __init__(self, owner):
                    self.owner = owner

                def write(self, text):
                    if not text:
                        return 0
                    for line in str(text).splitlines(True):
                        self.owner._append_calibration_output(line)
                    return len(text)

                def flush(self):
                    pass

            with contextlib.redirect_stdout(_CalibrationLogStream(self)), contextlib.redirect_stderr(_CalibrationLogStream(self)):
                result = run_structured_light_calibration(
                    capture_dir,
                    (proj_height, proj_width),
                    (columns, rows),
                    square_size,
                    graycode_step,
                    black_thr,
                    white_thr,
                    camera_json,
                )

            if result is not None and isinstance(result, dict):
                self._append_calibration_output(
                    f"Structured-light calibration complete: rms {result['rms']:.4f}, saved to {result['output_file']}"
                )
                self.status_var.set(
                    f"Structured-light calibration complete: rms {result['rms']:.4f}, saved to {result['output_file']}"
                )
            else:
                self._append_calibration_output(
                    "Structured-light calibration complete.")
                self.status_var.set("Structured-light calibration complete.")
        except Exception as exc:
            self._append_calibration_output(f"Calibration failed: {exc}")
            self.status_var.set(f"Calibration failed: {exc}")

    def start_calibration_from_sidebar(self):
        try:
            image_dir = Path(self.ensure_folder()).expanduser()

            # needs change to automatic read of its metadata
            # width_string = self.calibration_width_var.get().strip()
            # height_string = self.calibration_height_var.get().strip()
            width_string = self.foto_width_var.get().strip()
            height_string = self.foto_height_var.get().strip()

            # image = Image.open(path)
            # width, height = image.size

            columns = int(self.calibration_columns_var.get())
            rows = int(self.calibration_rows_var.get())
            square_size = float(self.calibration_square_size_var.get())

            if width_string or height_string:
                if not width_string or not height_string:
                    raise ValueError(
                        "Both picture width and picture height must be filled together."
                    )
                expected_size = (int(width_string), int(height_string))
                if expected_size[0] <= 0 or expected_size[1] <= 0:
                    raise ValueError(
                        "Picture width and height must be positive integers."
                    )
            else:
                expected_size = None

            if not image_dir.exists():
                raise FileNotFoundError(
                    f"Calibration folder does not exist: {image_dir}")
            if columns < 2 or rows < 2:
                raise ValueError(
                    "Board dimensions must be at least 2x2 inner corners.")
            if square_size <= 0:
                raise ValueError("Square size must be greater than zero.")

            self._clear_calibration_output()
            self._append_calibration_output(
                f"Starting calibration for {image_dir}...")
            self.status_var.set(f"Running calibration on {image_dir}...")

            class _CalibrationLogStream(io.TextIOBase):
                def __init__(self, owner):
                    self.owner = owner

                def write(self, text):
                    if not text:
                        return 0
                    for line in str(text).splitlines(True):
                        self.owner._append_calibration_output(line)
                    return len(text)

                def flush(self):
                    pass

            with contextlib.redirect_stdout(_CalibrationLogStream(self)), contextlib.redirect_stderr(_CalibrationLogStream(self)):
                result = run_camera_calibration(
                    image_dir, (columns, rows), square_size,
                    output_file=OUTPUT_FILE,
                    max_view_rmse_px=MAX_VIEW_RMSE_PX,
                    min_views=MIN_VIEWS,
                    expected_size=expected_size,
                )

            self._append_calibration_output(
                f"Calibration complete: rms {result['rms']:.4f}px, saved to {result['output_file']}"
            )
            self.status_var.set(
                f"Calibration complete: rms {result['rms']:.4f}px, saved to {result['output_file']}"
            )
        except Exception as exc:
            self._append_calibration_output(f"Calibration failed: {exc}")
            self.status_var.set(f"Calibration failed: {exc}")

    def validate_interval(self):
        try:
            interval = float(self.capture_interval_var.get())
            if interval <= 0:
                raise ValueError
            self.status_var.set(f"Interval set to {interval:.2f}s")

            try:
                address = self.osc_interval_time_cmd_var.get().strip() or "/camera/interval/time"
                self._send_osc_float(address, interval, port=7001)
            except Exception:
                pass

            return interval
        except (TypeError, ValueError):
            self.status_var.set("Enter a valid positive interval in seconds.")
            return None

    def toggle_interval_capture(self):
        if self.interval_active:
            self.stop_interval_capture()
            return

        interval = self.validate_interval()
        if interval is None:
            return

        self.interval_active = True
        self.interval_button.config(text="Stop interval")
        self.interval_duration = interval
        self.interval_start_time = time.monotonic()
        self._show_interval_progress()
        self.status_var.set(f"Interval capture active: {interval:.2f}s")
        self._schedule_interval_capture(interval)

    def _show_interval_progress(self):
        if self.take_button.winfo_exists():
            self.interval_progress.pack(
                fill=tk.X, pady=(0, 4), before=self.take_button)
            self.interval_label.pack(
                fill=tk.X, pady=(0, 4), before=self.take_button)
        else:
            self.interval_progress.pack(fill=tk.X, pady=(0, 4))
            self.interval_label.pack(fill=tk.X, pady=(0, 4))
        self.interval_progress.configure(value=0)
        self.interval_status_var.set("Next shot in 0.0s")

    def _hide_interval_progress(self):
        try:
            self.interval_progress.pack_forget()
        except Exception:
            pass
        try:
            self.interval_label.pack_forget()
        except Exception:
            pass
        self.interval_status_var.set("")

    def _update_interval_progress(self):
        if not self.interval_active or self.interval_duration <= 0:
            return

        elapsed = time.monotonic() - self.interval_start_time
        progress = min(100.0, max(
            0.0, (elapsed / self.interval_duration) * 100.0))
        self.interval_progress.configure(value=progress)
        remaining = max(0.0, self.interval_duration - elapsed)
        self.interval_status_var.set(f"Next shot in {remaining:.1f}s")

        if elapsed < self.interval_duration:
            self.interval_progress_after_id = self.root.after(
                50,
                self._update_interval_progress,
            )

    def _capture_in_background(self, capture_func, success_callback=None, error_callback=None, *args, **kwargs):
        def worker():
            try:
                with self.capture_lock:
                    result = capture_func(*args, **kwargs)
                if self.root.winfo_exists():
                    self.root.after(0, lambda: self._apply_capture_result(
                        result, success_callback, error_callback))
            except Exception as exc:
                if self.root.winfo_exists():
                    self.root.after(
                        0, lambda: self._apply_capture_error(exc, error_callback))

        threading.Thread(target=worker, daemon=True).start()

    def _apply_capture_result(self, result, success_callback=None, error_callback=None):
        if success_callback is not None:
            success_callback(result)

    def _apply_capture_error(self, exc, error_callback=None):
        if error_callback is not None:
            error_callback(exc)
            return
        self.status_var.set(f"Capture failed: {exc}")

    def _schedule_interval_capture(self, interval_seconds):
        if not self.interval_active:
            return

        self.interval_start_time = time.monotonic()
        self.interval_duration = interval_seconds
        self._update_interval_progress()
        self._capture_in_background(
            self._capture_current_frame_to_disk,
            success_callback=self._handle_capture_result,
            error_callback=self._handle_capture_error,
        )
        self.interval_after_id = self.root.after(
            max(250, int(interval_seconds * 1000)),
            lambda: self._schedule_interval_capture(interval_seconds),
        )

    def stop_interval_capture(self):
        self.interval_active = False
        self.interval_button.config(
            text="Interval Shots", font=("Helvetica", 10))
        if self.interval_after_id is not None:
            try:
                self.root.after_cancel(self.interval_after_id)
            except Exception:
                pass
            self.interval_after_id = None
        if self.interval_progress_after_id is not None:
            try:
                self.root.after_cancel(self.interval_progress_after_id)
            except Exception:
                pass
            self.interval_progress_after_id = None
        self.interval_duration = 0.0
        self.interval_start_time = None
        self._hide_interval_progress()
        self.status_var.set("Interval capture stopped.")

    def _capture_current_frame_to_disk(self):
        selected = self.camera_var.get()

        if selected == "Sony HTTP API":
            return self._capture_sony_image_to_disk()

        if self.latest_frame is None:
            raise RuntimeError("No live frame available yet.")

        folder = self.ensure_folder()
        path = build_capture_path(folder, selected)
        success = cv2.imwrite(path, self.latest_frame)
        if not success:
            raise RuntimeError("Could not save the image.")
        return path

    def _capture_sony_image_to_disk(self):
        result = sony_request("actTakePicture", [""])
        data = result.get("result", [])
        if not data or not data[0]:
            raise RuntimeError("Sony API returned no captured image.")

        image_url = data[0][0]
        with urllib.request.urlopen(image_url, timeout=10) as response:
            image_bytes = response.read()

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        folder = self.ensure_folder()
        path = build_capture_path(folder, "Sony HTTP API")
        success = cv2.imwrite(path, frame)
        if not success:
            raise RuntimeError("Could not save the Sony image.")
        return path

    def _handle_capture_result(self, path):
        self.status_var.set(f"Saved: {path}")
        self.refresh_gallery_from_folder()

    def _handle_capture_error(self, exc):
        self.status_var.set(str(exc))

    def take_picture(self):
        self._capture_in_background(
            self._capture_current_frame_to_disk,
            success_callback=self._handle_capture_result,
            error_callback=self._handle_capture_error,
        )

    def take_sony_picture(self):
        self._capture_in_background(
            self._capture_sony_image_to_disk,
            success_callback=self._handle_capture_result,
            error_callback=self._handle_capture_error,
        )

    def add_gallery_thumbnail(self, path):
        thumbnail_container = tk.Frame(self.gallery_frame, bg="#1a1a1a")
        thumbnail_container.pack(fill=tk.X, pady=2)
        thumbnail_container.grid_columnconfigure(0, weight=0)
        thumbnail_container.grid_columnconfigure(1, weight=1)

        try:
            image = Image.open(path)
            width, height = image.size
            image.thumbnail((90, 70))
            photo = ImageTk.PhotoImage(image)
        except Exception:
            width = height = 0
            photo = None

        thumbnail = tk.Label(thumbnail_container, bg="#1a1a1a", padx=4, pady=4)
        if photo is not None:
            thumbnail.configure(image=photo)
            thumbnail.image = photo
        else:
            thumbnail.configure(text="Image")
        thumbnail.grid(row=0, column=0, sticky="w")

        size_text = f"{width}x{height}" if width and height else "Unknown size"
        size_label = tk.Label(
            thumbnail_container,
            text=size_text,
            bg="#1a1a1a",
            fg="#dfe7f3",
            anchor="w",
            justify=tk.LEFT,
            padx=8,
        )
        size_label.grid(row=0, column=1, sticky="w")

    def zoom_in(self):
        self.zoom_factor = min(self.zoom_factor + 0.1, 2.0)
        self.status_var.set(f"Zoom: {self.zoom_factor:.1f}x")

    def zoom_out(self):
        self.zoom_factor = max(self.zoom_factor - 0.1, 1.0)
        self.status_var.set(f"Zoom: {self.zoom_factor:.1f}x")

    def on_close(self):
        self._cancel_update_loop()
        self.stop_osc_server()
        if self.interval_after_id is not None:
            try:
                self.root.after_cancel(self.interval_after_id)
            except Exception:
                pass
            self.interval_after_id = None
        if self.interval_progress_after_id is not None:
            try:
                self.root.after_cancel(self.interval_progress_after_id)
            except Exception:
                pass
            self.interval_progress_after_id = None
        self.interval_active = False
        self.close_local_source()
        self.close_sony_source()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = CameraCaptureApp(root)
    root.mainloop()
