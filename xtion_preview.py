from openni import openni2
import numpy as np
import cv2
import os
import platform

lib_dir = f"/usr/lib/{platform.machine()}-linux-gnu"
driver_dir = os.path.join(lib_dir, "OpenNI2", "Drivers")
os.environ["OPENNI2_REDIST"] = lib_dir
os.environ["OPENNI2_DRIVERS_PATH"] = driver_dir


PATTERN_SIZE = (8, 5)  # inside corners: columns, rows
MAX_DEPTH_MM = 5000
PATCH_RADIUS = 4       # sample a 9x9 area around the checkerboard center


def depth_at_center(depth_mm, x, y, radius=PATCH_RADIUS):
    """Return median nonzero depth in a small patch, or None if unavailable."""
    h, w = depth_mm.shape
    x, y = int(round(x)), int(round(y))

    x0, x1 = max(0, x - radius), min(w, x + radius + 1)
    y0, y1 = max(0, y - radius), min(h, y + radius + 1)

    patch = depth_mm[y0:y1, x0:x1]
    valid = patch[patch > 0]
    return float(np.median(valid)) if valid.size else None


def main():
    color_stream = None
    depth_stream = None
    initialized = False

    try:
        openni2.initialize(lib_dir)
        initialized = True

        # device = openni2.Device.open_any()

        # # Depth pixels must be aligned with color pixels for sampling the
        # # detected center in the depth image.
        # mode = openni2.IMAGE_REGISTRATION_DEPTH_TO_COLOR
        # if not device.is_image_registration_mode_supported(mode):
        #     raise RuntimeError(
        #         "This device/driver does not support depth-to-color "
        #         "registration; the center depth would not reliably match "
        #         "the detected color-image center."
        #     )
        # device.set_image_registration_mode(mode)

        # color_stream = device.create_color_stream()
        # depth_stream = device.create_depth_stream()
        # color_stream.start()
        # depth_stream.start()

        device = openni2.Device.open_any()

        color_stream = device.create_color_stream()
        depth_stream = device.create_depth_stream()

        mode = openni2.IMAGE_REGISTRATION_DEPTH_TO_COLOR
        if not device.is_image_registration_mode_supported(mode):
            raise RuntimeError("Depth-to-color registration is not supported")

        device.set_image_registration_mode(mode)

        color_stream.start()
        depth_stream.start()

        print("Press q or Esc to quit.")

        while True:
            color_frame = color_stream.read_frame()
            color_bgr = np.frombuffer(
                color_frame.get_buffer_as_uint8(), dtype=np.uint8
            ).reshape(color_frame.height, color_frame.width, 3)
            color_bgr = cv2.cvtColor(color_bgr, cv2.COLOR_RGB2BGR)

            depth_frame = depth_stream.read_frame()
            depth_mm = np.frombuffer(
                depth_frame.get_buffer_as_uint16(), dtype=np.uint16
            ).reshape(depth_frame.height, depth_frame.width)

            if depth_mm.shape != color_bgr.shape[:2]:
                raise RuntimeError(
                    f"Registered depth size {depth_mm.shape} does not match "
                    f"color size {color_bgr.shape[:2]}"
                )

            gray = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2GRAY)
            found, corners = cv2.findChessboardCorners(
                gray,
                PATTERN_SIZE,
                cv2.CALIB_CB_ADAPTIVE_THRESH
                + cv2.CALIB_CB_NORMALIZE_IMAGE,
            )

            display = color_bgr.copy()
            distance_mm = None

            if found:
                # Refine detected corner positions for a more stable center.
                corners = cv2.cornerSubPix(
                    gray,
                    corners,
                    (11, 11),
                    (-1, -1),
                    (
                        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                        30,
                        0.01,
                    ),
                )
                cv2.drawChessboardCorners(
                    display, PATTERN_SIZE, corners, found
                )

                # The average of the inside corners gives the board's image center.
                center_x, center_y = corners.reshape(-1, 2).mean(axis=0)
                distance_mm = depth_at_center(depth_mm, center_x, center_y)

                cx, cy = int(round(center_x)), int(round(center_y))
                cv2.circle(display, (cx, cy), 6, (0, 0, 255), -1)

                if distance_mm is not None:
                    label = f"Center depth: {distance_mm:.0f} mm"
                    print(
                        f"\rCenter depth: {distance_mm:.0f} mm "
                        f"({distance_mm / 1000:.3f} m)   ",
                        end="",
                        flush=True,
                    )
                else:
                    label = "Center depth unavailable"
                    print("\rCenter depth unavailable   ", end="", flush=True)
            else:
                label = "Checkerboard not found"
                print("\rCheckerboard not found      ", end="", flush=True)

            cv2.putText(
                display,
                label,
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            depth_8bit = np.clip(
                depth_mm.astype(np.float32) * (255.0 / MAX_DEPTH_MM),
                0,
                255,
            ).astype(np.uint8)
            depth_view = cv2.applyColorMap(depth_8bit, cv2.COLORMAP_TURBO)
            depth_view[depth_mm == 0] = 0

            cv2.imshow("Checkerboard and center depth", display)
            cv2.imshow("Depth preview", depth_view)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break

    finally:
        if color_stream is not None:
            color_stream.stop()
        if depth_stream is not None:
            depth_stream.stop()
        cv2.destroyAllWindows()
        if initialized:
            openni2.unload()
        print()


if __name__ == "__main__":
    main()
