############################ How to use this script##############
# in terminal
# python app.py <input_filename> <output_filename> -- width <projector_width> --height <projector_height> --unit-to-unity
# example
#   python exportUnityCalibration.py result_projector_calibration.xml UnityProjectorCalibration.xml --width 1920 --height 1080 --unit-to-unity 0.001
# real life example from root folder
# python exportUnityCalibration.py results/result_projector_calibration.xml results/UnityProjectorCalibration.xml --width 1920 --height 1080 --unit-to-unity 0.001
#
# Use 0.001 if the calibration translation is in millimetres and one Unity unit represents one metre; use 0.01 if it is in
# centimetres; use 1.0 if it is already in metres. The XML’s translation uses the calibration’s checkerboard-size units.
# The repo writes the projector intrinsics as proj_int and distortion as proj_dist



import argparse
import xml.etree.ElementTree as ET

import cv2
import numpy as np


def read_matrix(fs, name):
    node = fs.getNode(name)
    if node.empty():
        raise KeyError(f"Required XML node not found: {name}")

    matrix = node.mat()
    if matrix is None:
        raise ValueError(f"XML node '{name}' could not be read as a matrix")

    return np.asarray(matrix, dtype=np.float64)


def fmt(value):
    """Format a number using a dot decimal separator."""
    return format(float(value), ".17g")


def main():
    parser = argparse.ArgumentParser(
        description="Convert procam-calibration XML to Unity-friendly XML."
    )
    parser.add_argument("source_xml", help="procam-calibration result XML")
    parser.add_argument("output_xml", help="Output XML for Unity")
    parser.add_argument("--width", type=int, default=1920,
                        help="Projector output width in pixels")
    parser.add_argument("--height", type=int, default=1080,
                        help="Projector output height in pixels")
    parser.add_argument(
        "--unit-to-unity",
        type=float,
        required=True,
        help="Multiply calibration translation by this to get Unity units"
    )
    args = parser.parse_args()

    fs = cv2.FileStorage(args.source_xml, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise RuntimeError(f"Could not open {args.source_xml}")

    try:
        R = read_matrix(fs, "rotation")
        T = read_matrix(fs, "translation").reshape(3, 1)
        Kp = read_matrix(fs, "proj_int")
        Dp = read_matrix(fs, "proj_dist").reshape(-1)
    finally:
        fs.release()

    if R.shape != (3, 3):
        raise ValueError(f"'rotation' should be 3x3, got {R.shape}")
    if Kp.shape != (3, 3):
        raise ValueError(f"'proj_int' should be 3x3, got {Kp.shape}")

    # OpenCV camera axes: X right, Y down, Z forward.
    # Unity-style axes: X right, Y up, Z forward.
    S = np.diag([1.0, -1.0, 1.0])

    # XML transform maps camera coordinates to projector coordinates:
    # X_projector = R * X_camera + T.
    position_camera_axes = -R.T @ T
    position_unity_axes = S @ position_camera_axes
    position_unity_axes *= args.unit_to_unity

    # Projector-local-to-camera-reference rotation, converted to Unity axes.
    rotation_unity = S @ R.T @ S

    root = ET.Element(
        "projectorCalibration",
        {
            "schemaVersion": "1",
            "projectorWidth": str(args.width),
            "projectorHeight": str(args.height),
        },
    )

    ET.SubElement(
        root,
        "position",
        {
            "x": fmt(position_unity_axes[0, 0]),
            "y": fmt(position_unity_axes[1, 0]),
            "z": fmt(position_unity_axes[2, 0]),
        },
    )

    rotation_element = ET.SubElement(root, "rotation")
    for row in range(3):
        for col in range(3):
            rotation_element.set(
                f"m{row}{col}", fmt(rotation_unity[row, col])
            )

    ET.SubElement(
        root,
        "intrinsics",
        {
            "fx": fmt(Kp[0, 0]),
            "fy": fmt(Kp[1, 1]),
            "cx": fmt(Kp[0, 2]),
            "cy": fmt(Kp[1, 2]),
        },
    )

    distortion_element = ET.SubElement(root, "distortion")
    distortion_element.text = ",".join(fmt(x) for x in Dp)

    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(args.output_xml, encoding="utf-8", xml_declaration=True)

    print(f"Wrote {args.output_xml}")
    print("Projector position relative to the observing camera:")
    print(position_unity_axes.ravel())
    print("Projector Unity-axis rotation:")
    print(rotation_unity)
    print("Projector intrinsics:")
    print(Kp)


if __name__ == "__main__":
    main()
