import os
import struct
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

from camera_app import (
    BOARD_SIZE,
    CameraCaptureApp,
    build_capture_path,
    enumerate_camera_indexes,
    find_capture_directories,
    generate_graycode_pattern_images,
    run_camera_calibration,
)


class DummyRoot:
    def __init__(self):
        self.calls = []

    def winfo_exists(self):
        return True

    def after(self, delay, callback):
        self.calls.append((delay, callback))
        return 1


class CameraAppTest(unittest.TestCase):
    def test_graycode_fallback_decodes_pixel_coordinates(self):
        width, height = 8, 8
        graycode = __import__(
            "camera_app").GrayCodePatternFallback(width, height)
        images = []

        for bit in range(3):
            image = np.zeros((height, width), dtype=np.uint8)
            for y in range(height):
                for x in range(width):
                    pattern_value = x ^ (x >> 1)
                    image[y, x] = 255 if (
                        (pattern_value >> (3 - 1 - bit)) & 1) else 0
            images.append(image)

        for bit in range(3):
            image = np.zeros((height, width), dtype=np.uint8)
            for y in range(height):
                for x in range(width):
                    pattern_value = y ^ (y >> 1)
                    image[y, x] = 255 if (
                        (pattern_value >> (3 - 1 - bit)) & 1) else 0
            images.append(image)

        ok, pixel = graycode.getProjPixel(images, 5, 7)
        self.assertTrue(ok)
        self.assertEqual(pixel, (5.0, 7.0))

    def test_generate_graycode_pattern_images_creates_expected_file_count(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            files = generate_graycode_pattern_images(temp_dir, 1280, 720, 1)
            expected = 11 + 10 + 2
            self.assertEqual(len(files), expected)
            self.assertTrue(any(os.path.basename(
                path).startswith("pattern_") for path in files))

    def test_sidebar_column_layout_does_not_exceed_available_width(self):
        layout = CameraCaptureApp._calculate_sidebar_columns(1200)
        self.assertLessEqual(layout["total_width"], 1200)
        self.assertGreaterEqual(layout["column_width"], 180)

    def test_enumerate_camera_indexes(self):
        devices = enumerate_camera_indexes(limit=4)
        self.assertIsInstance(devices, list)
        self.assertTrue(all(isinstance(item, int) for item in devices))

    def test_enumerate_camera_indexes_ignores_unopenable_devices(self):
        with mock.patch("camera_app.Path.glob") as glob_mock, mock.patch(
            "camera_app.cv2.VideoCapture"
        ) as video_capture_ctor:
            glob_mock.return_value = [
                Path("/sys/class/video4linux/video0"),
                Path("/sys/class/video4linux/video1"),
            ]

            def fake_capture(index):
                cap = mock.Mock()
                cap.isOpened.return_value = index == 1
                cap.get.side_effect = lambda prop: (
                    1920 if prop == cv2.CAP_PROP_FRAME_WIDTH else 1080
                )
                return cap

            video_capture_ctor.side_effect = fake_capture

            self.assertEqual(enumerate_camera_indexes(limit=4), [1])

    def test_set_interval_from_osc_updates_active_interval(self):
        app = object.__new__(CameraCaptureApp)
        app.root = DummyRoot()
        app.capture_interval_var = type(
            "CaptureVar",
            (),
            {
                "__init__": lambda self: setattr(self, "value", "2.0"),
                "get": lambda self: self.value,
                "set": lambda self, value: setattr(self, "value", str(value)),
            },
        )()
        app.status_var = type(
            "StatusVar",
            (),
            {"__init__": lambda self: setattr(
                self, "value", "Ready"), "set": lambda self, value: setattr(self, "value", value)},
        )()
        app.interval_active = True
        app.interval_duration = 2.0
        app.interval_start_time = 0.0
        app.interval_after_id = 99
        app.interval_progress_after_id = 100
        app._show_interval_progress = lambda: None
        app._update_interval_progress = lambda: None
        app._schedule_interval_capture = lambda interval: setattr(
            app, "scheduled_interval", interval)

        app.set_interval_from_osc(4.5)

        self.assertEqual(app.capture_interval_var.get(), "4.5")
        self.assertEqual(app.interval_duration, 4.5)
        self.assertEqual(app.scheduled_interval, 4.5)
        self.assertTrue(app.interval_start_time > 0)

    def test_toggle_osc_can_expand_multiple_times(self):
        app = object.__new__(CameraCaptureApp)
        app.osc_expanded = False

        class FakeWidget:
            def __init__(self):
                self.packed = False
                self.forget_count = 0

            def pack(self, *args, **kwargs):
                self.packed = True

            def pack_forget(self):
                self.forget_count += 1
                self.packed = False

            def config(self, *args, **kwargs):
                pass

        app.osc_content = FakeWidget()
        app.osc_toggle_button = FakeWidget()
        app.status_label = FakeWidget()

        app.toggle_osc()
        app.toggle_osc()
        app.toggle_osc()

        self.assertEqual(app.osc_expanded, True)
        self.assertEqual(app.osc_content.packed, True)
        self.assertEqual(app.osc_content.forget_count, 1)

    def test_build_capture_path_uses_folder_and_camera_name(self):
        folder = os.path.join("tmp_capture_test", "photos")
        path = build_capture_path(folder, "cam_0")
        self.assertTrue(path.endswith(".jpg"))
        self.assertTrue(path.startswith(folder))
        self.assertIn("cam_0", os.path.basename(path))

    def test_interval_capture_is_scheduled_without_blocking_the_ui(self):
        app = object.__new__(CameraCaptureApp)
        app.root = DummyRoot()
        app.interval_active = True
        app.interval_duration = 0.0
        app.interval_start_time = 0.0
        app.interval_after_id = None
        app.interval_progress_after_id = None
        app.take_picture_called = False

        def fake_take_picture():
            app.take_picture_called = True

        app.take_picture = fake_take_picture
        app._update_interval_progress = lambda: None

        app._schedule_interval_capture(2.0)

        self.assertFalse(app.take_picture_called)
        self.assertGreaterEqual(len(app.root.calls), 1)

    def test_osc_packet_parses_capture_and_interval_commands(self):
        app = object.__new__(CameraCaptureApp)

        def make_packet(address, *args):
            data = address.encode("utf-8") + b"\x00"
            data += b"\x00" * ((4 - len(data) % 4) % 4)
            tags = "," + "".join({int: "i", float: "f",
                                 str: "s"}.get(type(arg), "s") for arg in args)
            data += tags.encode("utf-8") + b"\x00"
            data += b"\x00" * ((4 - len(data) % 4) % 4)
            for arg in args:
                if isinstance(arg, int):
                    data += struct.pack(">i", arg)
                elif isinstance(arg, float):
                    data += struct.pack(">f", arg)
                else:
                    value = str(arg).encode("utf-8") + b"\x00"
                    data += value
                    data += b"\x00" * ((4 - len(data) % 4) % 4)
            return data

        message = app._parse_osc_message(
            make_packet("/camera/interval/time", 3.5))
        self.assertEqual(message["address"], "/camera/interval/time")
        self.assertEqual(message["args"][0], 3.5)

        message = app._parse_osc_message(make_packet("/camera/take_picture"))
        self.assertEqual(message["address"], "/camera/take_picture")
        self.assertEqual(message["args"], [])

    def test_refresh_gallery_from_folder_includes_existing_images(self):
        app = object.__new__(CameraCaptureApp)
        app.gallery_frame = type(
            "GalleryFrame",
            (),
            {
                "widgets": [],
                "pack": lambda self, *args, **kwargs: None,
                "winfo_children": lambda self: [],
            },
        )()
        app.folder_var = type(
            "FolderVar",
            (),
            {
                "get": lambda self: os.path.join("tmp_gallery_test", "photos"),
                "set": lambda self, value: setattr(self, "value", value),
            },
        )()

        os.makedirs(app.folder_var.get(), exist_ok=True)
        existing_image = os.path.join(app.folder_var.get(), "existing.jpg")
        with open(existing_image, "wb") as fh:
            fh.write(b"not-a-real-jpg")

        app.refresh_gallery_from_folder()

        self.assertTrue(os.path.exists(existing_image))

    def test_projector_slideshow_interval_clamps_to_minimum(self):
        app = object.__new__(CameraCaptureApp)
        values = {"value": "1.5"}
        app.projector_slideshow_speed_var = type(
            "SpeedVar",
            (),
            {
                "get": lambda self: values["value"],
                "set": lambda self, value: values.__setitem__("value", str(value)),
            },
        )()

        self.assertEqual(app._get_projector_slideshow_interval(), 3.0)
        self.assertEqual(values["value"], "3.0")

    def test_projector_slideshow_skips_first_image_and_keeps_last_capture(self):
        app = object.__new__(CameraCaptureApp)
        app.projector_slideshow_running = True
        app.projector_slideshow_images = [
            "img_1.png", "img_2.png", "img_3.png"]
        app.projector_slideshow_index = 0
        app.projector_slideshow_speed_var = type(
            "SpeedVar",
            (),
            {"get": lambda self: "3.0", "set": lambda self, value: None},
        )()
        app.status_var = type(
            "StatusVar",
            (),
            {"set": lambda self, value: None},
        )()
        app.projector_slideshow_window = type(
            "Window",
            (),
            {"after": mock.Mock(return_value=42)},
        )()
        app._capture_projector_photo = mock.Mock()
        app.stop_projector_slideshow = mock.Mock()

        app._capture_projector_photo_delayed()

        app._capture_projector_photo.assert_not_called()
        app.stop_projector_slideshow.assert_not_called()
        self.assertEqual(app.projector_slideshow_index, 1)
        app.projector_slideshow_window.after.assert_called_once()

        app.projector_slideshow_index = 2
        app._capture_projector_photo_delayed()

        app._capture_projector_photo.assert_called_once()
        app.stop_projector_slideshow.assert_called_once()

    def test_validate_interval_sends_osc_time_to_port_7001(self):
        app = object.__new__(CameraCaptureApp)
        app.capture_interval_var = type(
            "CaptureVar",
            (),
            {"get": lambda self: "3.5", "set": lambda self, value: None},
        )()
        app.status_var = type(
            "StatusVar",
            (),
            {"set": lambda self, value: None},
        )()
        app.osc_bind_ip_var = type(
            "OscHostVar",
            (),
            {"get": lambda self: "127.0.0.1"},
        )()
        app.osc_interval_time_cmd_var = type(
            "OscCmdVar",
            (),
            {"get": lambda self: "/camera/interval/time"},
        )()

        class FakeSocket:
            def __init__(self):
                self.sent = []

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def sendto(self, data, target):
                self.sent.append((data, target))

        with mock.patch("camera_app.socket.socket", return_value=FakeSocket()) as socket_ctor:
            result = app.validate_interval()
            self.assertEqual(result, 3.5)
            socket_ctor.return_value.sent
            self.assertEqual(socket_ctor.return_value.sent[0][1][1], 7001)
            self.assertIn(b"/camera/interval/time\x00",
                          socket_ctor.return_value.sent[0][0])

    def test_get_camera_resolution_options_uses_detected_supported_sizes(self):
        app = object.__new__(CameraCaptureApp)
        app.camera_var = type(
            "CameraVar",
            (),
            {"get": lambda self: "Camera 0"},
        )()

        class FakeCap:
            def __init__(self, width, height):
                self.width = width
                self.height = height
                self.opened = True

            def isOpened(self):
                return self.opened

            def set(self, prop, value):
                if prop == cv2.CAP_PROP_FRAME_WIDTH:
                    self.width = int(value)
                elif prop == cv2.CAP_PROP_FRAME_HEIGHT:
                    self.height = int(value)
                return True

            def get(self, prop):
                if prop == cv2.CAP_PROP_FRAME_WIDTH:
                    return float(self.width)
                if prop == cv2.CAP_PROP_FRAME_HEIGHT:
                    return float(self.height)
                return 0.0

            def release(self):
                self.opened = False

        with mock.patch("camera_app.cv2.VideoCapture") as video_capture_ctor:
            def fake_capture(_index):
                return FakeCap(1920, 1080)

            video_capture_ctor.side_effect = fake_capture

            options = app.get_camera_resolution_options(0)

        self.assertIn("1920x1080", options)
        self.assertIn("1280x720", options)
        self.assertIn("640x480", options)

    def test_run_camera_calibration_uses_expected_image_size(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            image_dir = Path(tmp_dir)
            (image_dir / "calib_1.jpg").write_bytes(b"fake")

            dummy = np.zeros((720, 1280, 3), dtype=np.uint8)
            corners = np.zeros(
                (BOARD_SIZE[0] * BOARD_SIZE[1], 2), dtype=np.float32)

            with mock.patch("camera_app.cv2.imread", return_value=dummy), mock.patch(
                "camera_app.detect_corners", return_value=(True, corners)
            ), mock.patch(
                "camera_app.fit_camera",
                return_value=(
                    0.1,
                    np.eye(3),
                    np.zeros((5,)),
                    [np.zeros((1, 1, 3), dtype=np.float32)],
                    [np.zeros((1, 1, 3), dtype=np.float32)],
                ),
            ), mock.patch(
                "camera_app.per_view_rmse",
                return_value=np.asarray([0.1]),
            ):
                result = run_camera_calibration(
                    image_dir,
                    board_size=(BOARD_SIZE[0], BOARD_SIZE[1]),
                    square_size=29.0,
                    output_file=image_dir / "calibration.npz",
                    min_views=1,
                    expected_size=(1280, 720),
                )

        self.assertEqual(result["image_size"], (1280, 720))

    def test_calibration_log_output_appends_messages(self):
        app = object.__new__(CameraCaptureApp)

        class FakeLogWidget:
            def __init__(self):
                self.value = ""
                self.state = "normal"

            def configure(self, *args, **kwargs):
                if "state" in kwargs:
                    self.state = kwargs["state"]

            def insert(self, index, text):
                self.value += text

            def index(self, position):
                if position == "end-1c linestart":
                    return "1.0"
                return "1.0"

            def tag_add(self, *args, **kwargs):
                pass

            def see(self, *args, **kwargs):
                pass

            def delete(self, *args, **kwargs):
                self.value = ""

        app.calibration_output = FakeLogWidget()

        app._append_calibration_output("Processing image 1")
        app._append_calibration_output("Processing image 2")

        self.assertIn("Processing image 1", app.calibration_output.value)
        self.assertIn("Processing image 2", app.calibration_output.value)
        self.assertEqual(app.calibration_output.state, "normal")

    def test_get_attached_monitors_parses_xrandr_output(self):
        app = object.__new__(CameraCaptureApp)

        fake_result = mock.Mock()
        fake_result.returncode = 0
        fake_result.stdout = (
            "Monitors: 2\n"
            " 0: +*HDMI-0 1920/509x1080/286+0+0  HDMI-0\n"
            " 1: +DP-1 2560/677x1440/381+1920+0  DP-1\n"
        )

        with mock.patch("camera_app.subprocess.run", return_value=fake_result):
            monitors = app.get_attached_monitors()

        self.assertIn("HDMI-0", monitors)
        self.assertEqual(monitors["HDMI-0"], (0, 0, 1920, 1080))
        self.assertIn("DP-1", monitors)
        self.assertEqual(monitors["DP-1"], (1920, 0, 2560, 1440))

    def test_get_next_graycode_capture_index_counts_existing_files(self):
        app = object.__new__(CameraCaptureApp)

        class FakeStringVar:
            def __init__(self, value=""):
                self.value = value

            def get(self):
                return self.value

            def set(self, value):
                self.value = value

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir)
            (path / "graycode_00.png").write_bytes(b"x")
            (path / "graycode_01.png").write_bytes(b"x")
            (path / "other.txt").write_text("n")
            app.projector_capture_dir_var = FakeStringVar(str(path))
            self.assertEqual(app.get_next_graycode_capture_index(), 2)

    def test_find_projector_capture_dirs_lists_capture_prefix_folders(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            capture_root = Path(tmp_dir)
            (capture_root / "capture_01").mkdir()
            (capture_root / "capture_01" / "graycode_00.png").write_bytes(b"x")
            (capture_root / "capture_02").mkdir()
            (capture_root / "capture_02" / "graycode_00.png").write_bytes(b"x")
            (capture_root / "capture_03").mkdir()
            (capture_root / "capture_03" / "notes.txt").write_text("n")

            dirs = find_capture_directories(capture_root)

        self.assertEqual([str(path.name)
                         for path in dirs], ["capture_01", "capture_02"])


if __name__ == "__main__":
    unittest.main()
