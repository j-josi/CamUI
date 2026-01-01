# =========================
# System-level imports
# =========================
import os
import io
import json
import time
import re
import glob
import math
import tempfile
import zipfile
import copy
import logging
import threading
import subprocess
import argparse

from typing import Optional, Dict, List
from datetime import datetime, timedelta
from threading import Condition

# =========================
# Picamera2 imports
# =========================
from picamera2 import Picamera2
from picamera2.encoders import JpegEncoder
from picamera2.encoders import H264Encoder
from picamera2.outputs import FileOutput, PyavOutput, FfmpegOutput
from libcamera import Transform, controls

# =========================
# Image handling imports
# =========================
from PIL import (
    Image,
    ImageDraw,
    ImageFont,
    ImageEnhance,
    ImageOps,
    ExifTags,
)

# =========================
# Logging
# =========================
logger = logging.getLogger(__name__)

# =========================
# Constants
# =========================
H264_ENCODER_MAX_VID_RES = (1920, 1080)


class CameraObject:
    def __init__(
        self,
        camera: Dict,
        camera_module_info: Dict,
        upload_folder: str,
        last_config_path: str,
        controls_db_path: str,
        profile_folder: str,
    ) -> None:
        self.camera_info = camera
        self.camera_module_info = camera_module_info
        self.upload_folder = upload_folder
        self.last_config_path = last_config_path
        self.controls_db_path = controls_db_path
        self.profile_folder = profile_folder

        self.lock = threading.Lock()
        self.camera_num: int = self.camera_info["Num"]

        # Generate default camera profile
        self.camera_profile: Dict = self.generate_camera_profile()

        # Initialize Picamera2 with camera number
        self.picam2: Picamera2 = Picamera2(self.camera_num)

        # Camera module specification
        self.camera_module_spec: Optional[Dict] = self.get_camera_module_spec()

        # Sensor modes and resolutions
        self.sensor_modes: List[Dict] = self.picam2.sensor_modes
        self.camera_resolutions: List = self.generate_camera_resolutions()
        self.video_resolutions: List = self.generate_video_resolutions()

        # Stream encoder
        self.encoder_stream: H264Encoder = H264Encoder(bitrate=8_000_000)
        self.encoder_stream.audio = False
        self.encoder_stream.audio_output = {"codec_name": "libopus"}
        self.encoder_stream.audio_sync = 0

        # Stream output (processed by MediaMTX)
        self.output_stream: PyavOutput = PyavOutput(
            f"rtsp://127.0.0.1:8554/cam{self.camera_num}",
            format="rtsp",
        )

        # Stream activity flag
        self.active_stream: bool = False

        # Recording encoder
        self.encoder_recording: H264Encoder = H264Encoder(bitrate=8_000_000)
        self.output_recording = None
        self.filename_recording: Optional[str] = None
        self.active_recording: bool = False

        # Initialize camera configurations
        self.init_configure_camera()

        # Initialize and sanitize control definitions
        self.live_controls: Dict = self.initialize_controls_template(
            self.get_ui_controls(),
            self.controls_db_path,
        )

        # Capture flags
        self.active_capture_still: bool = False

        # Audio (placeholder for future support)
        self.audio: bool = False

        self.main_stream: str = "recording"
        self.lores_stream: str = "streaming"

        # Load saved camera profile if available
        self.load_saved_camera_profile()

        # Start streaming
        self.start_streaming()

        # Debug information
        logger.debug("Available Camera Controls: %s", self.get_ui_controls())
        logger.debug("Available Sensor Modes: %s", self.sensor_modes)
        logger.debug("Available Resolutions: %s", self.camera_resolutions)
        logger.info("Final Camera Profile: %s", self.camera_profile)

    # ------------------------------------------------------------------
    # Camera configuration functions
    # ------------------------------------------------------------------

    def get_ui_controls(self) -> Dict:
        """
        Copy camera controls from picamera2.camera_controls (inherited from
        libcamera) and crop them to reasonable UI ranges.
        """
        ui_controls: Dict = copy.deepcopy(self.picam2.camera_controls)

        if "ExposureTime" in ui_controls:
            min_exposure_time = 100        # microseconds
            max_exposure_time = 100_000    # microseconds
            default_exposure_time = 500
            ui_controls["ExposureTime"] = (
                min_exposure_time,
                max_exposure_time,
                default_exposure_time,
            )

        if "ColourTemperature" in ui_controls:
            min_color_temp = 100           # Kelvin
            max_color_temp = 10_000        # Kelvin
            default_color_temp = None
            ui_controls["ColourTemperature"] = (
                min_color_temp,
                max_color_temp,
                default_color_temp,
            )

        return ui_controls

    def init_configure_camera(self) -> None:
        self.still_config: Dict = self.picam2.create_still_configuration()
        self.video_config: Dict = self.picam2.create_video_configuration()

    def load_saved_camera_profile(self) -> None:
        """Load the saved camera profile if available."""
        logger.debug("load_saved_camera_profile")

        if self.camera_info.get("Has_Config") and self.camera_info.get("Config_Location"):
            self.load_camera_profile(self.camera_info["Config_Location"])

    def load_camera_profile(self, profile_filename: str) -> bool:
        """Load and apply a camera profile from a file."""
        profile_path = os.path.join(self.profile_folder, profile_filename)

        if not os.path.exists(profile_path):
            logger.warning("Profile file not found: %s", profile_path)
            return False

        try:
            with open(profile_path, "r") as f:
                profile_data = json.load(f)

            self.camera_profile = profile_data

            self.update_settings(
                "recording_resolution",
                self.camera_profile["resolutions"]["recording_resolution"],
                True,
            )
            self.update_settings(
                "streaming_resolution",
                self.camera_profile["resolutions"]["streaming_resolution"],
                True,
            )
            self.update_settings(
                "StillCaptureResolution",
                self.camera_profile["resolutions"]["StillCaptureResolution"],
                True,
            )
            self.update_settings("hflip", self.camera_profile["hflip"], True)
            self.update_settings("vflip", self.camera_profile["vflip"], True)
            self.update_settings("saveRAW", self.camera_profile["saveRAW"], True)

            self.reconfigure_video_pipeline()
            self.apply_profile_controls()
            self.sync_live_controls()

            # Update camera-last-config.json
            try:
                if os.path.exists(self.last_config_path):
                    with open(self.last_config_path, "r") as f:
                        last_config = json.load(f)
                else:
                    last_config = {"cameras": []}

                updated = False
                for camera in last_config["cameras"]:
                    if camera["Num"] == self.camera_num:
                        camera["Has_Config"] = True
                        camera["Config_Location"] = profile_filename
                        updated = True
                        break

                if not updated:
                    logger.warning(
                        "Camera %s not found in camera-last-config.json",
                        self.camera_num,
                    )

                with open(self.last_config_path, "w") as f:
                    json.dump(last_config, f, indent=4)

                logger.info(
                    "Loaded profile '%s' and updated camera-last-config.json",
                    profile_filename,
                )

            except Exception as e:
                logger.error(
                    "Error updating camera-last-config.json: %s",
                    e,
                    exc_info=True,
                )

            logger.info("Successfully loaded profile %s", profile_path)
            return True

        except Exception as e:
            logger.error(
                "Error loading camera profile '%s': %s",
                profile_filename,
                e,
                exc_info=True,
            )
            return False

    def generate_camera_profile(self) -> Dict:
        file_name = os.path.join(self.profile_folder, "camera-module-info.json")

        if not self.camera_info.get("Has_Config", False) or not os.path.exists(file_name):
            self.camera_profile = {
                "hflip": 0,
                "vflip": 0,
                "sensor_mode": 0,
                "model": self.camera_info.get("Model", "Unknown"),
                "resolutions": {
                    "StillCaptureResolution": 0,
                    "recording_resolution": 0,
                    "streaming_resolution": 0,
                },
                "saveRAW": False,
                "controls": {},
            }
        else:
            with open(file_name, "r") as file:
                self.camera_profile = json.load(file)

        return self.camera_profile

    def initialize_controls_template(
        self,
        picamera2_controls: Dict,
        controls_db_path: str,
    ) -> Dict:
        if os.path.isfile(controls_db_path):
            try:
                with open(controls_db_path, "r") as f:
                    cam_ctrl_json = json.load(f)
            except Exception as e:
                logger.error(
                    "Failed to extract JSON data from '%s': %s",
                    controls_db_path,
                    e,
                    exc_info=True,
                )
                return {}

            if "sections" not in cam_ctrl_json:
                logger.error("'sections' key not found in cam_ctrl_json!")
                return cam_ctrl_json
        else:
            logger.error("Controls DB file does not exist: %s", controls_db_path)
            return {}

        # Initialize empty controls in camera_profile
        self.camera_profile["controls"] = {}

        for section in cam_ctrl_json["sections"]:
            if "settings" not in section:
                logger.warning(
                    "Missing 'settings' key in section: %s",
                    section.get("title", "Unknown"),
                )
                continue

            section_enabled: bool = False

            for setting in section["settings"]:
                if not isinstance(setting, dict):
                    logger.warning("Unexpected setting format: %s", setting)
                    continue

                setting_id: Optional[str] = setting.get("id")
                source: Optional[str] = setting.get("source")
                original_enabled: bool = setting.get("enabled", False)

                if source == "controls":
                    if setting_id in picamera2_controls:
                        min_val, max_val, default_val = picamera2_controls[setting_id]

                        logger.debug(
                            "Updating control %s: min=%s max=%s default=%s",
                            setting_id,
                            min_val,
                            max_val,
                            default_val,
                        )

                        setting["min"] = min_val
                        setting["max"] = max_val

                        if default_val is not None:
                            setting["default"] = default_val
                        else:
                            default_val = False if isinstance(min_val, bool) else min_val

                        if setting.get("enabled"):
                            self.camera_profile["controls"][setting_id] = default_val

                        setting["enabled"] = original_enabled

                        if original_enabled:
                            section_enabled = True
                    else:
                        logger.debug(
                            "Disabling control %s: not found in picamera2_controls",
                            setting_id,
                        )
                        setting["enabled"] = False

                elif source == "generatedresolutions":
                    resolution_options = [
                        {
                            "value": i,
                            "label": f"{w} x {h}",
                            "enabled": True,
                        }
                        for i, (w, h) in enumerate(self.camera_resolutions)
                    ]
                    setting["options"] = resolution_options
                    section_enabled = True

                    logger.debug(
                        "Updated %s with generated resolutions",
                        setting_id,
                    )

                elif source == "video_resolutions":
                    resolution_options = [
                        {
                            "value": i,
                            "label": f"{w} x {h}",
                            "enabled": True,
                        }
                        for i, (w, h) in enumerate(self.video_resolutions)
                    ]
                    setting["options"] = resolution_options
                    section_enabled = True

                else:
                    logger.debug(
                        "Skipping %s: no source specified, keeping existing values",
                        setting_id,
                    )
                    section_enabled = True

                if "childsettings" in setting:
                    for child in setting["childsettings"]:
                        child_id: Optional[str] = child.get("id")
                        child_source: Optional[str] = child.get("source")

                        if (
                            child_source == "controls"
                            and child_id in picamera2_controls
                        ):
                            min_val, max_val, default_val = picamera2_controls[child_id]

                            logger.debug(
                                "Updating child control %s: min=%s max=%s default=%s",
                                child_id,
                                min_val,
                                max_val,
                                default_val,
                            )

                            child["min"] = min_val
                            child["max"] = max_val

                            self.camera_profile["controls"][child_id] = (
                                default_val if default_val is not None else min_val
                            )

                            if default_val is not None:
                                child["default"] = default_val

                            child["enabled"] = child.get("enabled", False)

                            if child["enabled"]:
                                section_enabled = True
                        else:
                            logger.debug(
                                "Skipping or disabling child setting %s",
                                child_id,
                            )

            section["enabled"] = section_enabled

        logger.debug(
            "Initialized camera_profile controls: %s",
            self.camera_profile,
        )
        return cam_ctrl_json

    def update_settings(
        self,
        setting_id: str,
        setting_value,
        init: bool = False,
    ):
        if setting_id in ["hflip", "vflip"]:
            try:
                self.camera_profile[setting_id] = bool(int(setting_value))

                if not init:
                    self.reconfigure_video_pipeline()

                logger.info(
                    "Applied transform: %s -> %s",
                    setting_id,
                    setting_value,
                )
            except ValueError as e:
                logger.error("Invalid value for %s: %s", setting_id, e)

        elif setting_id == "StillCaptureResolution":
            self.camera_profile["resolutions"][setting_id] = int(setting_value)

            logger.info(
                "Applied StillCaptureResolution index %s -> %s",
                setting_value,
                self.camera_resolutions[int(setting_value)],
            )

        elif setting_id == "recording_resolution":
            self.set_recording_resolution(int(setting_value))
            self.camera_profile["resolutions"][setting_id] = int(setting_value)

            logger.info(
                "Applied recording_resolution index %s -> %s",
                setting_value,
                self.video_resolutions[int(setting_value)],
            )

            if not init:
                self.reconfigure_video_pipeline()

        elif setting_id == "streaming_resolution":
            self.set_streaming_resolution(int(setting_value))
            self.camera_profile["resolutions"][setting_id] = int(setting_value)

            logger.info(
                "Applied streaming_resolution index %s -> %s",
                setting_value,
                self.video_resolutions[int(setting_value)],
            )

            if not init:
                self.reconfigure_video_pipeline()

        elif setting_id == "saveRAW":
            self.camera_profile[setting_id] = setting_value
            logger.info("Applied setting: %s -> %s", setting_id, setting_value)

        else:
            if "." in str(setting_value):
                setting_value = float(setting_value)
            else:
                setting_value = int(setting_value)

            self.picam2.set_controls({setting_id: setting_value})
            self.camera_profile.setdefault("controls", {})[setting_id] = setting_value

        updated: bool = False

        for section in self.live_controls.get("sections", []):
            for setting in section.get("settings", []):
                if setting["id"] == setting_id:
                    setting["value"] = setting_value
                    updated = True
                    break

                for child in setting.get("childsettings", []):
                    if child["id"] == setting_id:
                        child["value"] = setting_value
                        updated = True
                        break

            if updated:
                break

        if not updated:
            logger.warning(
                "Setting %s not found in live_controls",
                setting_id,
            )

        return setting_value

    def sync_live_controls(self) -> None:
        """Sync live_controls with camera_profile values."""
        for section in self.live_controls.get("sections", []):
            for setting in section.get("settings", []):
                setting_id = setting["id"]

                if setting_id in self.camera_profile["controls"]:
                    setting["value"] = self.camera_profile["controls"][setting_id]

                for child in setting.get("childsettings", []):
                    child_id = child["id"]
                    if child_id in self.camera_profile["controls"]:
                        child["value"] = self.camera_profile["controls"][child_id]

        logger.debug("Live controls synced with camera profile")

    def apply_profile_controls(self) -> None:
        if "controls" in self.camera_profile:
            try:
                for setting_id, setting_value in self.camera_profile["controls"].items():
                    self.picam2.set_controls({setting_id: setting_value})
                    self.update_settings(setting_id, setting_value)
                    logger.debug(
                        "Applied profile control: %s -> %s",
                        setting_id,
                        setting_value,
                    )

                logger.info("All profile controls applied successfully")
            except Exception as e:
                logger.error(
                    "Error applying profile controls: %s",
                    e,
                    exc_info=True,
                )

    def generate_video_resolutions(self) -> List:
        resolutions = set()

        for mode in self.sensor_modes:
            sw, sh = mode["size"]

            for w, h in [
                (1920, 1080),
                (1536, 864),
                (1280, 720),
                (1152, 648),
                (768, 432),
            ]:
                if (
                    w <= H264_ENCODER_MAX_VID_RES[0]
                    and h <= H264_ENCODER_MAX_VID_RES[1]
                    and w <= sw
                    and h <= sh
                ):
                    resolutions.add((w, h))

        return sorted(resolutions, reverse=True)

    def find_best_sensor_mode(self, video_resolution: tuple) -> Dict:
        tw, th = video_resolution

        candidates = [
            mode
            for mode in self.sensor_modes
            if mode["size"][0] >= tw and mode["size"][1] >= th
        ]

        if not candidates:
            raise ValueError("No suitable sensor mode found")

        # Prioritize smallest suitable resolution
        return min(candidates, key=lambda m: m["size"][0] * m["size"][1])


    def reconfigure_video_pipeline(self) -> bool:
        if self.active_capture_still or self.active_recording:
            logger.warning(
                "Failed to reconfigure video pipeline: active still capture or recording"
            )
            return False

        rec_index: int = self.camera_profile["resolutions"]["recording_resolution"]
        stream_index: int = self.camera_profile["resolutions"]["streaming_resolution"]

        rec = self.video_resolutions[rec_index]
        stream = self.video_resolutions[stream_index]

        # define main and lores streams
        if rec[0] * rec[1] >= stream[0] * stream[1]:
            main_size = rec
            lores_size = stream
            self.main_stream = "recording"
            self.lores_stream = "streaming"
        else:
            main_size = stream
            lores_size = rec
            self.main_stream = "streaming"
            self.lores_stream = "recording"

        # determine appropriate sensor mode
        mode = self.find_best_sensor_mode(main_size)

        was_streaming = self.active_stream
        if was_streaming:
            self.stop_streaming()

        with self.lock:
            self.picam2.stop()

            transform = Transform()
            transform.hflip = bool(self.camera_profile.get("hflip", False))
            transform.vflip = bool(self.camera_profile.get("vflip", False))

            self.video_config = self.picam2.create_video_configuration(
                main={"size": main_size},
                lores={"size": lores_size},
                transform=transform,
                sensor={
                    "output_size": mode["size"],
                    "bit_depth": mode["bit_depth"]
                },
            )

            self.picam2.configure(self.video_config)
            self.picam2.start()

        if was_streaming:
            self.start_streaming()

        # update sensor mode in profile
        self.camera_profile["sensor_mode"] = self.sensor_modes.index(mode)

        logger.info("✅ Video pipeline reconfigured")
        logger.info("   main:  %s (%s)", main_size, self.main_stream)
        logger.info("   lores: %s (%s)", lores_size, self.lores_stream)

        return True

    def get_recording_stream(self) -> str:
        return "main" if self.main_stream == "recording" else "lores"

    def get_streaming_stream(self) -> str:
        return "main" if self.main_stream == "streaming" else "lores"

    def get_recording_resolution(self) -> tuple:
        return self.video_resolutions[
            self.camera_profile["resolutions"]["recording_resolution"]
        ]

    def get_streaming_resolution(self) -> tuple:
        return self.video_resolutions[
            self.camera_profile["resolutions"]["streaming_resolution"]
        ]

    def set_recording_resolution(self, resolution_index: int) -> None:
        self.camera_profile["resolutions"]["recording_resolution"] = int(resolution_index)
        self.reconfigure_video_pipeline()

    def set_streaming_resolution(self, resolution_index: int) -> None:
        self.camera_profile["resolutions"]["streaming_resolution"] = int(resolution_index)
        self.reconfigure_video_pipeline()

    def create_profile(self, filename: str) -> bool:
        """Save the current camera profile and update camera-last-config.json."""
        try:
            logger.debug("Saving camera profile: %s", self.camera_profile)

            if filename.lower().endswith(".json"):
                filename = filename[:-5]

            profile_path = os.path.join(self.profile_folder, f"{filename}.json")

            with open(profile_path, "w") as f:
                json.dump(self.camera_profile, f, indent=4)

            # Update camera-last-config.json
            try:
                if os.path.exists(self.last_config_path):
                    with open(self.last_config_path, "r") as f:
                        last_config = json.load(f)
                else:
                    last_config = {"cameras": []}

                updated = False
                for camera in last_config["cameras"]:
                    if camera["Num"] == self.camera_num:
                        camera["Has_Config"] = True
                        camera["Config_Location"] = f"{filename}.json"
                        updated = True
                        break

                if not updated:
                    logger.warning(
                        "Camera %s not found in camera-last-config.json", self.camera_num
                    )

                with open(self.last_config_path, "w") as f:
                    json.dump(last_config, f, indent=4)

                logger.info(
                    "Updated camera-last-config.json for camera %s", self.camera_num
                )
            except Exception as e:
                logger.error("Error updating camera-last-config.json: %s", e, exc_info=True)

            return True
        except Exception as e:
            logger.error("Error saving profile: %s", e, exc_info=True)
            return False

    def reset_to_default(self) -> bool:
        """Reset camera settings to default values and apply them."""
        self.camera_profile = {
            "hflip": 0,
            "vflip": 0,
            "model": self.camera_info.get("Model", "Unknown"),
            "resolutions": {
                "StillCaptureResolution": 0,
                "recording_resolution": 0,
                "streaming_resolution": 0,
            },
            "saveRAW": False,
            "controls": {
                "AfMode": 0,
                "LensPosition": 1.0,
                "AfRange": 0,
                "AfSpeed": 0,
                "ExposureTime": 33000,
                "AnalogueGain": 1.1228070259094238,
                "AeEnable": 1,
                "ExposureValue": 0.0,
                "AeConstraintMode": 0,
                "AeExposureMode": 0,
                "AeMeteringMode": 0,
                "AeFlickerMode": 0,
                "AeFlickerPeriod": 100,
                "AwbEnable": 0,
                "AwbMode": 0,
                "Brightness": 0,
                "Contrast": 1.0,
                "Saturation": 1.0,
                "Sharpness": 1.0,
                "ColourTemperature": 4000,
            },
        }

        # Apply video resolutions and orientation
        self.reconfigure_video_pipeline()

        # Apply UI and camera controls
        self.update_settings("saveRAW", self.camera_profile["saveRAW"])
        self.apply_profile_controls()

        logger.info("Camera profile reset to default and settings applied")
        return True

    #-----
    # Camera Information Functions
    #-----

    def capture_metadata(self) -> dict:
        self.metadata = self.picam2.capture_metadata()
        logger.debug("Sensor resolution: %s", self.picam2.sensor_resolution)
        return self.metadata

    def get_camera_module_spec(self) -> Optional[dict]:
        """Return camera module details for this camera."""
        camera_module = next(
            (
                cam
                for cam in self.camera_module_info.get("camera_modules", [])
                if cam["sensor_model"] == self.camera_info["Model"]
            ),
            None,
        )
        return camera_module

    def get_sensor_mode(self) -> Optional[int]:
        """Return the index of the currently active sensor mode."""
        current_config = self.picam2.camera_configuration()
        active_mode = current_config.get("sensor", {})
        active_mode_index: Optional[int] = None

        for index, mode in enumerate(self.sensor_modes):
            if (
                mode["size"] == active_mode.get("output_size")
                and mode["bit_depth"] == active_mode.get("bit_depth")
            ):
                active_mode_index = index
                break

        logger.info("Active Sensor Mode: %s", active_mode_index)
        return active_mode_index

    def generate_camera_resolutions(self) -> List[tuple]:
        """Precompute available resolutions based on sensor modes."""
        if not self.sensor_modes:
            logger.warning("No sensor modes available!")
            return []

        resolutions = sorted(
            set(mode["size"] for mode in self.sensor_modes if "size" in mode),
            reverse=True,
        )

        if not resolutions:
            logger.warning("No valid resolutions found in sensor modes!")
            return []

        max_resolution = resolutions[0]
        aspect_ratio = max_resolution[0] / max_resolution[1]

        extra_resolutions = []
        for i in range(len(resolutions) - 1):
            w1, h1 = resolutions[i]
            w2, h2 = resolutions[i + 1]
            midpoint = ((w1 + w2) // 2, (h1 + h2) // 2)
            extra_resolutions.append(midpoint)

        last_w, last_h = resolutions[-1]
        half_res = (last_w // 2, last_h // 2)
        inbetween_res = ((last_w + half_res[0]) // 2, (last_h + half_res[1]) // 2)

        resolutions.extend(extra_resolutions)
        resolutions.append(inbetween_res)
        resolutions.append(half_res)

        self.available_resolutions = sorted(set(resolutions), reverse=True)
        return self.available_resolutions

    #-----
    # Camera Streaming Functions
    #-----

    def start_streaming(self) -> None:
        if self.active_stream:
            logger.info("Skip starting stream, already active")
            return

        stream_name = self.get_streaming_stream()
        with self.lock:
            self.picam2.start_recording(
                self.encoder_stream, output=self.output_stream, name=stream_name
            )
            self.active_stream = True
            logger.info("Streaming started on '%s' stream", stream_name)

    def stop_streaming(self) -> None:
        if not self.active_stream:
            logger.info("Skip stopping stream, no active stream")
            return

        with self.lock:
            self.picam2.stop_recording()
            self.active_stream = False
            logger.info("Streaming stopped")

    #-----
    # Camera Recording Functions
    #-----

    def start_recording(self, filename_recording: str) -> tuple[bool, Optional[str]]:
        if self.active_recording:
            logger.info("Skip starting recording, already active")
            return False, None

        stream_name = self.get_recording_stream()
        filepath = os.path.join(self.upload_folder, filename_recording)
        output_recording = FfmpegOutput(filepath, audio=self.audio)

        try:
            with self.lock:
                self.picam2.start_recording(
                    self.encoder_recording, output_recording, name=stream_name
                )
                self.output_recording = output_recording
                self.active_recording = True
                self.filename_recording = filename_recording
                logger.info("Recording '%s' started", filename_recording)
                return True, filename_recording
        except Exception as e:
            logger.error("Failed to start recording '%s': %s", filename_recording, e, exc_info=True)
            return False, None

    def stop_recording(self) -> bool:
        if not self.active_recording:
            logger.info("Skip stopping recording, no active recording")
            return False

        with self.lock:
            self.picam2.stop_recording()
            self.output_recording = None
            self.active_recording = False
            self.active_stream = False
            logger.info("Recording '%s' stopped", self.filename_recording)

        self.start_streaming()
        return True

    #-----
    # Camera Capture Functions
    #-----

    def capture_still(self, filename: str, raw: bool = False) -> Optional[str]:
        with self.lock:
            if self.active_capture_still:
                logger.warning(
                    "Skip capturing still image '%s', another capture is active", filepath
                )
                return False
            self.active_capture_still = True

        was_streaming = self.active_stream
        filepath = os.path.join(self.upload_folder, filename)

        still_index = self.camera_profile["resolutions"]["StillCaptureResolution"]
        still_resolution = self.camera_resolutions[still_index]

        rec_index = self.camera_profile["resolutions"]["recording_resolution"]
        recording_resolution = self.video_resolutions[rec_index]

        # choose sensor mode for capture
        if still_resolution[0] * still_resolution[1] >= recording_resolution[0] * recording_resolution[1]:
            mode = self.find_best_sensor_mode(still_resolution)
        else:
            mode = self.find_best_sensor_mode(recording_resolution)

        still_config = self.picam2.create_still_configuration(
            buffer_count=1,
            main={"size": still_resolution},
            sensor={"output_size": mode["size"], "bit_depth": mode["bit_depth"]},
            controls={"FrameDurationLimits": (100, 10_000_000_000)},
        )

        self.stop_recording()
        self.stop_streaming()

        with self.lock:
            self.picam2.stop()
            self.picam2.configure(still_config)
            self.camera_profile["sensor_mode"] = self.sensor_modes.index(mode)
            self.picam2.start()

            try:
                if raw:
                    logger.debug("Capturing raw image '%s'", filepath)
                    buffers, metadata = self.picam2.switch_mode_and_capture_buffers(
                        still_config, ["main", "raw"]
                    )
                    self.picam2.helpers.save(
                        self.picam2.helpers.make_image(buffers[0], still_config["main"]),
                        metadata,
                        filepath,
                    )
                    self.picam2.helpers.save_dng(buffers[1], metadata, still_config["raw"], filepath)
                else:
                    logger.debug("Capturing non-raw image '%s'", filepath)
                    buffers, metadata = self.picam2.switch_mode_and_capture_buffers(still_config, ["main"])
                    self.picam2.helpers.save(
                        self.picam2.helpers.make_image(buffers[0], still_config["main"]),
                        metadata,
                        filepath,
                    )
            finally:
                self.active_stream = was_streaming
                self.active_capture_still = False

        logger.info("Successfully captured image '%s'", filepath)
        return True

    def capture_still_from_feed(self, filename: str) -> Optional[str]:
        try:
            filepath = os.path.join(self.upload_folder, filename)
            request = self.picam2.capture_request()
            request.save("main", f"{filepath}")
            logger.info("Image captured successfully. Path: %s", filepath)
            return True
        except Exception as e:
            logger.error("Error capturing image '%s': %s", filepath, e, exc_info=True)
            return False