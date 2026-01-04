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
    """
    DOMAIN OBJECT

    - hält Canonical Camera State
    - kennt KEIN WebSocket / HTTP / Flask
    - erzeugt interne State-Events über _on_state_changed()
    """

    # ---------------------------------------------------
    # INIT
    # ---------------------------------------------------

    def __init__(
        self,
        camera: Dict,
        camera_module_info: Dict,
        upload_folder: str,
        camera_active_profile_path: str,
        controls_db_path: str,
        profile_folder: str,
    ) -> None:

        self.camera_info = camera
        self.camera_module_info = camera_module_info
        self.camera_num: int = camera["Num"]
        self.upload_folder = upload_folder
        self.camera_active_profile_path = camera_active_profile_path
        self.controls_db_path = controls_db_path
        self.profile_folder = profile_folder

        # temp
        self.camera_profile = {}


        self.lock = threading.Lock()

        # ------------------------------------------------
        # Canonical STATE (Single Source of Truth)
        # ------------------------------------------------
        self._state: Dict = {
            "camera_num": self.camera_num,
            "active_stream": False,
            "active_recording": False,
            "active_capture_still": False,
            "info": {
                "model": self.camera_info.get("Model"),
            },
            "controls": {},
            "config": {
                "hflip": False,
                "vflip": False,
                "saveRAW": False,
                "sensor_mode": None,
                "still_capture_resolution": 0,
                "recording_resolution": 0,
                "streaming_resolution": 0,
            },
        }

        # ------------------------------------------------
        # Camera init
        # ------------------------------------------------
        self.picam2 = Picamera2(self.camera_num)
        self.sensor_modes = self.picam2.sensor_modes
        self.camera_resolutions = self._generate_camera_resolutions()
        self.video_resolutions = self._generate_video_resolutions()

        # ------------------------------------------------
        # Encoders
        # ------------------------------------------------
        self.encoder_stream = H264Encoder(bitrate=8_000_000)
        self.encoder_stream.audio = False

        self.output_stream = PyavOutput(
            f"rtsp://127.0.0.1:8554/cam{self.camera_num}",
            format="rtsp",
        )

        self.encoder_recording = H264Encoder(bitrate=8_000_000)
        self.output_recording = None
        self.filename_recording = None

        self.main_stream = "recording"
        self.lores_stream = "streaming"

        # ------------------------------------------------
        # Config & Controls
        # ------------------------------------------------

        # Initialize and sanitize control definitions
        self.live_controls = self._init_controls_from_db(
            self.get_ui_controls(),
            self.controls_db_path,
        )

        # Load last profile (sets state through setters!)
        self.load_active_profile()

        # Initialize video configuration (stream and recording)
        self.reconfigure_video_pipeline()

        # Sync actual camera values
        self._sync_controls_from_camera()

        # Start stream
        self.start_streaming()

        # Debug information
        logger.debug("Available Camera Controls: %s", self.get_ui_controls())
        logger.debug("Available Sensor Modes: %s", self.sensor_modes)
        logger.debug("Available Resolutions: %s", self.camera_resolutions)

    # ===================================================
    # STATE API (ONLY mutation entry points)
    # ===================================================

    def set_state(self, key: str, value):
        if self._state.get(key) == value:
            return False
        self._state[key] = value
        self._on_state_changed()
        return True

    def set_control(self, name: str, value):
        current = self._state["controls"].get(name)
        if current == value:
            return False

        # Convert value to valid/supported type
        control_type = type(current) if current is not None else None
        if control_type is not None:
            try:
                if control_type is int:
                    value = int(value)
                elif control_type is float:
                    value = float(value)
                elif control_type is bool:
                    value = bool(value)
            except ValueError:
                logger.warning("Failed to convert value for control %s: %s", name, value)

        self.picam2.set_controls({name: value})
        self._state["controls"][name] = value
        self._on_state_changed()
        return True

    def get_state(self) -> Dict:
        return copy.deepcopy(self._state)

    def _on_state_changed(self):
        """
        Hook for infrastructure layer.
        Re-bound in camera_manager.py.
        """
        pass

    # ===================================================
    # CONTROLS / SYNC
    # ===================================================

    def _sync_controls_from_camera(self):
        try:
            metadata = self.picam2.capture_metadata()
            for key in self.picam2.camera_controls:
                if key in metadata:
                    self._state["controls"][key] = metadata[key]
        except Exception as e:
            logger.warning("Control sync failed: %s", e)

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

    def _init_camera_configuration(self):
        self.video_config = self.picam2.create_video_configuration()
        self.picam2.configure(self.video_config)
        self.picam2.start()

    def load_profile(self, profile_filename: str) -> bool:
        """Load and apply a camera profile created via save_profile().
        Model mismatch is allowed and is handled by the frontend.
        """
        profile_path = os.path.join(self.profile_folder, profile_filename)
        if not os.path.exists(profile_path):
            logger.warning("Profile file not found: %s", profile_path)
            return False

        try:
            with open(profile_path, "r") as f:
                profile = json.load(f)

            # Iterate over all top-level keys in the profile
            for top_key, sub_dict in profile.items():
                if isinstance(sub_dict, dict):
                    # Update existing keys in _state
                    if top_key not in self._state:
                        self._state[top_key] = {}
                    for key, value in sub_dict.items():
                        self._state[top_key][key] = value
                else:
                    # If the value is not a dict, assign it directly
                    self._state[top_key] = sub_dict

            # Reconfigure video pipeline and sync controls after loading
            self.reconfigure_video_pipeline()
            self.sync_live_controls()
            self._update_active_profile(profile_filename)
            logger.info("Camera profile '%s' loaded successfully", profile_filename)
            return True

        except Exception as e:
            logger.error("Error loading camera profile '%s': %s", profile_filename, e, exc_info=True)
            return False

    def load_active_profile(self) -> None:
        """Load the active camera profile if configured."""

        if not self.camera_info.get("Has_Config"):
            return

        profile_filename = self.camera_info.get("Config_Location")
        if profile_filename:
            self.load_profile(profile_filename)

    def save_profile(self, filename: str) -> bool:
        try:
            if filename.lower().endswith(".json"):
                filename = filename[:-5]

            profile_path = os.path.join(self.profile_folder, f"{filename}.json")
            state = self.get_state()

            profile = {
                "info": dict(self._state.get("info", {})),
                "config": dict(self._state.get("config", {})),
                "controls": dict(self._state.get("controls", {})),
            }

            with open(profile_path, "w") as f:
                json.dump(profile, f, indent=2)

            self._update_active_profile(f"{filename}.json")
            logger.info("Camera profile saved: %s", profile_path)
            return True

        except Exception as e:
            logger.error("Error saving profile: %s", e, exc_info=True)
            return False


    def _update_active_profile(self, profile_filename: str) -> None:
        """Update camera-active-profile.json with the latest profile."""
        try:
            active_config = {"cameras": []}
            if os.path.exists(self.camera_active_profile_path):
                with open(self.camera_active_profile_path, "r") as f:
                    active_config = json.load(f)

            for camera in active_config.get("cameras", []):
                if camera.get("Num") == self.camera_num:
                    camera["Has_Config"] = True
                    camera["Config_Location"] = profile_filename
                    break

            with open(self.camera_active_profile_path, "w") as f:
                json.dump(active_config, f, indent=4)

        except Exception as e:
            logger.error("Failed to update camera-active-profile.json: %s", e, exc_info=True)

    # def generate_profile(self) -> Dict:
    #     file_name = os.path.join(self.profile_folder, "camera-module-info.json")

    #     if not self.camera_info.get("Has_Config", False) or not os.path.exists(file_name):
    #         self.camera_profile = {
    #             "hflip": 0,
    #             "vflip": 0,
    #             "sensor_mode": 0,
    #             "model": self.camera_info.get("Model", "Unknown"),
    #             "resolutions": {
    #                 "still_capture_resolution": 0,
    #                 "recording_resolution": 0,
    #                 "streaming_resolution": 0,
    #             },
    #             "saveRAW": False,
    #             "controls": {},
    #         }
    #     else:
    #         with open(file_name, "r") as file:
    #             self.camera_profile = json.load(file)

    #     return self.camera_profile

    def _init_controls_from_db(
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

    # def update_settings(self, setting_id: str, setting_value, init: bool = False):
    #     """Update a camera setting or control in STATE."""
    #     try:
    #         if setting_id in ("hflip", "vflip"):
    #             self.set_state(setting_id, bool(setting_value))
    #             if not init:
    #                 self.reconfigure_video_pipeline()
    #             logger.info("Applied transform: %s -> %s", setting_id, setting_value)

    #         elif setting_id == "saveRAW":
    #             self.set_state(setting_id, bool(setting_value))
    #             logger.info("Applied setting: %s -> %s", setting_id, setting_value)

    #         elif setting_id in ("still_capture_resolution", "recording_resolution", "streaming_resolution"):
    #             self._state["config"][setting_id] = int(setting_value)
    #             if setting_id in ("recording_resolution", "streaming_resolution") and not init:
    #                 self.reconfigure_video_pipeline()
    #             logger.info("Applied resolution %s -> %s", setting_id, setting_value)

    #         else:
    #             # convert setting_value for camera controls from string to numeric value (int or float)
    #             if isinstance(setting_value, str) and "." in setting_value:
    #                 setting_value = float(setting_value)
    #             elif isinstance(setting_value, (int, float)):
    #                 pass
    #             elif isinstance(setting_value, bool):
    #                 pass
    #             else:
    #                 try:
    #                     setting_value = int(setting_value)
    #                 except Exception:
    #                     logger.warning("Cannot convert setting_value '%s' to int/float", setting_value)

    #             self.set_control(setting_id, setting_value)
    #             logger.info("Applied control %s -> %s", setting_id, setting_value)

    #         self.sync_live_controls()
    #         return setting_value

    #     except Exception as e:
    #         logger.error("Error updating setting '%s' with value '%s': %s", setting_id, setting_value, e)
    #         return None

    def sync_live_controls(self) -> None:
        """Sync live_controls with STATE controls."""
        for section in self.live_controls.get("sections", []):
            for setting in section.get("settings", []):
                setting_id = setting["id"]
                if setting_id in self._state["controls"]:
                    setting["value"] = self._state["controls"][setting_id]

                for child in setting.get("childsettings", []):
                    child_id = child["id"]
                    if child_id in self._state["controls"]:
                        child["value"] = self._state["controls"][child_id]

        logger.debug("Live controls synced with STATE")

    def apply_state_controls(self):
        """Apply all controls from STATE to the actual camera."""
        controls = self._state.get("controls", {})
        try:
            for ctrl, val in controls.items():
                self.picam2.set_controls({ctrl: val})
                logger.debug("Applied control from STATE: %s -> %s", ctrl, val)

            self.sync_live_controls()
            logger.info("All profile controls applied successfully")
        except Exception as e:
            logger.error("Error applying profile controls: %s", e, exc_info=True)

    def _generate_video_resolutions(self) -> List:
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

    def _find_best_sensor_mode(self, video_resolution: tuple) -> Dict:
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
        """Reconfigure video pipeline based on current STATE."""
        if self._state["active_recording"] or self._state["active_capture_still"]:
            return False

        rec = self.video_resolutions[self._state["config"]["recording_resolution"]]
        stream = self.video_resolutions[self._state["config"]["streaming_resolution"]]

        main_size, lores_size = (rec, stream) if rec[0]*rec[1] >= stream[0]*stream[1] else (stream, rec)
        self.main_stream, self.lores_stream = ("recording", "streaming") if rec[0]*rec[1] >= stream[0]*stream[1] else ("streaming", "recording")

        mode = self._find_best_sensor_mode(main_size)
        was_streaming = self._state["active_stream"]
        if was_streaming:
            self.stop_streaming()

        with self.lock:
            self.picam2.stop()
            self.picam2.configure(self.picam2.create_video_configuration(
                main={"size": main_size},
                lores={"size": lores_size},
                transform=Transform(hflip=self._state["config"]["hflip"], vflip=self._state["config"]["vflip"]),
                sensor={"output_size": mode["size"], "bit_depth": mode["bit_depth"]}
            ))
            self.picam2.start()

        if was_streaming:
            self.start_streaming()

        self._state["config"]["sensor_mode"] = self.sensor_modes.index(mode)
        logger.info("Video pipeline reconfigured: main=%s lores=%s", main_size, lores_size)
        return True

    def get_recording_stream(self) -> str:
        return "main" if self.main_stream == "recording" else "lores"

    def get_streaming_stream(self) -> str:
        return "main" if self.main_stream == "streaming" else "lores"

    def get_recording_resolution(self) -> tuple:
        return self.video_resolutions[self._state["config"]["recording_resolution"]]

    def get_streaming_resolution(self) -> tuple:
        return self.video_resolutions[self._state["config"]["streaming_resolution"]]

    def set_recording_resolution(self, resolution_index: int) -> None:
        self._state["config"]["recording_resolution"] = int(resolution_index)
        self.reconfigure_video_pipeline()

    def set_streaming_resolution(self, resolution_index: int) -> None:
        self._state["config"]["streaming_resolution"] = int(resolution_index)
        self.reconfigure_video_pipeline()

    def reset_to_default(self) -> bool:
        # TODO: fix this function for websocket
        """Reset camera settings to default STATE and apply them."""
        self._state.update({
            "config": {
                "hflip": False,
                "vflip": False,
                "saveRAW": False,
                "sensor_mode": None,
                "still_capture_resolution": 0,
                "recording_resolution": 0,
                "streaming_resolution": 0,
            },
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
        })

        # self.apply_state_controls()
        self.reconfigure_video_pipeline()
        # self.update_settings("saveRAW", self._state["saveRAW"])
        logger.info("Camera STATE reset to default and settings applied")
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
        try:
            current_config = self.picam2.camera_configuration()
            active_mode = current_config.get("sensor", {})

            for index, mode in enumerate(self.sensor_modes):
                if (
                    mode["size"] == active_mode.get("output_size")
                    and mode["bit_depth"] == active_mode.get("bit_depth")
                ):
                    logger.info("Active Sensor Mode: %s", index)
                    return index

            logger.info("No matching active sensor mode found")
            return None

        except Exception as e:
            logger.error("Error retrieving sensor mode: %s", e, exc_info=True)
            return None

    def _generate_camera_resolutions(self) -> List[tuple]:
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

    def start_streaming(self):
        with self.lock:
            if self._state["active_stream"]:
                logger.info("Skip starting stream, already active")
                return

            stream_name = self.get_streaming_stream()
            self.picam2.start_recording(
                self.encoder_stream,
                output=self.output_stream,
                name=stream_name,
            )
            self.set_state("active_stream", True)
            logger.info("Streaming started on '%s' stream", stream_name)

    def stop_streaming(self):
        with self.lock:
            if not self._state["active_stream"]:
                logger.info("Skip stopping stream, no active stream")
                return

            self.picam2.stop_recording()
            self.set_state("active_stream", False)
            logger.info("Streaming stopped")

    def start_recording(self, filename: str):
        with self.lock:
            if self._state["active_recording"]:
                logger.info("Skip starting recording, already active")
                return False, None

            path = os.path.join(self.upload_folder, filename)
            output = FfmpegOutput(path)

            self.picam2.start_recording(
                self.encoder_recording,
                output,
                name=self.get_recording_stream(),
            )

            self.output_recording = output
            self.filename_recording = filename
            self.set_state("active_recording", True)
            logger.info(f"Recording {filename} started")
            return True, filename

    def stop_recording(self):
        with self.lock:
            if not self._state["active_recording"]:
                logger.info("Skip stopping recording, no active recording")
                return False

            _filename = self.filename_recording

            # Stoppe Recording
            self.picam2.stop_recording()
            self.output_recording = None
            self.filename_recording = None
            self.set_state("active_recording", False)
            logger.info(f"Recording {_filename} stopped")

        return True

    #-----
    # Camera Capture Functions
    #-----
    def capture_still(self, filename: str, raw: bool = False) -> Optional[str]:
        filepath = os.path.join(self.upload_folder, filename)

        # Sperre gleich zu Beginn für alle _state-Zugriffe
        with self.lock:
            if self._state["active_capture_still"]:
                logger.warning(
                    "Skip capturing still image '%s', another capture is active", filename
                )
                return None
            # Markiere Capture als aktiv
            self._state["active_capture_still"] = True
            was_streaming = self._state["active_stream"]

        try:
            # Auflösungen ermitteln
            still_index = self._state["config"]["still_capture_resolution"]
            still_resolution = self.camera_resolutions[still_index]

            rec_index = self._state["config"]["recording_resolution"]
            recording_resolution = self.video_resolutions[rec_index]

            if still_resolution[0] * still_resolution[1] >= recording_resolution[0] * recording_resolution[1]:
                mode = self._find_best_sensor_mode(still_resolution)
            else:
                mode = self._find_best_sensor_mode(recording_resolution)

            still_config = self.picam2.create_still_configuration(
                buffer_count=1,
                main={"size": still_resolution},
                sensor={"output_size": mode["size"], "bit_depth": mode["bit_depth"]},
                controls={"FrameDurationLimits": (100, 10_000_000_000)},
            )

            # Aufnahme/Stream stoppen, ohne _state zu ändern
            self.stop_recording()
            self.stop_streaming()

            # Kamera konfigurieren
            with self.lock:
                self.picam2.stop()
                self.picam2.configure(still_config)
                self._state["config"]["sensor_mode"] = self.sensor_modes.index(mode)
                self.picam2.start()

            # Aufnahme durchführen
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

            logger.info("Successfully captured image '%s'", filepath)
            return filepath

        except Exception as e:
            logger.error("Error capturing still image '%s': %s", filepath, e, exc_info=True)
            return None

        finally:
            # Lock nur für _state-Update verwenden
            with self.lock:
                self._state["active_capture_still"] = False

            # Streaming außerhalb des Locks starten (kann blockieren)
            if was_streaming:
                self.start_streaming()

    def capture_still_from_feed(self, filename: str) -> Optional[str]:
        try:
            filepath = os.path.join(self.upload_folder, filename)
            request = self.picam2.capture_request()
            request.save("main", filepath)
            logger.info("Image captured successfully: %s", filepath)
            return filepath
        except Exception as e:
            logger.error("Error capturing image '%s': %s", filename, e, exc_info=True)
            return None