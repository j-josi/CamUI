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

from typing import Optional, Dict, List, Tuple, Union, Any
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

    # Registry of all CameraObject instances: {camera_num: CameraObject}
    cameras: Dict[int, "CameraObject"] = {}

    # -----------------------------
    # DEFAULT CONTROLS
    # -----------------------------
    DEFAULT_CONTROLS = {
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
    }

    # ---------------------------------------------------
    # INIT
    # ---------------------------------------------------

    def __init__(
        self,
        camera: Dict,
        camera_module_info: Dict,
        upload_folder: str,
        camera_ui_settings_db_path: str,
    ) -> None:

        self.camera_info = camera
        self.camera_module_info = camera_module_info
        self.camera_num: int = camera["Num"]
        self.upload_folder = upload_folder
        self.ui_settings_db_path = camera_ui_settings_db_path

        self.lock = threading.Lock()

        self.infos = {
            "model": self.camera_info.get("Model"),
        }

        self.configs = {
            "hflip": False,
            "vflip": False,
            "saveRAW": False,
            "sensor_mode": None,
            "still_capture_resolution": 0,
            "recording_resolution": 0,
            "streaming_resolution": 0,
        }

        self.controls = {}

        self.states = {
            "is_video_streaming": False,
            "is_video_recording": False,
            "is_capturing_still_image": False,
        }

        # ------------------------------------------------
        # Camera init
        # ------------------------------------------------
        self.picam2 = Picamera2(self.camera_num)
        CameraObject.cameras[self.camera_num] = self

        # Hardware-derived capabilities
        self.sensor_modes_supported = self.picam2.sensor_modes
        self.still_resolutions_supported = self._generate_still_resolutions_supported()
        self.video_resolutions_supported = self._generate_video_resolutions_supported()

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

        # Initialize and sanitize control definitions for UI / Frontend schema
        self.ui_settings = self._init_ui_settings_from_db(
            self._get_picam_control_capabilities(),
            self.ui_settings_db_path,
        )

        # Load last profile (sets state through setters!)
        # self.load_active_profile()

        # Initialize video configuration (stream and recording)
        self.reconfigure_video_pipeline()

        # Sync actual camera values
        self._sync_controls_from_camera()

        # Start stream
        self.start_streaming()

        # Debug information
        logger.debug("Available Camera Controls: %s", self._get_picam_control_capabilities())
        logger.debug("Available Sensor Modes: %s", self.sensor_modes_supported)
        logger.debug("Available Resolutions: %s", self.still_resolutions_supported)

    # ===================================================
    # STATE API (ONLY mutation entry points)
    # ===================================================

    def _set_state(self, name: str, value: bool) -> None:
        """Set a boolean state (internal). Calls _on_state_changed() if changed."""
        with self.lock:
            if self.states.get(name) != value:
                self.states[name] = value
                self._on_state_changed()

    def _coerce_control_value(self, name: str, value):
        """Convert incoming control values to the type expected by Picamera2."""
        current = self.controls.get(name)

        if current is None:
            return value

        try:
            if isinstance(current, bool):
                return bool(int(value)) if isinstance(value, str) else bool(value)
            if isinstance(current, int):
                return int(value)
            if isinstance(current, float):
                return float(value)
        except (ValueError, TypeError):
            logger.warning(
                "Failed to coerce control '%s' value '%s' to type %s",
                name,
                value,
                type(current),
            )

        return value

    def get_settings(self) -> Dict:
        return {
            "infos": copy.deepcopy(self.infos),
            "configs": copy.deepcopy(self.configs),
            "controls": copy.deepcopy(self.controls),
            "states": copy.deepcopy(self.states),
        }

    # def set_control(self, name: str, value) -> bool:
    #     """Set a single camera control (live). Returns True if updated."""
    #     with self.lock:
    #         current = self.controls.get(name)

    #         if current == value:
    #             return False

    #         # Convert value to the same type as current
    #         control_type = type(current) if current is not None else None
    #         if control_type is not None:
    #             try:
    #                 if control_type is int:
    #                     value = int(value)
    #                 elif control_type is float:
    #                     value = float(value)
    #                 elif control_type is bool:
    #                     value = bool(value)
    #             except ValueError:
    #                 logger.warning("Failed to convert value for control %s: %s", name, value)

    #         try:
    #             self.picam2.set_controls({name: value})
    #             self.controls[name] = value
    #             self._on_state_changed()
    #             return True
    #         except Exception as e:
    #             logger.error("Failed to set control %s: %s", name, e, exc_info=True)
    #             return False
    
    def set_control(self, name, value=None) -> bool:
        """
        Set camera control(s).

        - set_control("ExposureTime", 30000)
        - set_control({ "ExposureTime": 30000, "AnalogueGain": 2.0 })

        Returns True if at least one control was updated.
        """

        # -----------------------------
        # BULK MODE
        # -----------------------------
        if isinstance(name, dict):
            updated = False
            controls_to_apply = {}

            with self.lock:
                for ctrl_name, raw_value in name.items():
                    if ctrl_name not in self.controls:
                        logger.warning(
                            "Attempted to set unknown camera control '%s'", ctrl_name
                        )
                        continue

                    coerced = self._coerce_control_value(ctrl_name, raw_value)
                    current = self.controls.get(ctrl_name)

                    # Skip if value is unchanged
                    if current == coerced:
                        continue

                    controls_to_apply[ctrl_name] = coerced

                if not controls_to_apply:
                    return False

                try:
                    self.picam2.set_controls(controls_to_apply)
                    self.controls.update(controls_to_apply)
                    updated = True
                except Exception as exc:
                    logger.error(
                        "Failed to apply bulk camera controls %s: %s",
                        controls_to_apply,
                        exc,
                        exc_info=True,
                    )
                    return False

            if updated:
                self._on_state_changed()

            return updated

        # -----------------------------
        # SINGLE MODE
        # -----------------------------
        if not isinstance(name, str):
            raise TypeError("set_control expects str or dict")

        with self.lock:
            if name not in self.controls:
                logger.warning("Attempted to set unknown camera control '%s'", name)
                return False

            coerced = self._coerce_control_value(name, value)
            current = self.controls.get(name)

            if current == coerced:
                return False

            try:
                self.picam2.set_controls({name: coerced})
                self.controls[name] = coerced
                self._on_state_changed()
                return True
            except Exception as exc:
                logger.error(
                    "Failed to set control %s=%s: %s",
                    name,
                    coerced,
                    exc,
                    exc_info=True,
                )
                return False

    def get_control(self, name: Optional[str] = None):
        """Get a single live camera control value by name or all controls if name is None."""
        with self.lock:
            if name:
                return copy.deepcopy(self.controls.get(name))
            return copy.deepcopy(self.controls)

    # def set_config(self, name: str, value) -> bool:
    #     """Set a single camera configuration parameter. Returns True if updated"""
    #     with self.lock:
    #         current = self.configs.get(name)
    #         if name in self.configs:
    #             if current == value:
    #                 return False
    #             else:
    #                 self.configs[name] = value
    #                 return True
    #         else:
    #             logger.warning("Attempted to set unknown config parameter '%s'", name)
    #             return False

    def set_config(self, name: Union[str, Dict[str, Any]], value=None) -> bool:
        """
        Set camera config(s).

        - set_config("hflip", True)
        - set_config({ "hflip": False, "vflip": False })

        Returns True if at least one config value was updated.
        """
        # ---------- BULK MODE ----------
        if isinstance(name, dict):
            updated = False

            with self.lock:
                for key, val in name.items():
                    if key not in self.configs:
                        logger.warning("Unknown config parameter '%s', skipping", key)
                        continue

                    if self.configs.get(key) != val:
                        self.configs[key] = val
                        updated = True

            return updated

        # ---------- SINGLE MODE ----------
        if not isinstance(name, str):
            raise TypeError("set_config expects str or dict")

        with self.lock:
            if name not in self.configs:
                logger.warning("Attempted to set unknown config parameter '%s'", name)
                return False

            if self.configs[name] == value:
                return False

            self.configs[name] = value
            return True

    def get_config(self, name: Optional[str] = None):
        """Get a single config value by name or all configs if name is None."""
        with self.lock:
            if name:
                return copy.deepcopy(self.configs.get(name))
            return copy.deepcopy(self.configs)

    def get_info(self, name: Optional[str] = None):
        """Get a single info value by name or all infos if name is None."""
        with self.lock:
            if name:
                return copy.deepcopy(self.infos.get(name))
            return copy.deepcopy(self.infos)

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
        """Thread-safe: synchronize current camera controls into STATE dictionary."""
        try:
            metadata = self.picam2.capture_metadata()
            with self.lock:
                for key in self.picam2.camera_controls:
                    if key in metadata:
                        self.controls[key] = metadata[key]
            logger.debug("Camera controls synced from hardware: %s", self.controls)
        except Exception as e:
            logger.warning("Control sync failed: %s", e)

    # ------------------------------------------------------------------
    # Camera configuration functions
    # ------------------------------------------------------------------

    def _get_picam_control_capabilities(self) -> Dict:
        """
        Copy camera controls from picamera2.camera_controls (inherited from
        libcamera) and crop them to reasonable UI ranges.
        """
        controls: Dict = copy.deepcopy(self.picam2.camera_controls)

        if "ExposureTime" in controls:
            min_exposure_time = 100        # microseconds
            max_exposure_time = 100_000    # microseconds
            default_exposure_time = 500
            controls["ExposureTime"] = (
                min_exposure_time,
                max_exposure_time,
                default_exposure_time,
            )

        if "ColourTemperature" in controls:
            min_color_temp = 100           # Kelvin
            max_color_temp = 10_000        # Kelvin
            default_color_temp = None
            controls["ColourTemperature"] = (
                min_color_temp,
                max_color_temp,
                default_color_temp,
            )

        return controls

    def _init_camera_configuration(self):
        self.video_config = self.picam2.create_video_configuration()
        self.picam2.configure(self.video_config)
        self.picam2.start()

    def apply_profile(self, profile: Dict) -> None:
        """
        Apply a validated camera profile dict.
        Profile format:
        {
            "info": {...},
            "config": {...},
            "controls": {...}
        }
        """
        with self.lock:
            self.infos.update(profile.get("info", {}))
            self.configs.update(profile.get("config", {}))
            self.controls.update(profile.get("controls", {}))

        self.reconfigure_video_pipeline()
        self.apply_controls()
        self.sync_ui_settings()

    def _init_ui_settings_from_db(
        self,
        picamera2_controls: Dict,
        ui_settings_db_path: str,
    ) -> Dict:
        if os.path.isfile(ui_settings_db_path):
            try:
                with open(ui_settings_db_path, "r") as f:
                    cam_ctrl_json = json.load(f)
            except Exception as e:
                logger.error(
                    "Failed to extract JSON data from '%s': %s",
                    ui_settings_db_path,
                    e,
                    exc_info=True,
                )
                return {}

            if "sections" not in cam_ctrl_json:
                logger.error("'sections' key not found in cam_ctrl_json!")
                return cam_ctrl_json
        else:
            logger.error("Controls DB file does not exist: %s", ui_settings_db_path)
            return {}

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

                        setting["enabled"] = original_enabled

                        if original_enabled:
                            section_enabled = True
                    else:
                        logger.debug(
                            "Disabling control %s: not found in picamera2_controls",
                            setting_id,
                        )
                        setting["enabled"] = False

                elif source == "still_resolutions_supported":
                    resolution_options = [
                        {
                            "value": i,
                            "label": f"{w} x {h}",
                            "enabled": True,
                        }
                        for i, (w, h) in enumerate(self.still_resolutions_supported)
                    ]
                    setting["options"] = resolution_options
                    section_enabled = True

                    logger.debug(
                        "Updated %s with generated resolutions",
                        setting_id,
                    )

                elif source == "video_resolutions_supported":
                    resolution_options = [
                        {
                            "value": i,
                            "label": f"{w} x {h}",
                            "enabled": True,
                        }
                        for i, (w, h) in enumerate(self.video_resolutions_supported)
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
    #             self.configs[setting_id] = int(setting_value)
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

    #         self.sync_ui_settings()
    #         return setting_value

    #     except Exception as e:
    #         logger.error("Error updating setting '%s' with value '%s': %s", setting_id, setting_value, e)
    #         return None

    def sync_ui_settings(self) -> None:
        """Sync ui_settings with current camera controls and camera configs."""
        for section in self.ui_settings.get("sections", []):
            for setting in section.get("settings", []):
                setting_id = setting.get("id")
                if setting_id in self.controls:
                    setting["value"] = self.controls[setting_id]
                elif setting_id in self.configs:
                    setting["value"] = self.configs[setting_id]

                for child in setting.get("childsettings", []):
                    child_id = child.get("id")
                    if child_id in self.controls:
                        child["value"] = self.controls[child_id]
                    elif child_id in self.configs:
                        child["value"] = self.cofigs[child_id]

        logger.debug("Live controls synced with current controls")

    def apply_controls(self):
        """Thread-safe: apply all current controls (self.controls) to the camera hardware.
        To apply a camera (live) control parameter, no restart of the camera (Picamera2) recording is necessary."""
        with self.lock:
            try:
                self.picam2.set_controls(self.controls)
                logger.debug("Applied controls to hardware: %s", self.controls)
            except Exception as e:
                logger.error("Error applying profile controls: %s", e, exc_info=True)
                return False

        # Update UI controls outside the lock to avoid blocking
        self.sync_ui_settings()
        logger.info("All profile controls applied successfully")
        return True

    def _generate_video_resolutions_supported(self) -> List[Tuple[int, int]]:
        resolutions = set()

        for mode in self.sensor_modes_supported:
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
            for mode in self.sensor_modes_supported
            if mode["size"][0] >= tw and mode["size"][1] >= th
        ]

        if not candidates:
            raise ValueError("No suitable sensor mode found")

        # Prioritize smallest suitable resolution
        return min(candidates, key=lambda m: m["size"][0] * m["size"][1])

    def reconfigure_video_pipeline(self) -> bool:
        """Reconfigure video pipeline based on current (camera) configs."""
        if self.states["is_video_recording"] or self.states["is_capturing_still_image"]:
            return False

        rec = self.video_resolutions_supported[self.configs["recording_resolution"]]
        stream = self.video_resolutions_supported[self.configs["streaming_resolution"]]

        main_size, lores_size = (rec, stream) if rec[0]*rec[1] >= stream[0]*stream[1] else (stream, rec)
        self.main_stream, self.lores_stream = ("recording", "streaming") if rec[0]*rec[1] >= stream[0]*stream[1] else ("streaming", "recording")

        mode = self._find_best_sensor_mode(main_size)
        was_streaming = self.states["is_video_streaming"]
        if was_streaming:
            self.stop_streaming()

        with self.lock:
            self.picam2.stop()
            self.picam2.configure(self.picam2.create_video_configuration(
                main={"size": main_size},
                lores={"size": lores_size},
                transform=Transform(hflip=self.configs["hflip"], vflip=self.configs["vflip"]),
                sensor={"output_size": mode["size"], "bit_depth": mode["bit_depth"]}
            ))
            self.picam2.start()

        if was_streaming:
            self.start_streaming()

        self.configs["sensor_mode"] = self.sensor_modes_supported.index(mode)
        logger.info("Video pipeline reconfigured: main=%s lores=%s", main_size, lores_size)
        return True

    def get_recording_stream(self) -> str:
        return "main" if self.main_stream == "recording" else "lores"

    def get_streaming_stream(self) -> str:
        return "main" if self.main_stream == "streaming" else "lores"

    def get_recording_resolution(self) -> tuple:
        return self.video_resolutions_supported[self.configs["recording_resolution"]]

    def get_streaming_resolution(self) -> tuple:
        return self.video_resolutions_supported[self.configs["streaming_resolution"]]

    def set_recording_resolution(self, resolution_index: int) -> None:
        self.configs["recording_resolution"] = int(resolution_index)
        self.reconfigure_video_pipeline()

    def set_streaming_resolution(self, resolution_index: int) -> None:
        self.configs["streaming_resolution"] = int(resolution_index)
        self.reconfigure_video_pipeline()

    def reset_to_default(self) -> bool:
        """Reset camera configuration and controls to default values and apply them."""

        # --- 1. Reset persistent configuration ---
        self.configs.update({
            "hflip": False,
            "vflip": False,
            "saveRAW": False,
            "sensor_mode": 0,  # Setze explizit ersten Modus
            "still_capture_resolution": 0,
            "recording_resolution": 0,
            "streaming_resolution": 0,
        })
        logger.debug("Reset config to defaults: %s", self.configs)

        # --- 2. Reset all camera controls to known defaults ---
        self.controls.clear()
        self.controls.update(DEFAULT_CONTROLS)
        logger.debug("Reset controls to defaults: %s", self.controls)

        # --- 3. Reconfigure video pipeline based on defaults ---
        if not self.reconfigure_video_pipeline():
            logger.warning("Failed to reconfigure video pipeline during reset")

        # --- 4. Apply control defaults to camera hardware ---
        try:
            self.apply_controls()
        except Exception as e:
            logger.error("Failed to apply default camera controls: %s", e, exc_info=True)
            return False

        logger.info("Camera successfully reset to default configuration and controls")
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

            for index, mode in enumerate(self.sensor_modes_supported):
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

    def _generate_still_resolutions_supported(self) -> List[tuple]:
        """Precompute available resolutions based on sensor modes."""
        if not self.sensor_modes_supported:
            logger.warning("No sensor modes available!")
            return []

        resolutions = sorted(
            set(mode["size"] for mode in self.sensor_modes_supported if "size" in mode),
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
            if self.states["is_video_streaming"]:
                logger.info("Skip starting stream, already active")
                return

            stream_name = self.get_streaming_stream()
            self.picam2.start_recording(
                self.encoder_stream,
                output=self.output_stream,
                name=stream_name,
            )
        self._set_state("is_video_streaming", True)
        logger.info("Streaming started on '%s' stream", stream_name)

    def stop_streaming(self):
        with self.lock:
            if not self.states["is_video_streaming"]:
                logger.info("Skip stopping stream, no active stream")
                return

            self.picam2.stop_recording()
        
        self._set_state("is_video_streaming", False)
        logger.info("Streaming stopped")

    def start_recording(self, filename: str):
        with self.lock:
            if self.states["is_video_recording"]:
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

        self._set_state("is_video_recording", True)
        logger.info(f"Recording {filename} started")
        return True, filename

    def stop_recording(self):
        with self.lock:
            if not self.states["is_video_recording"]:
                logger.info("Skip stopping recording, no active recording")
                return False

            _filename = self.filename_recording

            # Stoppe Recording
            self.picam2.stop_recording()
            self.output_recording = None
            self.filename_recording = None

        self._set_state("is_video_recording", False)
        logger.info(f"Recording {_filename} stopped")

        return True

    #-----
    # Camera Capture Functions
    #-----
    def capture_still(self, filename: str, raw: bool = False) -> Optional[str]:
        filepath = os.path.join(self.upload_folder, filename)
        sucess = False

        # Sperre gleich zu Beginn für alle _state-Zugriffe
        with self.lock:
            if self.states["is_capturing_still_image"]:
                logger.warning(
                    "Skip capturing still image '%s', another capture is active", filename
                )
                return None
            # Markiere Capture als aktiv
            self.states["is_capturing_still_image"] = True
            was_streaming = self.states["is_video_streaming"]

        try:
            # Auflösungen ermitteln
            still_index = self.configs["still_capture_resolution"]
            still_resolution = self.still_resolutions_supported[still_index]

            rec_index = self.configs["recording_resolution"]
            recording_resolution = self.video_resolutions_supported[rec_index]

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
                self.configs["sensor_mode"] = self.sensor_modes_supported.index(mode)
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
            success = True

        except Exception as e:
            logger.error("Error capturing still image '%s': %s", filepath, e, exc_info=True)

        finally:
            # Lock nur für _state-Update verwenden
            with self.lock:
                self.states["is_capturing_still_image"] = False

            self.reconfigure_video_pipeline()
            
            # Streaming außerhalb des Locks starten (kann blockieren)
            if was_streaming:
                self.start_streaming()
            
            if success:
                return filepath
            else:
                return None

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