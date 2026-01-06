import os
import json
import threading
import logging
import copy
from typing import Dict, List, Optional
from picamera2 import Picamera2
from camera_object import CameraObject

# Flask-SocketIO Imports
from flask_socketio import SocketIO, emit, join_room, leave_room

logger = logging.getLogger(__name__)

####################
# CameraManager Class
####################

class CameraManager:

    DEFAULT_CONFIG = {
        "hflip": False,
        "vflip": False,
        "saveRAW": False,
        "sensor_mode": 0,
        "still_capture_resolution": 0,
        "recording_resolution": 0,
        "streaming_resolution": 0,
    }

    DEFAULT_CONTROLS = {
    "AfMode": 0,
    "LensPosition": 1.0,
    "AfRange": 0,
    "AfSpeed": 0,
    "ExposureTime": 33000,
    "AnalogueGain": 1.12,
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

    def __init__(
        self,
        camera_module_info_path: str,
        camera_active_profile_path: str,
        media_upload_folder: str,
        camera_ui_settings_db_path: str,
        camera_profile_folder: str,
        socketio: SocketIO,
    ):
        """
        :param camera_module_info_path: Path to camera-module-info.json
        :param camera_active_profile_path: Path to camera-active-profile.json
        :param media_upload_folder: Path to the folder where photos and videos are stored (media gallery)
        :camera_ui_settings_db_path: Path to file storing camera controls parameter, controllable via webui
        :camera_profile_folder: Path to folder storing files of saved camera profiles (.json)
        :socketio: Flask-SocketIO object used to snyc camera settings
        """

        self.camera_active_profile_path = camera_active_profile_path
        self.media_upload_folder = media_upload_folder
        self.camera_ui_settings_db_path = camera_ui_settings_db_path
        self.camera_profile_folder = camera_profile_folder
        self.socketio = socketio

        self.connected_cameras: List[dict] = []
        self.cameras: Dict[int, CameraObject] = {}
        self.lock = threading.Lock()


        try:
            with open(camera_module_info_path, "r") as f:
                self.camera_module_info = json.load(f)
        except Exception as exc:
            logger.warning(
                "Failed to load camera module info from '%s': %s",
                camera_module_info_path,
                exc,
            )
            self.camera_module_info = {}
        
        self._load_active_profiles_file(self.camera_active_profile_path)


    def _detect_connected_cameras(self) -> List[dict]:
        """
        Detect currently connected cameras using Picamera2 and
        determine whether they are Raspberry Pi cameras.
        """
        currently_connected = []

        for connected_camera in Picamera2.global_camera_info():
            matching_module = next(
                (
                    module
                    for module in self.camera_module_info.get("camera_modules", [])
                    if module["sensor_model"] == connected_camera["Model"]
                ),
                None,
            )

            is_pi_cam = bool(matching_module and matching_module.get("is_pi_cam", False))

            if is_pi_cam:
                logger.info(
                    "Detected Raspberry Pi Camera: model=%s",
                    connected_camera["Model"],
                )
            else:
                logger.info(
                    "Detected non-Pi camera or unknown model: model=%s",
                    connected_camera["Model"],
                )

            camera_info = {
                "Num": connected_camera["Num"],
                "Model": connected_camera["Model"],
                "Is_Pi_Cam": is_pi_cam,
                "Has_Config": False,
                "Config_Location": f"{connected_camera['Model']}_default.json",
            }

            currently_connected.append(camera_info)

        return currently_connected

    def _make_state_callback(self, camera):
        def callback():
            # TODO anpassen auf neue struktur
            state = camera.get_settings()
            room = f"camera_{camera.camera_num}"
            # print(f"DEBUG: _make_state_callback, state: {state}, room: {room}")
            self.socketio.emit(
                "camera_state",
                {"camera_num": camera.camera_num, "state": state},
                room=room,
            )
        return callback

    def init_cameras(self):
        """Create CameraObject instances for all connected cameras."""
        self.connected_cameras = self._detect_connected_cameras()
        self._update_active_profiles_file(self.connected_cameras)

        for cam_info in self.connected_cameras:
            try:
                camera_obj = CameraObject(
                    cam_info,
                    self.camera_module_info,
                    self.media_upload_folder,
                    self.camera_ui_settings_db_path,
                )

                camera_obj._on_state_changed = self._make_state_callback(camera_obj)
                self.cameras[cam_info["Num"]] = camera_obj
                # apply/load active profile -> set camera configs and controls
                self._load_active_profile(cam_info["Num"])
            except Exception as e:
                logger.error("Failed to initialize camera %s: %s", cam_info["Num"], e)

        for key, camera in self.cameras.items():
            logger.info("Initialized camera %s: %s", key, camera.camera_info)

    def join_camera_room(self, sid, camera_num):
        """ SocketIO-Client joins camera room """
        room = f"camera_{camera_num}"
        join_room(room, sid=sid)

    def leave_camera_room(self, sid, camera_num):
        """ SocketIO-Client leaves camera room """
        room = f"camera_{camera_num}"
        leave_room(room, sid=sid)

    def get_camera(self, cam_num: int) -> Optional[CameraObject]:
        """
        Return the CameraObject instance for the given camera number.
        Returns None if the request is invalid or the camera does not exist.
        """
        if not isinstance(cam_num, int):
            logger.warning("Invalid camera number requested: %r", cam_num)
            return None

        return self.cameras.get(cam_num)

    def list_cameras(self) -> Optional[List[dict]]:
        """
        Return a list of all connected cameras as dictionaries.
        Returns None if no cameras are present or the internal state is invalid.
        """
        if not isinstance(self.connected_cameras, list):
            logger.error("Invalid internal state: connected_cameras is not a list.")
            return None

        if not self.connected_cameras:
            logger.info("No connected cameras detected.")
            return None

        return self.connected_cameras

    # -------------- Camera Profile Management -------------------
    def _load_active_profiles_file(self, filepath: str):
        """
        Load the file storing informations about the active camera profiles
        or initalize file, if not already existing
        """
        if os.path.exists(filepath):
            try:
                with open(filepath, "r") as f:
                    self.camera_active_profile = json.load(f)
            except Exception as exc:
                logger.error(
                    "Failed to load active camera profile file '%s': %s",
                    filepath,
                    exc,
                    exc_info=True,
                )
                self.camera_active_profile = {"cameras": []}
        else:
            self.camera_active_profile = {"cameras": []}


    def _update_active_profiles_file(self, connected_cameras: List[dict]):
        """
        Compare currently connected cameras with the file storing informations
        about the active camera profiles (camera_active_profile_path)
        and update its settings/values if necessary.
        """
        existing_lookup = {
            cam["Num"]: cam for cam in self.camera_active_profile.get("cameras", [])
        }

        updated_cameras = []

        for cam in self.connected_cameras:
            cam_num = cam["Num"]

            if cam_num in existing_lookup:
                config_cam = existing_lookup[cam_num]

                if (
                    config_cam["Model"] != cam["Model"]
                    or config_cam.get("Is_Pi_Cam") != cam.get("Is_Pi_Cam")
                ):
                    logger.info(
                        "Camera %s changed (model or Pi Camera flag). Updating config.",
                        cam_num,
                    )
                    updated_cameras.append(cam)
                else:
                    updated_cameras.append(config_cam)
            else:
                logger.info("New camera detected and added to config: %s", cam)
                updated_cameras.append(cam)

        self.camera_active_profile = {"cameras": updated_cameras}
        with open(self.camera_active_profile_path, "w") as f:
            json.dump(self.camera_active_profile, f, indent=4)

        self.connected_cameras = updated_cameras
        return updated_cameras

    def load_profile(self, camera_num: int, profile_filename: str) -> bool:
        camera = self.get_camera(camera_num)
        if not camera:
            return False

        if profile_filename == self.camera_active_profile_path:
            profile_path = profile_filename
        else:
            profile_path = os.path.join(self.camera_profile_folder, profile_filename)
        if not os.path.exists(profile_path):
            logger.warning("Profile not found: %s", profile_path)
            return False

        try:
            with open(profile_path, "r") as f:
                profile = json.load(f)

            camera.set_config(profile["config"])
            camera.reconfigure_video_pipeline()
            camera.set_control(profile["controls"])

            self._set_active_profile(camera_num, profile_filename)
            return True

        except Exception as e:
            logger.error("Failed to load profile: %s", e, exc_info=True)
            return False

    def _set_active_profile(self, camera_num: int, filename: str):
        for cam in self.camera_active_profile["cameras"]:
            if cam["Num"] == camera_num:
                cam["Has_Config"] = True
                cam["Config_Location"] = filename
                break

        with open(self.camera_active_profile_path, "w") as f:
            json.dump(self.camera_active_profile, f, indent=4)

    def save_profile(self, camera_num: int, profile_name: str) -> bool:
        camera = self.get_camera(camera_num)
        if not camera:
            return False

        profile = {
            "info": camera.get_info(),
            "config": camera.get_config(),
            "controls": camera.get_control(),
        }

        path = os.path.join(self.camera_profile_folder, f"{profile_name}.json")
        with open(path, "w") as f:
            json.dump(profile, f, indent=2)

        self._set_active_profile(camera_num, f"{profile_name}.json")
        return True

    def _load_active_profile(self, camera_num: int) -> None:
        """
        Load and apply the active profile for a specific camera
        based on camera-active-profile.json.
        """

        # Load camera-active-profile.json if it exists
        if not os.path.exists(self.camera_active_profile_path):
            logger.info(
                "No active profile file found (%s). Using defaults for camera %s.",
                self.camera_active_profile_path,
                camera_num,
            )
            return

        try:
            with open(self.camera_active_profile_path, "r") as f:
                self.camera_active_profile = json.load(f)
        except Exception as exc:
            logger.error(
                "Failed to read active profile file '%s': %s",
                self.camera_active_profile_path,
                exc,
                exc_info=True,
            )
            return

        camera_entry = next(
            (c for c in self.camera_active_profile.get("cameras", []) if c.get("Num") == camera_num),
            None,
        )

        if not camera_entry:
            logger.info(
                "No active profile entry for camera %s. Using defaults.",
                camera_num,
            )
            return

        if not camera_entry.get("Has_Config"):
            logger.info(
                "Camera %s has no active profile configured. Using defaults.",
                camera_num,
            )
            return

        profile_filename = camera_entry.get("Config_Location")
        if not profile_filename:
            logger.warning(
                "Camera %s marked Has_Config but Config_Location is empty.",
                camera_num,
            )
            return

        if self.load_profile(camera_num, profile_filename):
            logger.debug(
                "Loaded active profile '%s' for camera %s",
                profile_filename,
                camera_num,
            )
        else:
            logger.error(
                "Failed to load active profile '%s' for camera %s",
                profile_filename,
                camera_num,
            )

    def reset_camera_to_defaults(self, camera_num: int) -> bool:
        camera = self.get_camera(camera_num)
        if not camera:
            return False

        logger.info("Resetting camera %s to default configuration", camera_num)

        was_streaming = camera.states["is_video_streaming"]

        # -------------------------------------------------
        # 1. Stop runtime activity
        # -------------------------------------------------
        try:
            camera.stop_streaming()
            camera.stop_recording()
        except Exception as exc:
            logger.error(
                "Failed to stop runtime activity for camera %s: %s",
                camera_num,
                exc,
                exc_info=True,
            )
            return False

        # -------------------------------------------------
        # 2. Apply default configs (canonical state only)
        # -------------------------------------------------
        updated_configs = camera.set_config(copy.deepcopy(CameraManager.DEFAULT_CONFIG))
        if updated_configs:
            try:
                camera.reconfigure_video_pipeline()
            except Exception as exc:
                logger.error(
                    "Failed to reconfigure pipeline for camera %s after config reset: %s",
                    camera_num,
                    exc,
                    exc_info=True,
                )
                return False

        # -------------------------------------------------
        # 3. Apply default controls (live hardware)
        # -------------------------------------------------
        success = camera.set_control(copy.deepcopy(CameraManager.DEFAULT_CONTROLS))
        if not success:
            logger.error(
                "Failed to apply default controls to camera %s",
                camera_num,
            )
            return False

        # -------------------------------------------------
        # 4. Restart streaming
        # -------------------------------------------------
        if was_streaming:
            camera.start_streaming()

        logger.info("Camera %s successfully reset to defaults", camera_num)
        return True